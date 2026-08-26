# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Always-on voice labeling with OpenAI Whisper: say "success" or "failure".

Audio comes from the machine's microphone via an ``arecord`` subprocess (16 kHz
mono S16), so no Python audio stack is needed — the operator just has to be
within speaking range of the machine (the headset microphone is not streamed to
the server by this teleop stack).

A reader thread segments the stream into utterances with a simple energy gate
(ambient noise is measured at startup), a worker thread transcribes each
utterance with a local Whisper model and turns keyword matches into events, and
the teleop loop drains the events with :meth:`VoiceLabeler.poll`. Every
transcription is printed, so mis-hearings are visible immediately.

Recognized commands: any word starting with "success"/"succeed" → ``"success"``;
any word starting with "fail" → ``"failure"``; "align" (or Whisper's common
mis-hearing "a line") → ``"align"``. An utterance matching more than one is
ignored (announced on the console).
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading

import numpy as np

_SAMPLE_RATE = 16000
_CHUNK_S = 0.1  # reader granularity
_CHUNK_BYTES = int(_SAMPLE_RATE * _CHUNK_S) * 2  # S16LE mono
_HIGHPASS_HZ = 80.0  # kill DC/infrasonic wander (laptop mics drift hugely below ~20 Hz)

_SUCCESS_RE = re.compile(r"\b(success\w*|succeed\w*)\b")
_FAILURE_RE = re.compile(r"\bfail\w*\b")
# "a line" is Whisper's most common mis-hearing of a spoken "align".
_ALIGN_RE = re.compile(r"\b(align\w*|a line)\b")


def parse_label(text: str) -> str | None:
    """Map a transcription to ``"success"`` / ``"failure"`` / ``"align"`` / ``None``.

    Returns None when nothing matches or the utterance is contradictory
    (more than one command recognized at once).
    """
    text = text.lower()
    matches = [
        name
        for name, regex in (("success", _SUCCESS_RE), ("failure", _FAILURE_RE), ("align", _ALIGN_RE))
        if regex.search(text)
    ]
    return matches[0] if len(matches) == 1 else None


class VoiceLabeler:
    """Background success/failure keyword listener (see module docstring).

    Args:
        model_name: Whisper model to load (e.g. ``base.en``, ``small.en``).
        device: torch device for Whisper. Default ``cpu`` so transcription never
            competes with the simulation and CloudXR encode for the GPU.
        mic_device: ALSA capture device passed to ``arecord -D``.
        min_utterance_s: Shortest speech burst considered an utterance.
        silence_s: Trailing silence that closes an utterance.
        max_utterance_s: Utterances are clipped to this length.
    """

    def __init__(
        self,
        model_name: str = "base.en",
        device: str = "cpu",
        mic_device: str = "default",
        min_utterance_s: float = 0.25,
        silence_s: float = 0.5,
        max_utterance_s: float = 6.0,
    ):
        import whisper

        print(f"[VOICE] Loading Whisper '{model_name}' on {device} ...")
        self._model = whisper.load_model(model_name, device=device)
        self._fp16 = device != "cpu"
        self._min_chunks = max(1, round(min_utterance_s / _CHUNK_S))
        self._silence_chunks = max(1, round(silence_s / _CHUNK_S))
        self._max_chunks = max(self._min_chunks, round(max_utterance_s / _CHUNK_S))

        self._proc = subprocess.Popen(
            ["arecord", "-q", "-D", mic_device, "-f", "S16_LE", "-r", str(_SAMPLE_RATE), "-c", "1", "-t", "raw"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._events: queue.Queue[str] = queue.Queue()
        self._clips: queue.Queue[np.ndarray] = queue.Queue()
        self._stop = threading.Event()
        #: Energy gate, set once ambient calibration finishes (None before).
        self.threshold: float | None = None
        #: Loudest chunk RMS since the last :meth:`take_peak` call (level metering).
        self._peak_rms = 0.0
        # One-pole high-pass y[n] = a*(y[n-1] + x[n] - x[n-1]); filter state is
        # carried across chunks so there is no per-chunk transient.
        a = float(np.exp(-2.0 * np.pi * _HIGHPASS_HZ / _SAMPLE_RATE))
        self._hp_b, self._hp_a = np.array([a, -a]), np.array([1.0, -a])
        self._hp_zi = np.zeros(1)
        self._reader = threading.Thread(target=self._read_loop, daemon=True, name="voice-reader")
        self._worker = threading.Thread(target=self._transcribe_loop, daemon=True, name="voice-transcriber")
        self._reader.start()
        self._worker.start()

    def poll(self) -> str | None:
        """Return the next ``"success"``/``"failure"`` event, or None."""
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def take_peak(self) -> float:
        """Return the loudest chunk RMS since the last call and reset the meter."""
        peak, self._peak_rms = self._peak_rms, 0.0
        return peak

    def close(self) -> None:
        self._stop.set()
        self._proc.terminate()

    # -- internals -----------------------------------------------------------

    def _read_chunk(self) -> np.ndarray | None:
        data = self._proc.stdout.read(_CHUNK_BYTES)
        if not data or len(data) < _CHUNK_BYTES:
            return None
        raw = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return self._highpass(raw)

    def _highpass(self, x: np.ndarray) -> np.ndarray:
        """Streaming one-pole high-pass; laptop mics carry near-full-scale infrasonic
        wander that would otherwise saturate the energy gate (and clip speech)."""
        from scipy.signal import lfilter

        y, self._hp_zi = lfilter(self._hp_b, self._hp_a, x, zi=self._hp_zi)
        return y.astype(np.float32)

    def _read_loop(self) -> None:
        # Calibrate the energy gate on ambient noise.
        ambient = []
        for _ in range(15):
            chunk = self._read_chunk()
            if chunk is None:
                print("[VOICE] Microphone stream ended during calibration; voice labeling disabled.")
                return
            ambient.append(np.sqrt(np.mean(chunk**2)))
        threshold = max(3.0 * float(np.median(ambient)), 0.002)
        if threshold > 0.2:
            print(
                f"[VOICE] WARNING: ambient level is very high (gate {threshold:.3f}); the microphone looks"
                " saturated or misrouted — voice labels are unlikely to trigger. Check capture gain"
                " (alsamixer) or pass a different --mic_device."
            )
        print(f"[VOICE] Listening (energy gate {threshold:.4f}). Say 'success' or 'failure' to label an episode.")
        self.threshold = threshold

        pre_roll: list[np.ndarray] = []
        utterance: list[np.ndarray] = []
        quiet = 0
        while not self._stop.is_set():
            chunk = self._read_chunk()
            if chunk is None:
                print("[VOICE] Microphone stream ended; voice labeling disabled.")
                return
            # Hysteresis: mid-utterance dips only need half the onset level, so a
            # sentence with soft syllables is not split into fragments.
            gate = threshold if not utterance else 0.5 * threshold
            rms = float(np.sqrt(np.mean(chunk**2)))
            self._peak_rms = max(self._peak_rms, rms)
            loud = rms > gate
            if not utterance:
                if loud:
                    utterance = [*pre_roll, chunk]
                    quiet = 0
                else:
                    pre_roll = [*pre_roll[-2:], chunk]  # keep ~0.3 s of context
            else:
                utterance.append(chunk)
                quiet = 0 if loud else quiet + 1
                if quiet >= self._silence_chunks or len(utterance) >= self._max_chunks:
                    speech_chunks = len(utterance) - quiet - len(pre_roll)
                    if speech_chunks >= self._min_chunks:
                        self._clips.put(np.concatenate(utterance))
                    utterance = []
                    pre_roll = []

    def _transcribe_loop(self) -> None:
        while not self._stop.is_set():
            try:
                clip = self._clips.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self._model.transcribe(
                    clip,
                    language="en",
                    fp16=self._fp16,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    initial_prompt="Robot teleoperation commands: success, failure, align.",
                )
            except Exception as exc:
                print(f"[VOICE] Transcription failed: {exc}")
                continue
            text = result["text"].strip()
            if not text:
                print(f"[VOICE] Heard a {len(clip) / _SAMPLE_RATE:.1f} s sound but no intelligible speech.")
                continue
            label = parse_label(text)
            if label is None:
                print(f'[VOICE] Heard: "{text}" (no label)')
            else:
                print(f'[VOICE] Heard: "{text}" -> {label.upper()}')
                self._events.put(label)
