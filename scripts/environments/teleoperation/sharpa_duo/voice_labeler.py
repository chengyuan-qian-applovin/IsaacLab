# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Always-on voice labeling with OpenAI Whisper: say "success" or "failure".

Audio is 16 kHz mono S16 from one of three sources: the machine's own
microphone via an ``arecord`` subprocess, the headset microphone streamed from a
page this process serves (:mod:`headset_mic`), or the same headset microphone
relayed by the always-on control app (:mod:`teleop_app`), which keeps capture
alive across teleop restarts.

A reader thread segments the stream into utterances with a simple energy gate
(ambient noise is measured at startup), a worker thread transcribes each
utterance with a local Whisper model and turns keyword matches into events, and
the teleop loop drains the events with :meth:`VoiceLabeler.poll`. Every
transcription is printed, so mis-hearings are visible immediately.

Recognized commands: "success"/"succeed…" → ``"success"``; "fail…" →
``"failure"``; "align" (or Whisper's common mis-hearing "a line") →
``"align"``; "play"/"start…" → ``"play"``; "reset" → ``"reset"``;
"next"/"skip" → ``"next"`` (advance to the next scene in the list). An
utterance matching more than one is ignored (announced on the console).
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
from dataclasses import dataclass

import numpy as np

_SAMPLE_RATE = 16000
_CHUNK_S = 0.1  # reader granularity
_CHUNK_BYTES = int(_SAMPLE_RATE * _CHUNK_S) * 2  # S16LE mono
_HIGHPASS_HZ = 80.0  # kill DC/infrasonic wander (laptop mics drift hugely below ~20 Hz)

_COMMAND_RES = (
    ("success", re.compile(r"\b(success\w*|succeed\w*)\b")),
    ("failure", re.compile(r"\bfail\w*\b")),
    # "a line" is Whisper's most common mis-hearing of a spoken "align".
    ("align", re.compile(r"\b(align\w*|a line)\b")),
    ("play", re.compile(r"\b(play\w*|start\w*)\b")),
    ("stop", re.compile(r"\b(stop|pause)\b")),
    ("reset", re.compile(r"\breset\w*\b")),
    ("next", re.compile(r"\b(next|skip)\b")),
    ("adjust", re.compile(r"\b(adjust|edit)\s+(objects?|poses?)\b")),
    ("done", re.compile(r"\b(done|finish\w*)\b")),
)


def parse_label(text: str) -> str | None:
    """Map a transcription to a command word, or None.

    Recognized commands: success, failure, align, play, stop, reset, next,
    adjust, done. Returns None when nothing matches or the utterance is
    contradictory (more than one command recognized at once).
    """
    text = text.lower()
    matches = [name for name, regex in _COMMAND_RES if regex.search(text)]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class VoiceEvent:
    """One closed utterance: what was transcribed and which command it mapped to.

    Every utterance the energy gate captures produces an event, including ones
    that match no command, so the caller can show the operator what was heard.

    Attributes:
        text: The transcription, or ``""`` when audio was captured but Whisper
            found no intelligible speech.
        command: The parsed command (success / failure / align / play / stop /
            reset / next / adjust / done), or None when the utterance matched none of them (or
            matched several, which is treated as contradictory).
    """

    text: str
    command: str | None


class VoiceLabeler:
    """Background success/failure keyword listener (see module docstring).

    Args:
        model_name: Whisper model to load (e.g. ``base.en``, ``small.en``).
        device: torch device for Whisper. Default ``cpu`` so transcription never
            competes with the simulation and CloudXR encode for the GPU.
        mic_device: ALSA capture device passed to ``arecord -D``;
            ``"quest"`` / ``"avp"`` (optionally ``"quest:<port>"`` /
            ``"avp:<port>"``, default port 8444) to receive audio from the
            headset instead — see :mod:`headset_mic` (Quest streams from a
            browser page, the AVP client app streams natively); or ``"hub"``
            (optionally ``"hub:<port>"`` or ``"hub:<host>:<port>"``, default
            ``127.0.0.1:8500``) to consume the relay of the always-on teleop
            app — see :mod:`teleop_app`. With any headset source, calibration
            waits until audio starts flowing and measures ambient from its
            first 1.5 s, so stay quiet right after the mic starts streaming.
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

        self._proc = None
        self._headset = None
        kind, _, spec = mic_device.partition(":")
        if kind == "hub":
            from headset_mic import MicHubClient

            # "hub", "hub:<port>" or "hub:<host>:<port>".
            host, _, port = spec.rpartition(":")
            self._headset = MicHubClient(host=host or "127.0.0.1", port=int(port) if port else 8500)
        elif kind in ("quest", "avp"):
            from headset_mic import HeadsetMicServer

            self._headset = HeadsetMicServer(port=int(spec) if spec else 8444, client=kind)
        else:
            self._proc = subprocess.Popen(
                ["arecord", "-q", "-D", mic_device, "-f", "S16_LE", "-r", str(_SAMPLE_RATE), "-c", "1", "-t", "raw"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        self._events: queue.Queue[VoiceEvent] = queue.Queue()
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
        self._reader = threading.Thread(target=self._guarded(self._read_loop), daemon=True, name="voice-reader")
        self._worker = threading.Thread(
            target=self._guarded(self._transcribe_loop), daemon=True, name="voice-transcriber"
        )
        self._reader.start()
        self._worker.start()

    def poll(self) -> VoiceEvent | None:
        """Return the next captured utterance, or None when none is pending.

        An event is emitted for every utterance the energy gate closes, so
        callers see mis-hearings too; act only on events whose
        :attr:`VoiceEvent.command` is not None.
        """
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def _guarded(self, fn):
        """A voice thread dying would silently kill voice commands — report it loudly."""

        def run():
            try:
                fn()
            except Exception:
                import traceback

                traceback.print_exc()
                print(f"[VOICE] ERROR: {fn.__name__} crashed (see traceback above); voice commands are DEAD.")

        return run

    def take_peak(self) -> float:
        """Return the loudest chunk RMS since the last call and reset the meter."""
        peak, self._peak_rms = self._peak_rms, 0.0
        return peak

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        if self._headset is not None:
            self._headset.close()

    # -- internals -----------------------------------------------------------

    def _read_chunk(self) -> np.ndarray | None:
        if self._headset is not None:
            raw = self._headset.read_chunk()  # blocks until the headset streams
            if raw is None:
                return None
        else:
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
                    initial_prompt="Robot teleoperation commands: success, failure, align, play, stop, reset, next.",
                )
            except Exception as exc:
                print(f"[VOICE] Transcription failed: {exc}")
                continue
            text = result["text"].strip()
            if not text:
                print(f"[VOICE] Heard a {len(clip) / _SAMPLE_RATE:.1f} s sound but no intelligible speech.")
                self._events.put(VoiceEvent("", None))
                continue
            label = parse_label(text)
            if label is None:
                print(f'[VOICE] Heard: "{text}" (no label)')
            else:
                print(f'[VOICE] Heard: "{text}" -> {label.upper()}')
            self._events.put(VoiceEvent(text, label))
