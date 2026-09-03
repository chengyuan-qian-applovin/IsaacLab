# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Always-on voice commands: say "success", "failure", "reset", ...

Audio is 16 kHz mono S16 from one of three sources: the machine's own
microphone via an ``arecord`` subprocess, the headset microphone streamed from a
page this process serves (:mod:`headset_mic`), or the same headset microphone
relayed by the always-on control app (:mod:`teleop_app`), which keeps capture
alive across teleop restarts.

A reader thread segments the stream into utterances with a simple energy gate
(ambient noise is measured at startup), a worker thread transcribes each
utterance and turns keyword matches into events, and the teleop loop drains the
events with :meth:`VoiceLabeler.poll`. Every transcription is printed, so
mis-hearings are visible immediately.

Transcription is OpenAI Whisper (``openai-whisper``), tuned for a closed
vocabulary of short commands rather than dictation:

* the decoder is primed with the command vocabulary as ``initial_prompt`` —
  Whisper is an autoregressive language model, so this prior is what turns an
  ambiguous half-second of audio into "reset" rather than "we're set";
* beam search (not greedy) — a few times slower but markedly more reliable on
  one-word clips, and still well under 0.5 s on a CPU for ``base.en``;
* Whisper's own confidence is honoured: a segment it flags as probably not
  speech (``no_speech_prob``) or decodes with a poor average log-probability is
  reported as "no intelligible speech" instead of being trusted — on noise the
  greedy decode otherwise happily invents "All right. All right.";
* output that cannot be real speech (impossible word rate, or the looping
  ``"reset, reset, reset, ..."`` an autoregressive decoder produces on a clip
  cut mid-word) is rejected by :func:`hallucination_reason`;
* a short utterance that matches no command exactly is matched fuzzily
  against the vocabulary, so "we're set" or "a lie" still count.

Recognized commands: "success"/"succeed…" → ``"success"``; "fail…" →
``"failure"``; "align" → ``"align"``; "play"/"start…" → ``"play"``;
"stop" → ``"stop"``; "reset" → ``"reset"``; "next"/"skip" →
``"next"`` (advance to the next scene in the list); "initial" →
``"adjust"``; "done"/"finish…" → ``"done"``. An utterance matching more than
one is ignored (announced on the console). A command said several times in one
breath ("reset, reset, reset") is emitted once per occurrence.
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
_MAX_WORDS_PER_S = 6.0  # fast speech is ~4 words/s; more than this in a clip is not speech
_MAX_COMPRESSION_RATIO = 2.4  # Whisper's own fallback threshold for a looping decoder
_BEAM_SIZE = 5
_NO_SPEECH_THRESHOLD = 0.6  # Whisper's default: above this the segment is probably not speech
_LOGPROB_THRESHOLD = -1.0  # Whisper's default: below this the decode is not trusted
_FUZZY_MAX_WORDS = 3  # only short utterances are fuzzy-matched; sentences must match exactly
_FUZZY_MIN_RATIO = 0.8  # difflib ratio a word must reach to count as a command
# Prompt text is an example transcript in the style expected, which is how
# Whisper's prompting works: it biases the decoder toward these spellings.
_INITIAL_PROMPT = "Reset. Align. Play. Start. Stop. Next. Skip. Success. Failure. Initial. Done. Finish."

_COMMAND_RES = (
    ("success", re.compile(r"\b(success\w*|succeed\w*)\b")),
    ("failure", re.compile(r"\bfail\w*\b")),
    # "a line" / "a lie" are the common mis-hearings of a spoken "align".
    ("align", re.compile(r"\b(align\w*|a line|a lie)\b")),
    ("play", re.compile(r"\b(play\w*|start\w*)\b")),
    ("stop", re.compile(r"\bstop\b")),
    ("reset", re.compile(r"\breset\w*\b")),
    ("next", re.compile(r"\b(next|skip)\b")),
    # "initial" (any suffix): open the initial-pose editor (internal command name "adjust").
    ("adjust", re.compile(r"\binitial\w*\b")),
    ("done", re.compile(r"\b(done|finish\w*)\b")),
)
# Spoken forms for fuzzy matching, when the exact patterns above find nothing.
_COMMAND_WORDS = {
    "success": ("success",),
    "failure": ("failure", "fail"),
    "align": ("align",),
    "play": ("play", "start"),
    "stop": ("stop",),
    "reset": ("reset",),
    "next": ("next", "skip"),
    "adjust": ("initial",),
    "done": ("done", "finish"),
}


def hallucination_reason(text: str, segments: list[dict], clip_s: float) -> str | None:
    """Return why ``text`` cannot be genuine speech from a ``clip_s``-second clip, or None.

    Whisper's decoder loops on garbled input — half a word cut off by a
    dropout, a cough, a door — and returns the nearest token repeated until
    the length limit. Two independent tells: a word rate no human could
    produce, and text so repetitive that it compresses far beyond prose (the
    statistic Whisper uses for its own temperature fallback, which the
    fixed-temperature decode here never triggers).
    """
    words = len(text.split())
    if words > _MAX_WORDS_PER_S * clip_s + 2:
        return f"{words} words in a {clip_s:.1f} s clip"
    ratio = max((seg.get("compression_ratio", 0.0) for seg in segments), default=0.0)
    if ratio > _MAX_COMPRESSION_RATIO:
        return f"repetitive output (compression ratio {ratio:.1f})"
    return None


def parse_label(text: str) -> str | None:
    """Map a transcription to a command word, or None.

    Recognized commands: success, failure, align, play, stop, reset, next,
    adjust, done. Returns None when nothing matches or the utterance is
    contradictory (more than one command recognized at once).
    """
    text = text.lower()
    matches = [name for name, regex in _COMMAND_RES if regex.search(text)]
    return matches[0] if len(matches) == 1 else None


def fuzzy_label(text: str) -> str | None:
    """Nearest command for a SHORT utterance that :func:`parse_label` rejected, or None.

    Mis-hearings of an isolated command word tend to be near-homophones —
    "we're set", "recent" for "reset"; "a lie" for "align". Each word, and the
    whole utterance with spaces removed, is compared to the spoken forms of
    every command; a single command above :data:`_FUZZY_MIN_RATIO` wins.
    Utterances longer than :data:`_FUZZY_MAX_WORDS` words are never fuzzy-matched:
    a sentence of room conversation must contain a command verbatim to count.
    """
    from difflib import SequenceMatcher

    words = re.findall(r"[a-z']+", text.lower())
    if not words or len(words) > _FUZZY_MAX_WORDS:
        return None
    letters = [w.replace("'", "") for w in words]  # "we're set" -> "wereset" vs "reset"
    candidates = [*letters, "".join(letters)]
    hits = set()
    for label, forms in _COMMAND_WORDS.items():
        for form in forms:
            if any(SequenceMatcher(None, c, form).ratio() >= _FUZZY_MIN_RATIO for c in candidates):
                hits.add(label)
                break
    return hits.pop() if len(hits) == 1 else None


def count_label(text: str, label: str) -> int:
    """How many times ``label`` was said in ``text`` (at least 1).

    "Reset, reset, reset" in one breath is one utterance but three orders; the
    operator expects three resets.
    """
    return max(1, len(dict(_COMMAND_RES)[label].findall(text.lower())))


@dataclass(frozen=True)
class VoiceEvent:
    """One closed utterance: what was transcribed and which command it mapped to.

    Every utterance the energy gate captures produces an event, including ones
    that match no command, so the caller can show the operator what was heard.
    A command said N times in one utterance produces N identical events.

    Attributes:
        text: The transcription, or ``""`` when audio was captured but no
            intelligible speech was found.
        command: The parsed command (success / failure / align / play / stop /
            reset / next / adjust / done), or None when the utterance matched
            none of them (or matched several, which is treated as contradictory).
    """

    text: str
    command: str | None


class _Whisper:
    """OpenAI Whisper configured for short command clips (see module docstring)."""

    def __init__(self, model_name: str, device: str):
        import whisper

        print(f"[VOICE] Loading Whisper '{model_name}' on {device} ...")
        self._model = whisper.load_model(model_name, device=device)
        self._fp16 = device != "cpu"

    def transcribe(self, clip: np.ndarray) -> tuple[str, list[dict]]:
        """Return ``(text, segments)``; ``text`` is ``""`` when Whisper itself is not confident.

        Segments carry ``no_speech_prob``, ``avg_logprob`` and
        ``compression_ratio``; the first two gate the result here, the last is
        consumed by :func:`hallucination_reason`.
        """
        result = self._model.transcribe(
            clip,
            language="en",
            fp16=self._fp16,
            temperature=0.0,
            beam_size=_BEAM_SIZE,
            condition_on_previous_text=False,
            initial_prompt=_INITIAL_PROMPT,
        )
        segments = result.get("segments", [])
        if segments and all(
            seg.get("no_speech_prob", 0.0) > _NO_SPEECH_THRESHOLD or seg.get("avg_logprob", 0.0) < _LOGPROB_THRESHOLD
            for seg in segments
        ):
            return "", segments
        return result["text"].strip(), segments


class VoiceLabeler:
    """Background voice-command listener (see module docstring).

    Args:
        model_name: Whisper model (``base.en`` default; ``small.en`` is more
            accurate and ~3x slower on a CPU).
        device: torch device for Whisper. Default ``cpu`` so transcription
            never competes with the simulation and CloudXR encode for the GPU.
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
        self._whisper = _Whisper(model_name, device)
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
                text, segments = self._whisper.transcribe(clip)
            except Exception as exc:
                print(f"[VOICE] Transcription failed: {exc}")
                continue
            clip_s = len(clip) / _SAMPLE_RATE
            if not text:
                print(f"[VOICE] Heard a {clip_s:.1f} s sound but no intelligible speech.")
                self._events.put(VoiceEvent("", None))
                continue
            reason = hallucination_reason(text, segments, clip_s)
            if reason is not None:
                brief = text if len(text) <= 60 else text[:57] + "..."
                print(f'[VOICE] Rejected ASR output "{brief}" as a hallucination: {reason}.')
                self._events.put(VoiceEvent("", None))
                continue
            label = parse_label(text)
            fuzzy = label is None and (label := fuzzy_label(text)) is not None
            if label is None:
                print(f'[VOICE] Heard: "{text}" (no label)')
                self._events.put(VoiceEvent(text, None))
                continue
            if fuzzy:
                print(f'[VOICE] Heard: "{text}" -> {label.upper()} (fuzzy match)')
                self._events.put(VoiceEvent(text, label))
                continue
            # One event per time the command was said: the consumer acts on one
            # event per loop iteration, so each repeat becomes its own execution.
            count = count_label(text, label)
            print(f'[VOICE] Heard: "{text}" -> {label.upper()}' + (f" x{count}" if count > 1 else ""))
            for _ in range(count):
                self._events.put(VoiceEvent(text, label))
