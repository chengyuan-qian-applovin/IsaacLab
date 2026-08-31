# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headset-microphone audio source: a WSS server the headset streams mic audio to.

Nothing in the CloudXR/IsaacTeleop stack delivers the headset microphone to the
server (the runtime's mic machinery is internal-only, and NVIDIA's own voice
flow runs on-device), so this module provides the audio path ourselves.
:class:`HeadsetMicServer` accepts a WSS stream of 16 kHz mono PCM16 chunks on
one port; TLS reuses the CloudXR WSS proxy's self-signed certificate
(``~/.cloudxr/certs``) when present — the same warning flow the headset already
went through — and generates one otherwise. Two headsets speak this protocol:

- **Quest/Pico** (``client="quest"``): the headset has no native hook, so plain
  HTTP GETs on the same port serve a small mic-capture page (embedded below)
  that grabs the browser microphone with ``getUserMedia``, resamples to 16 kHz
  in an AudioWorklet, and ships int16 chunks over the websocket, auto-
  reconnecting. Usage: open ``https://<workstation-ip>:<port>/`` in the Quest
  browser, accept the certificate warning, tap "Start microphone", grant the
  mic permission — then connect the CloudXR client as usual. The page keeps
  streaming from the background tab.
- **Apple Vision Pro** (``client="avp"``): the Isaac XR Teleop Sample Client
  (``feature/avp-voice-mic`` branch) captures the mic natively and streams the
  same chunks to ``wss://<workstation-ip>:<port>/audio`` by itself once its
  "Stream microphone" toggle is on and the CloudXR session connects — nothing
  to open on the headset.

The ``client`` parameter only selects the operator instructions printed at
startup; the wire protocol and server behavior are identical, so either
headset is accepted whichever was named.

The server exposes :meth:`HeadsetMicServer.read_chunk`, blocking until the next
0.1 s chunk arrives, which :class:`voice_labeler.VoiceLabeler` consumes in
place of ``arecord`` when ``--mic_device quest`` or ``--mic_device avp`` is
selected.
"""

from __future__ import annotations

import asyncio
import contextlib
import http
import os
import queue
import socket
import ssl
import subprocess
import threading
import time

import numpy as np

_SAMPLE_RATE = 16000
_CHUNK_SAMPLES = 1600  # 0.1 s, matching voice_labeler's reader granularity

_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Teleop microphone</title>
<style>
  body { font-family: sans-serif; background: #111; color: #eee; text-align: center; padding: 3em 1em; }
  button { font-size: 1.6em; padding: 0.6em 1.2em; border-radius: 0.5em; border: 0; }
  #status { margin-top: 1.5em; font-size: 1.2em; }
  #level { width: 60%; height: 1em; background: #333; margin: 1em auto; border-radius: 0.5em; overflow: hidden; }
  #bar { height: 100%; width: 0; background: #4c4; }
</style>
<h1>Teleop microphone</h1>
<button id="btn">Start microphone</button>
<div id="level"><div id="bar"></div></div>
<div id="status">idle</div>
<script>
const RATE = 16000, CHUNK = 1600;
const status = t => document.getElementById("status").textContent = t;
let ws = null, started = false;

function connect() {
  ws = new WebSocket("wss://" + location.host + "/audio");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => status("streaming to " + location.host);
  ws.onclose = () => { status("disconnected - retrying"); setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}

let lastAudioMs = 0;

async function start() {
  if (started) return;
  started = true;
  document.getElementById("btn").textContent = "Microphone running";
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const ctx = new AudioContext();
  // Keep-alive: a tab that is PLAYING audio is exempt from Chromium's
  // background-tab freezing/throttling, which otherwise stops the capture a
  // few minutes into an immersive CloudXR session. Emit an inaudible tone.
  const keepAlive = ctx.createOscillator();
  const keepAliveGain = ctx.createGain();
  keepAliveGain.gain.value = 0.001;
  keepAlive.frequency.value = 30;
  keepAlive.connect(keepAliveGain).connect(ctx.destination);
  keepAlive.start();
  // Self-heal: resume whenever the OS suspends/interrupts the context.
  ctx.onstatechange = () => { if (ctx.state !== "running") ctx.resume(); };
  // Self-heal: restart capture if the OS reclaims the microphone track.
  stream.getTracks()[0].onended = () => { status("mic lost - restarting"); started = false; start(); };
  const workletCode = `
    registerProcessor("grab", class extends AudioWorkletProcessor {
      process(inputs) {
        if (inputs[0] && inputs[0][0]) this.port.postMessage(inputs[0][0]);
        return true;
      }
    });`;
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([workletCode], { type: "text/javascript" })));
  const node = new AudioWorkletNode(ctx, "grab");
  ctx.createMediaStreamSource(stream).connect(node);
  // Watchdog: if no audio flowed for 3 s, poke the context and report.
  setInterval(() => {
    if (lastAudioMs && Date.now() - lastAudioMs > 3000) {
      status("audio stalled - resuming (" + ctx.state + ")");
      ctx.resume();
    }
  }, 2000);

  const ratio = ctx.sampleRate / RATE;
  let buf = [], acc = 0, out = new Int16Array(CHUNK), oi = 0, peak = 0, frames = 0;
  node.port.onmessage = (e) => {
    lastAudioMs = Date.now();
    const x = e.data;
    for (let i = 0; i < x.length; i++) buf.push(x[i]);
    // Linear resample ctx.sampleRate -> 16 kHz.
    while (acc + ratio < buf.length) {
      const j = Math.floor(acc), f = acc - j;
      const v = buf[j] * (1 - f) + buf[j + 1] * f;
      out[oi++] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
      peak = Math.max(peak, Math.abs(v));
      acc += ratio;
      if (oi === CHUNK) {
        if (ws && ws.readyState === 1) ws.send(out.buffer.slice(0));
        oi = 0;
        if (++frames % 10 === 0) {
          document.getElementById("bar").style.width = Math.min(100, peak * 300) + "%";
          peak = 0;
        }
      }
    }
    const drop = Math.floor(acc);
    buf = buf.slice(drop);
    acc -= drop;
  };
  connect();
}
document.getElementById("btn").onclick = () => start().catch(e => status("error: " + e));
</script>
"""


def _lan_ips() -> list[str]:
    """All global IPv4 addresses of this host (VPN tunnels included; the LAN one is what the headset needs)."""
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"], check=True, capture_output=True, text=True
        ).stdout
        ips = [line.split()[3].split("/")[0] for line in out.splitlines() if len(line.split()) > 3]
        if ips:
            return ips
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return [ip]
    except OSError:
        return [socket.gethostbyname(socket.gethostname())]


def _ssl_context() -> ssl.SSLContext:
    """TLS context from the CloudXR proxy's cert, generating a self-signed one if absent."""
    cert_dir = os.path.expanduser("~/.cloudxr/certs")
    cert, key = os.path.join(cert_dir, "server.crt"), os.path.join(cert_dir, "server.key")
    if not (os.path.exists(cert) and os.path.exists(key)):
        os.makedirs(cert_dir, exist_ok=True)
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
             "-keyout", key, "-out", cert, "-subj", "/CN=teleop-mic"],
            check=True, capture_output=True,
        )  # fmt: skip
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return ctx


class HeadsetMicServer:
    """Receives the headset's 16 kHz PCM stream (and serves the Quest mic page).

    Args:
        port: TCP port for the WSS audio endpoint (and the Quest capture page).
        client: Which headset the operator uses — ``"quest"`` or ``"avp"``.
            Only changes the instructions printed at startup and in the stall
            watchdog; the audio endpoint accepts either client regardless.
    """

    def __init__(self, port: int = 8444, client: str = "quest"):
        self._port = port
        self._client = client
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._connected = False
        self._last_rx = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True, name="headset-mic-server")
        self._thread.start()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="headset-mic-watchdog")
        self._watchdog.start()
        if client == "avp":
            ips = " or ".join(_lan_ips())
            print(
                f"[VOICE] AVP microphone: turn on 'Stream microphone' in the Isaac XR Teleop client"
                f" (feature/avp-voice-mic build); it streams to wss://{ips}:{port}/audio by itself"
                " once the CloudXR session connects."
            )
        else:
            urls = " or ".join(f"https://{ip}:{port}/" for ip in _lan_ips())
            print(
                f"[VOICE] Quest microphone: open {urls} (the LAN address) in the headset browser,"
                " accept the certificate, tap 'Start microphone', then connect the CloudXR client."
            )

    def read_chunk(self) -> np.ndarray | None:
        """Next 0.1 s float32 chunk; blocks until audio arrives. None once stopped."""
        while not self._stop.is_set():
            try:
                return self._chunks.get(timeout=0.5)
            except queue.Empty:
                continue
        return None

    def close(self) -> None:
        self._stop.set()

    # -- internals -----------------------------------------------------------

    def _watch(self) -> None:
        """Announce audio stalls: a frozen headset tab keeps the socket open but stops sending."""
        stalled = False
        while not self._stop.is_set():
            time.sleep(2.0)
            if not self._connected or self._last_rx == 0.0:
                continue
            silent_s = time.monotonic() - self._last_rx
            if silent_s > 4.0 and not stalled:
                stalled = True
                hint = (
                    "check the 'Stream microphone' toggle in the Isaac XR Teleop client"
                    if self._client == "avp"
                    else "the headset browser tab was likely frozen — bring the mic page to the"
                    " foreground once, or reload it"
                )
                print(
                    f"[VOICE] WARNING: headset microphone stalled ({silent_s:.0f} s without audio, socket"
                    f" still open) — {hint}."
                )
            elif silent_s < 1.0 and stalled:
                stalled = False
                print("[VOICE] Headset microphone audio resumed.")

    def _serve(self) -> None:
        asyncio.run(self._serve_async())

    async def _serve_async(self) -> None:
        from websockets.asyncio.server import serve

        def process_request(connection, request):
            # Plain HTTP GETs receive the capture page; websocket upgrades pass through.
            if "upgrade" not in request.headers.get("Connection", "").lower():
                response = connection.respond(http.HTTPStatus.OK, _PAGE)
                response.headers["Content-Type"] = "text/html; charset=utf-8"
                return response
            return None

        async def handler(connection):
            self._connected = True
            print("[VOICE] Headset microphone connected.")
            leftover = np.zeros(0, dtype=np.float32)
            try:
                async for message in connection:
                    if not isinstance(message, bytes):
                        continue
                    self._last_rx = time.monotonic()
                    pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    data = np.concatenate([leftover, pcm])
                    while len(data) >= _CHUNK_SAMPLES:
                        # A stalled consumer drops audio rather than lagging.
                        with contextlib.suppress(queue.Full):
                            self._chunks.put_nowait(data[:_CHUNK_SAMPLES].copy())
                        data = data[_CHUNK_SAMPLES:]
                    leftover = data
            finally:
                self._connected = False
                print("[VOICE] Headset microphone disconnected (the client reconnects automatically).")

        async with serve(handler, "0.0.0.0", self._port, ssl=_ssl_context(), process_request=process_request):
            while not self._stop.is_set():
                await asyncio.sleep(0.5)
