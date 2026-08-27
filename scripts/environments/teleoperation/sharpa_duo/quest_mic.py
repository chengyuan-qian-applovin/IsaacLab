# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quest-microphone audio source: a mic capture page streamed from the headset's browser.

Nothing in the CloudXR/IsaacTeleop stack delivers the headset microphone to the
server (the runtime's mic machinery is internal-only, and NVIDIA's own voice
flow runs on-device), so this module provides the audio path ourselves:

- :class:`QuestMicServer` serves a small HTTPS page and accepts a WSS stream of
  16 kHz mono PCM16 on the SAME port (plain GETs get the page, websocket
  upgrades get the audio channel). TLS reuses the CloudXR WSS proxy's
  self-signed certificate (``~/.cloudxr/certs``) when present — the same
  warning flow the headset already went through — and generates one otherwise.
- The page (embedded below) captures the browser microphone with
  ``getUserMedia``, resamples to 16 kHz in an AudioWorklet, and ships int16
  chunks over the websocket. It auto-reconnects.

Usage on the headset: open ``https://<workstation-ip>:<port>/`` in the Quest
browser, accept the certificate warning, tap "Start microphone", grant the mic
permission — then connect the CloudXR client as usual. The page keeps
streaming from the background tab.

The server exposes :meth:`QuestMicServer.read_chunk`, blocking until the next
0.1 s chunk arrives, which :class:`voice_labeler.VoiceLabeler` consumes in
place of ``arecord`` when ``--mic_device quest`` is selected.
"""

from __future__ import annotations

import asyncio
import http
import os
import queue
import socket
import ssl
import subprocess
import threading

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

async function start() {
  if (started) return;
  started = true;
  document.getElementById("btn").textContent = "Microphone running";
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const ctx = new AudioContext();
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

  const ratio = ctx.sampleRate / RATE;
  let buf = [], acc = 0, out = new Int16Array(CHUNK), oi = 0, peak = 0, frames = 0;
  node.port.onmessage = (e) => {
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


class QuestMicServer:
    """Serves the mic page and receives the headset's 16 kHz PCM stream."""

    def __init__(self, port: int = 8444):
        self._port = port
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True, name="quest-mic-server")
        self._thread.start()
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
            print("[VOICE] Quest microphone connected.")
            leftover = np.zeros(0, dtype=np.float32)
            try:
                async for message in connection:
                    if not isinstance(message, bytes):
                        continue
                    pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                    data = np.concatenate([leftover, pcm])
                    while len(data) >= _CHUNK_SAMPLES:
                        try:
                            self._chunks.put_nowait(data[:_CHUNK_SAMPLES].copy())
                        except queue.Full:
                            pass  # consumer stalled; drop rather than lag
                        data = data[_CHUNK_SAMPLES:]
                    leftover = data
            finally:
                self._connected = False
                print("[VOICE] Quest microphone disconnected (the page reconnects automatically).")

        async with serve(handler, "0.0.0.0", self._port, ssl=_ssl_context(), process_request=process_request):
            while not self._stop.is_set():
                await asyncio.sleep(0.5)
