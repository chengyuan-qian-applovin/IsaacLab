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

_SUPERSEDED = 4001
"""Private websocket close code telling an older page a newer one took over.

In the 4000-4999 application range. The page treats it as terminal: it releases
the microphone and stops reconnecting, so a forgotten tab cannot keep stealing
the mic from the page actually in use."""

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
const RATE = 16000, CHUNK = 1600, SUPERSEDED = 4001;
const status = t => document.getElementById("status").textContent = t;
let ws = null, started = false, micStream = null;

// Release the mic and stand down. Used when the server reports that a newer
// page took over: a stale tab that kept its capture alive would go on stealing
// the microphone back, restarting the real page's capture every few seconds.
function standDown(msg) {
  status(msg);
  started = false;
  document.getElementById("btn").textContent = "Start microphone";
  document.getElementById("bar").style.width = "0%";
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
}

function connect() {
  // Drop any previous socket first: start() re-runs whenever the OS reclaims
  // the mic, and without this each restart leaks another open connection.
  if (ws) { ws.onclose = null; ws.onerror = null; try { ws.close(); } catch (e) {} }
  ws = new WebSocket("wss://" + location.host + "/audio");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => status("streaming to " + location.host);
  ws.onclose = (ev) => {
    if (ev.code === SUPERSEDED) { standDown("superseded by a newer mic page - close this tab"); return; }
    status("disconnected - retrying");
    setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
}

let lastAudioMs = 0;

async function start() {
  if (started) return;
  // Claim the mic BEFORE latching: getUserMedia rejects whenever another copy
  // of this page already holds the microphone, and latching first would leave
  // the button reading "running" with every later tap a no-op until a reload.
  let stream;
  document.getElementById("btn").textContent = "Starting...";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (err) {
    document.getElementById("btn").textContent = "Start microphone";
    status("mic failed: " + err.name + " - close any other copy of this page, then tap again");
    return;
  }
  started = true;
  micStream = stream;
  document.getElementById("btn").textContent = "Microphone running";
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


class QuestMicServer:
    """Serves the mic page and receives the headset's 16 kHz PCM stream."""

    def __init__(self, port: int = 8444):
        self._port = port
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._active = None  # the one connection allowed to stream (see handler)
        self._connected = False
        self._last_rx = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True, name="quest-mic-server")
        self._thread.start()
        self._watchdog = threading.Thread(target=self._watch, daemon=True, name="quest-mic-watchdog")
        self._watchdog.start()
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
                print(
                    f"[VOICE] WARNING: Quest microphone stalled ({silent_s:.0f} s without audio, socket still"
                    " open) — the headset browser tab was likely frozen. Bring the mic page to the"
                    " foreground once, or reload it."
                )
            elif silent_s < 1.0 and stalled:
                stalled = False
                print("[VOICE] Quest microphone audio resumed.")

    def _serve(self) -> None:
        asyncio.run(self._serve_async())

    async def _serve_async(self) -> None:
        from websockets.asyncio.server import serve
        from websockets.exceptions import ConnectionClosed

        def process_request(connection, request):
            # Plain HTTP GETs receive the capture page; websocket upgrades pass through.
            if "upgrade" not in request.headers.get("Connection", "").lower():
                response = connection.respond(http.HTTPStatus.OK, _PAGE)
                response.headers["Content-Type"] = "text/html; charset=utf-8"
                return response
            return None

        async def handler(connection):
            # Newest page wins. The headset grants its microphone to one page at
            # a time, so leftover tabs do not merely idle — they keep stealing
            # the mic back, and every theft restarts the loser's capture. Evict
            # the previous connection with _SUPERSEDED so it releases the mic
            # and stops retrying, instead of both sides fighting forever.
            previous, self._active = self._active, connection
            if previous is not None:
                with contextlib.suppress(Exception):
                    await previous.close(code=_SUPERSEDED, reason="superseded")
                print("[VOICE] Superseded an older Quest microphone page (a stale tab was still open).")
            self._connected = True
            print("[VOICE] Quest microphone connected.")
            leftover = np.zeros(0, dtype=np.float32)
            try:
                async for message in connection:
                    if connection is not self._active:
                        break  # a newer page took over mid-stream
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
            except ConnectionClosed:
                # Routine: the page navigated away, or we superseded it. The 4001
                # close is deliberate, so let it end the handler quietly instead
                # of surfacing as websockets' "connection handler failed".
                pass
            finally:
                if connection is self._active:
                    self._active = None
                    self._connected = False
                    print("[VOICE] Quest microphone disconnected (the page reconnects automatically).")

        async with serve(handler, "0.0.0.0", self._port, ssl=_ssl_context(), process_request=process_request):
            while not self._stop.is_set():
                await asyncio.sleep(0.5)


class MicHubClient:
    """Consumes microphone audio relayed by the teleop app (``--mic_device hub``).

    Exposes the same :meth:`read_chunk`/:meth:`close` pair as
    :class:`QuestMicServer`, so :class:`~voice_labeler.VoiceLabeler` cannot tell
    the two apart. The difference is ownership: here the headset page and the
    microphone belong to :mod:`teleop_app`, which outlives any single teleop
    run, so the operator taps "Start microphone" once per headset session rather
    than once per run. If the app restarts underneath us, this end reconnects on
    its own instead of dying with a stale socket.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8500):
        self._uri = f"wss://{host}:{port}/subscribe"
        self._chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mic-hub-client")
        self._thread.start()
        print(f"[VOICE] Using the teleop app's microphone relay at {self._uri}.")
        print("[VOICE] Tap 'Start session' (or 'Microphone only') on the app page in the headset.")

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

    def _run(self) -> None:
        asyncio.run(self._consume())

    async def _consume(self) -> None:
        from websockets.asyncio.client import connect as ws_connect

        # The app serves the same self-signed CloudXR cert; this is a loopback
        # hop on the machine that issued it, so verification buys nothing.
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        announced = False
        while not self._stop.is_set():
            try:
                async with ws_connect(self._uri, ssl=ctx, open_timeout=5) as ws:
                    if not announced:
                        print("[VOICE] Connected to the teleop app's microphone relay.")
                        announced = True
                    leftover = np.zeros(0, dtype=np.float32)
                    while not self._stop.is_set():
                        message = await ws.recv()
                        if not isinstance(message, bytes):
                            continue
                        pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                        data = np.concatenate([leftover, pcm])
                        while len(data) >= _CHUNK_SAMPLES:
                            with contextlib.suppress(queue.Full):
                                self._chunks.put_nowait(data[:_CHUNK_SAMPLES].copy())
                            data = data[_CHUNK_SAMPLES:]
                        leftover = data
            except Exception:
                if announced:
                    print("[VOICE] Lost the microphone relay; retrying. Is the teleop app still running?")
                    announced = False
                await asyncio.sleep(2.0)
