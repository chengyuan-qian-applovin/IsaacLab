// Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// Microphone capture inside the CloudXR client page.
//
// The teleop app serves NVIDIA's WebXR client from its own origin (/client/) and
// adds this script to it. The CloudXR window is the one window the headset keeps
// in the foreground during an immersive session; a page in the background has
// its capture paused about a minute after it is hidden and its track ended
// whenever the operator switches windows, which is what made the microphone die
// mid-session when the app page owned it. Here the microphone rides in the
// foreground page and streams to the same hub (wss://<host>/audio, same origin)
// as 16 kHz mono PCM16 chunks of 1600 samples; the hub evicts the app page,
// which takes the microphone back when this window closes.
//
// The client itself is opened with mic=0 so its own audio passthrough never asks
// for the device. Text frames on the socket are diagnostics for the app log; the
// input device follows the choice the app page remembered (same origin, same
// localStorage keys).
(() => {
  const RATE = 16000, CHUNK = 1600, SUPERSEDED = 4001;
  let ws = null, micStream = null, audioCtx = null, reacquiring = false, stopped = false;
  let frames = 0, lastPeak = 0, curChunks = 0, seenChunks = 0, lastFrameAt = 0, stalled = false;

  function deviceId() {
    try { return localStorage.getItem("mic_device") || localStorage.getItem("mic_device_auto") || ""; } catch (e) { return ""; }
  }
  function constraints() {
    const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
    const id = deviceId();
    if (id) audio.deviceId = { exact: id };
    return { audio };
  }

  function trackState() {
    const t = micStream && micStream.getTracks()[0];
    if (!t) return { track: null };
    const s = t.getSettings ? t.getSettings() : {};
    return { label: t.label, muted: t.muted, enabled: t.enabled, readyState: t.readyState,
             deviceId: (s.deviceId || "").slice(0, 8), sampleRate: s.sampleRate, hidden: document.hidden,
             visibility: document.visibilityState };
  }
  let pendingDiag = [];
  function diag(event, extra) {
    const msg = JSON.stringify({ t: new Date().toISOString().slice(11, 19), source: "client", event, ...(extra || {}), ...trackState() });
    if (ws && ws.readyState === 1) ws.send(msg); else pendingDiag.push(msg);
  }

  function connect() {
    if (stopped) return;
    ws = new WebSocket("wss://" + location.host + "/audio?source=client");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      pendingDiag.forEach(m => ws.send(m)); pendingDiag = []; diag("socket open");
      // Only now ask for the microphone: connecting made the hub evict the app
      // page, which drops its track, so the two pages never hold the mic at once.
      if (!micStream && !reacquiring) reacquire("socket open");
    };
    ws.onclose = (ev) => {
      if (ev.code === SUPERSEDED) { diag("superseded"); stop(); return; }  // a newer page owns the mic now
      if (!stopped) setTimeout(connect, 1000);
    };
    ws.onerror = () => ws.close();
  }

  function dropTrack() {
    if (micStream) { micStream.getTracks().forEach(t => { t.onended = null; t.stop(); }); micStream = null; }
    if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  }
  function stop() {
    stopped = true;
    dropTrack();
    if (ws) { ws.onclose = null; ws.onerror = null; try { ws.close(); } catch (e) {} ws = null; }
  }

  async function acquire() {
    const stream = await navigator.mediaDevices.getUserMedia(constraints());
    if (stopped) { stream.getTracks().forEach(t => t.stop()); return; }
    attachTrack(stream);
  }

  // The headset ended our track; ask for it back, waiting longer each time. A
  // remembered input that is gone falls back to the default at once.
  async function reacquire(reason) {
    if (stopped || reacquiring) return;
    reacquiring = true;
    try {
      let delay = 1000;
      while (!stopped) {
        try { await acquire(); return; }
        catch (err) {
          diag("reacquire failed", { reason, error: err.name, retry_ms: delay });
          if (err.name === "NotAllowedError") return;  // needs a gesture or permission: the next tap retries
          if ((err.name === "OverconstrainedError" || err.name === "NotFoundError") && deviceId()) {
            try { localStorage.removeItem("mic_device"); localStorage.removeItem("mic_device_auto"); } catch (e) {}
            continue;
          }
          await new Promise(r => setTimeout(r, delay));
          delay = Math.min(delay * 2, 10000);
        }
      }
    } finally { reacquiring = false; }
  }

  function attachTrack(stream) {
    dropTrack();
    micStream = stream;
    curChunks = 0; seenChunks = 0; lastFrameAt = Date.now(); stalled = false;
    const track = stream.getTracks()[0];
    track.onended = () => { diag("track ended"); micStream = null; reacquire("ended"); };
    track.onmute = () => diag("track muted");
    track.onunmute = () => diag("track unmuted");
    diag("track started", { chosen: deviceId().slice(0, 8) || "default" });

    // Resample whatever the device delivers (48 kHz on Quest) to 16 kHz PCM16.
    let buf = [], acc = 0, out = new Int16Array(CHUNK), oi = 0, peak = 0;
    const feed = (x, srcRate) => {
      const ratio = srcRate / RATE;
      for (let i = 0; i < x.length; i++) buf.push(x[i]);
      while (acc + ratio < buf.length) {
        const j = Math.floor(acc), f = acc - j;
        const v = buf[j] * (1 - f) + buf[j + 1] * f;
        out[oi++] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
        peak = Math.max(peak, Math.abs(v));
        acc += ratio;
        if (oi === CHUNK) {
          if (ws && ws.readyState === 1) ws.send(out.buffer.slice(0));
          oi = 0; curChunks++;
          if (++frames % 5 === 0) { lastPeak = peak; peak = 0; }
        }
      }
      const drop = Math.floor(acc); buf = buf.slice(drop); acc -= drop;
    };

    if (window.MediaStreamTrackProcessor) {
      // Preferred: frames straight off the track, no AudioContext to be suspended.
      const reader = new MediaStreamTrackProcessor({ track }).readable.getReader();
      (async () => {
        let f32 = new Float32Array(0);
        for (;;) {
          const { value, done } = await reader.read();
          if (done) { diag("track processor ended"); break; }
          if (micStream !== stream) { value.close(); break; }
          const n = value.numberOfFrames;
          if (f32.length < n) f32 = new Float32Array(n);
          try {
            value.copyTo(f32, { planeIndex: 0, format: "f32-planar" });
          } catch (e) {
            if (value.format === "s16" || value.format === "s16-planar") {
              const ch = value.format === "s16" ? value.numberOfChannels : 1;
              const s16 = new Int16Array(n * ch); value.copyTo(s16, { planeIndex: 0 });
              for (let i = 0; i < n; i++) f32[i] = s16[i * ch] / 32768;
            } else {
              const ch = value.format === "f32" ? value.numberOfChannels : 1;
              const raw = new Float32Array(n * ch); value.copyTo(raw, { planeIndex: 0 });
              for (let i = 0; i < n; i++) f32[i] = raw[i * ch];
            }
          }
          const rate = value.sampleRate; value.close();
          feed(f32.subarray(0, n), rate);
        }
      })().catch(e => diag("track processor error", { error: String(e) }));
    } else {
      // Fallback: AudioWorklet.
      const ctx = audioCtx = new AudioContext();
      ctx.onstatechange = () => { if (ctx.state !== "running") ctx.resume(); diag("audiocontext " + ctx.state); };
      const code = `registerProcessor("grab", class extends AudioWorkletProcessor {
          process(i) { if (i[0] && i[0][0]) this.port.postMessage(i[0][0]); return true; } });`;
      ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([code], { type: "text/javascript" }))).then(() => {
        const node = new AudioWorkletNode(ctx, "grab");
        ctx.createMediaStreamSource(stream).connect(node);
        node.port.onmessage = (e) => feed(e.data, ctx.sampleRate);
      });
    }
  }

  // A live track that stops delivering frames is dead in all but name. Hidden
  // (the operator left VR and went to the app page), this window is now the
  // background one the headset starves, so it lets go of the microphone - the
  // socket closes and the app page, which is in front, takes the mic back.
  // Visible, it fetches a new track instead.
  let released = false;
  function release() {
    released = true; dropTrack();
    if (ws) { ws.onclose = null; ws.onerror = null; try { ws.close(); } catch (e) {} ws = null; }
  }
  function watch() {
    if (stopped || released || !micStream || reacquiring) return;
    if (curChunks !== seenChunks) { seenChunks = curChunks; lastFrameAt = Date.now(); if (stalled) { stalled = false; diag("frames resumed"); } return; }
    if (Date.now() - lastFrameAt < 5000) return;
    if (!stalled) { stalled = true; diag("frames stalled"); }
    if (document.hidden) { diag("released while hidden"); release(); return; }
    lastFrameAt = Date.now(); dropTrack(); reacquire("stall");
  }

  function start() {
    connect();
    setInterval(watch, 1000);
    setInterval(() => diag("tick", { peak: +lastPeak.toFixed(4), frames }), 20000);
    // Some browsers want a gesture for a page's first capture; the operator
    // taps Connect anyway, so a tap retries if the load-time attempt was refused.
    const onTap = () => { if (!micStream && !reacquiring && !stopped) reacquire("gesture"); };
    window.addEventListener("pointerdown", onTap, true);
    // The XR session's start and end go to the log next to the track events.
    if (navigator.xr && navigator.xr.requestSession) {
      const orig = navigator.xr.requestSession.bind(navigator.xr);
      navigator.xr.requestSession = async (mode, opts) => {
        const session = await orig(mode, opts);
        diag("xr session start", { mode });
        session.addEventListener("end", () => diag("xr session end"));
        if (released) { released = false; stalled = false; connect(); }
        else if (!micStream && !reacquiring && !stopped) reacquire("xr start");
        return session;
      };
    }
  }
  window.__teleopMic = { state: () => ({ stopped, reacquiring, stalled, frames, socket: ws ? ws.readyState : null, ...trackState() }) };

  document.addEventListener("visibilitychange", () => {
    if (stopped) return;
    diag("visibility");
    if (document.hidden) return;
    // Back in front: take the microphone again (the socket's open handler asks
    // for it, which also evicts the app page), or replace a stalled track.
    if (released) { released = false; stalled = false; connect(); return; }
    if (stalled && micStream && !reacquiring) { dropTrack(); reacquire("visible"); }
  });
  // Free the hub slot at once so the app page can take the microphone back.
  window.addEventListener("pagehide", () => { diag("pagehide"); stop(); });

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
  // Kill switches: mic=1 means the bundle captures for CloudXR's own audio
  // passthrough (we would only fight it); appmic=0 disables this script for A/B.
  const q = new URLSearchParams(location.search);
  if (q.get("mic") === "1" || q.get("appmic") === "0") return;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
