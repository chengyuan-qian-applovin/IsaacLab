# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Always-on teleop control app, driven from the headset browser.

One page, one bookmark, one tap. It replaces the pair of hand-maintained
shortcuts whose URLs went stale whenever the host's address or a port changed.

The app is a **supervisor**: it runs on its own fixed port, independent of any
teleop run, so it stays reachable when teleop is stopped, wedged, or crashed —
which is exactly when you need it. From the headset it can:

- start, restart and kill ``make_teleop_scene.py`` (killing the whole process
  group, plus this user's leftover CloudXR runtime, so nothing keeps holding
  49100/48322 the way a bare Ctrl-Z or a stray orphan does);
- capture the headset microphone in the page itself and relay it to whichever
  teleop process is running. Because the mic lives here rather than inside
  teleop, it **survives teleop restarts** — no re-tapping between runs;
- hand out the CloudXR client link with the ports the running session actually
  uses, read at request time rather than baked into a bookmark.

Bookmark it once as ``https://<hostname>.local:8500/`` — mDNS keeps that
address correct across DHCP leases, and the port is fixed by this service.

Run it directly for a quick trial::

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py

or install it as a user service so it comes back after a reboot::

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py --install-service

Pair it with ``--mic_device hub`` on the teleop side (see :mod:`quest_mic`),
which makes teleop pull audio from this app instead of serving its own page.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from quest_mic import _lan_ips, _ssl_context  # noqa: E402  (shares the CloudXR cert)

DEFAULT_PORT = 8500
"""Fixed port for the app. Bookmarked once, so it must not move."""

_TELEOP_SCRIPT = os.path.join(_HERE, "make_teleop_scene.py")
_ISAACLAB_SH = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "isaaclab.sh"))

_SUPERSEDED = 4001
"""Close code telling an older mic page a newer one took over (see :mod:`quest_mic`)."""

_STOP_GRACE_S = 12.0
"""Seconds to wait for a SIGINT (the Ctrl-C equivalent) before escalating."""

_CONTROL_HEADER = "X-Teleop-Control"
"""Header the page sends on start/stop/restart, so navigation alone cannot fire them."""


class TeleopProcess:
    """Owns the ``make_teleop_scene.py`` child: start, status, and a real kill.

    The child is started in its **own session** so the whole tree can be
    signalled at once. A plain ``kill`` on the launcher misses the Isaac Sim
    process and the CloudXR runtime, which is how ports end up held by
    survivors after an apparently clean exit.
    """

    def __init__(self, extra_args: list[str], env_overrides: dict[str, str]):
        self._extra_args = list(extra_args)
        self._env_overrides = dict(env_overrides)
        self._proc: subprocess.Popen | None = None
        # A run inherited from a previous app instance (see _adopt_orphan). Not
        # our child, so it is tracked by pid and polled through /proc.
        self._adopted_pid: int | None = None
        self._started_at = 0.0
        log_dir = os.path.join(os.path.expanduser("~/.cache"), "teleop_app")
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "last_run.log")
        self._log_file = None
        self._adopt_orphan()

    def _adopt_orphan(self) -> None:
        """Take over a teleop run that outlived the app instance that started it.

        The service is restarted with ``KillMode=process`` precisely so that a
        running teleop survives an app update; the price is that the new
        instance would otherwise not know about it, showing "stopped" while the
        port is busy and leaving "Kill" with nothing to kill. Only runs we
        launched are adopted: they are session leaders (``start_new_session``),
        which a run typed into a terminal never is, so that one stays untouched.
        """
        try:
            out = subprocess.run(
                ["ps", "-o", "pid=,sid=,etimes=,cmd=", "-u", str(os.getuid())],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 4 or _TELEOP_SCRIPT not in parts[3]:
                continue
            pid, sid, etimes = int(parts[0]), int(parts[1]), float(parts[2])
            if pid != sid:
                continue
            self._adopted_pid = pid
            self._started_at = time.monotonic() - etimes
            print(f"[APP] Adopted teleop run already in progress (pid {pid}, up {int(etimes)}s).", flush=True)
            return

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> tuple[bool, str]:
        """Launch teleop if it is not already up. Returns ``(ok, message)``."""
        if self.is_running():
            return False, f"already running (pid {self.pid})"
        cmd = [_ISAACLAB_SH, "-p", _TELEOP_SCRIPT, *self._extra_args]
        if not os.path.exists(_ISAACLAB_SH):
            return False, f"launcher not found: {_ISAACLAB_SH}"
        # Desktop vars first so explicit overrides still win.
        env = {**os.environ, **_desktop_env(), **self._env_overrides}
        self._log_file = open(self._log_path, "wb")  # noqa: SIM115  (closed in _reap)
        try:
            self._proc = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(_ISAACLAB_SH),
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # own process group, so stop() can signal the tree
                env=env,
            )
        except OSError as exc:
            self._log_file.close()
            self._log_file = None
            return False, f"failed to launch: {exc}"
        self._adopted_pid = None
        self._started_at = time.monotonic()
        return True, f"started (pid {self._proc.pid}); log: {self._log_path}"

    def stop(self) -> tuple[bool, str]:
        """Kill the whole teleop tree, then sweep this user's CloudXR leftovers.

        Escalates SIGINT (what Ctrl-C would send) → SIGTERM → SIGKILL, giving
        Isaac Sim a chance to release the XR session before being forced.
        """
        notes = []
        if self.is_running():
            pgid = os.getpgid(self.pid)
            for sig, wait in ((signal.SIGINT, _STOP_GRACE_S), (signal.SIGTERM, 5.0), (signal.SIGKILL, 3.0)):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pgid, sig)
                if self._wait_for_exit(wait):
                    notes.append(f"exited on {sig.name}")
                    break
            else:
                notes.append("did not exit even after SIGKILL")
            # Only sweep runtimes younger than the run we just killed. An
            # unscoped sweep would also take down a teleop session started by
            # hand in a terminal, which is not ours to end.
            swept = _sweep_cloudxr_runtime(max_age_s=time.monotonic() - self._started_at + 30.0)
            if swept:
                notes.append(f"killed {swept} leftover CloudXR runtime process(es)")
        else:
            notes.append("teleop was not running (nothing swept)")
        self._reap()
        return True, "; ".join(notes)

    def restart(self) -> tuple[bool, str]:
        """Stop (if up) and start again."""
        _, stopped = self.stop()
        time.sleep(1.0)  # let the ports actually come back before rebinding
        ok, started = self.start()
        return ok, f"{stopped}; {started}"

    # -- state --------------------------------------------------------------

    @property
    def pid(self) -> int | None:
        """Pid of the run we control, whether spawned or adopted."""
        if self._proc is not None:
            return self._proc.pid
        return self._adopted_pid

    def is_running(self) -> bool:
        """Whether the run is alive."""
        if self._proc is not None:
            return self._proc.poll() is None
        if self._adopted_pid is not None:
            return _pid_alive(self._adopted_pid)
        return False

    def status(self) -> dict:
        """JSON-ready snapshot for the page."""
        running = self.is_running()
        return {
            "running": running,
            "pid": self.pid if running else None,
            "uptime_s": round(time.monotonic() - self._started_at, 1) if running else 0.0,
            "exit_code": None if running else (self._proc.returncode if self._proc else None),
            "log": self._log_path,
            "command": " ".join(shlex.quote(a) for a in [_ISAACLAB_SH, "-p", _TELEOP_SCRIPT, *self._extra_args]),
        }

    def tail_log(self, lines: int = 40) -> str:
        """Last ``lines`` of the child's combined output, for in-headset triage."""
        try:
            with open(self._log_path, "rb") as f:
                return "\n".join(f.read().decode("utf-8", "replace").splitlines()[-lines:])
        except OSError:
            return "(no log yet)"

    # -- internals ----------------------------------------------------------

    def _wait_for_exit(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                return True
            time.sleep(0.2)
        return not self.is_running()

    def _reap(self) -> None:
        if self._log_file is not None:
            with contextlib.suppress(OSError):
                self._log_file.close()
            self._log_file = None
        # An adopted run is gone for good once stopped; forget it so a later
        # start() is not mistaken for "already running".
        self._adopted_pid = None


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` exists and is not a zombie (we cannot reap a non-child)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def _sweep_cloudxr_runtime(max_age_s: float) -> int:
    """Kill this user's stray ``cloudxr.runtime`` processes; return how many.

    The runtime routinely outlives the teleop process that spawned it and keeps
    holding 49100/48322, which is the usual reason the next run cannot bind.
    Killing it is the point of this sweep, but two guards keep it narrow:

    - **This uid only.** The workstation is shared; another user's runtime is
      neither ours to kill nor killable.
    - **Younger than ``max_age_s``.** A runtime older than the run we just
      stopped belongs to somebody else's session — typically one started by
      hand in a terminal — and must survive.

    Args:
        max_age_s: Kill only runtimes started within this many seconds.
    """
    try:
        # etimes is elapsed seconds since start, so age filtering needs no clock math.
        out = subprocess.run(
            ["ps", "-u", str(os.getuid()), "-o", "pid=,etimes=,cmd="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return 0
    killed = 0
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3 or "cloudxr.runtime" not in parts[2]:
            continue
        pid, etimes = int(parts[0]), int(parts[1])
        if etimes > max_age_s:
            print(f"[APP] Leaving CloudXR runtime {pid} alone ({etimes}s old — predates this run).")
            continue
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)
            killed += 1
    return killed


_DESKTOP_VARS = ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "DBUS_SESSION_BUS_ADDRESS")


def _desktop_env() -> dict[str, str]:
    """Display variables borrowed from this user's running desktop session.

    A systemd user service has no ``DISPLAY``, but teleop launched from a
    terminal always did, and parts of the stack (the retargeting tuning UI via
    GLFW) open windows. Copying the variables from any live desktop process
    makes a service launch behave like the terminal launch it replaces. Returns
    an empty dict when no desktop session is running, which is also fine.
    """
    if os.environ.get("DISPLAY"):
        return {k: os.environ[k] for k in _DESKTOP_VARS if k in os.environ}
    uid = os.getuid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = dict(kv.split(b"=", 1) for kv in f.read().split(b"\0") if b"=" in kv)
        except OSError:
            continue
        if env.get(b"DISPLAY"):
            return {k: env[k.encode()].decode() for k in _DESKTOP_VARS if k.encode() in env}
    return {}


def _port_in_use(port: int) -> bool:
    """Whether anything is listening on ``port`` (any user)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _preferred_host() -> str:
    """Stable address for links handed to the headset.

    Prefers ``<hostname>.local`` so a DHCP lease change cannot break the
    bookmark; falls back to the first LAN IP when mDNS is unavailable.
    """
    if subprocess.run(["systemctl", "is-active", "--quiet", "avahi-daemon"], check=False).returncode == 0:
        host = socket.gethostname().split(".")[0]
        if host:
            return f"{host}.local"
    ips = _lan_ips()
    return ips[0] if ips else "127.0.0.1"


class MicHub:
    """Receives the headset microphone in the page and fans it out to teleop.

    Owning the microphone here rather than inside ``make_teleop_scene.py`` is
    the point: capture keeps running across teleop restarts, so the operator
    taps "Start microphone" once per headset session instead of once per run.

    Exactly one page may stream at a time — the headset grants its mic to a
    single page, and leftover tabs otherwise keep stealing it back — so a new
    page evicts the previous one with :data:`_SUPERSEDED`.
    """

    def __init__(self):
        self._active = None
        self._subscribers: set = set()
        self._last_rx = 0.0

    def connected(self) -> bool:
        """Whether a page is currently streaming."""
        return self._active is not None

    def seconds_since_audio(self) -> float | None:
        """Age of the newest audio chunk, or None if none has ever arrived."""
        return None if self._last_rx == 0.0 else round(time.monotonic() - self._last_rx, 1)

    def status(self) -> dict:
        """JSON-ready snapshot for the page."""
        return {
            "mic_connected": self.connected(),
            "mic_silent_s": self.seconds_since_audio(),
            "consumers": len(self._subscribers),
        }

    async def ingest(self, connection) -> None:
        """Handle ``/audio``: one page streams PCM16, older pages are evicted."""
        from websockets.exceptions import ConnectionClosed

        previous, self._active = self._active, connection
        if previous is not None:
            with contextlib.suppress(Exception):
                await previous.close(code=_SUPERSEDED, reason="superseded")
            print("[APP] Superseded an older mic page (a stale tab was still open).")
        print("[APP] Microphone page connected.")
        try:
            async for message in connection:
                if connection is not self._active:
                    break  # a newer page took over
                if not isinstance(message, bytes):
                    continue
                self._last_rx = time.monotonic()
                for sub in list(self._subscribers):
                    # Never let a slow or dead consumer stall capture.
                    with contextlib.suppress(Exception):
                        await sub.send(message)
        except ConnectionClosed:
            pass  # routine: page navigated away, or we superseded it
        finally:
            if connection is self._active:
                self._active = None
                print("[APP] Microphone page disconnected.")

    async def subscribe(self, connection) -> None:
        """Handle ``/subscribe``: a teleop process consuming the relayed audio."""
        from websockets.exceptions import ConnectionClosed

        self._subscribers.add(connection)
        print(f"[APP] Teleop subscribed to microphone audio ({len(self._subscribers)} consumer(s)).")
        try:
            await connection.wait_closed()
        except ConnectionClosed:
            pass
        finally:
            self._subscribers.discard(connection)
            print("[APP] Teleop unsubscribed from microphone audio.")


_MANIFEST = {
    "name": "Teleop",
    "short_name": "Teleop",
    "start_url": "/",
    "scope": "/",
    "display": "standalone",
    "background_color": "#111111",
    "theme_color": "#111111",
    "description": "Start, monitor and kill duo teleop from the headset.",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
        # Deliberately not "maskable": that crops to a circle, and the visor is
        # wide enough that its ends would be shaved off.
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
    ],
}
"""Web app manifest, so the Quest can install this to the library as an app icon."""

_SERVICE_WORKER = """// Chromium only offers "install" for pages backed by a service worker with a
// fetch handler. Nothing is cached: the app is useless offline, since the whole
// point is talking to a workstation on the LAN.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", () => {});
"""

_ICON_CACHE: dict[int, bytes] = {}


def _icon_png(size: int) -> bytes:
    """A VR-visor glyph, drawn at ``size`` px, for the manifest and the tab icon.

    Generated rather than checked in so no binary asset has to live in the repo.
    """
    if size in _ICON_CACHE:
        return _ICON_CACHE[size]
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=size * 0.22, fill=(17, 17, 17, 255))
    visor_w, visor_h = size * 0.68, size * 0.36
    x0, y0 = (size - visor_w) / 2, (size - visor_h) / 2
    draw.rounded_rectangle([x0, y0, x0 + visor_w, y0 + visor_h], radius=visor_h * 0.38, fill=(76, 204, 76, 255))
    lens_r, cy = visor_h * 0.20, y0 + visor_h * 0.44
    for cx in (x0 + visor_w * 0.30, x0 + visor_w * 0.70):
        draw.ellipse([cx - lens_r, cy - lens_r, cx + lens_r, cy + lens_r], fill=(17, 17, 17, 255))
    # Nose notch, which is what makes the shape read as a headset at tile size.
    notch_half_w, bottom = visor_w * 0.07, y0 + visor_h
    draw.polygon(
        [(size / 2 - notch_half_w, bottom), (size / 2 + notch_half_w, bottom), (size / 2, y0 + visor_h * 0.74)],
        fill=(17, 17, 17, 255),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    _ICON_CACHE[size] = buf.getvalue()
    return _ICON_CACHE[size]


_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Teleop</title>
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" type="image/png" href="/icon-192.png">
<meta name="theme-color" content="#111111">
<style>
  body { font-family: sans-serif; background: #111; color: #eee; text-align: center; padding: 1.5em 1em; }
  button, a.btn { font-size: 1.3em; padding: 0.6em 1em; border-radius: 0.5em; border: 0; margin: 0.3em;
                  display: inline-block; background: #ddd; color: #111; text-decoration: none; }
  /* Highlighted once the session is up and only the CloudXR tap is left. */
  a.ready { background: #4c4; font-size: 1.7em; padding: 0.8em 1.6em; animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity: 0.55; } }
  a.disabled { pointer-events: none; opacity: 0.35; }
  a.ok { background: #2a5; color: #fff; }
  .primary { background: #4c4; font-size: 1.7em; padding: 0.8em 1.6em; }
  .danger { background: #c44; color: #fff; }
  #level { width: 60%; height: 0.9em; background: #333; margin: 0.8em auto; border-radius: 0.5em; overflow: hidden; }
  #bar { height: 100%; width: 0; background: #4c4; }
  .row { margin: 0.8em 0; }
  .dot { display: inline-block; width: 0.8em; height: 0.8em; border-radius: 50%; margin-right: 0.4em; }
  .on { background: #4c4; } .off { background: #666; }
  #log { text-align: left; font-family: monospace; font-size: 0.75em; white-space: pre-wrap;
         background: #000; padding: 0.6em; border-radius: 0.4em; max-height: 12em; overflow-y: auto; display: none; }
  #status { margin-top: 0.6em; font-size: 1.05em; color: #bbb; }
</style>
<h1>Teleop</h1>
<div class="row"><span id="teleop_dot" class="dot off"></span><span id="teleop_state">checking...</span></div>
<div class="row"><span id="mic_dot" class="dot off"></span><span id="mic_state">microphone idle</span></div>
<div id="level"><div id="bar"></div></div>

<button id="go" class="primary">Start session</button>
<div class="row">
  <button id="mic">Microphone only</button>
  <button id="micstop">Stop microphone</button>
  <!-- A real link, not window.open: a popup opened after the readiness wait is
       no longer tied to the tap and gets blocked silently. -->
  <a id="cxr" class="btn disabled" target="_blank" rel="noopener" href="#">Open CloudXR</a>
  <!-- One-time per port: the NVIDIA-hosted client cannot open a socket back
       here until this self-signed certificate is accepted for this origin. -->
  <a id="cert" class="btn disabled" target="_blank" rel="noopener" href="#">Teleop not running</a>
</div>
<div class="row">
  <button id="restart">Restart teleop</button>
  <button id="kill" class="danger">Kill teleop</button>
  <button id="logbtn">Log</button>
</div>
<div id="status">idle</div>
<pre id="log"></pre>

<script>
// Registering this is what lets the headset offer "Install"/"Add to library".
// It fails harmlessly on a self-signed origin the browser deems insecure, in
// which case a bookmark tile still works.
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});

const RATE = 16000, CHUNK = 1600, SUPERSEDED = 4001;
const $ = id => document.getElementById(id);
const status = t => $("status").textContent = t;
let ws = null, started = false, micStream = null, cxrUrl = null, audioCtx = null;

async function api(path, method) {
  const r = await fetch(path, { method: method || "GET", headers: { "X-Teleop-Control": "1" } });
  return await r.json();
}

// ---- status polling -------------------------------------------------------
// Whether this browser already trusts the proxy's certificate. A no-cors fetch
// resolves (opaque) once the certificate has been accepted for that host and
// port, and rejects while it has not. Only meaningful while the port is up,
// which is why it is gated on cloudxr_ready wherever it is called.
let certOk = null, lastProbe = 0;
async function probeCert(url) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), 4000);
  try { await fetch(url, { mode: "no-cors", cache: "no-store", signal: ctl.signal }); return true; }
  catch (e) { return false; }
  finally { clearTimeout(t); }
}

async function refresh() {
  try {
    const s = await api("/status");
    cxrUrl = s.cloudxr_url;
    $("cxr").href = cxrUrl;  // kept current, so the link is always tappable
    const cert = $("cert"), cxr = $("cxr");
    cert.href = s.cert_url;
    if (!s.cloudxr_ready) {
      // The proxy port is not listening yet. The certificate button doubles as
      // the boot indicator here: tapping it now would hang on a dead port for
      // as long as teleop takes to come up, so it is inert until then. The
      // CloudXR link likewise sheds any "ready" glow left over from a previous
      // run, which otherwise survives a kill/restart and lies.
      cert.classList.add("disabled"); cert.classList.remove("ready", "ok");
      cert.textContent = s.running ? "Preparing teleop..." : "Teleop not running";
      cxr.classList.add("disabled"); cxr.classList.remove("ready");
      certOk = null;
    } else {
      cert.classList.remove("disabled");
      cxr.classList.remove("disabled");
      if (certOk !== true && Date.now() - lastProbe > 3000) {
        lastProbe = Date.now();
        certOk = await probeCert(s.cert_url);
      }
      if (certOk) {
        cert.textContent = "Teleop ready - certificate OK";
        cert.classList.add("ok"); cert.classList.remove("ready");
      } else {
        cert.textContent = "Teleop ready - accept certificate";
        cert.classList.remove("ok");
      }
    }
    $("teleop_dot").className = "dot " + (s.running ? "on" : "off");
    $("teleop_state").textContent = s.running
      ? "teleop running (pid " + s.pid + ", up " + Math.round(s.uptime_s) + "s)"
      : "teleop stopped" + (s.exit_code === null ? "" : " (exit " + s.exit_code + ")");
    $("mic_dot").className = "dot " + (s.mic_connected ? "on" : "off");
    if (s.mic_connected && !started) {
      // The server sees a page streaming, but it is not this one: either a
      // second app tab, or a socket this page just closed that the server has
      // not timed out yet. Saying "streaming" here read as Stop having failed.
      $("mic_state").textContent = "microphone streaming from another tab";
    } else if (s.mic_connected) {
      const age = s.mic_silent_s;
      $("mic_state").textContent = (age !== null && age > 4)
        ? "microphone connected but silent " + age + "s" : "microphone streaming";
    } else {
      $("mic_state").textContent = started ? "microphone starting..." : "microphone idle";
    }
  } catch (e) { $("teleop_state").textContent = "app unreachable"; }
}
setInterval(refresh, 1500);
refresh();

// ---- microphone -----------------------------------------------------------
function standDown(msg) {
  status(msg); started = false; $("bar").style.width = "0%";
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
}

function connect() {
  if (ws) { ws.onclose = null; ws.onerror = null; try { ws.close(); } catch (e) {} }
  ws = new WebSocket("wss://" + location.host + "/audio");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => status("microphone streaming");
  ws.onclose = (ev) => {
    if (ev.code === SUPERSEDED) { standDown("superseded by a newer tab - close this one"); return; }
    status("mic disconnected - retrying"); setTimeout(connect, 1000);
  };
  ws.onerror = () => ws.close();
}

async function startMic() {
  if (started) return true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
  } catch (err) {
    status("mic failed: " + err.name + " - close any other copy of this page, then tap again");
    return false;
  }
  started = true; micStream = stream;
  const ctx = audioCtx = new AudioContext();
  // A tab that is PLAYING audio is exempt from Chromium's background-tab
  // freezing, which otherwise kills capture minutes into an immersive session.
  const osc = ctx.createOscillator(), g = ctx.createGain();
  g.gain.value = 0.001; osc.frequency.value = 30;
  osc.connect(g).connect(ctx.destination); osc.start();
  ctx.onstatechange = () => { if (ctx.state !== "running") ctx.resume(); };
  stream.getTracks()[0].onended = () => { status("mic lost - restarting"); started = false; startMic(); };
  const code = `registerProcessor("grab", class extends AudioWorkletProcessor {
      process(i) { if (i[0] && i[0][0]) this.port.postMessage(i[0][0]); return true; } });`;
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
  const node = new AudioWorkletNode(ctx, "grab");
  ctx.createMediaStreamSource(stream).connect(node);
  const ratio = ctx.sampleRate / RATE;
  let buf = [], acc = 0, out = new Int16Array(CHUNK), oi = 0, peak = 0, frames = 0;
  node.port.onmessage = (e) => {
    const x = e.data;
    for (let i = 0; i < x.length; i++) buf.push(x[i]);
    while (acc + ratio < buf.length) {
      const j = Math.floor(acc), f = acc - j;
      const v = buf[j] * (1 - f) + buf[j + 1] * f;
      out[oi++] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
      peak = Math.max(peak, Math.abs(v));
      acc += ratio;
      if (oi === CHUNK) {
        if (ws && ws.readyState === 1) ws.send(out.buffer.slice(0));
        oi = 0;
        if (++frames % 5 === 0) { $("bar").style.width = Math.min(100, peak * 400) + "%"; peak = 0; }
      }
    }
    const drop = Math.floor(acc); buf = buf.slice(drop); acc -= drop;
  };
  connect();
  return true;
}

// ---- buttons --------------------------------------------------------------
$("mic").onclick = () => startMic();

function stopMic() {
  // Detach handlers before tearing down: onclose would otherwise reconnect and
  // the track's onended would treat this as a lost mic and start it again.
  if (ws) { ws.onclose = null; ws.onerror = null; try { ws.close(); } catch (e) {} ws = null; }
  if (micStream) { micStream.getTracks().forEach(t => { t.onended = null; t.stop(); }); micStream = null; }
  if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
  started = false; $("bar").style.width = "0%";
  status("microphone stopped");
}
$("micstop").onclick = stopMic;

$("go").onclick = async () => {
  const go = $("go");
  go.disabled = true;
  try {
    status("starting microphone...");
    if (!await startMic()) return;
    let s = await api("/status");
    if (!s.running) { status("starting teleop..."); await api("/start", "POST"); }
    // The process existing is not the same as the XR stack being up; opening
    // the client early lands on a page that cannot connect yet.
    const deadline = Date.now() + 180000;
    while (!s.cloudxr_ready && Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 2000));
      s = await api("/status");
      if (!s.running && s.exit_code !== null) {
        status("teleop exited (code " + s.exit_code + ") - tap Log to see why");
        return;
      }
      status("waiting for teleop to come up... " + Math.round((Date.now() - (deadline - 180000)) / 1000) + "s");
    }
    if (!s.cloudxr_ready) { status("teleop did not come up in 3 min - tap Log"); return; }
    // Certificate gate. The NVIDIA-hosted client cannot open its socket to the
    // proxy until this browser has accepted the proxy's certificate, and that
    // acceptance is not always remembered between sessions. Only ask when the
    // probe says it is actually missing, then advance as soon as it is granted.
    if (!(await probeCert(s.cert_url))) {
      const cert = $("cert");
      cert.classList.remove("disabled"); cert.classList.add("ready");
      status("tap the green certificate button, accept the warning, then come back to this tab");
      const certDeadline = Date.now() + 300000;
      while (Date.now() < certDeadline && !(await probeCert(s.cert_url))) {
        await new Promise(r => setTimeout(r, 2000));
      }
      cert.classList.remove("ready");
      if (!(await probeCert(s.cert_url))) {
        status("certificate still not accepted - tap the certificate button, then 'Open CloudXR'");
        return;
      }
    }
    certOk = true;
    // Try to open it directly, but the tap that started this is long expired,
    // so the browser may refuse. window.open returns null when it does.
    const w = window.open(s.cloudxr_url, "_blank");
    const cxr = $("cxr");
    cxr.href = s.cloudxr_url;
    if (w) {
      status("running - keep THIS tab open, the mic runs here");
    } else {
      cxr.classList.add("ready");
      status("ready - tap 'Open CloudXR' below (keep THIS tab open, the mic runs here)");
    }
  } finally {
    go.disabled = false;
  }
};
$("restart").onclick = async () => { status("restarting..."); status((await api("/restart", "POST")).message); };
$("kill").onclick = async () => { status("killing..."); status((await api("/stop", "POST")).message); };
$("logbtn").onclick = async () => {
  const el = $("log");
  el.style.display = el.style.display === "none" ? "block" : "none";
  if (el.style.display === "block") { el.textContent = (await api("/log")).log; el.scrollTop = el.scrollHeight; }
};
</script>
"""


class TeleopApp:
    """The HTTPS/WSS service: control page, mic hub, and teleop supervision."""

    def __init__(self, port: int, teleop: TeleopProcess):
        self._port = port
        self._teleop = teleop
        self._mic = MicHub()

    @staticmethod
    def _proxy_port() -> int:
        """Port of the CloudXR WSS proxy, which moves when the host is shared."""
        return int(os.environ.get("PROXY_PORT", "").strip() or 48322)

    @staticmethod
    def _client_base() -> str:
        """Origin of the WebXR client, which NVIDIA hosts rather than this machine.

        Resolved from the installed ``isaacteleop`` so the client matches the
        runtime it talks to (1.3.x gets the ``release-1.3.x`` build).
        """
        try:
            from isaacteleop.cloudxr.oob_teleop_env import (
                default_web_client_origin,
                web_client_base_override_from_env,
            )

            return (web_client_base_override_from_env() or default_web_client_origin()).rstrip("/")
        except Exception:
            return "https://nvidia.github.io/IsaacTeleop/client/main"

    def cert_url(self) -> str:
        """Page that exists solely to let the headset accept the proxy's certificate.

        The client runs on an NVIDIA origin but opens a secure socket back here,
        and the browser refuses that until this self-signed certificate has been
        accepted for this exact host and port. Visiting this once per port is
        what unblocks it; it answers "Certificate Accepted" and nothing else.
        """
        return f"https://{_preferred_host()}:{self._proxy_port()}/"

    def cloudxr_url(self) -> str:
        """WebXR client URL pointed at this machine's current ports.

        Built per request rather than bookmarked, which is what kept going stale:
        ``PROXY_PORT`` moves when sharing the host with another user.
        """
        host, proxy = _preferred_host(), self._proxy_port()
        # serverIP/port are query params the client reads on load, so its
        # connection fields arrive pre-filled instead of typed in the headset.
        return f"{self._client_base()}/?serverIP={host}&port={proxy}"

    def status(self) -> dict:
        """Everything the page polls, in one round trip."""
        return {
            **self._teleop.status(),
            **self._mic.status(),
            "cloudxr_url": self.cloudxr_url(),
            "cert_url": self.cert_url(),
            # Teleop takes ~30-60 s to boot, and the process existing says
            # nothing about the XR stack being up. The proxy port going live is
            # the signal that the client has something to connect to.
            "cloudxr_ready": _port_in_use(self._proxy_port()),
        }

    async def serve(self) -> None:
        """Run until cancelled."""
        from websockets.asyncio.server import serve

        def process_request(connection, request):
            path = (request.path or "/").split("?")[0]
            if "upgrade" in request.headers.get("Connection", "").lower():
                return None  # websocket routes are handled in the handler
            if path == "/":
                r = connection.respond(http.HTTPStatus.OK, _PAGE)
                r.headers["Content-Type"] = "text/html; charset=utf-8"
                # The page changes whenever the app is updated; a reload on the
                # headset must always fetch it, never replay a cached copy.
                r.headers["Cache-Control"] = "no-store"
                return r
            if path == "/manifest.webmanifest":
                r = connection.respond(http.HTTPStatus.OK, json.dumps(_MANIFEST))
                r.headers["Content-Type"] = "application/manifest+json"
                return r
            if path == "/sw.js":
                r = connection.respond(http.HTTPStatus.OK, _SERVICE_WORKER)
                r.headers["Content-Type"] = "text/javascript"
                r.headers["Cache-Control"] = "no-store"
                return r
            if path in ("/icon-192.png", "/icon-512.png"):
                return self._binary(_icon_png(512 if "512" in path else 192), "image/png")
            if path == "/status":
                return self._json(connection, self.status())
            if path == "/log":
                return self._json(connection, {"log": self._teleop.tail_log()})
            if path in ("/start", "/stop", "/restart"):
                # The websockets server admits any method here, so a stray
                # reload or a link prefetch of /stop would otherwise kill a live
                # session. A custom header cannot be produced by navigation or
                # by a cross-origin form, so only our own page can act.
                if request.headers.get(_CONTROL_HEADER) is None:
                    return connection.respond(
                        http.HTTPStatus.FORBIDDEN, f"{path} needs the {_CONTROL_HEADER} header; use the app page.\n"
                    )
                action = {"/start": self._teleop.start, "/stop": self._teleop.stop, "/restart": self._teleop.restart}
                ok, message = action[path]()
                print(f"[APP] {path[1:]}: {message}")
                return self._json(connection, {"ok": ok, "message": message})
            return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")

        async def handler(connection):
            path = (connection.request.path or "/").split("?")[0]
            if path == "/audio":
                await self._mic.ingest(connection)
            elif path == "/subscribe":
                await self._mic.subscribe(connection)
            else:
                await connection.close(code=1008, reason="unknown path")

        host = _preferred_host()
        # Short ping cycle so a headset tab that vanished without a close frame
        # is dropped in ~10 s, not the default ~40 s, and the indicator follows.
        async with serve(
            handler,
            "0.0.0.0",
            self._port,
            ssl=_ssl_context(),
            process_request=process_request,
            ping_interval=5,
            ping_timeout=5,
        ):
            print(f"[APP] Teleop app ready. Bookmark this on the headset:\n[APP]     https://{host}:{self._port}/")
            print(f"[APP] CloudXR client link it will hand out: {self.cloudxr_url()}")
            await asyncio.Future()  # run forever

    @staticmethod
    def _json(connection, payload: dict):
        r = connection.respond(http.HTTPStatus.OK, json.dumps(payload))
        r.headers["Content-Type"] = "application/json"
        return r

    @staticmethod
    def _binary(body: bytes, content_type: str):
        """Serve bytes, which ``connection.respond`` cannot do (it text-encodes)."""
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        headers = Headers()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(body))
        headers["Cache-Control"] = "public, max-age=86400"
        return Response(http.HTTPStatus.OK.value, "OK", headers, body)


_SERVICE_UNIT = """[Unit]
Description=Teleop control app (sharpa_duo)
After=network-online.target avahi-daemon.service

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStart={python} -u {script} --port {port} {teleop_args}
Restart=always
RestartSec=3
# Stop/restart the app without taking teleop with it. systemd's default
# (control-group) would kill the whole cgroup, including the child run, which
# defeats the point of supervising it from a process that can be restarted.
KillMode=process

[Install]
WantedBy=default.target
"""


def _install_service(port: int, teleop_args: list[str]) -> None:
    """Write and enable a **user** systemd unit, so the app survives reboots.

    Runs the interpreter directly rather than through ``isaaclab.sh``: a user
    unit gets a minimal environment with no active virtualenv, and the app
    itself needs no Isaac Sim — only the teleop child it launches does.
    """
    unit_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(unit_dir, exist_ok=True)
    unit_path = os.path.join(unit_dir, "teleop-app.service")
    body = _SERVICE_UNIT.format(
        python=sys.executable,
        workdir=os.path.dirname(_ISAACLAB_SH),
        script=os.path.abspath(__file__),
        port=port,
        teleop_args=" ".join(f"--teleop-arg {shlex.quote(a)}" for a in teleop_args),
    )
    with open(unit_path, "w") as f:
        f.write(body)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "teleop-app.service"], check=False)
    # "enable --now" would leave an already-running copy on the old code; restart
    # picks up the new script (and, with KillMode=process, spares any teleop).
    subprocess.run(["systemctl", "--user", "restart", "teleop-app.service"], check=False)
    print(f"[APP] Installed {unit_path} and (re)started it.")
    print("[APP] Survive logout with:  sudo loginctl enable-linger $USER")
    print("[APP] Follow it with:       journalctl --user -u teleop-app -f")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Fixed port for the app page and mic hub.")
    parser.add_argument(
        "--teleop-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Argument forwarded to make_teleop_scene.py; repeat once per argument.",
    )
    parser.add_argument("--install-service", action="store_true", help="Install and start the systemd user service.")
    args = parser.parse_args()

    # The app owns the microphone, so teleop must consume the relay rather than
    # serving a competing page of its own.
    teleop_args = list(args.teleop_arg)
    if not any(a.startswith("--mic_device") for a in teleop_args):
        teleop_args += ["--mic_device", f"hub:127.0.0.1:{args.port}"]
    if "--headless" not in teleop_args:
        teleop_args.append("--headless")

    if args.install_service:
        _install_service(args.port, args.teleop_arg)
        return

    if _port_in_use(args.port):
        raise SystemExit(f"[APP] Port {args.port} is already in use — another copy of the app is probably running.")

    app = TeleopApp(args.port, TeleopProcess(teleop_args, env_overrides={}))
    try:
        asyncio.run(app.serve())
    except KeyboardInterrupt:
        print("\n[APP] Shutting down; leaving any running teleop alone (use Kill in the page to stop it).")


if __name__ == "__main__":
    main()
