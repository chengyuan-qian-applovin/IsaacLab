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
  uses, read at request time rather than baked into a bookmark;
- **configure the session** from the headset — the same things the desktop
  launcher (:mod:`teleop_launcher`) offers: the teleop parameters, the record
  directory, and a **scene source** (a local scene directory with this
  machine's per-scene demo counts, or the fleet server with its live
  progress and who is collecting what), ticking the scenes to collect. Both
  UIs share :mod:`session_config` — one schema, one settings file
  (``~/.config/duo_teleop_launcher.json``) — so a choice made on either side
  shows up on the other. "Start session" launches ``make_teleop_scene.py``
  from the saved settings.

Bookmark it once as ``https://<hostname>.local:8500/`` — mDNS keeps that
address correct across DHCP leases, and the port is fixed by this service.

Run it directly for a quick trial::

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py

or install it as a user service so it comes back after a reboot::

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py --install-service

The app passes ``--mic_device hub`` to every teleop it launches (see
:mod:`headset_mic`), which makes teleop pull audio from this app instead of
serving its own page. Anything the settings page does not cover can still be
forwarded once per argument with ``--teleop-arg``.
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
import threading
import time
import urllib.parse
from collections import deque
from collections.abc import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fleet_client import FleetMonitor  # noqa: E402
from headset_mic import _lan_ips, _ssl_context  # noqa: E402  (shares the CloudXR cert)
from session_config import (  # noqa: E402
    PORTS_NOTE,
    TELEOP_SCRIPT,
    LaunchSpec,
    SceneTable,
    build_env,
    build_launch,
    load_settings,
    merge_settings,
    normalize_server_url,
    param_groups,
    save_settings,
    scene_table,
    schema,
)

DEFAULT_PORT = 8500
"""Fixed port for the app. Bookmarked once, so it must not move."""

_TELEOP_SCRIPT = TELEOP_SCRIPT
_ISAACLAB_SH = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", "isaaclab.sh"))
_PAGE_PATH = os.path.join(_HERE, "teleop_app_page.html")
"""The control page, read per request so an edit shows on the next reload."""

_CSS_PATH = os.path.join(_HERE, "teleop_app_page.css")
"""The page's stylesheet (served at ``/app.css``), read per request like the page."""

_CLIENT_MIC_PATH = os.path.join(_HERE, "client_mic.js")
"""Microphone capture script added to the WebXR client the app serves at ``/client/``."""

_SUPERSEDED = 4001
"""Close code telling an older mic page a newer one took over (see :mod:`headset_mic`)."""
_PEAK_WINDOW_CHUNKS = 20
"""Chunks (0.1 s each) over which the page's "muted" verdict looks for any sound."""

_SUBSCRIBER_BACKLOG_BYTES = 32 * 1024
"""Unsent audio (about 1 s) a teleop consumer may have queued before further chunks are dropped for it."""

_SOUND_PEAK = 0.02
"""Chunk peak (0..1) above the headset's noise floor: the operator is speaking, not just present."""
"""Close code telling an older mic page a newer one took over (see :mod:`headset_mic`)."""


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

    Every launch asks ``launch_provider`` for the :class:`LaunchSpec` built
    from the saved settings (scene source, selection, parameters, ports);
    ``base_args`` are the service-level ``--teleop-arg`` extras, placed first
    so the settings win where both name a flag.
    """

    def __init__(self, base_args: list[str], launch_provider: Callable[[], LaunchSpec]):
        self._base_args = list(base_args)
        self._launch_provider = launch_provider
        self._last_launch: LaunchSpec | None = None
        self._last_command: list[str] = [_ISAACLAB_SH, "-p", _TELEOP_SCRIPT, *self._base_args]
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
        """Launch teleop from the saved settings if it is not already up. Returns ``(ok, message)``."""
        if self.is_running():
            return False, f"already running (pid {self.pid})"
        if not os.path.exists(_ISAACLAB_SH):
            return False, f"launcher not found: {_ISAACLAB_SH}"
        try:
            spec = self._launch_provider()
        except ValueError as exc:  # operator-facing: nothing selected, bad port, no server URL...
            return False, str(exc)
        cmd = [_ISAACLAB_SH, "-p", _TELEOP_SCRIPT, *self._base_args, *spec.args]
        # Desktop vars first so the settings' port variables still win.
        env = {**os.environ, **_desktop_env(), **spec.env}
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
        self._last_launch, self._last_command = spec, cmd
        shown_env = " ".join(f"{k}={'***' if k == 'FLEET_TOKEN' else v}" for k, v in spec.env.items())
        print("[APP] Launching: " + " ".join(filter(None, [shown_env, *(shlex.quote(a) for a in cmd)])), flush=True)
        return True, f"started (pid {self._proc.pid}) with {spec.summary}; log: {self._log_path}"

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

    def launch_env(self) -> dict[str, str]:
        """The extra environment of the run we started (empty for an adopted run or before the first start)."""
        return dict(self._last_launch.env) if self._last_launch is not None else {}

    def status(self) -> dict:
        """JSON-ready snapshot for the page."""
        running = self.is_running()
        return {
            "running": running,
            "pid": self.pid if running else None,
            "uptime_s": round(time.monotonic() - self._started_at, 1) if running else 0.0,
            "exit_code": None if running else (self._proc.returncode if self._proc else None),
            "log": self._log_path,
            "command": " ".join(shlex.quote(a) for a in self._last_command),
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


def _pcm16_peak(chunk: bytes) -> float:
    """Loudest |sample| of a little-endian PCM16 chunk, normalised to 0..1."""
    if len(chunk) < 2:
        return 0.0
    samples = memoryview(chunk)[: len(chunk) // 2 * 2].cast("h")
    return max(abs(min(samples)), abs(max(samples))) / 32768.0


def _frame_source(message: str) -> str | None:
    """The ``source`` a page stamps on its diagnostic frames, if it is one we know."""
    try:
        source = json.loads(message).get("source")
    except (ValueError, AttributeError):
        return None
    return source if source in ("app", "client") else None


_CLIENT_MIC_TAG = b'<script defer="defer" src="mic.js"></script>'


def _inject_mic_script(index_html: bytes) -> bytes:
    """The client's ``index.html`` with the microphone script tag added after its bundle."""
    marker = b'<script defer="defer" src="bundle.js"></script>'
    if marker in index_html:
        return index_html.replace(marker, marker + _CLIENT_MIC_TAG, 1)
    if b"</body>" in index_html:
        return index_html.replace(b"</body>", _CLIENT_MIC_TAG + b"</body>", 1)
    return index_html + _CLIENT_MIC_TAG


def _client_mic_js() -> bytes:
    """The capture script served into the client page (``client_mic.js`` next to this file, read per request)."""
    try:
        with open(_CLIENT_MIC_PATH, "rb") as f:
            return f.read()
    except OSError as exc:
        return f"console.error({json.dumps(f'cannot read {_CLIENT_MIC_PATH}: {exc}')});".encode()


class WebClientAssets:
    """NVIDIA's WebXR client, served by the app so the microphone script can ride along.

    The headset keeps exactly one window in the foreground during an immersive
    session: the CloudXR client. A page behind it has its capture paused about a
    minute after it is hidden and its track ended whenever the operator switches
    windows, which is what made the microphone die mid-session while the app
    page owned it. Serving the client from here and adding ``client_mic.js`` to
    it puts the microphone in the window that stays awake.

    The files come from the same versioned nvidia.github.io origin ``isaacteleop``
    resolves for the installed runtime (``default_web_client_origin``), cached
    under ``~/.cache/teleop_app/client/<release>/`` with their ETags, and
    re-checked with conditional GETs because the "release-X.Y.x" build is a
    moving target. ``TELEOP_WEB_CLIENT_STATIC_DIR`` (isaacteleop's own knob)
    pins a directory to serve verbatim instead. While nothing is available
    (offline first start) the app hands out the nvidia.github.io link and the
    microphone stays on the app page.
    """

    _RETRY_S = 60.0
    _REFRESH_S = 6 * 3600.0
    _MAX_BYTES = 32 * 1024 * 1024
    _FILES = ("index.html", "bundle.js")

    def __init__(self):
        self._lock = threading.Lock()
        self._files: dict[str, bytes] = {}
        self._etags: dict[str, str] = {}
        self._last_attempt = 0.0
        self._pinned_dir = os.environ.get("TELEOP_WEB_CLIENT_STATIC_DIR", "").strip() or None

    def available(self) -> bool:
        """Whether both files are in memory."""
        return all(name in self._files for name in self._FILES)

    def index(self) -> bytes:
        """The client's ``index.html`` with the microphone script tag added."""
        return _inject_mic_script(self._files.get("index.html", b""))

    def bundle(self) -> bytes:
        return self._files.get("bundle.js", b"")

    def etag(self, name: str) -> str | None:
        return self._etags.get(name)

    @staticmethod
    def _origin() -> str | None:
        try:
            from isaacteleop.cloudxr.oob_teleop_env import default_web_client_origin

            return default_web_client_origin()
        except Exception:  # noqa: BLE001 — isaacteleop missing or too old: no local client
            return None

    def _cache_dir(self, origin: str) -> str:
        release = origin.rstrip("/").rsplit("/", 1)[-1] or "client"
        return os.path.join(os.path.expanduser("~/.cache"), "teleop_app", "client", release)

    def load(self) -> bool:
        """Read the cache, then refresh it from the origin when due. Blocking: run off the event loop.

        Returns :meth:`available`. Failures are logged once per attempt and retried
        after :attr:`_RETRY_S`; a stale cache keeps being served meanwhile.
        """
        with self._lock:
            if time.monotonic() - self._last_attempt < self._RETRY_S:
                return self.available()
            self._last_attempt = time.monotonic()
            if self._pinned_dir:
                return self._read_dir(self._pinned_dir, pinned=True)
            origin = self._origin()
            if origin is None:
                return False
            cache = self._cache_dir(origin)
            if not self._files:
                self._read_dir(cache, pinned=False)
            self._refresh(origin, cache)
            return self.available()

    def _read_dir(self, directory: str, pinned: bool) -> bool:
        files, etags = {}, {}
        for name in self._FILES:
            try:
                with open(os.path.join(directory, name), "rb") as f:
                    files[name] = f.read()
                with open(os.path.join(directory, name + ".etag"), encoding="utf-8") as f:
                    etags[name] = f.read().strip()
            except OSError:
                if name not in files:
                    if pinned:
                        print(f"[APP] TELEOP_WEB_CLIENT_STATIC_DIR={directory} lacks {name}; no local WebXR client.")
                    return False
        if all(files.get(n) for n in self._FILES):
            self._files, self._etags = files, etags
            if pinned:
                print(f"[APP] Serving the pinned WebXR client from {directory} at /client/ with the microphone script.")
            return True
        return False

    def _refresh(self, origin: str, cache: str) -> None:
        """Conditional GET of each file; new bodies are written atomically and swapped in."""
        import urllib.error
        import urllib.request

        changed = []
        for name in self._FILES:
            url = origin.rstrip("/") + "/" + name
            req = urllib.request.Request(url, headers={"User-Agent": "teleop-app"})
            if name in self._files and self._etags.get(name):
                req.add_header("If-None-Match", self._etags[name])
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read(self._MAX_BYTES + 1)
                    etag = resp.headers.get("ETag", "")
            except urllib.error.HTTPError as exc:
                if exc.code == 304:
                    continue
                print(
                    f"[APP] WebXR client: {url} -> HTTP {exc.code}; keeping the cached copy."
                    if name in self._files
                    else f"[APP] WebXR client not available ({url} -> HTTP {exc.code}); handing out the nvidia link."
                )
                return
            except (OSError, ValueError) as exc:
                print(
                    f"[APP] WebXR client: could not reach {url} ({exc}); keeping the cached copy."
                    if name in self._files
                    else f"[APP] WebXR client not available ({exc}); handing out the nvidia.github.io link."
                )
                return
            if not body or len(body) > self._MAX_BYTES:
                print(f"[APP] WebXR client: {url} returned {len(body)} bytes; ignored.")
                return
            if name == "index.html" and b"Content-Security-Policy" in body:
                print(
                    "[APP] WebXR client: index.html now has a Content-Security-Policy; the mic script may be blocked."
                )
            try:
                os.makedirs(cache, exist_ok=True)
                for fname, data in ((name, body), (name + ".etag", etag.encode())):
                    tmp = os.path.join(cache, fname + ".part")
                    with open(tmp, "wb") as f:
                        f.write(data)
                    os.replace(tmp, os.path.join(cache, fname))
            except OSError as exc:
                print(f"[APP] WebXR client: cannot write {cache} ({exc}); serving from memory only.")
            self._files[name], self._etags[name] = body, etag
            changed.append(name)
        if changed:
            print(
                f"[APP] WebXR client {origin} cached at {cache} ({', '.join(changed)} updated);"
                " served at /client/ with the microphone script."
            )
        elif self.available():
            print(f"[APP] WebXR client {origin} is current (cache {cache}); served at /client/ with the mic script.")

    def refresh_loop(self) -> None:
        """Keep the cache current for as long as the app runs (daemon thread)."""
        while True:
            try:
                self.load()
            except Exception as exc:  # noqa: BLE001 — a refresh must never kill the app
                print(f"[APP] WebXR client refresh failed: {exc}")
            time.sleep(self._REFRESH_S if self.available() else self._RETRY_S)


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


class SessionConfigurator:
    """The app's side of :mod:`session_config`: settings on disk, the fleet monitor, table rows, launches.

    Holds the settings the page edits (persisted in the file shared with the
    desktop launcher — re-read whenever that file changes underneath), runs a
    :class:`~fleet_client.FleetMonitor` while the scene source is the fleet
    server, and turns the saved state into the :class:`LaunchSpec` every
    "Start session" runs. Everything here may block on disk or network, so the
    server calls it from a worker thread.
    """

    _TABLE_TTL_S = 5.0
    """How long a scanned scene table is reused for the (frequent) status summary."""

    def __init__(self, mic_device: str | None):
        self._mic_device = mic_device
        self._lock = threading.RLock()
        self._settings = load_settings()
        self._settings_mtime = self._file_mtime()
        self._monitor: FleetMonitor | None = None
        self._table: SceneTable | None = None
        self._table_at = 0.0
        self._sync_monitor()

    # -- settings -------------------------------------------------------------

    @staticmethod
    def _file_mtime() -> float:
        from session_config import SETTINGS_PATH

        try:
            return os.stat(SETTINGS_PATH).st_mtime
        except OSError:
            return 0.0

    def settings(self) -> dict:
        """The current settings; picks up edits made by the desktop launcher to the shared file."""
        with self._lock:
            mtime = self._file_mtime()
            if mtime != self._settings_mtime:
                self._settings = load_settings()
                self._settings_mtime = mtime
                self._table = None
                self._sync_monitor()
            return self._settings

    def update(self, patch: dict) -> dict:
        """Merge ``patch`` (the page's settings) into the saved settings and persist them."""
        if not isinstance(patch, dict):
            raise ValueError("settings must be an object")
        with self._lock:
            settings = merge_settings(self.settings(), patch)
            settings["fleet_server"] = normalize_server_url(settings.get("fleet_server", ""))
            save_settings(settings)
            self._settings, self._settings_mtime = settings, self._file_mtime()
            self._table = None
            self._sync_monitor()
            return settings

    # -- fleet ----------------------------------------------------------------

    def _sync_monitor(self) -> None:
        """Run exactly one monitor, for the saved server, only while the source is the fleet server."""
        settings = self._settings
        server = (
            normalize_server_url(settings.get("fleet_server", "")) if settings.get("scene_source") == "server" else ""
        )
        token = str(settings.get("fleet_token", "")).strip() or None
        if self._monitor is not None and (not server or not self._monitor.matches(server, token)):
            self._monitor.stop()
            self._monitor = None
        if server and self._monitor is None:
            self._monitor = FleetMonitor(server, token)
            self._monitor.start()

    def connect_fleet(self) -> None:
        """Poll the fleet server now (the operator pressed Connect)."""
        with self._lock:
            self.settings()
            if self._monitor is None:
                if self._settings.get("scene_source") != "server":
                    raise ValueError("Switch the scene source to 'Fleet server' first.")
                raise ValueError("Enter the fleet server URL first (e.g. http://fleet-host:8080).")
            self._monitor.poll_now()

    def fleet_status(self) -> dict | None:
        with self._lock:
            return self._monitor.status() if self._monitor is not None else None

    # -- scenes & launch --------------------------------------------------------

    def table(self, max_age_s: float = 0.0) -> SceneTable:
        """The active source's scene table, rescanned unless a copy younger than ``max_age_s`` exists."""
        with self._lock:
            settings = self.settings()
            if self._table is not None and time.monotonic() - self._table_at <= max_age_s:
                return self._table
            scenes = self._monitor.scenes if self._monitor is not None else None
            self._table = scene_table(settings, scenes)
            self._table_at = time.monotonic()
            return self._table

    def launch(self) -> LaunchSpec:
        """The launch for the saved settings; raises ``ValueError`` with an operator-facing message."""
        with self._lock:
            settings = self.settings()
            if settings.get("scene_source") == "server":
                fleet = self.fleet_status()
                if fleet is None or not fleet["connected"]:
                    raise ValueError("Connect to the fleet server before starting (Scenes tab).")
            return build_launch(settings, self.table(), mic_device=self._mic_device)

    def port_env(self) -> dict[str, str]:
        """The port variables the saved settings would give a launch (for links before the first start)."""
        try:
            return build_env(self.settings().get("params", {}))
        except ValueError:
            return {}

    def summary(self) -> str:
        """One line for the session tab: what "Start session" would launch."""
        settings = self.settings()
        table = self.table(max_age_s=self._TABLE_TTL_S)
        robot = settings.get("params", {}).get("--embodiment", "?")
        if table.source == "server":
            fleet = self.fleet_status()
            server = settings.get("fleet_server") or "(no server URL)"
            if fleet is None or not fleet["connected"]:
                return f"Fleet server {server}: not connected — robot {robot}"
            t = fleet["totals"]
            return (
                f"Fleet server {server}: {len(table.selected)} of {len(table.rows)} scenes selected,"
                f" {t['successes_toward_target']}/{t['target_successes']} successes fleet-wide — robot {robot}"
            )
        scene_dir = settings.get("scene_dir") or ""
        return (
            f"Local scenes: {len(table.selected)} of {len(table.rows)} selected under"
            f" {os.path.basename(scene_dir.rstrip('/')) or scene_dir} — robot {robot}"
        )

    def state(self) -> dict:
        """Everything the configuration tabs render, in one message."""
        settings = self.settings()
        table = self.table()
        return {
            "settings": settings,
            "schema": schema(),
            "groups": param_groups(),
            "ports_note": PORTS_NOTE,
            # The app relays the headset mic itself, so the mic port is not a knob here.
            "hidden_params": ["port:mic"] if self._mic_device else [],
            "table": {"source": table.source, "rows": table.rows, "untagged": table.untagged},
            "fleet": self.fleet_status(),
            "summary": self.summary(),
        }

    def close(self) -> None:
        with self._lock:
            if self._monitor is not None:
                self._monitor.stop()
                self._monitor = None


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
        # Loudest sample of each of the newest chunks, 0..1. The page reads the
        # window's maximum: a single 0.1 s chunk is silent between every two
        # words once the headset's noise suppression is on, and judging from it
        # made the label flip to "muted" whenever a poll landed in a pause.
        self._peaks: deque[float] = deque(maxlen=_PEAK_WINDOW_CHUNKS)
        self._chunks = 0
        self._last_sound = 0.0  # when a chunk last carried speech-level sound
        self._source: str | None = None  # "app" (control page) or "client" (CloudXR page), from its diagnostics

    def connected(self) -> bool:
        """Whether a page is currently streaming."""
        return self._active is not None

    def seconds_since_audio(self) -> float | None:
        """Age of the newest audio chunk, or None if none has ever arrived."""
        return None if self._last_rx == 0.0 else round(time.monotonic() - self._last_rx, 1)

    def seconds_since_sound(self) -> float | None:
        """Age of the newest chunk that carried speech-level sound, or None if none has yet."""
        return None if self._last_sound == 0.0 else round(time.monotonic() - self._last_sound, 1)

    def status(self) -> dict:
        """JSON-ready snapshot for the page."""
        return {
            "mic_connected": self.connected(),
            "mic_silent_s": self.seconds_since_audio(),
            # Silence is the normal state between voice commands (the headset's
            # noise suppression gates it to exact zeros), so the page reports
            # how long ago a voice was last heard rather than calling silence
            # "muted".
            "mic_heard_s": self.seconds_since_sound(),
            # Which page holds the microphone: the app's control page, or the
            # CloudXR client page the app serves with its own capture script.
            "mic_source": self._source if self.connected() else None,
            # Tells silence apart from "no frames": frames arriving with a
            # peak near zero for the whole window mean the headset delivers a
            # muted track.
            "mic_peak": round(max(self._peaks, default=0.0), 4),
            "mic_chunks": self._chunks,
            "consumers": len(self._subscribers),
        }

    async def ingest(self, connection) -> None:
        """Handle ``/audio``: one page streams PCM16, older pages are evicted."""
        from websockets.exceptions import ConnectionClosed

        previous, self._active = self._active, connection
        # The page names itself in the socket URL (?source=app|client) so the
        # eviction below can tell the loser who took over; older pages that
        # do not still stamp their diagnostic frames.
        query = urllib.parse.urlparse(connection.request.path or "").query
        self._source = _frame_source(json.dumps(dict(urllib.parse.parse_qsl(query))))
        if previous is not None:
            with contextlib.suppress(Exception):
                await previous.close(code=_SUPERSEDED, reason=f"superseded by {self._source or 'page'}")
            print(f"[APP] Microphone: {self._source or 'a newer page'} took over from the previous page.")
        print(f"[APP] Microphone page connected ({self._source or 'unnamed'}).")
        try:
            async for message in connection:
                if connection is not self._active:
                    break  # a newer page took over
                if not isinstance(message, bytes):
                    # Text frames are the page's own diagnostics: audio-track
                    # state (label, muted, readyState), why capture restarted,
                    # tab visibility. They tell headset-side mic trouble apart
                    # from ours, so they go to the log verbatim. Their "source"
                    # field says which page is speaking.
                    self._source = _frame_source(message) or self._source
                    print(f"[MIC] {self._source or 'page'}: {message[:400]}")
                    continue
                self._last_rx = time.monotonic()
                self._chunks += 1
                peak = _pcm16_peak(message)
                self._peaks.append(peak)
                if peak > _SOUND_PEAK:
                    self._last_sound = self._last_rx
                for sub in list(self._subscribers):
                    # Never let a slow or dead consumer stall capture. ``send``
                    # waits for the socket's write buffer to drain once it is
                    # over the high-water mark, and while it waits this loop is
                    # not reading the page either. A consumer that has already
                    # fallen a second behind loses this chunk instead.
                    transport = getattr(sub, "transport", None)
                    if transport is not None and transport.get_write_buffer_size() > _SUBSCRIBER_BACKLOG_BYTES:
                        continue
                    with contextlib.suppress(Exception):
                        await sub.send(message)
        except ConnectionClosed:
            pass  # routine: page navigated away, or we superseded it
        finally:
            if connection is self._active:
                self._active = None
                # A fresh page must not inherit this one's silence age or
                # chunk count, or it reads as "frozen" before its first frame.
                self._last_rx = 0.0
                self._last_sound = 0.0
                self._chunks = 0
                self._peaks.clear()
                self._source = None
                # Who hung up matters for triage: a close the page sent means
                # the browser gave up; one we sent is our keepalive timing out;
                # neither means the TCP connection died under us, which on a
                # headset is the tab being frozen or the device going to sleep.
                rcvd_first = getattr(connection.protocol, "close_rcvd_then_sent", None)
                who = {True: "page", False: "app", None: "transport"}[rcvd_first]
                cause = getattr(connection.protocol, "close_exc", None)
                print(
                    f"[APP] Microphone page disconnected (closed by {who}, code {connection.close_code},"
                    f" reason {connection.close_reason!r}, {cause})."
                )

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


def _page_html() -> str:
    """The control page (``teleop_app_page.html`` next to this file), read per request."""
    try:
        with open(_PAGE_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        return f"<!doctype html><meta charset=utf-8><pre>cannot read {_PAGE_PATH}: {exc}</pre>"


def _page_css() -> str:
    """The stylesheet shared by the control page and the log page, read per request."""
    try:
        with open(_CSS_PATH, encoding="utf-8") as f:
            return f.read()
    except OSError as exc:
        return f"/* cannot read {_CSS_PATH}: {exc} */"


_LOG_PAGE = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Teleop log</title>
<link rel="stylesheet" href="/app.css">
<style>
  body { text-align: left; }
  .bar { display: flex; align-items: center; gap: 0.8em; margin-bottom: 0.8em; flex-wrap: wrap; }
  #log { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 1.3em; line-height: 1.35;
         white-space: pre-wrap; word-break: break-word; background: #000; color: #dfe3ea; padding: 0.8em;
         border-radius: 0.6em; border: 1px solid var(--border); }
</style>
<div class="bar">
  <a class="btn" href="/">&#8592; Back to teleop</a>
  <button id="pause" class="btn">Pause</button>
  <span id="meta" class="muted">loading...</span>
</div>
<pre id="log"></pre>
<script>
const $ = id => document.getElementById(id);
let live = true;
async function load() {
  if (!live) return;
  try {
    const r = await fetch("/log?lines=200", { cache: "no-store" });
    const atEnd = window.innerHeight + window.scrollY >= document.body.offsetHeight - 40;
    $("log").textContent = (await r.json()).log;
    $("meta").textContent = "last 200 lines - " + new Date().toLocaleTimeString();
    // Follow the tail unless the reader has scrolled up to look at something.
    if (atEnd) window.scrollTo(0, document.body.scrollHeight);
  } catch (e) { $("meta").textContent = "app unreachable"; }
}
$("pause").onclick = () => { live = !live; $("pause").textContent = live ? "Pause" : "Resume"; if (live) load(); };
setInterval(load, 2000);
load().then(() => window.scrollTo(0, document.body.scrollHeight));
</script>
"""


class TeleopApp:
    """The HTTPS/WSS service: control page, mic hub, session configuration, and teleop supervision."""

    def __init__(self, port: int, teleop: TeleopProcess, config: SessionConfigurator):
        self._port = port
        self._teleop = teleop
        self._config = config
        self._mic = MicHub()
        self._client = WebClientAssets()
        self._action_lock = threading.Lock()  # one start/stop/restart at a time

    def _proxy_port(self) -> int:
        """Port of the CloudXR WSS proxy, which moves when the host is shared.

        The run we launched knows its own; before the first start (or for an
        adopted run) the saved settings' port applies, then the environment.
        """
        for source in (self._teleop.launch_env(), self._config.port_env(), os.environ):
            value = str(source.get("PROXY_PORT", "")).strip()
            if value:
                return int(value)
        return 48322

    def _client_base(self) -> str:
        """Origin of the WebXR client handed to the headset.

        An operator override (``TELEOP_WEB_CLIENT_BASE``, isaacteleop's own knob)
        wins and is handed out untouched. Otherwise our copy with the microphone
        script (see :class:`WebClientAssets`) when it is available, else NVIDIA's
        hosted build for the installed ``isaacteleop`` (1.3.x gets ``release-1.3.x``).
        """
        override = os.environ.get("TELEOP_WEB_CLIENT_BASE", "").strip()
        if override:
            return override.rstrip("/")
        if self._client.available():
            return f"https://{_preferred_host()}:{self._port}/client"
        try:
            from isaacteleop.cloudxr.oob_teleop_env import default_web_client_origin

            return default_web_client_origin().rstrip("/")
        except Exception:
            return "https://nvidia.github.io/IsaacTeleop/client/main"

    def client_local(self) -> bool:
        """Whether the link points at our copy of the client (the one with the microphone script)."""
        return self._client_base().startswith(f"https://{_preferred_host()}:{self._port}/")

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
        base = self._client_base()
        # serverIP/port are query params the client reads on load, so its
        # connection fields arrive pre-filled instead of typed in the headset.
        # mic=0 keeps the client's own audio passthrough from capturing the
        # headset microphone: the Quest grants the mic to one page at a time,
        # so with it on, the bundle and our script (or the app page) steal the
        # mic from each other every ~15 s and every hand-over chops the audio
        # the voice labeler hears.
        return f"{base}/?serverIP={host}&port={proxy}&mic=0"

    def status(self) -> dict:
        """Everything the page polls, in one round trip."""
        try:
            summary = self._config.summary()
            scene_source = self._config.settings().get("scene_source", "local")
            fleet = self._config.fleet_status()
        except Exception as exc:  # noqa: BLE001 — a broken settings file must not take the page down
            summary, scene_source, fleet = f"configuration unavailable: {exc}", "local", None
        return {
            **self._teleop.status(),
            **self._mic.status(),
            "cloudxr_url": self.cloudxr_url(),
            "client_local": self.client_local(),
            "cert_url": self.cert_url(),
            # Teleop takes ~30-60 s to boot, and the process existing says
            # nothing about the XR stack being up. The proxy port going live is
            # the signal that the client has something to connect to.
            "cloudxr_ready": _port_in_use(self._proxy_port()),
            "proxy_port": self._proxy_port(),
            "summary": summary,
            "scene_source": scene_source,
            # Just enough of the fleet monitor for the status bar; the Scenes
            # tab fetches the full picture (rows, totals) over /control.
            "fleet": None
            if fleet is None
            else {
                "connected": fleet["connected"],
                "error": fleet["error"],
                "online": len(fleet["online"]),
                "server_url": fleet["server_url"],
            },
        }

    # -- control channel -----------------------------------------------------

    def _control_op(self, op: str, request: dict) -> dict:
        """One ``/control`` request (runs on a worker thread: settings, scans and the fleet may block)."""
        if op == "state":
            pass
        elif op == "save":
            self._config.update(request.get("settings"))
        elif op == "connect":
            self._config.connect_fleet()
        elif op != "refresh":
            raise ValueError(f"unknown op '{op}'")
        return {"state": self._config.state()}

    async def control(self, connection) -> None:
        """Handle ``/control``: JSON requests ``{id, op, ...}`` answered with ``{id, ok, ...}``.

        The HTTP layer cannot read request bodies, so this is how the page
        sends anything that carries data (settings, the scene selection).
        """
        from websockets.exceptions import ConnectionClosed

        try:
            async for message in connection:
                request_id, op = None, None
                try:
                    request = json.loads(message)
                    request_id, op = request.get("id"), request.get("op")
                    result = await asyncio.to_thread(self._control_op, op, request)
                    reply = {"id": request_id, "ok": True, **result}
                except Exception as exc:  # noqa: BLE001 — reported to the page, never fatal for the app
                    reply = {"id": request_id, "ok": False, "error": str(exc)}
                    if not isinstance(exc, ValueError):
                        print(f"[APP] control '{op}' failed: {exc!r}", flush=True)
                await connection.send(json.dumps(reply))
        except ConnectionClosed:
            pass

    async def serve(self) -> None:
        """Run until cancelled."""
        from websockets.asyncio.server import serve

        async def process_request(connection, request):
            path, _, query = (request.path or "/").partition("?")
            if "upgrade" in request.headers.get("Connection", "").lower():
                return None  # websocket routes are handled in the handler
            # The page and its stylesheet change whenever the app is updated; a
            # reload on the headset must always fetch them, never a cached copy.
            if path == "/":
                return self._text(connection, _page_html(), "text/html; charset=utf-8")
            if path == "/app.css":
                return self._text(connection, _page_css(), "text/css; charset=utf-8")
            if path == "/client":
                r = connection.respond(http.HTTPStatus.MOVED_PERMANENTLY, "")
                r.headers["Location"] = "/client/" + (f"?{query}" if query else "")
                return r
            if path in ("/client/", "/client/index.html"):
                # The WebXR client with the microphone script; fetched on first
                # use if the startup fetch has not got it yet.
                if not self._client.available() and not await asyncio.to_thread(self._client.load):
                    return connection.respond(
                        http.HTTPStatus.SERVICE_UNAVAILABLE,
                        "The WebXR client is not available locally; open the nvidia.github.io link from the app page\n",
                    )
                return self._binary(self._client.index(), "text/html; charset=utf-8", cache="no-store")
            if path == "/client/bundle.js":
                if not self._client.available():
                    return connection.respond(http.HTTPStatus.NOT_FOUND, "not fetched yet\n")
                # 9.7 MB: let the browser revalidate instead of re-downloading per session.
                etag = self._client.etag("bundle.js") or f'"{len(self._client.bundle())}"'
                if request.headers.get("If-None-Match") == etag:
                    r = connection.respond(http.HTTPStatus.NOT_MODIFIED, "")
                    r.headers["ETag"] = etag
                    return r
                r = self._binary(self._client.bundle(), "text/javascript", cache="no-cache")
                r.headers["ETag"] = etag
                return r
            if path == "/client/mic.js":
                return self._binary(_client_mic_js(), "text/javascript", cache="no-store")
            if path == "/client/favicon.ico":
                return self._binary(_icon_png(64), "image/png")
            if path == "/manifest.webmanifest":
                return self._text(connection, json.dumps(_MANIFEST), "application/manifest+json")
            if path == "/sw.js":
                return self._text(connection, _SERVICE_WORKER, "text/javascript")
            if path in ("/icon-192.png", "/icon-512.png"):
                return self._binary(_icon_png(512 if "512" in path else 192), "image/png")
            if path == "/status":
                return self._json(connection, self.status())
            if path == "/log.html":
                return self._text(connection, _LOG_PAGE, "text/html; charset=utf-8")
            if path == "/log":
                lines = urllib.parse.parse_qs(query).get("lines", ["40"])[0]
                lines = max(1, min(int(lines), 2000)) if lines.isdigit() else 40
                return self._json(connection, {"log": self._teleop.tail_log(lines)})
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
                # stop() waits up to _STOP_GRACE_S for Isaac Sim to exit and
                # start() shells out to nvidia-smi; run in the loop they would
                # freeze the microphone relay and every status poll meanwhile.
                # The lock keeps two taps from stopping and starting at once.
                ok, message = await asyncio.to_thread(self._run_action, action[path])
                print(f"[APP] {path[1:]}: {message}")
                return self._json(connection, {"ok": ok, "message": message})
            return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")

        async def handler(connection):
            path = (connection.request.path or "/").split("?")[0]
            if path == "/audio":
                await self._mic.ingest(connection)
            elif path == "/subscribe":
                await self._mic.subscribe(connection)
            elif path == "/control":
                await self.control(connection)
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
            # Fetching the client can take a while; never hold the page for it.
            threading.Thread(target=self._client.refresh_loop, daemon=True, name="webxr-client-cache").start()
            print(f"[APP] CloudXR client link it will hand out: {self.cloudxr_url()}")
            await asyncio.Future()  # run forever

    @staticmethod
    def _text(connection, body: str, content_type: str):
        """Serve text with the given type, uncached.

        ``connection.respond`` stamps ``text/plain`` on its own, and assigning
        the header again *adds* a second value rather than replacing it. The
        browser then sees the first one, which is fatal for a stylesheet (the
        MIME check is strict there), so the default is removed first.
        """
        r = connection.respond(http.HTTPStatus.OK, body)
        del r.headers["Content-Type"]
        r.headers["Content-Type"] = content_type
        r.headers["Cache-Control"] = "no-store"
        return r

    @classmethod
    def _json(cls, connection, payload: dict):
        return cls._text(connection, json.dumps(payload), "application/json")

    def _run_action(self, action: Callable[[], tuple[bool, str]]) -> tuple[bool, str]:
        with self._action_lock:
            return action()

    @staticmethod
    def _json(connection, payload: dict):
        r = connection.respond(http.HTTPStatus.OK, json.dumps(payload))
        r.headers["Content-Type"] = "application/json"
        return r

    @staticmethod
    def _binary(body: bytes, content_type: str, cache: str = "public, max-age=86400"):
        """Serve bytes, which ``connection.respond`` cannot do (it text-encodes)."""
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        headers = Headers()
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(body))
        headers["Cache-Control"] = cache
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
        # "=" form: a forwarded flag such as "--device" would otherwise be read
        # by argparse as an option of the app itself, not as the value.
        teleop_args=" ".join(f"--teleop-arg={shlex.quote(a)}" for a in teleop_args),
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
        help=(
            "Extra argument forwarded to make_teleop_scene.py, for flags the settings page does not cover;"
            " repeat once per argument and use the '=' form for flags, e.g. --teleop-arg=--device"
            " --teleop-arg=cuda:1. Scenes, recording and the parameters on the page come from the saved"
            " settings, so do not pass --scene_list/--scene_usda/--record_dir here."
        ),
    )
    parser.add_argument("--install-service", action="store_true", help="Install and start the systemd user service.")
    args = parser.parse_args()

    if args.install_service:
        _install_service(args.port, args.teleop_arg)
        return

    # The app owns the microphone, so teleop consumes the relay rather than
    # serving a competing page of its own — unless a --teleop-arg says otherwise.
    teleop_args = list(args.teleop_arg)
    mic_device = f"hub:127.0.0.1:{args.port}"
    if "--mic_device" in teleop_args:
        index = teleop_args.index("--mic_device")
        mic_device = teleop_args[index + 1] if index + 1 < len(teleop_args) else mic_device
        del teleop_args[index : index + 2]

    if _port_in_use(args.port):
        raise SystemExit(f"[APP] Port {args.port} is already in use — another copy of the app is probably running.")

    config = SessionConfigurator(mic_device=mic_device)
    app = TeleopApp(args.port, TeleopProcess(teleop_args, launch_provider=config.launch), config)
    try:
        asyncio.run(app.serve())
    except KeyboardInterrupt:
        print("\n[APP] Shutting down; leaving any running teleop alone (use Kill in the page to stop it).")
    finally:
        config.close()


if __name__ == "__main__":
    main()
