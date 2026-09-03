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

_SUPERSEDED = 4001
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
                # Who hung up matters for triage: a close the page sent means
                # the browser gave up; one we sent is our keepalive timing out.
                rcvd_first = getattr(connection.protocol, "close_rcvd_then_sent", None)
                who = {True: "page", False: "app", None: "transport"}[rcvd_first]
                print(
                    f"[APP] Microphone page disconnected (closed by {who}, code {connection.close_code},"
                    f" reason {connection.close_reason!r})."
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
        # mic=0 keeps the client from capturing the headset microphone for
        # CloudXR's own audio passthrough (unused here): the Quest grants the
        # mic to one page at a time, so with it on, the client and this app's
        # page steal the mic from each other every ~15 s and every hand-over
        # chops the audio the voice labeler hears.
        return f"{self._client_base()}/?serverIP={host}&port={proxy}&mic=0"

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

        def process_request(connection, request):
            path, _, query = (request.path or "/").partition("?")
            if "upgrade" in request.headers.get("Connection", "").lower():
                return None  # websocket routes are handled in the handler
            # The page and its stylesheet change whenever the app is updated; a
            # reload on the headset must always fetch them, never a cached copy.
            if path == "/":
                return self._text(connection, _page_html(), "text/html; charset=utf-8")
            if path == "/app.css":
                return self._text(connection, _page_css(), "text/css; charset=utf-8")
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
        help=(
            "Extra argument forwarded to make_teleop_scene.py, for flags the settings page does not cover;"
            " repeat once per argument. Scenes, recording and the parameters on the page come from the saved"
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
