# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What a teleop session is launched with — shared by the desktop launcher and the headset app.

Both UIs (:mod:`teleop_launcher`, the tkinter desktop launcher, and
:mod:`teleop_app`, the headset browser app) let the operator pick the same
things: the teleop parameters, the record directory, a **scene source** (a
local directory, or the fleet server) and which of its scenes to collect. This
module is the single definition of all of that, so the two UIs cannot drift:

- the parameter schema (:data:`PARAMS`), the headset devices (:data:`DEVICES`)
  and the network ports (:data:`PORTS`);
- the persisted settings file (:func:`load_settings` / :func:`save_settings`),
  shared by both UIs — a choice made on the desktop shows up in the headset
  and vice versa;
- scanning: scene files under a directory (:func:`scan_scene_dir`), per-scene
  demo counts under the record directory (:func:`scan_record_dir`), and the
  scene table rows each UI renders (:func:`local_scene_rows`,
  :func:`server_scene_rows`);
- turning settings plus a scene selection into the ``make_teleop_scene.py``
  command line and environment (:func:`build_launch`).

Pure standard library apart from the optional ``h5py`` import in
:func:`scan_record_dir`; no simulator, no UI toolkit.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_ISAACLAB_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))

TELEOP_SCRIPT = os.path.join(_HERE, "make_teleop_scene.py")
"""The script every launch runs."""

DEFAULT_SCENE_DIR = os.path.join(_HERE, "scenes")
DEFAULT_RECORD_DIR = os.path.join(_ISAACLAB_ROOT, "datasets", "duo_teleop")
"""Mirrors ``make_teleop_scene.py --record_dir``'s default when run from the Isaac Lab root."""

SETTINGS_PATH = os.environ.get(
    "DUO_TELEOP_LAUNCHER_SETTINGS", os.path.join(os.path.expanduser("~"), ".config", "duo_teleop_launcher.json")
)
"""Where the settings persist (parameters, directories, scene source, fleet connection, selection)."""

SCENE_SUFFIXES = (".usdz", ".usda", ".usd")
SCENE_LIST_NAME = "launcher.scene_list.json"
"""The scene-list JSON a local-directory launch writes into the record directory."""

DEVICES: dict[str, tuple[str, str]] = {
    "meta quest": ("quest", "cloudxrjs"),
    "avp": ("avp", "avp"),
}
"""Headset choice -> ``(--mic_device, --cloudxr_env)``.

One choice sets BOTH the microphone source and the CloudXR client profile.
Quest streams its mic from a browser page; the AVP's Isaac XR Teleop client
streams it natively once its CloudXR session connects (see ``headset_mic.py``).
The headset app overrides the mic part with its own relay (``hub``).
"""

PORTS: dict[str, tuple[str | None, str, str]] = {
    # Runtime signaling. Default depends on the CloudXR device profile:
    # 49100 for the WebRTC (Quest browser) profile, 48010 for the AVP native one.
    "port:cxr_server": ("NV_CXR_SERVER_PORT", "tcp", "49100 Quest / 48010 AVP"),
    # Runtime media stream (video/audio/input) — UDP.
    "port:cxr_media": ("NV_CXR_MEDIA_PORT", "udp", "47998"),
    # WSS/HTTPS proxy: the Quest browser client connects here (wss://<ip>:<port>),
    # AVP "secure mode" too; the proxy forwards to the signaling port above.
    "port:wss_proxy": ("PROXY_PORT", "tcp", "48322"),
    # headset_mic.py: the Quest mic page and the WSS PCM endpoint for both
    # headsets. Irrelevant when the headset app relays the mic (it has its own port).
    "port:mic": (None, "tcp", "8444"),
}
"""Every port a teleop session opens, keyed by pseudo-parameter.

Values are ``(env var or None, protocol, default shown when blank)``. The
CloudXR runtime and its WSS proxy read their ports from the environment
(``make_teleop_scene.py`` inherits it from whichever UI launched it); the
headset-mic port is the ``:<port>`` suffix of ``--mic_device``. Not listed: the
fleet server's port (part of its URL) and the USB-tethered OOB ports
(``USB_UI_PORT``, ``USB_BACKEND_PORT``, ``USB_TURN_PORT``), which the teleop
script never enables.
"""


@dataclass(frozen=True)
class Param:
    """One row of the parameter schema.

    ``kind`` is ``str``, ``float``, ``bool``, ``port`` or ``choice:<a,b,c>``.
    ``default`` mirrors the argparse default of ``make_teleop_scene.py`` so
    :func:`build_args` can recognise unchanged values and leave them off the
    command line. ``device`` and ``port:*`` are pseudo-parameters (see
    :data:`DEVICES` / :data:`PORTS`).
    """

    flag: str
    label: str
    default: object
    kind: str
    group: str

    @property
    def choices(self) -> list[str] | None:
        return self.kind.split(":", 1)[1].split(",") if self.kind.startswith("choice:") else None


def _gpu_choices() -> list[str]:
    """``cuda:N`` for every GPU ``nvidia-smi`` reports, so the UIs only offer devices that exist.

    Falls back to ``cuda:0`` (the AppLauncher default) when nvidia-smi is missing.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        indices = [int(line.strip()) for line in out.splitlines() if line.strip().isdigit()]
    except (OSError, subprocess.TimeoutExpired):
        indices = []
    return [f"cuda:{i}" for i in indices] or ["cuda:0"]


PARAMS: list[Param] = [
    Param("device", "Headset", "meta quest", "choice:meta quest,avp", "Operator & voice"),
    Param("--embodiment", "Robot embodiment", "yam_duo", "choice:franka_duo,yam_duo", "Operator & voice"),
    Param("--user", "User name (hand calibration)", "", "str", "Operator & voice"),
    Param("--whisper_model", "Whisper model", "base.en", "str", "Operator & voice"),
    Param("--no_voice", "Disable voice commands", False, "bool", "Operator & voice"),
    Param("--no_auto_start", "Disable auto-start", False, "bool", "Session start"),
    Param("--auto_start_pos_tol", "Auto-start position tolerance [m]", 0.10, "float", "Session start"),
    Param("--auto_start_rot_tol", "Auto-start rotation tolerance [deg]", 25.0, "float", "Session start"),
    Param("--debug_auto_start", "Debug auto-start (frames + errors)", False, "bool", "Session start"),
    Param("--no_dr", "Disable domain randomization", False, "bool", "Domain randomization"),
    Param("--dr_arm_jitter", "Arm start-pose jitter [rad]", 0.08, "float", "Domain randomization"),
    Param("--dr_object_xy", "Object position range [m]", 0.05, "float", "Domain randomization"),
    Param("--dr_object_yaw", "Object yaw range [deg]", 180.0, "float", "Domain randomization"),
    Param("--dr_object_bias", "Shift objects toward the robot [m]", 0.0, "float", "Domain randomization"),
    Param("--settle_time", "Object settling time after reset [s]", 1.0, "float", "Domain randomization"),
    Param("--arm_kp", "Arm kp (stiffness) [N·m/rad]", 400.0, "float", "Control gains"),
    Param("--arm_kd", "Arm kd (damping) [N·m·s/rad]", 80.0, "float", "Control gains"),
    Param("--hand_kp", "Hand kp (stiffness) [N·m/rad]", 400.0, "float", "Control gains"),
    Param("--hand_kd", "Hand kd (damping) [N·m·s/rad]", 4.0, "float", "Control gains"),
    Param("--arm_visual", "Arm rendering", "transparent", "choice:transparent,hidden,normal", "Visuals"),
    Param("--visualize_hands", "Show tracked hand joints", False, "bool", "Visuals"),
    Param("--no_task_display", "Hide the task-description panel", False, "bool", "Visuals"),
    Param("--device", "Simulation GPU", "cuda:0", "choice:" + ",".join(_gpu_choices()), "Advanced"),
    Param("--no_adjust", "Disable the adjust-object mode", False, "bool", "Advanced"),
    Param("--episode_length_s", "Episode timeout [s]", 300.0, "float", "Advanced"),
    Param("--render_frequency", "Render frequency [Hz]", 30.0, "float", "Advanced"),
    Param("--no_record", "Disable recording", False, "bool", "Advanced"),
    Param("port:cxr_server", "CloudXR signaling [tcp]", "", "port", "Network ports"),
    Param("port:cxr_media", "CloudXR media [udp]", "", "port", "Network ports"),
    Param("port:wss_proxy", "WSS proxy / Quest client [tcp]", "", "port", "Network ports"),
    Param("port:mic", "Headset microphone [tcp]", "", "port", "Network ports"),
]
"""The teleop knobs both UIs expose, grouped by concern, in display order."""

PARAM_BY_FLAG: dict[str, Param] = {p.flag: p for p in PARAMS}

PORTS_NOTE = (
    "Blank keeps the default. Every operator sharing this workstation needs distinct TCP ports;"
    " open them in the firewall (e.g. sudo ufw allow <port>/tcp)."
)


def param_groups() -> list[str]:
    """The parameter groups in display order."""
    seen: list[str] = []
    for p in PARAMS:
        if p.group not in seen:
            seen.append(p.group)
    return seen


def schema() -> list[dict]:
    """The parameter schema as JSON-ready rows, for a UI that renders itself from it."""
    return [
        {
            "flag": p.flag,
            "label": p.label,
            "default": p.default,
            "kind": "choice" if p.choices else p.kind,
            "choices": p.choices,
            "group": p.group,
            "port_default": PORTS[p.flag][2] if p.kind == "port" else None,
        }
        for p in PARAMS
    ]


# -- persisted settings ---------------------------------------------------------


def default_settings() -> dict:
    """A complete settings dict with every parameter at its default."""
    return {
        "params": {p.flag: p.default for p in PARAMS},
        "record_dir": DEFAULT_RECORD_DIR,
        "scene_dir": DEFAULT_SCENE_DIR,
        "scene_source": "local",
        "fleet_server": "",
        "collector_id": "",
        "fleet_token": "",
        # Per-source scene selection, by scene name. Names not listed take the
        # source's default (local: collect; server: collect if the fleet still needs it).
        "selection": {"local": {}, "server": {}},
    }


def merge_settings(base: dict, patch: dict) -> dict:
    """``base`` updated with ``patch``; ``params`` and ``selection`` merge key-wise, unknown keys are kept."""
    out = dict(base)
    for key, value in patch.items():
        if key == "params" and isinstance(value, dict):
            out["params"] = {**out.get("params", {}), **value}
        elif key == "selection" and isinstance(value, dict):
            merged = {k: dict(v) for k, v in out.get("selection", {}).items() if isinstance(v, dict)}
            for source, names in value.items():
                if isinstance(names, dict):
                    merged[source] = {**merged.get(source, {}), **names}
            out["selection"] = merged
        else:
            out[key] = value
    return out


def load_settings(path: str = SETTINGS_PATH) -> dict:
    """The persisted settings over the defaults; a missing or unreadable file yields the defaults."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    return merge_settings(default_settings(), data if isinstance(data, dict) else {})


def save_settings(settings: dict, path: str = SETTINGS_PATH) -> None:
    """Persist ``settings`` (owner-readable only: the fleet token is in there)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(settings, f, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def normalize_server_url(url: str) -> str:
    """Strip whitespace and default to ``http://`` when no scheme was typed; empty stays empty."""
    url = str(url or "").strip().rstrip("/")
    if url and not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url


# -- scanning -------------------------------------------------------------------------


def scan_scene_dir(scene_dir: str) -> list[str]:
    """All scene files (``*.usdz``, ``*.usda``, ``*.usd``) under ``scene_dir``, recursively, sorted."""
    hits = []
    if not scene_dir or not os.path.isdir(scene_dir):
        return hits
    for root, _dirs, files in os.walk(scene_dir):
        hits += [os.path.join(root, f) for f in files if f.endswith(SCENE_SUFFIXES)]
    return sorted(hits)


def scan_record_dir(record_dir: str) -> dict[str, tuple[int, int]]:
    """Per-scene ``(success, failure)`` demo counts across the record directory.

    Counts every demo in every ``*.hdf5`` under ``record_dir`` (one file per
    episode nowadays; legacy multi-demo session files are counted the same
    way). Demos are grouped by their ``scene`` attribute; demos without one
    land under ``"<untagged>"``. Unreadable files (e.g. an episode being
    written right now) are skipped, as is everything when ``h5py`` is missing.
    """
    counts: dict[str, list[int]] = {}
    if not record_dir or not os.path.isdir(record_dir):
        return {}
    try:
        import h5py
    except ImportError:
        return {}
    for root, dirs, files in os.walk(record_dir):
        dirs[:] = [d for d in dirs if d not in ("fleet_cache", "fleet_scenes")]  # scene caches hold no demos
        for name in sorted(files):
            if not name.endswith(".hdf5"):
                continue
            path = os.path.join(root, name)
            try:
                with h5py.File(path, "r") as f:
                    if "data" not in f:
                        continue
                    for _name, group in f["data"].items():
                        scene = group.attrs.get("scene", "<untagged>")
                        if isinstance(scene, bytes):
                            scene = scene.decode()
                        entry = counts.setdefault(str(scene), [0, 0])
                        entry[0 if bool(group.attrs.get("success", False)) else 1] += 1
            except OSError as exc:
                if path not in _unreadable_reported:  # the app rescans often; say it once per file
                    _unreadable_reported.add(path)
                    print(f"[SESSION] Could not read {path}: {exc}")
    return {k: (v[0], v[1]) for k, v in counts.items()}


_unreadable_reported: set[str] = set()


# -- scene table rows ---------------------------------------------------------------------


@dataclass
class SceneTable:
    """What a UI renders for the active scene source."""

    source: str
    rows: list[dict]
    untagged: tuple[int, int] | None = None
    """Local mode: ``(success, failure)`` counts of demos with no scene tag, if any."""

    @property
    def selected(self) -> list[dict]:
        return [row for row in self.rows if row["selected"]]


def _selected(settings: dict, source: str, name: str, default: bool) -> bool:
    value = settings.get("selection", {}).get(source, {}).get(name)
    return bool(value) if isinstance(value, bool) else default


def local_scene_rows(settings: dict) -> SceneTable:
    """The local-directory table: every scene file with this machine's success/failure counts.

    A scene not yet in the selection is collected by default.
    """
    counts = scan_record_dir(settings.get("record_dir", ""))
    rows = []
    for path in scan_scene_dir(settings.get("scene_dir", "")):
        name = os.path.basename(path)
        success, failure = counts.get(name, (0, 0))
        rows.append(
            {
                "name": name,
                "path": path,
                "success": success,
                "failure": failure,
                "selected": _selected(settings, "local", name, True),
            }
        )
    return SceneTable("local", rows, untagged=counts.get("<untagged>"))


def scene_needed(scene: dict) -> bool:
    """Whether the fleet still needs demos of a server scene row (under target and not retired)."""
    return not scene.get("retired", False) and scene["successes"] < scene["target_successes"]


def server_scene_rows(settings: dict, fleet_scenes: dict[str, dict] | None) -> SceneTable:
    """The fleet-server table: the SERVER's scenes with the server's numbers only.

    ``fleet_scenes`` maps scene id to the server's scene row (from a status
    snapshot); ``None`` means "not connected" and yields no rows. A scene not
    yet in the selection is collected by default when the fleet still needs it.
    """
    rows = []
    for name in sorted(fleet_scenes or {}):
        scene = fleet_scenes[name]
        needed = scene_needed(scene)
        progress = f"{scene['successes']}/{scene['target_successes']}" + (" (retired)" if scene.get("retired") else "")
        rows.append(
            {
                "name": name,
                "path": None,
                "progress": progress,
                "workers": list(scene.get("active_workers", [])),
                "done": not needed and not scene.get("retired", False),
                "needed": needed,
                "selected": _selected(settings, "server", name, needed),
            }
        )
    return SceneTable("server", rows)


def scene_table(settings: dict, fleet_scenes: dict[str, dict] | None = None) -> SceneTable:
    """The table for the settings' active scene source."""
    if settings.get("scene_source") == "server":
        return server_scene_rows(settings, fleet_scenes)
    return local_scene_rows(settings)


# -- command line -------------------------------------------------------------------------


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def port_value(params: dict, flag: str) -> int | None:
    """The port typed into a ``port:*`` parameter, or ``None`` when it is blank.

    Raises:
        ValueError: If the value is not an integer in 1..65535.
    """
    raw = str(params.get(flag, "") or "").strip()
    if not raw:
        return None
    label = PARAM_BY_FLAG[flag].label
    try:
        port = int(raw)
    except ValueError:
        raise ValueError(f"{label}: '{raw}' is not a port number.") from None
    if not 1 <= port <= 65535:
        raise ValueError(f"{label}: {port} is out of range (1-65535).")
    return port


def validate_ports(params: dict) -> str | None:
    """An error message when a port is invalid or two TCP ports collide, else ``None``."""
    try:
        ports = {flag: port_value(params, flag) for flag in PORTS}
    except ValueError as exc:
        return str(exc)
    # The blank (default) signaling and proxy ports differ, and the mic default
    # is far from both; only explicit values can collide.
    seen: dict[int, str] = {}
    for flag, port in ports.items():
        if port is None or PORTS[flag][1] != "tcp":
            continue
        if port in seen:
            return f"Port {port} is used for both '{seen[port]}' and '{flag.split(':', 1)[1]}'."
        seen[port] = flag.split(":", 1)[1]
    return None


def build_env(params: dict) -> dict[str, str]:
    """Environment variables for the filled-in ``port:*`` parameters (see :data:`PORTS`)."""
    env = {}
    for flag, (env_var, _proto, _default) in PORTS.items():
        port = port_value(params, flag)
        if env_var is not None and port is not None:
            env[env_var] = str(port)
    return env


def build_args(params: dict, mic_device: str | None = None) -> list[str]:
    """``make_teleop_scene.py`` arguments for every parameter that differs from its default.

    The ``device`` pseudo-parameter always expands to explicit ``--mic_device``
    and ``--cloudxr_env`` flags (via :data:`DEVICES`) so the session
    unambiguously matches the chosen headset; a filled-in mic port becomes the
    ``--mic_device`` ``:<port>`` suffix. ``mic_device`` overrides the mic part
    entirely (the headset app passes its relay, ``hub:...``). ``port:*``
    parameters are environment variables (:func:`build_env`), not flags.
    """
    args: list[str] = []
    for p in PARAMS:
        value = params.get(p.flag, p.default)
        if p.flag == "device":
            device_mic, cloudxr_env = DEVICES.get(str(value), DEVICES["meta quest"])
            mic = mic_device
            if mic is None:
                mic = device_mic
                mic_port = port_value(params, "port:mic")
                if mic_port is not None:
                    mic = f"{mic}:{mic_port}"
            args += ["--mic_device", mic, "--cloudxr_env", cloudxr_env]
        elif p.kind == "port":
            continue
        elif p.kind == "bool":
            if _as_bool(value):
                args.append(p.flag)
        elif str(value).strip() != "" and str(value).strip() != str(p.default):
            args += [p.flag, str(value).strip()]
    return args


@dataclass
class LaunchSpec:
    """A ready-to-run teleop launch: arguments after the script path, extra environment, and a one-line summary."""

    args: list[str]
    env: dict[str, str]
    summary: str

    def command(self, python: str) -> list[str]:
        return [python, TELEOP_SCRIPT, *self.args]


def build_launch(settings: dict, table: SceneTable, mic_device: str | None = None) -> LaunchSpec:
    """Turn the settings and the active source's table into a launch.

    The two scene sources are exclusive by construction: local mode writes the
    selected files to a scene-list JSON in the record directory and passes NO
    fleet flags (fully standalone); server mode passes the selected server
    scene ids plus the connection flags and NO local scene list — the run
    downloads the scenes from the server and uploads every labeled episode.
    The fleet token travels in the ``FLEET_TOKEN`` environment variable, never
    on the (``ps``-visible) command line.

    Raises:
        ValueError: With an operator-facing message when nothing is selected,
            a port is invalid, or server mode has no server URL.
    """
    params = settings.get("params", {})
    selected = table.selected
    if not selected:
        raise ValueError("Select at least one scene to collect.")
    port_error = validate_ports(params)
    if port_error:
        raise ValueError(port_error)
    record_dir = os.path.abspath(os.path.expanduser(settings.get("record_dir") or DEFAULT_RECORD_DIR))
    os.makedirs(record_dir, exist_ok=True)
    env = build_env(params)

    if table.source == "server":
        server = normalize_server_url(settings.get("fleet_server", ""))
        if not server:
            raise ValueError("Enter the fleet server URL first (e.g. http://fleet-host:8080).")
        scene_args = ["--fleet_server", server, "--fleet_scene_ids", *[row["name"] for row in selected]]
        collector_id = str(settings.get("collector_id", "")).strip()
        if collector_id:
            scene_args += ["--collector_id", collector_id]
        token = str(settings.get("fleet_token", "")).strip()
        if token:
            env["FLEET_TOKEN"] = token
        summary = f"{len(selected)} fleet scene(s) from {server}"
    else:
        scene_list_path = os.path.join(record_dir, SCENE_LIST_NAME)
        with open(scene_list_path, "w") as f:
            json.dump({"scenes": [row["path"] for row in selected]}, f, indent=2)
        scene_args = ["--scene_list", scene_list_path]
        summary = f"{len(selected)} local scene(s)"

    args = [*scene_args, "--record_dir", record_dir, "--headless", *build_args(params, mic_device=mic_device)]
    return LaunchSpec(args=args, env=env, summary=summary)
