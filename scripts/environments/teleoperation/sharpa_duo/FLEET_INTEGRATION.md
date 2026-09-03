# Fleet collection in the headset-app flow — what changed and why

This document records how the fleet-collection work (branch
`cyqian/fleet-collection`: per-episode HDF5, the `duo-fleet-server` client,
the two-source desktop launcher) was brought onto the headset-app flow of
`cyqian/duo-usda-teleop` (systemd service + browser page + adjust mode), and
how the desktop launcher's functions were reproduced on that page. It is a
design note for whoever touches these files next; [README.md](README.md) and
[SETUP.md](SETUP.md) describe how to *use* the result.

## The two flows that had to meet

| | `cyqian/duo-usda-teleop` (kept) | `cyqian/fleet-collection` (integrated) |
|---|---|---|
| How a run starts | `teleop_app.py` user service; the operator taps **Start session** on a page in the headset browser | `teleop_launcher.py` (tkinter) on the desktop, or a terminal |
| What the UI configures | Nothing — fixed `--teleop-arg`s baked into the service unit | Parameters, record dir, scene source (local dir / fleet server), scene ticks, network ports; persisted |
| Recording | One appendable HDF5 per session/dataset | One HDF5 per labeled episode (fleet upload unit) |
| Fleet | — | `fleet_client.py`: check-in, presence, sha256-verified scene download, crash-safe upload outbox |

The requirement: keep the headset-driven flow, add the fleet features, and put
the launcher's configuration abilities on the web page — without two copies of
the configuration logic.

## Result: one configuration core, two front ends

```
 session_config.py  ── schema (PARAMS/PORTS/DEVICES), settings file, scanning,
        │              scene-table rows, settings → command line (LaunchSpec)
        ├── teleop_launcher.py   desktop (tkinter): renders the schema, Tk widgets only
        └── teleop_app.py        headset service: SessionConfigurator + /control JSON channel
              └── teleop_app_page.html   Session | Scenes | Parameters tabs
 fleet_client.py   ── FleetClient (used by make_teleop_scene.py during a run)
                      FleetMonitor (read-only status poller used by both UIs)
```

Both UIs read and write the **same settings file**
(`~/.config/duo_teleop_launcher.json`): a choice made on the desktop shows up
in the headset and vice versa. The app re-reads the file whenever its mtime
changes, so the two never fight.

## File-by-file

### New

- **`session_config.py`** — extracted from the fleet launcher, made
  UI-agnostic. `PARAMS` (a `Param` dataclass per knob, grouped), `PORTS`,
  `DEVICES`; `load_settings`/`save_settings`/`merge_settings` (key-wise merge
  for `params` and `selection`, unknown keys preserved, file mode 0600 because
  the fleet token is in it); `scan_scene_dir`, `scan_record_dir` (per-scene
  success/failure counts from the HDF5 `scene`/`success` attributes, reported
  once per unreadable file); `local_scene_rows` / `server_scene_rows` /
  `scene_table` (what a table renders, with the persisted selection applied:
  local rows default to *collect*, server rows default to *collect if the
  fleet still needs it*); `build_args` / `build_env` / `validate_ports`; and
  `build_launch(settings, table, mic_device) -> LaunchSpec`, the single place
  that turns settings into a command line. Schema defaults mirror
  `make_teleop_scene.py` (`--embodiment yam_duo`, `--dr_object_bias 0.0`;
  `--no_adjust` added).
- **`teleop_app_page.html`** — the control page, moved out of the Python
  string into a file the app reads per request (an edit shows on reload, no
  service restart). The Session tab is the upstream page as of "Add session
  controls and a log view" (Start/Finish session, the certificate and CloudXR
  links, Restart/Kill/Start mic/Stop mic, the Log link to `/log.html`, the
  starting/running split of the status line) with two tabs added. The log
  page itself stays an inline constant in `teleop_app.py`, as upstream wrote
  it.
- **`fleet_client.py`**, **`fleet_push_scenes.py`** — copied from
  `cyqian/fleet-collection` (pure stdlib). `FleetMonitor` was added to
  `fleet_client.py`: a thread that polls `/api/status` every 15 s and exposes
  a JSON-ready `status()`; both UIs use it instead of hand-rolled polling.
- **`FLEET_INTEGRATION.md`** — this file.

### Changed

- **`make_teleop_scene.py`** — the fleet hunks applied by 3-way merge onto
  the adjust-mode version: `--fleet_server`, `--collector_id`,
  `--fleet_token`, `--fleet_scenes`, `--fleet_scene_ids` (exclusive with
  `--scene_list`); `--dataset_file` removed (superseded by one file per
  episode); the recorder uses `PerEpisodeHDF5DatasetFileHandler`;
  `EpisodeFlow` gets a `fleet` client and `export_episode` mints the episode
  UUID, names the file `<scene>_<timestamp>_<uuid8>.hdf5`, stamps
  `episode_uuid`/`collector_id`, closes the file and queues it for upload;
  `run_teleop(..., scene_usda, fleet=None)` keeps upstream's `scene_usda`
  (adjust-mode sidecar) *and* takes `fleet`; `load_scene_list(fleet)` adds the
  two fleet sources; `main()` checks in, syncs docs, declares presence per
  scene and drains the outbox at exit; the CloudXR launch calls
  `patch_cloudxr_wss_backend_port()` so a moved signaling port reaches the
  WSS proxy.
- **`recording.py`** — `AppendableHDF5DatasetFileHandler` replaced by
  `PerEpisodeHDF5DatasetFileHandler` (one file per `write_episode`).
- **`task_display.py`** — scene ↔ task-description matching by basename
  *without extension*, so a `.usdz` from the fleet server matches JSONs
  authored against the generator's `.usda`.
- **`teleop_launcher.py`** — the fleet version, refactored to consume
  `session_config` and `FleetMonitor`: it no longer owns a schema, scan
  functions, settings I/O or a polling thread — only Tk widgets and gestures.
  Settings now include the scene selection and are shared with the app.
- **`teleop_app.py`**
  - `TeleopProcess(base_args, launch_provider)`: every start asks the
    provider for a `LaunchSpec` (args + env) and runs
    `isaaclab.sh -p make_teleop_scene.py <--teleop-arg extras> <spec.args>`
    with `{**os.environ, **desktop_env, **spec.env}`. Validation errors
    (`ValueError`) come back as the `/start` message the page shows.
  - `SessionConfigurator`: the app's side of `session_config` — settings
    (re-read on external change), the fleet monitor (runs only while the
    saved source is the fleet server and the URL is set), the scene table (a
    5 s cache serves the frequent `/status` summary), `launch()`, `state()`.
  - `/control` WebSocket: JSON requests `{id, op, ...}` → `{id, ok, ...}`
    with ops `state`, `save {settings}`, `connect`, `refresh`. This exists
    because the app's HTTP layer (`websockets` `process_request`) cannot read
    request bodies, and the settings/selection payload does not fit a query
    string safely. Ops run on a worker thread (`asyncio.to_thread`).
  - `/status` gains `summary` (what Start would launch); the CloudXR proxy
    port now comes from the running launch's env, then the saved settings,
    then the environment.
  - `--teleop-arg` is now for flags the page does not cover; it goes first so
    saved settings win. `--mic_device` in `--teleop-arg` still overrides the
    relay.
- **`source/isaaclab_teleop`** — `patch_cloudxr_wss_backend_port()` exposed
  from `session_lifecycle.py` (the monkey-patch `TeleopSessionLifecycle`
  already applied), with tests and a changelog fragment, so a script that
  owns the CloudXR runtime itself can apply it.
- **`README.md`**, **`SETUP.md`** — fleet section, ports section, the
  launcher and app descriptions, daily-use steps, troubleshooting rows.

### Deliberately not ported

- The `cyqian/fleet-collection` commit that unsets `DISPLAY` under
  `--headless` and removes the retargeter tuning UI. On that branch the fix
  targeted `ssh -Y` (GLXBadFBConfig via a forwarded X display). The app flow
  instead *copies* the desktop session's `DISPLAY` into the child on purpose
  (`_desktop_env`) so the GLFW tuning UI can open; the two intents conflict,
  and the service flow does not go through a forwarded display. Revisit if
  the service launch ever hits GLXBadFBConfig.

## Behaviour worth knowing

- **Scene sources are exclusive by construction.** Local mode writes the
  ticked files to `<record_dir>/launcher.scene_list.json` and passes no fleet
  flags; server mode passes `--fleet_server` + `--fleet_scene_ids` and no
  scene list. `make_teleop_scene.py` rejects both at once.
- **The fleet token is never on the command line.** `build_launch` puts it in
  `FLEET_TOKEN` (which `--fleet_token` defaults to); the app's launch log
  masks it.
- **Selection persistence.** `settings.selection.{local,server}` maps scene
  name → bool. Names not present take the source default, so new scenes on
  the server appear ticked while the fleet needs them, and new local files
  appear ticked.
- **Start from the page saves first.** Unsaved edits on the Scenes/Parameters
  tabs are saved before `/start`; the Session tab shows an "unsaved changes"
  banner meanwhile.
- **Adopted runs.** An app restart adopts a running teleop as before; its
  launch env is unknown, so the CloudXR link falls back to the saved
  settings' proxy port.
- **Drag-to-paint selection** exists only on the desktop launcher; the page
  uses tap-to-toggle plus Select all / none / needed (touch has no drag on
  table rows in the Quest browser).

## Verification done

- `session_config` unit-exercised: default/changed args, app-mode mic
  override, port validation and collisions, local and server tables, both
  launch specs, settings round-trip with merge, 0600 mode.
- `teleop_app.py` run on a spare port: page served with three tabs; `/status`
  with summary; `/control` `state`/`save`/`connect`/unknown-op paths
  including the operator-facing errors; server mode with an unreachable URL
  reports the fleet error; `/start` in unconnected server mode refuses with
  the message; the proxy port in `cert_url` follows the saved settings.
- `teleop_launcher.py` driven against the local display: source switching
  keeps each source's ticks, Connect to an unreachable server shows the
  error, `build_launch` from the UI state.
- `make_teleop_scene.py --help` parses; all files compile; pre-commit passes;
  `isaaclab_teleop` lifecycle tests pass (25).
- Not exercised here: an end-to-end run against a live fleet server and a
  headset (no server or headset in this environment).

## Follow-ups

- The vendored `scenes/` on this branch were regenerated (tables raised to
  1 m). Anything pushed to a fleet server from the older scenes is stale by
  sha256; repackage to `.usdz` and re-push.
- `teleop_launcher.py --dataset_file` no longer exists; older shortcuts
  passing it must drop the flag.
