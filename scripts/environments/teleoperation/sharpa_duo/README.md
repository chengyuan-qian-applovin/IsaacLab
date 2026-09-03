# SharpaWave Duo USDA Teleop

Load any scene USDA, drop a bimanual SharpaWave rig into it, and teleoperate
the rig with XR hand tracking:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --scene_usda ~/sim_benchmark/scene/taco_hoi_178_023.usda --headless
```

## Robot embodiments

`--embodiment` selects the robot (default `franka_duo`); both embodiments carry
two 22-DoF SharpaWave hands and share the same 58-D action space, recording
format, and voice/auto-start flow:

- `franka_duo` — the FR3 Duo: a fixed torso with two 7-DoF Panda arms. Default
  placement stands the torso south of the table facing +y.
- `yam_duo` — two 6-DoF [I2RT YAM Ultra (v2)](https://i2rt.com) arms on a slim
  mounting rail, bases 56.5 cm apart. Default placement puts the rail ON the
  tabletop at the table's near edge (`(0, -0.55, 1.0)` on the raised TACO
  table), arms reaching over the table. The wrist IK commands the SharpaWave
  wrist bodies directly, so hand alignment is independent of the arm mount.
  Rebuild the robot USD from the vendored I2RT URDF with
  `assets/robots/yam_ultra/make_yam_duo_assets.py`.

Recorded demos are stamped with an `embodiment` HDF5 attribute; the replay
script reads it back automatically.

Put the headset on, open the CloudXR client (Quest/Pico: the CloudXR.js web
client, e.g. `https://nvidia.github.io/IsaacTeleop/client/release-1.3.x`; Apple
Vision Pro: pass `--cloudxr_env avp` and use the Isaac XR Teleop Sample Client),
connect, and press **Play**. Your wrists drive the two arms through
differential IK and all ten fingers are retargeted onto the SharpaWave hands.
The arms render 5% transparent by default so they don't block your view
(`--arm_visual normal|hidden` to change; `--visualize_hands` draws markers on
the tracked joints).

## Recording episodes, hands-free

Recording is on by default (`--no_record` disables it). Every labeled episode
becomes its **own** robomimic-style HDF5 file under `--record_dir` (default
`./datasets/duo_teleop`), named `<scene>_<timestamp>_<uuid8>.hdf5` and holding
a single `demo_0` — one trajectory per file, so each can be uploaded to the
fleet server the moment it is labeled, and an interrupted write can never
corrupt earlier demos:

1. **Play** starts teleop and the episode buffer — by the client button, the
   voice command, or **auto-start**: hold both wrists at the robot's hand
   poses (within 5 cm / 20° for 0.5 s, `--auto_start_*_tol` to tune,
   `--no_auto_start` to disable) and teleop engages by itself with zero
   initial IK error, so the robot never snaps to distant hands. After a stop
   it re-arms only once you move your hands clearly away. The match is
   checked at the SharpaWave hands' wrists (where your own wrist maps onto
   the robot), not the arm flange. While auto-start is waiting, axis frames
   mark the poses to match — large on the robot's two hand wrists, small on
   your calibrated wrist targets (robot wrists only while a hand is
   untracked); they disappear when teleop engages (`--debug_auto_start` keeps
   them up and prints the errors).
2. **Say "success" or "failure"** to end the episode: the demo is exported
   with that label, the scene resets, and teleop ends in the stopped state —
   press Play (or match the start pose) for the next episode.
3. **Reset** (headset button or saying "reset") discards the in-flight
   episode instead and leaves teleop stopped — start the next episode via
   Play or auto-start; an episode timeout also discards.

Each demo carries per-step robot joint states, tracked object poses, the 58-D
actions, the raw XR hand poses (`obs/xr_hands`, (T, 2, 26, 7), the retargeter
input — enough to re-tune retargeting offline), the PD drive setpoints
(`obs/joint_setpoints`, (T, 58), the differential-IK output), and HDF5
attributes: the boolean `success` label, the `scene` name, the `episode_uuid`
(the identity the fleet server also keys on), and the drive gains it was
recorded with (`arm_kp`/`arm_kd`/`hand_kp`/`hand_kd` — the `--arm_kp` etc.
flags, tunable from the "Control gains" group of either UI).

## Fleet collection (multiple headsets, one coordinator)

For campaigns with several Quest/AVP collectors running at once, the
`duo-fleet-server` repo (a small FastAPI + SQLite service kept separate from
IsaacLab; see its README for setup) is the single source of truth: which scenes
need how many successful demos, who is working on what right now, and every
labeled trajectory. From the headset, pick **Fleet server** as the scene
source on the app's Scenes tab (URL, optional collector id and token, tap
Connect, tick scenes) and start the session as usual. From a terminal:

```bash
export FLEET_TOKEN=change-me   # or pass --fleet_token
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --fleet_server http://fleet-host:8080 --collector_id ws1-quest --headless
```

What fleet mode does, in the order it happens:

1. **Startup sync**: the collector checks in and prints the fleet-wide status
   (progress toward targets, who else is online). The loose scene-doc JSONs
   from the server's `scenes/` (e.g. `scene_instruct.json` with the task
   descriptions) are always re-downloaded into the local cache, so the copies
   next to the cached scene files are the latest by construction.
2. **Scene download**: the scenes to work on come from the server —
   `--fleet_scene_ids` names them explicitly (what both UIs pass for the
   ticked scenes; mutually exclusive with `--scene_list`), or with no
   selection at all the server picks the `--fleet_scenes` most-needed ones
   (highest priority, fewest active collectors, most demos remaining). Every
   scene on the server is one **self-contained `.usdz` package** (flattened,
   geometry and textures inside, no external references except the
   runtime-resolved `OmniPBR.mdl`), so a scene is exactly one download into
   `<record_dir>/fleet_cache/scenes/` — there is no asset tree to mirror, on
   the server or here. Downloads are sha256-verified against the server's
   scene row: a cached file whose hash already matches is skipped, a changed
   one is re-fetched, and a hash mismatch after download is a hard error.
   Scenes cycle with "next" as usual.
3. **Presence**: entering a scene declares "this collector works here" — pure
   information, never a lock: any number of collectors may share a scene, and
   a crashed collector can never block anyone (its presence just goes stale
   after 120 s without a heartbeat).
4. **Immediate upload**: the moment an episode is voice-labeled, it is queued
   in a local outbox (`<record_dir>/fleet_outbox.sqlite3`) and a background
   thread uploads it — file first, then the metadata commit, keyed by the
   episode UUID so retries are idempotent. The teleop loop never waits on the
   network; if the server is unreachable, episodes stay queued (across
   restarts) and sync when it returns. The reply prints the scene's live
   progress and says when it hits its target.

Seed the server with scenes from any collector machine:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/fleet_push_scenes.py \
    --fleet_server http://fleet-host:8080 --scene_dir ~/scenes_usdz --target 20
```

and watch the live dashboard at `http://fleet-host:8080/`. Push
self-contained `.usdz` packages only: the generator's `.usda` scenes reference
a separate `02_mesh/...` tree that collectors never receive, so convert them
first (dependency closure → flatten → usdz, documented in the
teleop-data-server README under "File organization convention"). The push
script warns on any non-`.usdz` file. The "adjust object" pose sidecars
(`<scene>.poses.json`) are loose JSON docs too: pushed into the server's
`scenes/` they reach every collector through the doc sync.

## Replaying episodes

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/replay_teleop_scene.py \
    --scene_usda ~/sim_benchmark/scene/taco_hoi_178_023.usda --headless
```

Loads the same USDA, replays the newest recorded dataset (or `--dataset
<file-or-dir>`) **kinematically** — every frame is a prescribed recorded state
pushed through forward kinematics; `env.step()` is never called and every PhysX
solver knob is floored, so nothing is solved or collided. One third-person
camera (`--cam_eye`/`--cam_lookat`) writes `<output_dir>/<demo>/video.mp4` at
30 fps plus a `meta.json` with the success label. `--episodes
all|success|failure|0,3,7` selects demos. No domain randomization.

### Voice commands (local Whisper)

Voice commands are transcribed locally by OpenAI Whisper (`--whisper_model`,
default `base.en`; `small.en` is more accurate and ~3x slower), on the CPU so
it never competes with the sim and CloudXR for the GPU (`--whisper_device cuda`
if you would rather spend ~1 GB of VRAM for lower latency). Whisper is tuned
for this closed vocabulary rather than dictation: the decoder is primed with the
command words (`initial_prompt`), decodes with beam search, and its own
confidence is honoured — a clip it flags as probably-not-speech or decodes
poorly is reported as "no intelligible speech" rather than trusted. Output that
cannot be real speech (an impossible word rate, or one word looping for a page,
which Whisper emits when a clip is cut mid-word) is rejected and logged. A
short utterance that matches no command exactly is matched fuzzily against the
vocabulary ("we're set" → reset, logged as `(fuzzy match)`); sentences are
never fuzzy-matched, so room conversation must contain a command verbatim to
count. Besides the labels, saying **"align"** (while teleop
is stopped) re-anchors the XR session: it rotates the world about your head
until you face the robot's forward axis, moves you to `--align_head_xy`
(default: the TACO table's near edge) and sets your head `--align_head_z`
(default 1.5 m) above the scene floor — pass 0 to keep the headset's own
floor calibration instead — the port of the source branch's AVP Align
button, with voice replacing the button. The head pose is
queried from XRCore on demand; do NOT put a head tracker in the retargeting
pipeline (it makes every session step fail on this stack). The new anchor is
pushed to the renderer and the retargeting pipeline at once, and half a
second later the two are compared: if the headset view did not follow (the
symptom is the small wrist-target frames jumping away from your hands while
the scene stays put), the log says so, the compositor is re-bound to the
scene's anchor prim, and you say "align" again. That re-binding also happens
at every scene start: Kit keeps its XR session across "next" while each scene
rebuilds the stage, so without it the compositor stays attached to the
previous scene's deleted anchor prim.
Every transcription is printed to the console, labels and mis-hearings alike,
and echoed in the headset on a floating panel: `Heard: "..."` for an utterance
that matched no command, and `Detected "..." - executed: ...` naming the effect
(or the reason it was ignored) when one ran. The panel hides itself after
`--voice_display_seconds` (default 4); `--voice_display_pos` moves it and
`--no_voice_display` turns it off. Since it shows mis-hearings too, a panel
that stays blank while you talk means the audio never reached the recognizer —
check the microphone rather than the wording. Every spoken command runs once
per time you say it: "reset … reset … reset" resets three times, whether you
pause between them (three utterances) or say them in one breath (one
utterance, logged as `-> RESET x3`).
The full voice vocabulary: **"success"** / **"failure"** (label + export the
episode), **"align"** (re-anchor; only while teleop is stopped — say "stop"
first if it is running),
**"play"** (or "start" — starts teleop, driven through the same state machine
as the client button), **"stop"** (pauses teleop, keeping the episode
buffer; resume with "play" or auto-start — "pause" is deliberately not a
synonym: it was mis-heard in room conversation far too often),
**"reset"** (discards the in-flight episode and resets the scene),
and **"next"** (or "skip" — advance to the next scene in the `--scene_list`,
wrapping at the end; an unlabeled in-flight episode is discarded, a
label-pending one must be labeled first). An utterance matching more than one
command is ignored.

## Launcher UI

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_launcher.py
```

The desktop counterpart of the headset app below: a two-page tkinter launcher
(Isaac Sim only starts when you press Start) that renders the same schema,
scene sources and settings file (`session_config.py`), so a choice made here
shows up in the headset and vice versa. Page 1 groups the teleop parameters
by concern (operator & voice, session start, domain randomization, control
gains, visuals, advanced, network ports); page 2 picks the record directory
and the **scene source** — a radio choice between two mutually exclusive
modes:

- **Local directory** (default): scan a directory recursively for scene files
  (`*.usdz`, `*.usda`, `*.usd`) and tick the scenes to collect; the table
  shows the per-machine success/failure demo counts recorded under the record
  directory. The run is fully standalone — no fleet server is involved at all.
- **Fleet server**: enter the server URL (plus optional collector id/token)
  and press **Connect**. The table then lists the *server's* scenes with the
  server's numbers only: live **Fleet progress** (`successes/target`, green
  when met) and **Working now** (who is collecting each scene right now)
  columns, auto-refreshed every 15 s; newly listed scenes come pre-ticked
  when the fleet still needs them, and **Select needed** re-derives that
  ticking on demand. Start passes the ticked scene ids as
  `--fleet_scene_ids`: the run downloads them from the server
  (sha256-verified) and uploads every labeled episode as it happens.

Selection works the same way in both modes (click toggles one scene,
dragging paints the toggle over consecutive rows, and Shift+Click extends
the last toggle over the whole range, Excel-style). All settings —
parameters, directories, scene source, fleet connection, scene selection,
network ports, window geometry — persist across runs in
`~/.config/duo_teleop_launcher.json` (shared with the headset app), and a
remembered fleet-server source reconnects automatically. Cycle the selected
scenes with the "next" voice command; when the run exits, the launcher
returns to the table with refreshed counts.

### Network ports

A teleop session listens on four ports; the **Network ports** group (in
either UI) edits all of them (blank keeps the default), and they are passed to
`make_teleop_scene.py` as environment variables — the same variables work on
the command line. Two operators sharing one workstation each need their own
TCP ports; open whatever you pick in the firewall (`sudo ufw allow <port>/tcp`).

| Port | Default | Set via | Who connects |
|---|---|---|---|
| CloudXR signaling (TCP) | 49100 with the Quest/WebRTC profile, 48010 with the AVP native profile | `NV_CXR_SERVER_PORT` | The WSS proxy (Quest) or the AVP client directly; CloudXR refuses to start if it is taken |
| CloudXR media (UDP) | 47998 | `NV_CXR_MEDIA_PORT` | The headset's video/input/audio stream |
| WSS proxy (TCP) | 48322 | `PROXY_PORT` | Quest: the CloudXR.js browser page (`https://<ip>:48322/`); AVP in secure mode. Forwards to the signaling port |
| Headset microphone (TCP) | 8444 | `--mic_device quest:<port>` / `avp:<port>` | The Quest mic page and the AVP client's mic stream (`wss://<ip>:<port>/audio`); not used when the headset app relays the mic |

```bash
# Second operator on the same workstation, e.g. from a second Linux account:
NV_CXR_SERVER_PORT=49101 NV_CXR_MEDIA_PORT=47999 PROXY_PORT=48323 \
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --mic_device quest:8445 --headless ...
```

The WSS proxy is told about a moved signaling port through
`isaaclab_teleop.patch_cloudxr_wss_backend_port` (isaacteleop's launcher
otherwise leaves it dialing 49100). The AVP native client keeps its own port
setting, so a moved signaling port has to be entered in the Isaac XR Teleop
client as well. The USB-tethered ports (`USB_UI_PORT` 8080,
`USB_BACKEND_PORT`, `USB_TURN_PORT` 3478) belong to isaacteleop's OOB mode,
which this pipeline does not use; the fleet server's port is part of its URL.

## Headset control app

```bash
# try it in a terminal
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py
# or install it as a user service that comes back after a reboot
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py --install-service
sudo loginctl enable-linger $USER   # keep it running when you are logged out
```

One page in the headset browser that starts a session, holds the microphone,
and kills teleop — replacing the pair of hand-edited bookmarks whose URLs went
stale whenever an address or a port moved. [SETUP.md](SETUP.md) has the
from-scratch procedure for a new workstation and for onboarding each headset.

On Quest software v74 or newer, the browser's ⋮ menu offers **Add this page to
my Library**, which puts the page in Library → Apps as its own tile (named
"Teleop", with the icon the app serves) that opens in a window of its own. The
page also ships a web app manifest, so where the browser offers "Install app"
that works too; it only does so on an origin it deems secure, which a
self-signed certificate is not always judged to be.

Bookmark or tile **`https://<hostname>.local:8500/`** once. mDNS keeps the host correct
across DHCP leases and the port is fixed by the service, so the bookmark does
not go stale; the app looks up the CloudXR client's ports when you ask for the
link rather than baking them in, and hands the client `serverIP` and `port` as
query params so its connection fields arrive filled in.

It runs as a **supervisor**, independent of any teleop run, so it is still
reachable when teleop is stopped or wedged — which is when you need it. **Start
session** starts the microphone, launches teleop if it is down, and opens the
CloudXR client in a new tab; **Finish session** next to it is the mirror image,
stopping the microphone and teleop together. The rows below control the parts
one at a time: the certificate and CloudXR links, then **Restart teleop**,
**Kill teleop**, **Start microphone** and **Stop microphone**, then **Log**, which opens
the last 200 lines of the run in its own live-updating page at a font readable
inside the headset. **Kill teleop** signals the whole process group
(SIGINT, then SIGTERM, then SIGKILL) and then reaps the CloudXR runtime that
normally outlives it holding 49100/48322 — the thing `Ctrl-C` alone leaves
behind. The sweep only touches runtimes younger than the run it just stopped
and owned by you, so a session you started by hand in a terminal, or one
belonging to another user of the workstation, survives untouched.

The page has three tabs. **Session** is the one-tap flow above. **Scenes** and
**Parameters** are the desktop launcher's two pages, reproduced for the
headset: the record directory; the scene source (**Local directory** with the
scene table and this machine's success/failure counts, or **Fleet server**
with URL, collector id, token, a Connect button, the fleet-wide progress and
"working now" columns refreshed every 15 s, and **Select needed**); tap a row
to toggle it, or use Select all / none; and every teleop parameter grouped as
on the desktop, including the network ports. Save persists to the same
`~/.config/duo_teleop_launcher.json` the desktop launcher reads, and **Start
session** launches from the saved settings (saving first if the tabs have
unsaved edits): a scene-list JSON of the ticked local files, or
`--fleet_server` + `--fleet_scene_ids` for the ticked server scenes, plus
`--record_dir` and only the parameters that differ from their defaults. The
Session tab shows a one-line summary of what Start would launch, and Start
refuses with a clear message when nothing is ticked, a port is invalid, or the
fleet server is not connected. The fleet token travels to teleop in the
`FLEET_TOKEN` environment variable, not on the command line.

The app owns the microphone and relays it to teleop over `--mic_device hub`,
which it passes automatically to anything it launches. Because capture lives in
the app rather than in the teleop process, **it survives teleop restarts**: tap
"Start microphone" once per headset session, not once per run. Only the newest
page streams — an older tab is evicted with close code 4001 and told to release
the microphone, which is what stops a pile of stale tabs from fighting over it.

Keep the app tab open while you teleoperate. Capture stops if that page is
closed, so the CloudXR client is deliberately opened in a *new* tab rather than
navigated to. Flags the Parameters tab does not cover can still be forwarded
once per argument with `--teleop-arg` (e.g. `--teleop-arg --robot_pos ...`);
they go first on the command line, so the saved settings win where both name
a flag, and scenes must come from the Scenes tab rather than `--teleop-arg
--scene_list`. Both the app
and the CloudXR proxy use the same self-signed certificate, but the headset
may still ask you to accept it for the proxy's port. The page checks for you.
The certificate button doubles as the boot indicator: it reads "Teleop not
running", then "Preparing teleop..." while the run boots (inert, since tapping
would only wait on a dead port), then "Teleop ready - certificate OK" when the
browser already trusts the proxy or "Teleop ready - accept certificate" when
it does not. The status line at the top agrees with it: "teleop starting -
loading the simulator" (amber, pulsing) while the process boots, "teleop
running" (green) only once the proxy listens, "teleop stopped" otherwise.
"Open CloudXR" is likewise inert until the proxy is up. During
"Start session" the certificate is only asked for when the check says it is
missing, and the flow moves on to "Open CloudXR" by itself as soon as you have
accepted. Restarting the app itself (for a code update) leaves a running
teleop alive and the new instance adopts it, so status and "Kill" keep
working across the restart. The one tap that
cannot be removed is "Connect" in the NVIDIA page: WebXR requires a user
gesture to enter VR, and this client build only *reads* the server address
from the URL, it does not connect on its own. (NVIDIA's `--setup-oob` gets to
zero taps by driving the headset browser over `adb`, which needs developer
mode and a USB link.)

## Multi-scene sessions

`--scene_list scenes/scene_list.json` teleops through a list of scenes (JSON:
a list of USDA paths or `{"scenes": [...]}`, relative to the JSON's
directory); the session starts at the first and **"next"** advances. Each
scene switch rebuilds the environment (the CloudXR runtime, headset
connection, ASR model, and any "align" adjustment all survive it) and
opens a fresh dataset file named `dataset_<time>_<scene>.hdf5`; every demo
also carries a `scene` HDF5 attribute naming the scene it was recorded in.

Scene-list entries may also be scene-generation dicts (`{"scene": ...,
"task_description": ...}`), so a run's instructions JSON — e.g.
`scenes/scenegen/04_episode_scenegen/runs/test_scenes_50/scenes_first50.json`
— works directly as a `--scene_list`. Absolute scene paths authored on
another machine are resolved by basename next to the JSON.

## Task description in the headset

When a scene has a task description — from the scene-list entry, or looked up
in any instructions JSON sitting next to the scene file — it is shown as a
floating emissive panel in the world (`task_display.py`), so the operator
reads the task in XR. `--task_display_pos` moves it (default: past the table
at head height, facing the operator); `--no_task_display` hides it. The text
is also printed to the console at every scene start. The voice-feedback panel
is a second billboard from the same module, sitting below this one by default.

## Settling period

After every scene reset (including the domain-randomized ones), physics runs
for `--settle_time` seconds (default 1.0) with the robot held still so the
objects drop and come to rest on the table before the episode starts. The
recorded `initial_state` is taken after settling, so demos start from the
scene the operator actually saw. `--settle_time 0` disables it.

## Domain randomization

On by default (`--no_dr` disables), applied at every episode reset:

- **Arm start pose**: each arm joint gets a uniform offset within
  `--dr_arm_jitter` (0.08 rad) around the ready pose. Auto-start adapts —
  you match the robot wherever it actually is.
- **Object placement**: each tracked object gets a uniform xy offset within
  `--dr_object_xy` (5 cm) and a yaw within `--dr_object_yaw` (180°) around
  its authored pose. `--dr_object_bias` additionally shifts the whole
  randomization center that many metres horizontally toward the robot base
  (never past it), to bring objects within easier reach; it is **0 by default**
  so a scene's authored layout — including one saved by the "initial" editor — is
  used as the center as-is. Turn it on only for scenes authored out of reach,
  and be aware that it compounds: adjust mode saves where objects currently
  are, so re-saving a shifted layout shifts it again on the next run.
  Draws are rejection-sampled against bounding-circle
  overlap (footprints from the USD bounds + 1 cm margin — the collision model
  of sim_benchmark's scenegen solvers), never demanding more clearance than
  the authored layout had; after 50 failed draws the authored poses are kept.
  Stacked arrangements (xy-coincident objects, e.g. the ARCTIC box lid on its
  base) move as one group and skip yaw so they are never knocked apart.

### Adjusting where objects start

**Quick start.** Say **"initial"**: the robot disappears, the desk turns
translucent, and green ghosts mark where each object starts. **Pinch** an
object (thumb and index tip touching) to grab it — move your hand to move it,
twirl your fingertips to rotate it, open your fingers to drop it where it is;
the table is a hard floor. Pinch empty air with both hands to move, rotate, or
zoom your view. Hold a **Quest controller** and flick the **thumbstick** to set
the randomization: left/right = xy range (1 cm steps), up/down = yaw range (5°
steps); the floating panel shows the values. "reset" puts objects back on their
ghosts. **"finish"** writes `<scene>.usda.poses.json` to disk **immediately** —
poses and ranges together — and it takes effect at once: the next reset uses
the new layout and ranges, and every later launch of the scene loads it.

Saying **"initial"** opens a pose-authoring mode that lets you re-author
a scene's object layout from inside the headset, without editing the USDA.

It is a **kinematic pose editor** — nothing is simulated while it is open, so
nothing can be flung, topple, or drift. On entry the arm rig parks out of
sight, the desk turns translucent, a translucent green **ghost** of every
object marks where it stood, and your tracked hand becomes the cursor (small
red joint spheres). **Pinch directly on an object to grab it**: it is
**welded to your fingertips 1:1** like a real object held at the pinch point —
the grabbed spot stays under your fingers, and rotation follows the pinch
grip itself: twirl your fingertips like turning a small dial, or turn the
whole hand; either pivots the object about the grip with the natural lever
arm (large turns: re-grab and keep turning, as in real life), and tilting
leans it (e.g. a brush resting only its head on the table). It stays exactly
where you release it, verbatim:
nothing settles or snaps, so what you place is what is saved. The one
constraint is the **tabletop, which is a hard floor**: no part of a mesh can
be dragged below it, so pushing down rests the object exactly on the surface —
the easiest way to place something flat. A pinch registers when thumb and
index tips close within 1 cm and releases past 3 cm. One object per hand; a
pinch on a panel key is a tap, anywhere else near an object is a grab.
**Pinching empty air with both hands grabs the world**: move your hands to
pan the view, turn them to rotate it (full 3-DoF — say "align" to re-level if
you lose the horizon), and spread/close them to zoom (a dolly toward/away
from you). Pure view motion: objects never move, and "done" restores the
original view exactly so teleop alignment is untouched. Nothing is recorded,
and whatever the recorder held is thrown away — authoring a layout never
leaves a demo behind.

While the mode is open, a translucent blue square on the tabletop previews the
XY randomization region per object (the yaw range is shown only as a number on
the range panel). It follows the objects as you move them and resizes live as
you retune the ranges (no overlay with `--no_dr`).

- **"finish"** (or "done") writes the edited poses **and the randomization ranges you
  tuned** (`xy_range` in m, `yaw_range_deg`) to a sidecar
  `<scene>.usda.poses.json` next to the scene file, and restores the rig
  exactly where it stood. The USDA is never modified, and the next load picks
  the sidecar up automatically — both the poses (as the new randomization
  centre) and the ranges (over the `--dr_object_xy`/`--dr_object_yaw`
  defaults; a value explicitly changed on the command line still wins). Range
  edits are live from the moment you make them; "done" is what persists them.
  (Poses are saved exactly as placed: an object left hovering will drop on the
  next reset, and deliberately overlapped objects get separated by stock
  physics there — delete the sidecar to fall back to the USDA's authored poses
  and the default ranges.)
- **"reset"** means *undo* while the mode is open: every object snaps back
  onto its ghost. It does not reset the scene.

A floating range panel appears alongside showing both values (`xy` in meters,
`yaw` in **degrees**), its `-`/`+` keys tapped with a thumb-index pinch
(±1 cm / ±5°). Holding a **Quest controller** (which replaces hand tracking on
that side) is the quickest way to retune: the thumbstick steps the ranges
directly — left/right = `xy_range` −/+, up/down = `yaw_range` +/− —
auto-repeating while held, with the panel updating live; the trigger with the
controller tip at a panel key also works as a tap. Exact values can be typed
at the terminal (`xy_range=0.08` / `yaw_range=45`, degrees). `--no_adjust`
disables the whole mode — no panels or controller plumbing, leaving teleop
exactly as it was without the feature. Full desk/ghost transparency needs the
default `--arm_visual transparent`; other modes render them solid.

The panel is placed within arm's reach at your station, just above the
tabletop — it must be physically reachable, because a tap is a pinch at the
key's world position (tight in the panel plane, ~12 cm of slack in depth). The
last step lights its key up on the panel; a pinch that lands near a key but
misses prints the miss distance on the console, and `--debug_adjust_buttons`
draws a sphere at every key's true hit center. `--range_panel_pos` moves the
panel if the default lands badly in an unusual scene.

Audio can come from two places:

- **Workstation microphone** (default): captured via `arecord`
  (`--mic_device` selects the ALSA device); stay within speaking range.
- **Headset microphone** (`--mic_device quest` or `--mic_device avp`): nothing
  in the CloudXR stack streams the headset mic to the server, so
  `headset_mic.py` provides the path — a WSS server the headset streams 16 kHz
  PCM to, putting the mic at your mouth instead of across the room. Open the
  port first (`sudo ufw allow 8444/tcp`; `quest:<port>` / `avp:<port>` changes
  it). Stay quiet for the first ~2 s after the stream starts — the energy gate
  calibrates on that ambient. The two clients differ in how the stream starts:
  - **Quest** (`--mic_device quest`): the script serves a small HTTPS mic
    page and prints its URL; open it in the Quest browser *before* connecting
    the CloudXR client, accept the certificate (it reuses the CloudXR
    proxy's), tap **Start microphone**, grant the mic permission, then
    connect the CloudXR client as usual. The page keeps streaming from the
    background tab.
  - **Apple Vision Pro** (`--mic_device avp`, pair with `--cloudxr_env avp`):
    the Isaac XR Teleop Sample Client — built from its `feature/avp-voice-mic`
    branch — captures the mic natively and streams it to the same server by
    itself once its **Stream microphone** toggle (on by default) is enabled
    and the CloudXR session connects; grant the mic permission on first use.
    Nothing to open on the headset. (The wire protocol is identical, so a
    quest/avp mix-up only prints the wrong instructions — audio still works.)

Test the mic + ASR chain without starting the simulator:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --scene_usda unused --voice_test 20
```

Microphone notes for this machine (Legion, ALC287 codec): the capture channel
carries near-full-scale infrasonic wander, so **do not max the input volume** —
at 100% the wander clips and drowns speech. Keep the source around 10–25%
(`wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.15`); the labeler high-passes the
stream at 80 Hz and calibrates its energy gate on ambient noise at startup (do
that in a quiet moment). If it warns about a saturated microphone, lower the
capture volume. `--mic_device` selects another ALSA capture device.

Sanity-check an installation or a new scene without a headset:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --scene_usda <scene.usda> --smoke 120 --headless
```

This creates the environment, holds the rig's ready pose for 120 control steps,
then commands both flanges 3 cm up and verifies the IK follows (it fails loudly
if the action space or IK wiring is broken). `--smoke_adjust 120` validates the
adjust-mode pose editor the same way: it grabs an object with a synthetic
pinch, drags it through a full 6-DoF motion (translate, lift, yaw, roll), and
checks the final pose, that no other object moved, the exact robot restore,
and the panel placement (the scene's pose sidecar is backed up and restored,
so a smoke never re-authors it).

## What the robot is

One PhysX articulation (58 actuated joints): a fixed torso, two 7-DoF Franka
Panda arms, and two 22-DoF SharpaWave dexterous hands on the arm flanges. The
robot USD and the SharpaWave URDF/DexPilot configs are **vendored** under
`assets/` (~46 MB) so this directory is self-contained — they originally lived
in the untracked `sim_benchmark` checkout and were copied verbatim.

## How a hand pose becomes robot motion

```
headset (CloudXR) → OpenXR hand tracking (26 joints per hand)
  │  IsaacTeleop retargeting pipeline (duo_teleop_pipeline.py)
  ├─ wrists  → Se3AbsRetargeter × 2 → absolute flange pose targets, world frame
  ├─ fingers → DexHandRetargeter × 2 → DexPilot QP against the vendored
  │            SharpaWave URDFs → 22 joint angles per hand
  └─ TensorReorderer → 58-D action: [L wrist 7 | R wrist 7 | L fingers 22 | R fingers 22]
  │  make_teleop_scene.py teleop loop
  ├─ to_root_frame(): world-frame wrists → robot-root frame
  └─ env.step()
  │  action terms (duo_robot.py)
  ├─ left_arm / right_arm: damped-least-squares differential IK → 7+7 arm joint targets
  │  (solved once per control step, not per physics substep — OncePerStepDiffIKAction)
  └─ left_hand / right_hand: direct joint-position targets → 22+22 finger targets
  ▼ PhysX: 240 Hz physics, 60 Hz control, 30 Hz render (configurable)
```

Two conventions worth knowing:

- **Quaternions are xyzw everywhere** — Isaac Lab 3.0's data and math stack
  uses the Warp layout, and `Se3AbsRetargeter` happens to emit the same, so
  poses flow through without reordering.
- **The IK action terms expect root-frame commands** and do no frame conversion
  themselves; the teleop loop owns the world→root conversion.

The wrist rotation offsets (left roll −180°/yaw 45°, right roll 180°/yaw 135°)
map the OpenXR wrist frame onto the `panda_link8` flange frame. They were
calibrated on the original branch at the rig's ready pose; the clean ±45°
structure is the rig's flange-mount rotations. If the hands ever track with a
constant twist, these offsets in `duo_teleop_pipeline.py` are the knob.

### Operator hand-shape calibration

Finger retargeting uses the operator's hand-shape calibration from the source
branch (`assets/dex_retargeting/hand_calibration.yml`, measured with flat
hands; loaded by default, `--hand_calibration ''` disables). Per hand it holds
a global rotation + scale (~1.18 — wrist-pinned Procrustes on
index/middle/ring tips) and thumb/pinky length ratios + tip-direction
rotations. Recalibrate per operator with the ported calibration scene:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/calibrate_hand_shape.py \
    --user alice --headless
```

Connect the headset, press Play, hold both hands flat with fingers straight;
5 s later the frame is captured, solved, and written to
`assets/dex_retargeting/hand_calibration_alice.yml`. Teleop then loads it with
`--user alice` (an explicit `--hand_calibration` wins over `--user`).

**Fingertip convention**: Quest/OpenXR `*_TIP` joints sit at the center of the
fingertip capsule — about one tip-radius inside the skin — while the MANO
keypoints the DexPilot configs expect are on the skin surface, so raw Quest
fingers read ~1 cm short and touching fingertips still read ~2 cm apart. New
calibrations therefore extend each tip along its distal-bone direction by the
runtime-reported joint radius (`--no_tip_extension` opts out); the choice is
stamped into the yml and the teleop retargeter mirrors it automatically, so
calibration and runtime always use the same convention. Old calibration files
without the stamp keep the original center-tip behavior.

Applied exactly as on the source branch (`sharpa_retargeting.py`):
the rotation composes wrist-side into the flange offset (the arm tilts the
robot hand until its fingers align with yours), the scale multiplies the
DexPilot keypoints (the yml `scaling_factor` is overridden to 1.0), thumb and
pinky get their per-finger corrections afterwards, and DexPilot's pinch
project/escape hysteresis runs on your RAW fingertip distances so calibration
never shifts pinch timing.

## Files

| File | Role |
|---|---|
| `make_teleop_scene.py` | CLI entrypoint: builds the env from a USDA + CLI args, runs the XR teleop loop (or `--smoke`/`--smoke_adjust`/`--voice_test`). |
| `replay_teleop_scene.py` | Kinematic replay of recorded demos in the same USDA scene, one camera to MP4. |
| `duo_env.py` | Env config shared by teleop and replay (scene skeleton + managers). |
| `duo_robot.py` | The rig: articulation config (actuators, ready pose) and the 58-D action space, including the once-per-step IK optimization. |
| `duo_teleop_pipeline.py` | The IsaacTeleop retargeting pipeline (hand tracking → 58-D action). |
| `sharpa_retargeting.py` | Calibrated DexPilot finger retargeting (hand-shape calibration, raw-distance pinch hysteresis). |
| `usda_scene.py` | References the scene USDA into the env and registers its rigid bodies so resets restore their poses. |
| `adjust_mode.py` | The "initial" (adjust-mode) kinematic pose editor: pinch-grab objects in full 6-DoF, ghosts, translucent desk, robot park/restore, sidecar save. |
| `region_overlay.py` | Adjust mode's in-headset preview of the object-randomization region (XY square per object). |
| `recording.py` | Recorder terms + the per-episode HDF5 handler (one file per labeled trajectory). |
| `session_config.py` | What a session is launched with, shared by both UIs: parameter schema, ports, headset devices, the persisted settings file, scene/record-dir scanning, scene table rows, and settings → command line. |
| `teleop_app.py` / `teleop_app_page.html` | Always-on headset control app: start/restart/kill teleop, own the microphone across runs, hand out an up-to-date CloudXR link, and configure the session (scenes, fleet server, parameters) from the headset. |
| `teleop_launcher.py` | Desktop (tkinter) launcher with the same pages: parameters, scene source, per-scene counts, fleet progress. |
| `fleet_client.py` | Client for the duo-fleet-server: startup check-in, presence, scene download, crash-safe episode upload outbox; plus the read-only status monitor both UIs poll. |
| `fleet_push_scenes.py` | Seeds the fleet server with self-contained scene packages (`.usdz`), targets, and task descriptions over HTTP. |
| `SETUP.md` | From-scratch procedure: workstation install, app service, per-headset onboarding, terminal-only use, troubleshooting. |
| `FLEET_INTEGRATION.md` | What changed to bring fleet collection and the launcher's functions into the headset-app flow, and why. |
| `assets/robots/` | Vendored robot USD (torso + arms + hands + skin material). |
| `assets/dex_retargeting/` | Vendored SharpaWave URDFs + DexPilot YAMLs for the finger retargeting. |

## Vendored example scenes

`scenes/` ships ready-to-use scenes (~13 MB, git-lfs), so nothing outside the
repo is needed:

- `scenes/taco/scene/taco_hoi_178_023.usda` — the TACO brush-and-bowl tabletop
  (the default `--scene_usda`), with its object USDs alongside.
- `scenes/scenegen/04_episode_scenegen/runs/scenes/*.usda` — six scenegen
  scenes (ARCTIC box, HOI4D toy car / trash can, OakInk USB hub + stick, two
  more TACO tasks) copied from
  `gs://foundational-research/yjw/example_usda/`, with only the eleven object
  payloads they reference mirrored under
  `scenes/scenegen/02_mesh/06_usd_conversion/runs/usd/`. The GCS directory
  layout is preserved because the scene files reference their payloads by
  relative path — keep it intact when adding more scenes.
- `scenes/scenegen/04_episode_scenegen/runs/test_scenes_50/` — the 50-scene
  benchmark set (TACO / GigaHands / OakInk-v2) with its instructions JSON
  `scenes_first50.json` (per-scene task descriptions; use it directly as
  `--scene_list`) and all payloads mirrored alongside the six above. Note:
  a few GigaHands scenes author tall objects (e.g. a pan on a stand) inside
  the arms' ready-pose workspace at the default robot placement — adjust
  `--robot_pos` for those.

## Scene requirements and placement

- The USDA must have a **default prim**; everything it authors (geometry,
  lights, physics, materials) is referenced in unmodified.
- Its rigid bodies are auto-discovered and re-posed on every env reset
  (disable with `--no_track_objects`).
- The robot's pose in the scene is `--robot_pos` / `--robot_rot`; the XR
  anchor (where *you* stand) is `--anchor_pos` / `--anchor_rot`. Defaults are
  the validated TACO-tabletop placement: the torso south of the table facing
  +y, you at the torso. Keep the anchor yaw equal to the robot yaw so the
  robot's arms line up with yours.

## Provenance and scope

This is a port of the `teleop_taco_scene.py` pipeline from the
`feature/robolab-xr-teleop` branch (which targets Isaac Lab 2.x) onto
`release/3.0.0-beta2`, restructured to be scene-agnostic and self-contained.
Deliberately **not** ported (yet): the Success/Failure client dialog and the
client-side Align button (both replaced by voice commands), the
`calibrate_hand_shape.py` capture script (its output yml is vendored and
used), self-collision contact filtering, and the domain-randomized
four-camera replay.

What changed in the port, beyond reorganization:

- 2.x `OpenXRDevice` + custom `RetargeterBase` (deprecated on this branch) →
  the native IsaacTeleop pipeline + `IsaacTeleopDevice`, matching how the
  GR1T2 teleop tasks work here. The SharpaWave DexPilot retargeter with the
  operator calibration is ported as a custom pipeline node
  (`sharpa_retargeting.py`) driving the same vendored URDFs/configs.
- 2.x `sim.physx.*` settings → `sim.physics = PhysxCfg(...)` (multi-backend
  split); asset data reads go through `.torch` views; quaternions wxyz → xyzw.

Known caveats:

- CCD is requested but PhysX disables it under GPU dynamics (warning at
  startup); run with `--device cpu` if fast-motion tunneling matters more
  than simulation speed.
- The wrist offsets were carried over on the argument that the GR1T2 offsets
  are identical between the 2.x and IsaacTeleop stacks; if the very first live
  session shows twisted wrists, recalibrate (see above) before blaming IK.
