# Sim teleop from scratch

How to stand up the SharpaWave Duo teleop stack on a new workstation, put the
headset control app on it, and onboard any number of Meta Quest headsets (or
run without the app at all). [README.md](README.md) explains what the pieces
do; this is the ordered, copy-paste procedure.

What you end up with:

```
 headset(s)  ──Wi-Fi──▶  workstation
   browser tile             teleop-app.service  (always on, port 8500)
   "Teleop"                    ├─ control page + microphone relay
                               └─ starts / kills  make_teleop_scene.py
                                      ├─ Isaac Sim + the robot scene
                                      └─ CloudXR runtime  (49100/tcp, 47998/udp)
                                         + WSS proxy      (48322/tcp)
   NVIDIA CloudXR client  ◀──────────── video / hand tracking ────────────┘
   (nvidia.github.io, opened by the page)
```

Only the workstation is set up per machine. A headset needs nothing installed:
it opens one URL, and every Quest on the same network uses the same URL.

---

## 1. Workstation (once per machine)

All commands run on the workstation, from the Isaac Lab checkout, as the user
who will operate teleop.

### 1.1 Isaac Lab with teleop support

Follow the top-level install for Isaac Sim 6.0 and Isaac Lab 3.0, then make
sure the teleop extras are in the environment:

```bash
cd ~/IsaacLab
./isaaclab.sh -i            # core + mimic/teleop + extras
./isaaclab.sh -p -c "import isaaclab_teleop, isaacteleop, websockets, whisper, PIL; print('ok')"
```

The last line must print `ok`. `isaacteleop` brings the CloudXR runtime and
its WSS proxy; `websockets` serves the app page; `openai-whisper` does voice
commands (its `base.en` weights, ~140 MB, download into `~/.cache/whisper`
the first time teleop starts — do that once while online); Pillow draws the
app icon.

### 1.2 System packages

```bash
sudo apt install -y alsa-utils ffmpeg avahi-daemon openssl
systemctl is-active avahi-daemon     # must say: active
hostname                             # this, plus ".local", is the address headsets use
```

- `avahi-daemon` publishes `<hostname>.local` over mDNS so headset URLs stay
  valid across DHCP leases. Without it you bookmark the IP and re-bookmark
  whenever it changes.
- `ffmpeg` is required by Whisper; `alsa-utils` (`arecord`) only if you ever
  use a microphone plugged into the workstation instead of the headset's.
- `openssl` creates the self-signed certificate on first use.

### 1.3 Firewall

If `ufw` is active, open the CloudXR ports, the app port and mDNS:

```bash
sudo ufw allow 8500/tcp    # teleop app page + microphone relay
sudo ufw allow 49100/tcp   # CloudXR WebRTC signaling
sudo ufw allow 48322/tcp   # CloudXR WSS proxy (certificate page lives here too)
sudo ufw allow 47998/udp   # CloudXR media stream
sudo ufw allow 5353/udp    # mDNS, so <hostname>.local resolves from headsets
sudo ufw allow 48010/tcp   # only for Apple Vision Pro (native signaling)
sudo ufw status
```

### 1.4 Install the app as a service

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py --install-service
sudo loginctl enable-linger $USER   # keep it running when nobody is logged in
```

This writes `~/.config/systemd/user/teleop-app.service`, enables it at boot
and starts it. Verify:

```bash
systemctl --user status teleop-app.service          # active (running)
curl -sk https://127.0.0.1:8500/status -H "X-Teleop-Control: 1"   # JSON with "running": false
```

The first start generates `~/.cloudxr/certs/server.{crt,key}` if it does not
exist; the CloudXR proxy reuses the same certificate, so headsets accept one
certificate for the whole machine.

**Choosing what the app launches.** Everything a session runs with is set on
the page itself — the **Scenes** tab (record directory; a local scene
directory with the scenes ticked, or the fleet server) and the **Parameters**
tab (robot embodiment, voice, domain randomization, gains, ports, ...) — and
saved to `~/.config/duo_teleop_launcher.json`, which the desktop launcher
(`teleop_launcher.py`) shares. Fresh out of the box a session runs the
scenes under `scenes/` with the script's defaults, `--mic_device hub` and
`--headless`. Only flags the page does not offer need `--teleop-arg` (once
per argument, then reinstall):

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py --install-service \
    --teleop-arg --robot_pos --teleop-arg 0 --teleop-arg -0.6 --teleop-arg 0
```

**Several operators on one workstation.** Each user installs their own service
under their own account. Give each a distinct app port (`--port 8501`, ...)
and distinct CloudXR ports in the Parameters tab's **Network ports** group
(signaling, media and WSS proxy), otherwise two sessions fight over
49100/48322. The ports become environment variables of the launched teleop;
the page's CloudXR link follows them.

**Fleet server (optional).** On the Scenes tab choose **Fleet server**, enter
its URL (e.g. `http://fleet-host:8080`), the collector id this workstation
should report as (default: hostname) and the token, tap **Connect**. The
table shows the server's scenes with fleet-wide progress and who is working
on each; scenes the fleet still needs come pre-ticked. Sessions then download
the ticked scenes from the server and upload every labeled episode as it
happens (queued locally and retried if the server is down). README.md,
"Fleet collection", has the details; nothing else is installed for it.

### 1.5 Updating the app

After editing `teleop_app.py` (or `session_config.py`, `fleet_client.py`):

```bash
systemctl --user restart teleop-app.service
```

An edit to the page itself, `teleop_app_page.html`, or to its stylesheet,
`teleop_app_page.css`, needs no restart: the app reads both from disk on
every request, so a reload on the headset shows it.

A running teleop survives this (the unit uses `KillMode=process`) and the new
instance adopts it, so status and Kill keep working. Headset tabs that are
already open keep the old page until reloaded.

Useful commands:

```bash
journalctl --user -u teleop-app.service -f      # follow the app's log
tail -f ~/.cache/teleop_app/last_run.log        # the current teleop run's output
systemctl --user stop teleop-app.service        # stop the app (teleop keeps running)
systemctl --user disable --now teleop-app.service   # uninstall from boot
```

---

## 2. Each headset (once per Quest, per account)

Nothing is installed on the headset. Do this logged in as the account that
will operate; a different account on the same headset repeats it (the browser
profile, certificate acceptance and Library are per account).

1. **Same network.** The headset must be on the same LAN as the workstation
   (same Wi-Fi as the workstation's wired network is fine). mDNS does not
   cross routers: if `https://<hostname>.local:8500/` does not load, use the
   workstation's IP instead (`hostname -I` on the workstation).
2. **Open the app.** In Meta Quest Browser go to `https://<hostname>.local:8500/`
   (e.g. `https://axon-1100.local:8500/`). Accept the self-signed certificate
   warning (Advanced → Proceed).
3. **Make it a Library tile.** ⋮ menu (top right) → **Add this page to my
   Library**. It appears in Library → Apps as "Teleop" with the visor icon and
   opens in its own window. Needs Quest software v74 or newer
   (Settings → System → Software update).
4. **Microphone.** The first "Start session" asks for microphone permission;
   allow it.
5. **Hand tracking.** Settings → Movement tracking → Hand tracking on. Put the
   controllers down before entering the scene: with a controller awake the
   browser starts the XR session without hands.
6. **First entry.** The first time this account connects, the browser asks
   whether the NVIDIA page may enter VR; allow it. It also asks you to accept
   the certificate for the proxy port once — the app page tells you when and
   walks you through it (section 3).

---

## 3. Daily use, from the headset

1. Open the **Teleop** tile. The top lines show whether teleop is running,
   what a session would launch (scene source, how many scenes are ticked,
   the robot), and whether a microphone is streaming.
2. To change what gets collected, open the **Scenes** tab: tick or untick
   scenes (tap a row; Select all / none / needed), switch between the local
   directory and the fleet server, tap **Save**. Robot, voice, randomization,
   gains and ports live on the **Parameters** tab. Start session saves any
   unsaved edits by itself. Changes apply to the *next* run — **Restart
   teleop** if one is already up.
3. Tap **Start session**. The microphone starts (green level bar), teleop
   launches if it is down, and the status line counts the boot (30–90 s).
   If nothing is ticked, a port is invalid or the fleet server is not
   connected, the status line says so instead and nothing starts.
   The certificate button reads **Preparing teleop...** meanwhile; it and
   **Open CloudXR** are inert until the proxy is actually up.
4. When teleop is up the button turns into one of:
   - **Teleop ready - certificate OK** (solid green): nothing to do.
   - **Teleop ready - accept certificate** (pulsing): tap it, accept the
     warning in the tab that opens ("Certificate Accepted"), come back. The
     page notices by itself and moves on.
5. **Open CloudXR** pulses: tap it. On the NVIDIA page the address and port are
   already filled in; tap **Connect**. That tap is the one WebXR requires and
   cannot be removed.
6. Hold your wrists at the robot's hands until teleop auto-starts (or say
   "play"). Voice: **success** / **failure** end an episode, **reset**
   discards it, **next** switches scene, **align** adjusts the workspace
   offset while stopped.
7. Keep the app window open the whole time: the microphone runs there.

When you are done: **Finish session** (the red button beside Start) stops the
microphone and the whole teleop process tree, CloudXR runtime included.
Whenever things look stuck: **Kill teleop** does the teleop half of that on
its own, and **Restart teleop** does the same and starts fresh. **Log** opens
the run's output in a separate page with large text that follows the tail.

**If you left the scene** (took the headset off, exited VR, lost the tab), tap
**Restart teleop** before connecting again. Rejoining a live run often hangs at
"Starting XR session"; a fresh run connects reliably.

---

## 4. Without the app

Everything the app does can be done from a terminal on the workstation. This
is also the path for headsets that are not a Quest.

```bash
# start teleop, serving the headset-microphone page yourself on port 8444
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/make_teleop_scene.py \
    --mic_device quest --headless [--embodiment yam_duo] [--scene_usda ...]
```

Then on the headset:

- microphone: open `https://<hostname>.local:8444/` and tap "Start microphone"
  (`sudo ufw allow 8444/tcp` once);
- video: open `https://nvidia.github.io/IsaacTeleop/client/release-1.3.x/?serverIP=<hostname>.local&port=48322&mic=0`,
  accept the certificate at `https://<hostname>.local:48322/` first if asked,
  tap Connect. `mic=0` matters: without it the CloudXR client grabs the
  headset microphone for its own audio passthrough and fights the mic page
  for it every ~15 s, chopping the audio the voice commands hear.

Other headsets: Apple Vision Pro uses the Isaac XR Teleop Sample Client with
`--cloudxr_env avp` (port 48010/tcp). A workstation microphone instead of the
headset's: `--mic_device default` (or an ALSA name from `arecord -L`).

Stopping by hand: `Ctrl-C` in the terminal, then `pkill -f "cloudxr[.]runtime"`
for the runtime that routinely outlives it. Never `Ctrl-Z`: a suspended teleop
keeps 48322 and 8444 bound and the next start fails with "port already in use".

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `https://<host>.local:8500/` does not load | Headset on a different network, or avahi down (`systemctl is-active avahi-daemon`), or 8500/5353 blocked. Try the IP. |
| App page says "app unreachable" | Service down: `systemctl --user status teleop-app.service`, `journalctl --user -u teleop-app.service -n 50`. |
| Start session → "teleop exited (code N)" | Tap **Log** on the page, or `tail -100 ~/.cache/teleop_app/last_run.log`. |
| `WSS proxy port 48322 is already in use` | A previous run is still alive (often a `Ctrl-Z`'d one): `ss -tlnp \| grep 48322`, then `kill -CONT <pid>; kill -TERM <pid>` (escalate to `-9`). Or **Kill teleop** in the app. |
| Certificate asked again after every session | Normal for the Quest browser after it was closed or the headset slept: its exceptions live only as long as its process. The page only asks when the check says it is missing. |
| Connect turns grey, "Starting XR session" forever | Rejoin of a live run after a disconnect. **Restart teleop**, then connect. |
| No hands in the scene | Controller awake, hand tracking off in headset settings, or VR permission denied for the NVIDIA page. Server log shows `XDev does not support hand tracking` when the headset never offered hands. |
| Mic shows "streaming from another tab" | Another app tab is streaming, or one you just closed has not timed out (≤10 s). Close extra tabs. |
| Voice never triggers | Stay quiet the first 1.5 s after the mic starts (ambient calibration). Check the level bar moves when you speak. |
| Robot is Franka but you expected YAM | Set **Robot embodiment** on the Parameters tab, Save, then Restart teleop. |
| Start session says "Select at least one scene" / "Connect to the fleet server" | Scenes tab: tick scenes, or Connect to the server first (check the URL/token; the status line shows the server's error). |
| Fleet table stays empty after Connect | Server unreachable from the workstation (`curl http://fleet-host:8080/api/status`), wrong token, or the source is still "Local directory". |
| Episodes not appearing on the fleet dashboard | They are queued in `<record_dir>/fleet_outbox.sqlite3`; the run's log (Log button) prints `[FLEET] Upload failed ...` with the reason and retries by itself. |

Ports in one place: 8500/tcp app, 49100/tcp signaling, 48322/tcp WSS proxy +
certificate page, 47998/udp media, 5353/udp mDNS, 8444/tcp mic page (only
without the app), 48010/tcp Apple Vision Pro.
