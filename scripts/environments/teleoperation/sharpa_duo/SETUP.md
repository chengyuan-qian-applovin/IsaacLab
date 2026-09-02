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
commands; Pillow draws the app icon.

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

**Choosing what the app launches.** By default a session runs
`make_teleop_scene.py --mic_device hub --headless` with the script's default
scene and embodiment. Forward extra arguments once each with `--teleop-arg`
and reinstall:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_app.py --install-service \
    --teleop-arg --embodiment --teleop-arg yam_duo \
    --teleop-arg --scene_list --teleop-arg scripts/environments/teleoperation/sharpa_duo/scenes/scene_list.json
```

**Several operators on one workstation.** Each user installs their own service
under their own account. Give each a distinct app port (`--port 8501`, ...)
and distinct CloudXR ports via `NV_CXR_SERVER_PORT` in the environment the
service inherits, otherwise two sessions fight over 49100/48322.

### 1.5 Updating the app

After editing `teleop_app.py`:

```bash
systemctl --user restart teleop-app.service
```

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

1. Open the **Teleop** tile. The top two lines show whether teleop is running
   and whether a microphone is streaming.
2. Tap **Start session**. The microphone starts (green level bar), teleop
   launches if it is down, and the status line counts the boot (30–90 s).
   The certificate button reads **Preparing teleop...** meanwhile; it and
   **Open CloudXR** are inert until the proxy is actually up.
3. When teleop is up the button turns into one of:
   - **Teleop ready - certificate OK** (solid green): nothing to do.
   - **Teleop ready - accept certificate** (pulsing): tap it, accept the
     warning in the tab that opens ("Certificate Accepted"), come back. The
     page notices by itself and moves on.
4. **Open CloudXR** pulses: tap it. On the NVIDIA page the address and port are
   already filled in; tap **Connect**. That tap is the one WebXR requires and
   cannot be removed.
5. Hold your wrists at the robot's hands until teleop auto-starts (or say
   "play"). Voice: **success** / **failure** end an episode, **reset**
   discards it, **next** switches scene, **align** adjusts the workspace
   offset while stopped.
6. Keep the app window open the whole time: the microphone runs there.

When you are done, or whenever things look stuck: **Kill teleop** ends the
whole process tree and the CloudXR runtime. **Restart teleop** does the same
and starts fresh.

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
- video: open `https://nvidia.github.io/IsaacTeleop/client/release-1.3.x/?serverIP=<hostname>.local&port=48322`,
  accept the certificate at `https://<hostname>.local:48322/` first if asked,
  tap Connect.

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
| Robot is Franka but you expected YAM | The service launches the script's default; reinstall with `--teleop-arg --embodiment --teleop-arg yam_duo` (section 1.4). |

Ports in one place: 8500/tcp app, 49100/tcp signaling, 48322/tcp WSS proxy +
certificate page, 47998/udp media, 5353/udp mDNS, 8444/tcp mic page (only
without the app), 48010/tcp Apple Vision Pro.
