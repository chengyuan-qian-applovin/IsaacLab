# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop launcher UI: tune parameters, pick scenes and a dataset, start teleop.

A plain-tkinter launcher (no Isaac Sim involved until Start is pressed):

- **Page 1 — Parameters**: the teleop knobs, grouped by concern (operator &
  voice, session start, domain randomization, control gains, visuals, advanced).
- **Page 2 — Scenes & dataset**: pick a scene directory (scanned recursively
  for ``*.usda``) and a dataset HDF5 file; a table lists every scene with the
  number of success/failure trajectories already collected for it in that
  dataset, and checkboxes select the scenes for this session.
- **Start teleop** writes the selection to a scene-list JSON and launches
  ``make_teleop_scene.py`` with ``--dataset_file`` (demos from all scenes and
  sessions append to the chosen file, each tagged with its scene) — cycle the
  selected scenes with the "next" voice command. Console output stays in the
  terminal the launcher was started from.

Run:

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_launcher.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

_HERE = os.path.dirname(os.path.abspath(__file__))
_TELEOP_SCRIPT = os.path.join(_HERE, "make_teleop_scene.py")
_DEFAULT_SCENE_DIR = os.path.join(_HERE, "scenes")
_DEFAULT_DATASET = os.path.abspath(os.path.join(os.getcwd(), "datasets", "duo_teleop", "duo_teleop.hdf5"))


def scan_scene_dir(scene_dir: str) -> list[str]:
    """All ``*.usda`` scene files under ``scene_dir``, recursively, sorted."""
    hits = []
    for root, _dirs, files in os.walk(scene_dir):
        hits += [os.path.join(root, f) for f in files if f.endswith(".usda")]
    return sorted(hits)


def scan_dataset(dataset_path: str) -> dict[str, tuple[int, int]]:
    """Per-scene ``(success, failure)`` demo counts in an HDF5 dataset.

    Demos are grouped by their ``scene`` attribute (written by the teleop
    script); demos without one are grouped under ``"<untagged>"``. Returns an
    empty dict when the file does not exist or holds no demos.
    """
    import h5py

    counts: dict[str, list[int]] = {}
    if not os.path.exists(dataset_path):
        return {}
    try:
        with h5py.File(dataset_path, "r") as f:
            if "data" not in f:
                return {}
            for _name, group in f["data"].items():
                scene = group.attrs.get("scene", "<untagged>")
                if isinstance(scene, bytes):
                    scene = scene.decode()
                entry = counts.setdefault(str(scene), [0, 0])
                entry[0 if bool(group.attrs.get("success", False)) else 1] += 1
    except OSError as exc:
        print(f"[LAUNCHER] Could not read {dataset_path}: {exc}")
        return {}
    return {k: (v[0], v[1]) for k, v in counts.items()}


# The headset choice sets BOTH the microphone source and the CloudXR client in
# one go: label -> (--mic_device, --cloudxr_env). Quest streams its mic from
# the CloudXR.js browser page; the AVP's Isaac XR Teleop client streams the mic
# natively once its CloudXR session connects (see headset_mic.py).
_DEVICES = {
    "meta quest": ("quest", "cloudxrjs"),
    "avp": ("avp", "avp"),
}

# Parameter schema: (flag, label, default, kind, group). Kind: str|float|bool|choice:<a,b,c>.
# Defaults mirror make_teleop_scene.py's argparse defaults. The "device" entry
# is a pseudo-parameter: _collect_args expands it through _DEVICES instead of
# passing it through as a flag.
_PARAMS = [
    ("device", "Device", "meta quest", "choice:meta quest,avp", "Operator & voice"),
    ("--embodiment", "Robot embodiment", "franka_duo", "choice:franka_duo,yam_duo", "Operator & voice"),
    ("--user", "User name (hand calibration)", "", "str", "Operator & voice"),
    ("--whisper_model", "Whisper model", "base.en", "str", "Operator & voice"),
    ("--no_voice", "Disable voice commands", False, "bool", "Operator & voice"),
    ("--no_auto_start", "Disable auto-start", False, "bool", "Session start"),
    ("--auto_start_pos_tol", "Auto-start position tolerance [m]", 0.10, "float", "Session start"),
    ("--auto_start_rot_tol", "Auto-start rotation tolerance [deg]", 25.0, "float", "Session start"),
    ("--debug_auto_start", "Debug auto-start (frames + errors)", False, "bool", "Session start"),
    ("--no_dr", "Disable domain randomization", False, "bool", "Domain randomization"),
    ("--dr_arm_jitter", "Arm start-pose jitter [rad]", 0.08, "float", "Domain randomization"),
    ("--dr_object_xy", "Object position range [m]", 0.05, "float", "Domain randomization"),
    ("--dr_object_yaw", "Object yaw range [deg]", 180.0, "float", "Domain randomization"),
    ("--dr_object_bias", "Shift objects toward the robot [m]", 0.3, "float", "Domain randomization"),
    ("--settle_time", "Object settling time after reset [s]", 1.0, "float", "Domain randomization"),
    ("--arm_kp", "Arm kp (stiffness) [N·m/rad]", 400.0, "float", "Control gains"),
    ("--arm_kd", "Arm kd (damping) [N·m·s/rad]", 80.0, "float", "Control gains"),
    ("--hand_kp", "Hand kp (stiffness) [N·m/rad]", 400.0, "float", "Control gains"),
    ("--hand_kd", "Hand kd (damping) [N·m·s/rad]", 4.0, "float", "Control gains"),
    ("--arm_visual", "Arm rendering", "transparent", "choice:transparent,hidden,normal", "Visuals"),
    ("--visualize_hands", "Show tracked hand joints", False, "bool", "Visuals"),
    ("--no_task_display", "Hide the task-description panel", False, "bool", "Visuals"),
    ("--episode_length_s", "Episode timeout [s]", 300.0, "float", "Advanced"),
    ("--render_frequency", "Render frequency [Hz]", 30.0, "float", "Advanced"),
    ("--no_record", "Disable recording", False, "bool", "Advanced"),
]

# UI prefills that intentionally differ from the teleop script's argparse
# defaults. The schema default above stays the argparse default so
# ``_collect_args`` recognizes these as changed and passes them on the
# command line.
_INITIAL_OVERRIDES: dict[str, object] = {}

# Palette (light, one blue accent).
_BG = "#eef0f4"  # window background
_PANEL = "#ffffff"  # group boxes, table
_FG = "#1c2430"  # main text
_MUTED = "#5c6675"  # secondary text
_ACCENT = "#2563eb"  # primary buttons
_ACCENT_ACTIVE = "#1d4ed8"
_STRIPE = "#f5f7fa"  # odd table rows
_DANGER = "#dc2626"  # stop button


class TeleopLauncher(tk.Tk):
    """Two-page launcher; see the module docstring."""

    def __init__(self):
        super().__init__()
        self.title("Duo Teleop Launcher")
        self._apply_style()
        self.geometry(f"{self._px(980)}x{self._px(820)}")
        self.minsize(self._px(820), self._px(620))
        self._proc: subprocess.Popen | None = None
        self._param_vars: dict[str, tuple[tk.Variable, object, str]] = {}
        self._scene_rows: dict[str, dict] = {}  # basename -> {path, selected(BooleanVar)}

        self._pages = {}
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        for name in ("params", "scenes", "running"):
            frame = ttk.Frame(container, padding=self._px(16))
            frame.grid(row=0, column=0, sticky="nsew")
            self._pages[name] = frame
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self._build_params_page(self._pages["params"])
        self._build_scenes_page(self._pages["scenes"])
        self._build_running_page(self._pages["running"])
        self._show("params")

    def _px(self, logical: int) -> int:
        """Scale a logical pixel size to the screen (HiDPI-aware)."""
        return int(round(logical * self._scale))

    def _apply_style(self) -> None:
        """Readable fonts scaled to the screen, one clean theme for every widget.

        Tk's defaults are ~10 pt and ignore HiDPI screens entirely, which is why
        the stock launcher rendered tiny. Sizes here are in PIXELS (negative tk
        font sizes) scaled from the screen height, so the layout looks the same
        on a 1080p and a 4K/HiDPI panel.
        """
        # The 0.8 keeps the UI comfortably compact (full HiDPI scaling felt big).
        self._scale = 0.8 * min(2.0, max(1.0, self.winfo_screenheight() / 1080))
        base = self._px(16)  # body text height in pixels

        # Best available proportional family. Conda's Tk is often built without
        # Xft and sees only legacy X11 core fonts (no DejaVu, no unicode arrows
        # or check marks — which is also why the UI text sticks to ASCII), so
        # fall through to the scalable Type1 face those setups do have.
        available = {f.lower(): f for f in tkfont.families(self)}
        family = None
        for candidate in ("dejavu sans", "liberation sans", "noto sans", "ubuntu", "nimbus sans l", "helvetica"):
            if candidate in available:
                family = available[candidate]
                break
        for name in (
            "TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
            "TkIconFont", "TkTooltipFont", "TkCaptionFont", "TkFixedFont",
        ):  # fmt: skip
            try:
                font = tkfont.nametofont(name)
                font.configure(size=-base, **({"family": family} if family else {}))
            except tk.TclError:
                pass
        self._font_title = tkfont.Font(font=tkfont.nametofont("TkDefaultFont"))
        self._font_title.configure(size=-self._px(24), weight="bold")
        self._font_bold = tkfont.Font(font=tkfont.nametofont("TkDefaultFont"))
        self._font_bold.configure(weight="bold")

        style = ttk.Style(self)
        style.theme_use("clam")
        self.configure(background=_BG)
        style.configure(".", background=_BG, foreground=_FG, bordercolor="#d4d9e1", focuscolor=_ACCENT)
        style.configure("Title.TLabel", font=self._font_title)
        style.configure("Muted.TLabel", foreground=_MUTED)
        style.configure("TLabelframe", background=_PANEL, relief="solid", borderwidth=1, padding=self._px(12))
        style.configure("TLabelframe.Label", background=_PANEL, foreground=_MUTED, font=self._font_bold)
        style.configure("Panel.TFrame", background=_PANEL)
        style.configure("Panel.TLabel", background=_PANEL)
        style.configure("Panel.TCheckbutton", background=_PANEL)
        style.map("Panel.TCheckbutton", background=[("active", _PANEL)])
        style.configure("TCheckbutton", indicatorsize=self._px(18), indicatormargin=(0, 0, self._px(8), 0))
        style.configure("TCombobox", arrowsize=self._px(22))
        pad_x, pad_y = self._px(14), self._px(7)
        style.configure("TButton", padding=(pad_x, pad_y))
        style.configure("Accent.TButton", background=_ACCENT, foreground="#ffffff", padding=(pad_x, pad_y))
        style.map(
            "Accent.TButton",
            background=[("pressed", _ACCENT_ACTIVE), ("active", _ACCENT_ACTIVE), ("disabled", "#9db4e8")],
        )
        style.configure("Danger.TButton", background=_DANGER, foreground="#ffffff", padding=(pad_x, pad_y))
        style.map("Danger.TButton", background=[("pressed", "#b91c1c"), ("active", "#b91c1c")])
        style.configure("TEntry", fieldbackground="#ffffff", padding=self._px(4))
        style.configure("TCombobox", fieldbackground="#ffffff", padding=self._px(4))
        style.configure("Treeview", background=_PANEL, fieldbackground=_PANEL, rowheight=self._px(32), borderwidth=1)
        style.configure("Treeview.Heading", font=self._font_bold, padding=(self._px(8), self._px(8)))
        style.map("Treeview", background=[("selected", "#dbe6fb")], foreground=[("selected", _FG)])

    def _show(self, name: str) -> None:
        self._pages[name].tkraise()

    # -- page 1: parameters ---------------------------------------------------

    def _build_params_page(self, page: ttk.Frame) -> None:
        ttk.Label(page, text="Teleop parameters", style="Title.TLabel").pack(anchor="w")
        ttk.Label(page, text="Only settings you change are passed on the command line.", style="Muted.TLabel").pack(
            anchor="w", pady=(0, self._px(6))
        )
        # Packed before the (expanding) parameter grid so it can never be
        # clipped off the bottom on a small window.
        ttk.Button(
            page, text="Select scenes  >>", style="Accent.TButton",
            command=lambda: self._show("scenes"),
        ).pack(side="bottom", anchor="e", pady=self._px(8))  # fmt: skip
        columns = ttk.Frame(page)
        columns.pack(fill="both", expand=True, pady=self._px(8))
        groups: dict[str, ttk.LabelFrame] = {}
        order = []
        for _flag, _label, _default, _kind, group in _PARAMS:
            if group not in groups:
                order.append(group)
                groups[group] = ttk.LabelFrame(columns, text=f" {group} ")
        for i, group in enumerate(order):
            groups[group].grid(row=i // 2, column=i % 2, sticky="nsew", padx=self._px(8), pady=self._px(8))
        columns.columnconfigure((0, 1), weight=1)

        for flag, label, default, kind, group in _PARAMS:
            initial = _INITIAL_OVERRIDES.get(flag, default)
            row = ttk.Frame(groups[group], style="Panel.TFrame")
            row.pack(fill="x", pady=self._px(3))
            if kind == "bool":
                var = tk.BooleanVar(value=initial)
                ttk.Checkbutton(row, text=label, variable=var, style="Panel.TCheckbutton").pack(anchor="w")
            elif kind.startswith("choice:"):
                var = tk.StringVar(value=initial)
                ttk.Label(row, text=label, style="Panel.TLabel").pack(side="left")
                ttk.Combobox(
                    row, textvariable=var, values=kind.split(":", 1)[1].split(","), width=12, state="readonly"
                ).pack(side="right")
            else:
                var = tk.StringVar(value=str(initial))
                ttk.Label(row, text=label, style="Panel.TLabel").pack(side="left")
                ttk.Entry(row, textvariable=var, width=12).pack(side="right")
            self._param_vars[flag] = (var, default, kind)

    def _collect_args(self) -> list[str]:
        """CLI arguments for every parameter that differs from its default.

        The "device" pseudo-parameter always expands to explicit ``--mic_device``
        and ``--cloudxr_env`` flags (via ``_DEVICES``), so the launched teleop
        session unambiguously matches the selected headset.
        """
        args = []
        for flag, (var, default, kind) in self._param_vars.items():
            value = var.get()
            if flag == "device":
                mic_device, cloudxr_env = _DEVICES[str(value)]
                args += ["--mic_device", mic_device, "--cloudxr_env", cloudxr_env]
            elif kind == "bool":
                if value:
                    args.append(flag)
            elif str(value) != str(default) and str(value).strip() != "":
                args += [flag, str(value).strip()]
        return args

    # -- page 2: scenes & dataset ----------------------------------------------

    def _build_scenes_page(self, page: ttk.Frame) -> None:
        ttk.Label(page, text="Scenes & dataset", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            page,
            text=(
                "Click a row to toggle whether that scene is collected this session;\n"
                "drag to toggle a whole run of rows, or Shift+Click to extend the last toggle up to a row."
            ),
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, self._px(6)))

        picker = ttk.Frame(page)
        picker.pack(fill="x", pady=self._px(8))
        self._scene_dir = tk.StringVar(value=_DEFAULT_SCENE_DIR)
        self._dataset = tk.StringVar(value=_DEFAULT_DATASET)
        for row_i, (label, var, browse) in enumerate(
            (
                ("Scene directory", self._scene_dir, self._browse_scene_dir),
                ("Dataset file (created if missing)", self._dataset, self._browse_dataset),
            )
        ):
            ttk.Label(picker, text=label).grid(row=row_i, column=0, sticky="w", pady=self._px(3))
            ttk.Entry(picker, textvariable=var).grid(row=row_i, column=1, sticky="ew", padx=self._px(8))
            ttk.Button(picker, text="Browse", command=browse).grid(row=row_i, column=2, pady=self._px(3))
        picker.columnconfigure(1, weight=1)

        self._table = ttk.Treeview(page, columns=("sel", "success", "failure"), height=14)
        self._table.heading("#0", text="Scene")
        self._table.heading("sel", text="Collect?")
        self._table.heading("success", text="Success")
        self._table.heading("failure", text="Failure")
        self._table.column("#0", width=self._px(500))
        for col in ("sel", "success", "failure"):
            self._table.column(col, width=self._px(100), anchor="center", stretch=False)
        self._table.tag_configure("odd", background=_STRIPE)
        self._table.tag_configure("off", foreground=_MUTED)
        self._table.pack(fill="both", expand=True, pady=self._px(8))
        # Excel-style toggling: click one row, drag to paint a run, Shift+Click
        # to extend the last toggle over a range (see _on_table_press).
        self._table.configure(selectmode="none")
        self._table.bind("<Button-1>", self._on_table_press)
        self._table.bind("<B1-Motion>", self._on_table_drag)
        self._table.bind("<ButtonRelease-1>", lambda _event: self._end_table_drag())
        self._toggle_anchor: str | None = None  # last individually toggled row
        self._anchor_state = True  # the state that toggle applied
        self._drag_origin: str | None = None  # row where the current drag started
        self._pre_drag: dict[str, bool] = {}  # states when the drag started

        buttons = ttk.Frame(page)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="<<  Parameters", command=lambda: self._show("params")).pack(side="left")
        ttk.Button(buttons, text="Refresh", command=self.refresh_table).pack(side="left", padx=self._px(8))
        ttk.Button(buttons, text="Select all", command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(buttons, text="Select none", command=lambda: self._set_all(False)).pack(
            side="left", padx=self._px(8)
        )
        ttk.Button(buttons, text="Start teleop", style="Accent.TButton", command=self.start_teleop).pack(side="right")

        self.refresh_table()

    def _browse_scene_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self._scene_dir.get() or _HERE)
        if path:
            self._scene_dir.set(path)
            self.refresh_table()

    def _browse_dataset(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=os.path.dirname(self._dataset.get()) or os.getcwd(),
            defaultextension=".hdf5",
            filetypes=[("HDF5 datasets", "*.hdf5")],
            confirmoverwrite=False,
        )
        if path:
            self._dataset.set(path)
            self.refresh_table()

    def _row_tags(self, name: str) -> tuple[str, ...]:
        row = self._scene_rows[name]
        tags = ("odd",) if row["odd"] else ()
        return tags if row["selected"].get() else (*tags, "off")

    def refresh_table(self) -> None:
        """Re-scan the scene directory and the dataset counts."""
        previous = {name: row["selected"].get() for name, row in self._scene_rows.items()}
        self._table.delete(*self._table.get_children())
        self._scene_rows.clear()
        counts = scan_dataset(self._dataset.get())
        for i, path in enumerate(scan_scene_dir(self._scene_dir.get())):
            name = os.path.basename(path)
            success, failure = counts.get(name, (0, 0))
            selected = tk.BooleanVar(value=previous.get(name, True))
            self._scene_rows[name] = {"path": path, "selected": selected, "odd": i % 2 == 1}
            self._table.insert(
                "", "end", iid=name, text=name, tags=self._row_tags(name),
                values=("Yes" if selected.get() else "", success, failure),
            )  # fmt: skip
        untagged = counts.get("<untagged>")
        if untagged:
            self._table.insert(
                "", "end", iid="<untagged>", text="(demos without a scene tag)", tags=("off",), values=("", *untagged)
            )

    def _set_row(self, name: str, value: bool) -> None:
        self._scene_rows[name]["selected"].set(value)
        self._table.set(name, "sel", "Yes" if value else "")
        self._table.item(name, tags=self._row_tags(name))

    def _rows_between(self, a: str, b: str) -> list[str]:
        """The consecutive scene rows from ``a`` to ``b`` (inclusive, any order)."""
        rows = [iid for iid in self._table.get_children() if iid in self._scene_rows]
        i, j = rows.index(a), rows.index(b)
        return rows[min(i, j) : max(i, j) + 1]

    def _on_table_press(self, event) -> None:
        """Click toggles a row; Shift+Click extends the last toggle over the range."""
        row_id = self._table.identify_row(event.y)
        if row_id not in self._scene_rows:
            return
        if event.state & 0x0001 and self._toggle_anchor in self._scene_rows:  # Shift held
            touched = self._rows_between(self._toggle_anchor, row_id)
            for name in touched:
                self._set_row(name, self._anchor_state)
            self._table.selection_set(touched)  # highlight what the gesture touched
            return
        state = not self._scene_rows[row_id]["selected"].get()
        self._toggle_anchor, self._anchor_state = row_id, state
        self._drag_origin = row_id
        self._pre_drag = {name: r["selected"].get() for name, r in self._scene_rows.items()}
        self._set_row(row_id, state)
        self._table.selection_set(row_id)

    def _on_table_drag(self, event) -> None:
        """Dragging paints the clicked toggle over every row passed; backing up restores."""
        if self._drag_origin is None:
            return
        row_id = self._table.identify_row(event.y)
        if row_id not in self._scene_rows:
            return
        painted = self._rows_between(self._drag_origin, row_id)
        painted_set = set(painted)
        for name in self._scene_rows:
            target = self._anchor_state if name in painted_set else self._pre_drag[name]
            if self._scene_rows[name]["selected"].get() != target:
                self._set_row(name, target)
        self._table.selection_set(painted)

    def _end_table_drag(self) -> None:
        self._drag_origin = None
        self._pre_drag = {}

    def _set_all(self, value: bool) -> None:
        for name in self._scene_rows:
            self._set_row(name, value)

    # -- page 3: running --------------------------------------------------------

    def _build_running_page(self, page: ttk.Frame) -> None:
        self._running_label = ttk.Label(page, text="", style="Title.TLabel")
        self._running_label.pack(anchor="w", pady=self._px(8))
        ttk.Label(
            page,
            style="Muted.TLabel",
            text=(
                "Console output (voice transcripts, saved-demo messages) is in the terminal\n"
                "the launcher was started from. Say 'next' to advance through the selected scenes."
            ),
        ).pack(anchor="w")
        ttk.Button(page, text="Stop teleop", style="Danger.TButton", command=self.stop_teleop).pack(
            anchor="w", pady=self._px(16)
        )

    def start_teleop(self) -> None:
        selected = [row["path"] for row in self._scene_rows.values() if row["selected"].get()]
        if not selected:
            messagebox.showerror("No scenes", "Select at least one scene to collect.")
            return
        dataset = os.path.abspath(self._dataset.get())
        os.makedirs(os.path.dirname(dataset), exist_ok=True)
        scene_list_path = os.path.splitext(dataset)[0] + ".scene_list.json"
        with open(scene_list_path, "w") as f:
            json.dump({"scenes": selected}, f, indent=2)

        command = [
            sys.executable,
            _TELEOP_SCRIPT,
            "--scene_list", scene_list_path,
            "--dataset_file", dataset,
            "--headless",
            *self._collect_args(),
        ]  # fmt: skip
        print("[LAUNCHER] " + " ".join(command))
        self._proc = subprocess.Popen(command)
        self._running_label.config(text=f"Teleop running (pid {self._proc.pid}), {len(selected)} scene(s) selected.")
        self._show("running")
        self.after(1000, self._poll_process)

    def _poll_process(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self.after(1000, self._poll_process)
            return
        code = self._proc.returncode
        self._proc = None
        print(f"[LAUNCHER] Teleop exited with code {code}.")
        self.refresh_table()  # pull in the freshly collected counts
        self._show("scenes")

    def stop_teleop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


if __name__ == "__main__":
    TeleopLauncher().mainloop()
