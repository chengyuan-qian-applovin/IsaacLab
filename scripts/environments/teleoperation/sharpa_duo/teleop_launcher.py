# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop launcher UI: tune parameters, pick scenes and a dataset, start teleop.

A plain-tkinter launcher (no Isaac Sim involved until Start is pressed):

- **Page 1 — Parameters**: the teleop knobs, grouped by concern (operator &
  voice, session start, domain randomization, stop gesture, visuals, advanced).
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


# Parameter schema: (flag, label, default, kind, group). Kind: str|float|bool|choice:<a,b,c>.
# Defaults mirror make_teleop_scene.py's argparse defaults.
_PARAMS = [
    ("--user", "User name (hand calibration)", "", "str", "Operator & voice"),
    ("--mic_device", "Microphone (default / quest / ALSA name)", "default", "str", "Operator & voice"),
    ("--whisper_model", "Whisper model", "base.en", "str", "Operator & voice"),
    ("--no_voice", "Disable voice commands", False, "bool", "Operator & voice"),
    ("--cloudxr_env", "Headset client", "cloudxrjs", "choice:cloudxrjs,avp,none", "Operator & voice"),
    ("--no_auto_start", "Disable auto-start", False, "bool", "Session start"),
    ("--auto_start_pos_tol", "Auto-start position tolerance [m]", 0.10, "float", "Session start"),
    ("--auto_start_rot_tol", "Auto-start rotation tolerance [deg]", 25.0, "float", "Session start"),
    ("--debug_auto_start", "Debug auto-start (frames + errors)", False, "bool", "Session start"),
    ("--no_dr", "Disable domain randomization", False, "bool", "Domain randomization"),
    ("--dr_arm_jitter", "Arm start-pose jitter [rad]", 0.08, "float", "Domain randomization"),
    ("--dr_object_xy", "Object position range [m]", 0.05, "float", "Domain randomization"),
    ("--dr_object_yaw", "Object yaw range [deg]", 180.0, "float", "Domain randomization"),
    ("--gesture_touch_cm", "Stop gesture touch distance [cm]", 2.0, "float", "Stop gesture"),
    ("--gesture_hold_s", "Stop gesture hold time [s]", 0.5, "float", "Stop gesture"),
    ("--arm_visual", "Arm rendering", "transparent", "choice:transparent,hidden,normal", "Visuals"),
    ("--visualize_hands", "Show tracked hand joints", False, "bool", "Visuals"),
    ("--episode_length_s", "Episode timeout [s]", 300.0, "float", "Advanced"),
    ("--render_frequency", "Render frequency [Hz]", 30.0, "float", "Advanced"),
    ("--no_record", "Disable recording", False, "bool", "Advanced"),
]


class TeleopLauncher(tk.Tk):
    """Two-page launcher; see the module docstring."""

    def __init__(self):
        super().__init__()
        self.title("Duo Teleop Launcher")
        self.geometry("860x640")
        self._proc: subprocess.Popen | None = None
        self._param_vars: dict[str, tuple[tk.Variable, object, str]] = {}
        self._scene_rows: dict[str, dict] = {}  # basename -> {path, selected(BooleanVar)}

        self._pages = {}
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        for name in ("params", "scenes", "running"):
            frame = ttk.Frame(container, padding=12)
            frame.grid(row=0, column=0, sticky="nsew")
            self._pages[name] = frame
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self._build_params_page(self._pages["params"])
        self._build_scenes_page(self._pages["scenes"])
        self._build_running_page(self._pages["running"])
        self._show("params")

    def _show(self, name: str) -> None:
        self._pages[name].tkraise()

    # -- page 1: parameters ---------------------------------------------------

    def _build_params_page(self, page: ttk.Frame) -> None:
        ttk.Label(page, text="Teleop parameters", font=("", 14, "bold")).pack(anchor="w")
        columns = ttk.Frame(page)
        columns.pack(fill="both", expand=True, pady=8)
        groups: dict[str, ttk.LabelFrame] = {}
        order = []
        for _flag, _label, _default, _kind, group in _PARAMS:
            if group not in groups:
                order.append(group)
                groups[group] = ttk.LabelFrame(columns, text=group, padding=8)
        for i, group in enumerate(order):
            groups[group].grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
        columns.columnconfigure((0, 1), weight=1)

        for flag, label, default, kind, group in _PARAMS:
            row = ttk.Frame(groups[group])
            row.pack(fill="x", pady=1)
            if kind == "bool":
                var = tk.BooleanVar(value=default)
                ttk.Checkbutton(row, text=label, variable=var).pack(anchor="w")
            elif kind.startswith("choice:"):
                var = tk.StringVar(value=default)
                ttk.Label(row, text=label).pack(side="left")
                ttk.Combobox(
                    row, textvariable=var, values=kind.split(":", 1)[1].split(","), width=12, state="readonly"
                ).pack(side="right")
            else:
                var = tk.StringVar(value=str(default))
                ttk.Label(row, text=label).pack(side="left")
                ttk.Entry(row, textvariable=var, width=14).pack(side="right")
            self._param_vars[flag] = (var, default, kind)

        ttk.Button(page, text="Select scenes  \N{RIGHTWARDS ARROW}", command=lambda: self._show("scenes")).pack(
            anchor="e", pady=6
        )

    def _collect_args(self) -> list[str]:
        """CLI arguments for every parameter that differs from its default."""
        args = []
        for flag, (var, default, kind) in self._param_vars.items():
            value = var.get()
            if kind == "bool":
                if value:
                    args.append(flag)
            elif str(value) != str(default) and str(value).strip() != "":
                args += [flag, str(value).strip()]
        return args

    # -- page 2: scenes & dataset ----------------------------------------------

    def _build_scenes_page(self, page: ttk.Frame) -> None:
        ttk.Label(page, text="Scenes & dataset", font=("", 14, "bold")).pack(anchor="w")

        picker = ttk.Frame(page)
        picker.pack(fill="x", pady=6)
        self._scene_dir = tk.StringVar(value=_DEFAULT_SCENE_DIR)
        self._dataset = tk.StringVar(value=_DEFAULT_DATASET)
        for row_i, (label, var, browse) in enumerate(
            (
                ("Scene directory", self._scene_dir, self._browse_scene_dir),
                ("Dataset file (created if missing)", self._dataset, self._browse_dataset),
            )
        ):
            ttk.Label(picker, text=label).grid(row=row_i, column=0, sticky="w")
            ttk.Entry(picker, textvariable=var, width=72).grid(row=row_i, column=1, sticky="ew", padx=6)
            ttk.Button(picker, text="Browse", command=browse).grid(row=row_i, column=2)
        picker.columnconfigure(1, weight=1)

        self._table = ttk.Treeview(page, columns=("sel", "success", "failure"), height=14)
        self._table.heading("#0", text="Scene")
        self._table.heading("sel", text="Collect?")
        self._table.heading("success", text="Success")
        self._table.heading("failure", text="Failure")
        self._table.column("#0", width=460)
        for col in ("sel", "success", "failure"):
            self._table.column(col, width=80, anchor="center")
        self._table.pack(fill="both", expand=True, pady=6)
        self._table.bind("<Button-1>", self._on_table_click)

        buttons = ttk.Frame(page)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="\N{LEFTWARDS ARROW}  Parameters", command=lambda: self._show("params")).pack(
            side="left"
        )
        ttk.Button(buttons, text="Refresh", command=self.refresh_table).pack(side="left", padx=6)
        ttk.Button(buttons, text="Select all", command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(buttons, text="Select none", command=lambda: self._set_all(False)).pack(side="left", padx=6)
        ttk.Button(buttons, text="Start teleop", command=self.start_teleop).pack(side="right")

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

    def refresh_table(self) -> None:
        """Re-scan the scene directory and the dataset counts."""
        previous = {name: row["selected"].get() for name, row in self._scene_rows.items()}
        self._table.delete(*self._table.get_children())
        self._scene_rows.clear()
        counts = scan_dataset(self._dataset.get())
        for path in scan_scene_dir(self._scene_dir.get()):
            name = os.path.basename(path)
            success, failure = counts.get(name, (0, 0))
            selected = tk.BooleanVar(value=previous.get(name, True))
            self._scene_rows[name] = {"path": path, "selected": selected}
            self._table.insert(
                "", "end", iid=name, text=name, values=("\N{CHECK MARK}" if selected.get() else "", success, failure)
            )
        untagged = counts.get("<untagged>")
        if untagged:
            self._table.insert("", "end", iid="<untagged>", text="(demos without a scene tag)", values=("", *untagged))

    def _on_table_click(self, event) -> None:
        row_id = self._table.identify_row(event.y)
        if row_id in self._scene_rows:
            var = self._scene_rows[row_id]["selected"]
            var.set(not var.get())
            self._table.set(row_id, "sel", "\N{CHECK MARK}" if var.get() else "")

    def _set_all(self, value: bool) -> None:
        for name, row in self._scene_rows.items():
            row["selected"].set(value)
            self._table.set(name, "sel", "\N{CHECK MARK}" if value else "")

    # -- page 3: running --------------------------------------------------------

    def _build_running_page(self, page: ttk.Frame) -> None:
        self._running_label = ttk.Label(page, text="", font=("", 12))
        self._running_label.pack(anchor="w", pady=8)
        ttk.Label(
            page,
            text=(
                "Console output (voice transcripts, saved-demo messages) is in the terminal\n"
                "the launcher was started from. Say 'next' to advance through the selected scenes."
            ),
        ).pack(anchor="w")
        ttk.Button(page, text="Stop teleop", command=self.stop_teleop).pack(anchor="w", pady=12)

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
