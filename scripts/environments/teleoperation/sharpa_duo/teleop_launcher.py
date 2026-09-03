# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Desktop teleop launcher: tune parameters, pick scenes and a dataset, start teleop.

A plain-tkinter launcher (no Isaac Sim involved until Start is pressed). It is
the desktop counterpart of the headset app (:mod:`teleop_app`): both render the
same parameter schema, scene sources and persisted settings from
:mod:`session_config`, so a choice made here shows up in the headset and vice
versa.

- **Page 1 — Parameters**: the teleop knobs, grouped by concern (operator &
  voice, session start, domain randomization, control gains, visuals,
  advanced, network ports). The **Network ports** group covers every port a
  teleop session listens on — CloudXR signaling and media, the WSS proxy, the
  headset-microphone stream — so several operators can share one workstation
  (each with distinct ports) or dodge a port another program holds. Blank
  fields keep the runtime defaults.
- **Page 2 — Scenes & dataset**: pick a record directory (one HDF5 file per
  labeled episode) and a **scene source** — a radio choice between two
  mutually exclusive modes:

  - **Local directory**: scan a directory recursively for scene files
    (``*.usdz``, ``*.usda``, ``*.usd``) and tick the scenes to collect; the
    table shows the per-machine success/failure demo counts recorded under
    the record directory. The run is fully standalone — no fleet server is
    involved at all.
  - **Fleet server**: connect to the fleet coordination server (URL, optional
    collector id and token); the table then lists the SERVER's scenes with
    the server's numbers only — fleet-wide progress (``successes/target``)
    and who is collecting each scene right now, auto-refreshed every 15 s —
    and you tick which of those to collect. The run downloads the ticked
    scenes from the server and uploads every labeled episode as it happens.
- **Start teleop** launches ``make_teleop_scene.py`` with the active source's
  arguments — a local scene-list JSON, or ``--fleet_server`` +
  ``--fleet_scene_ids``, never both — and the scenes cycle with the "next"
  voice command. Console output stays in the terminal the launcher was
  started from.

Every setting (parameters, directories, scene source, fleet connection, scene
selection, window geometry) is remembered across runs in the shared settings
file (``~/.config/duo_teleop_launcher.json``; saved on close and on Start); a
remembered fleet-server source reconnects automatically. The window sizes
itself to fit all content on first launch.

Run:

    ./isaaclab.sh -p scripts/environments/teleoperation/sharpa_duo/teleop_launcher.py
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fleet_client import FleetMonitor  # noqa: E402
from session_config import (  # noqa: E402
    PARAMS,
    PORTS,
    PORTS_NOTE,
    SceneTable,
    build_launch,
    load_settings,
    local_scene_rows,
    normalize_server_url,
    param_groups,
    save_settings,
    scene_needed,
    server_scene_rows,
)

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
        self.minsize(self._px(700), self._px(520))
        self._proc: subprocess.Popen | None = None
        self._param_vars: dict[str, tuple[tk.Variable, str]] = {}  # flag -> (var, kind)
        self._scene_rows: dict[str, dict] = {}  # name -> {path, selected(BooleanVar), odd, row(dict)}
        self._rows_source: str | None = None  # which scene source the rows above belong to
        self._monitor: FleetMonitor | None = None
        self._monitor_seen = 0.0  # polled_at of the last snapshot applied to the UI

        self._pages = {}
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        for name in ("params", "scenes", "running"):
            frame = ttk.Frame(container, padding=self._px(16))
            frame.grid(row=0, column=0, sticky="nsew")
            self._pages[name] = frame
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self._settings = load_settings()
        self._build_params_page(self._pages["params"])
        self._build_scenes_page(self._pages["scenes"])
        self._build_running_page(self._pages["running"])
        self._apply_settings(self._settings)
        self._show("params")

        # Size the window to fit ALL content (the tallest page wins, clamped to
        # the screen) unless a remembered geometry exists; no manual resizing
        # needed to see everything.
        self.update_idletasks()
        geometry = self._settings.get("geometry")
        if not isinstance(geometry, str) or "x" not in geometry:
            width = min(self.winfo_reqwidth(), int(self.winfo_screenwidth() * 0.95))
            height = min(self.winfo_reqheight(), int(self.winfo_screenheight() * 0.92))
            geometry = f"{width}x{height}"
        self.geometry(geometry)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(500, self._watch_monitor)

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
        # 0.96 = the original compact 0.8 enlarged by 20% for readability.
        self._scale = 0.96 * min(2.0, max(1.0, self.winfo_screenheight() / 1080))
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
        style.configure("PanelMuted.TLabel", background=_PANEL, foreground=_MUTED)
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

    # -- persisted settings ------------------------------------------------------

    def _apply_settings(self, settings: dict) -> None:
        """Restore the remembered launcher state; missing or unknown keys are ignored."""
        for flag, value in settings.get("params", {}).items():
            if flag in self._param_vars:
                with contextlib.suppress(tk.TclError):  # e.g. a bool flag remembered as a bad type
                    self._param_vars[flag][0].set(value)
        for key, var in (
            ("record_dir", self._record_dir),
            ("scene_dir", self._scene_dir),
            ("fleet_server", self._fleet_server_var),
            ("collector_id", self._collector_id_var),
            ("fleet_token", self._fleet_token_var),
        ):
            if isinstance(settings.get(key), str):
                var.set(settings[key])
        if settings.get("scene_source") in ("local", "server"):
            self._scene_source.set(settings["scene_source"])
        self._on_source_change()  # shows the restored source's pane and re-renders the table
        # The remembered source is the fleet server: reconnect right away.
        if self._scene_source.get() == "server" and normalize_server_url(self._fleet_server_var.get()):
            self.connect_fleet()

    def _current_settings(self) -> dict:
        """The settings dict as the UI currently shows it (the remembered selection updated with the table)."""
        selection = {k: dict(v) for k, v in self._settings.get("selection", {}).items()}
        if self._rows_source is not None:  # the table's toggles belong to the source it was rendered for
            selection.setdefault(self._rows_source, {}).update(
                {name: row["selected"].get() for name, row in self._scene_rows.items()}
            )
        return {
            **self._settings,
            "params": {flag: var.get() for flag, (var, _kind) in self._param_vars.items()},
            "record_dir": self._record_dir.get(),
            "scene_dir": self._scene_dir.get(),
            "scene_source": self._scene_source.get(),
            "fleet_server": self._fleet_server_var.get(),
            "collector_id": self._collector_id_var.get(),
            "fleet_token": self._fleet_token_var.get(),
            "selection": selection,
        }

    def _save_settings(self) -> None:
        self._settings = {**self._current_settings(), "geometry": self.geometry()}
        try:
            save_settings(self._settings)
        except OSError as exc:
            print(f"[LAUNCHER] Could not save settings: {exc}")

    def _on_close(self) -> None:
        self._save_settings()
        if self._monitor is not None:
            self._monitor.stop()
        self.destroy()

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
        for i, group in enumerate(param_groups()):
            groups[group] = ttk.LabelFrame(columns, text=f" {group} ")
            groups[group].grid(row=i // 2, column=i % 2, sticky="nsew", padx=self._px(8), pady=self._px(8))
        columns.columnconfigure((0, 1), weight=1)

        for p in PARAMS:
            row = ttk.Frame(groups[p.group], style="Panel.TFrame")
            row.pack(fill="x", pady=self._px(3))
            if p.kind == "bool":
                var = tk.BooleanVar(value=bool(p.default))
                ttk.Checkbutton(row, text=p.label, variable=var, style="Panel.TCheckbutton").pack(anchor="w")
            elif p.choices:
                var = tk.StringVar(value=str(p.default))
                ttk.Label(row, text=p.label, style="Panel.TLabel").pack(side="left")
                ttk.Combobox(row, textvariable=var, values=p.choices, width=12, state="readonly").pack(side="right")
            elif p.kind == "port":
                var = tk.StringVar(value=str(p.default))
                ttk.Label(row, text=p.label, style="Panel.TLabel").pack(side="left")
                ttk.Entry(row, textvariable=var, width=7).pack(side="right")
                ttk.Label(row, text=f"default {PORTS[p.flag][2]}", style="PanelMuted.TLabel").pack(
                    side="right", padx=(0, self._px(8))
                )
            else:
                var = tk.StringVar(value=str(p.default))
                ttk.Label(row, text=p.label, style="Panel.TLabel").pack(side="left")
                ttk.Entry(row, textvariable=var, width=12).pack(side="right")
            self._param_vars[p.flag] = (var, p.kind)
        ttk.Label(groups["Network ports"], style="PanelMuted.TLabel", text=PORTS_NOTE.replace("; ", ";\n")).pack(
            anchor="w", pady=(self._px(4), 0)
        )

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
        self._record_dir = tk.StringVar(value=self._settings["record_dir"])
        ttk.Label(picker, text="Record directory (one HDF5 per episode)").grid(
            row=0, column=0, sticky="w", pady=self._px(3)
        )
        ttk.Entry(picker, textvariable=self._record_dir).grid(row=0, column=1, sticky="ew", padx=self._px(8))
        ttk.Button(picker, text="Browse", command=self._browse_record_dir).grid(row=0, column=2, pady=self._px(3))
        picker.columnconfigure(1, weight=1)

        # -- scene source: local directory XOR fleet server ---------------------
        # The two modes are exclusive by construction: the radio button decides
        # which pane is shown, which rows the table lists, and which arguments
        # Start passes (see session_config.build_launch).
        source_box = ttk.LabelFrame(page, text=" Scene source ")
        source_box.pack(fill="x", pady=self._px(4))
        self._scene_source = tk.StringVar(value="local")
        radios = ttk.Frame(source_box, style="Panel.TFrame")
        radios.pack(fill="x")
        for value, label in (("local", "Local directory"), ("server", "Fleet server")):
            ttk.Radiobutton(
                radios, text=label, value=value, variable=self._scene_source,
                style="Panel.TCheckbutton", command=self._on_source_change,
            ).pack(side="left", padx=(0, self._px(24)))  # fmt: skip

        # Both panes live in the same grid cell; _on_source_change shows one.
        panes = ttk.Frame(source_box, style="Panel.TFrame")
        panes.pack(fill="x", pady=(self._px(6), 0))
        panes.columnconfigure(0, weight=1)

        self._local_pane = ttk.Frame(panes, style="Panel.TFrame")
        self._local_pane.grid(row=0, column=0, sticky="ew")
        self._scene_dir = tk.StringVar(value=self._settings["scene_dir"])
        ttk.Label(self._local_pane, text="Scene directory", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", pady=self._px(3)
        )
        ttk.Entry(self._local_pane, textvariable=self._scene_dir).grid(row=0, column=1, sticky="ew", padx=self._px(8))
        ttk.Button(self._local_pane, text="Browse", command=self._browse_scene_dir).grid(row=0, column=2)
        self._local_pane.columnconfigure(1, weight=1)

        self._server_pane = ttk.Frame(panes, style="Panel.TFrame")
        self._server_pane.grid(row=0, column=0, sticky="ew")
        self._fleet_server_var = tk.StringVar(value="")
        self._collector_id_var = tk.StringVar(value="")
        self._fleet_token_var = tk.StringVar(value="")
        ttk.Label(self._server_pane, text="Server URL", style="Panel.TLabel").grid(
            row=0, column=0, sticky="w", pady=self._px(3)
        )
        ttk.Entry(self._server_pane, textvariable=self._fleet_server_var).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=self._px(8)
        )
        self._fleet_btn = ttk.Button(self._server_pane, text="Connect", command=self.connect_fleet)
        self._fleet_btn.grid(row=0, column=4)
        ttk.Label(self._server_pane, text="Collector ID (default: hostname)", style="Panel.TLabel").grid(
            row=1, column=0, sticky="w", pady=self._px(3)
        )
        ttk.Entry(self._server_pane, textvariable=self._collector_id_var, width=16).grid(
            row=1, column=1, sticky="w", padx=self._px(8)
        )
        ttk.Label(self._server_pane, text="Token (default: $FLEET_TOKEN)", style="Panel.TLabel").grid(
            row=1, column=2, sticky="e"
        )
        ttk.Entry(self._server_pane, textvariable=self._fleet_token_var, width=16, show="*").grid(
            row=1, column=3, sticky="w", padx=self._px(8)
        )
        self._fleet_status = ttk.Label(self._server_pane, text="Not connected.", style="Panel.TLabel")
        self._fleet_status.config(foreground=_MUTED)
        self._fleet_status.grid(row=2, column=0, columnspan=5, sticky="w", pady=(self._px(3), 0))
        self._server_pane.columnconfigure(1, weight=1)
        self._server_pane.grid_remove()  # local mode is the default

        self._table = ttk.Treeview(page, columns=("sel", "success", "failure", "fleet", "workers"), height=12)
        self._table.heading("#0", text="Scene")
        self._table.heading("sel", text="Collect?")
        self._table.heading("success", text="Success")
        self._table.heading("failure", text="Failure")
        self._table.heading("fleet", text="Fleet progress")
        self._table.heading("workers", text="Working now")
        self._table.column("#0", width=self._px(420))
        for col in ("sel", "success", "failure"):
            self._table.column(col, width=self._px(90), anchor="center", stretch=False)
        self._table.column("fleet", width=self._px(120), anchor="center", stretch=False)
        self._table.column("workers", width=self._px(150), anchor="w", stretch=False)
        self._table.tag_configure("odd", background=_STRIPE)
        self._table.tag_configure("off", foreground=_MUTED)
        self._table.tag_configure("fleet_done", foreground="#1a7f4e")
        self._table.configure(displaycolumns=("sel", "success", "failure"))  # fleet columns appear in server mode
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

        # The button row packs BEFORE the table (side bottom): when the window
        # is short, the table shrinks instead of the buttons getting clipped.
        buttons = ttk.Frame(page)
        buttons.pack(side="bottom", fill="x")
        ttk.Button(buttons, text="<<  Parameters", command=lambda: self._show("params")).pack(side="left")
        ttk.Button(buttons, text="Refresh", command=self.refresh_table).pack(side="left", padx=self._px(8))
        ttk.Button(buttons, text="Select all", command=lambda: self._set_all(True)).pack(side="left")
        ttk.Button(buttons, text="Select none", command=lambda: self._set_all(False)).pack(
            side="left", padx=self._px(8)
        )
        self._select_needed_btn = ttk.Button(
            buttons, text="Select needed", command=self._select_needed, state="disabled"
        )
        self._select_needed_btn.pack(side="left")
        ttk.Button(buttons, text="Start teleop", style="Accent.TButton", command=self.start_teleop).pack(side="right")
        self._table.pack(fill="both", expand=True, pady=self._px(8))

    def _browse_scene_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self._scene_dir.get() or _HERE)
        if path:
            self._scene_dir.set(path)
            self.refresh_table()

    def _browse_record_dir(self) -> None:
        path = filedialog.askdirectory(initialdir=self._record_dir.get() or os.getcwd())
        if path:
            self._record_dir.set(path)
            self.refresh_table()

    def _fleet_connected(self) -> bool:
        return self._monitor is not None and self._monitor.status()["connected"]

    def _on_source_change(self) -> None:
        """Swap the visible source pane, the table columns, and the helper buttons."""
        server_mode = self._scene_source.get() == "server"
        if server_mode:
            self._local_pane.grid_remove()
            self._server_pane.grid()
            # Server mode shows only the server's numbers (fleet-wide progress);
            # the local per-machine demo counts are a local-directory concern.
            self._table.configure(displaycolumns=("sel", "fleet", "workers"))
        else:
            self._server_pane.grid_remove()
            self._local_pane.grid()
            self._table.configure(displaycolumns=("sel", "success", "failure"))
        self._select_needed_btn.config(state="normal" if server_mode and self._fleet_connected() else "disabled")
        self.refresh_table()

    def _row_tags(self, name: str) -> tuple[str, ...]:
        row = self._scene_rows[name]
        tags = ("odd",) if row["odd"] else ()
        if not row["selected"].get():
            return (*tags, "off")
        if row["row"].get("done"):  # at target on the server
            tags = (*tags, "fleet_done")
        return tags

    def _current_table(self) -> SceneTable:
        """The scene table for the active source, computed from the UI's current settings."""
        settings = self._current_settings()
        if self._scene_source.get() == "server":
            return server_scene_rows(settings, self._monitor.scenes if self._monitor is not None else None)
        return local_scene_rows(settings)

    def refresh_table(self) -> None:
        """Re-render the table for the active scene source.

        Local mode lists the scene files under the scene directory with the
        per-machine success/failure demo counts recorded under the record
        directory. Server mode lists the fleet server's scenes (from the last
        status poll) with the SERVER's numbers only — fleet-wide progress and
        live workers.
        """
        table = self._current_table()  # reads the current toggles before they are cleared
        self._settings = self._current_settings()  # ...and keeps them for the other source
        self._table.delete(*self._table.get_children())
        self._scene_rows.clear()
        self._rows_source = table.source
        if table.source == "server" and not self._fleet_connected():
            self._table.insert(
                "", "end", iid="<hint>", text="(press Connect to list the fleet server's scenes)",
                tags=("off",), values=("", "", "", "", ""),
            )  # fmt: skip
            return
        for i, row in enumerate(table.rows):
            name = row["name"]
            selected = tk.BooleanVar(value=row["selected"])
            self._scene_rows[name] = {"path": row["path"], "selected": selected, "odd": i % 2 == 1, "row": row}
            if table.source == "server":
                values = ("Yes" if selected.get() else "", "", "", row["progress"], ", ".join(row["workers"]))
            else:
                values = ("Yes" if selected.get() else "", row["success"], row["failure"], "", "")
            self._table.insert("", "end", iid=name, text=name, tags=self._row_tags(name), values=values)
        if table.untagged:
            self._table.insert(
                "", "end", iid="<untagged>", text="(demos without a scene tag)", tags=("off",),
                values=("", *table.untagged),
            )  # fmt: skip

    # -- fleet connection -------------------------------------------------------

    def connect_fleet(self) -> None:
        """Connect to (or immediately re-poll) the fleet server named in the server pane."""
        server = normalize_server_url(self._fleet_server_var.get())
        if not server:
            messagebox.showerror("No fleet server", "Enter the fleet server URL first (e.g. http://fleet-host:8080).")
            return
        self._fleet_server_var.set(server)
        token = str(self._fleet_token_var.get()).strip() or None
        if self._monitor is None or not self._monitor.matches(server, token):
            if self._monitor is not None:
                self._monitor.stop()
            self._monitor = FleetMonitor(server, token)
            self._monitor_seen = 0.0
            self._monitor.start()
        self._fleet_btn.config(state="disabled")
        self._fleet_status.config(text=f"Connecting to {server} ...", foreground=_MUTED)
        self._monitor.poll_now()

    def _watch_monitor(self) -> None:
        """Apply new fleet snapshots to the UI (the monitor polls on its own thread; tkinter must never block)."""
        self.after(500, self._watch_monitor)
        if self._monitor is None:
            return
        status = self._monitor.status()
        if status["polled_at"] <= self._monitor_seen:
            return
        self._monitor_seen = status["polled_at"]
        self._fleet_btn.config(state="normal")
        if status["error"] is not None:
            self._fleet_status.config(text=f"Unreachable — {status['error']}", foreground=_DANGER)
            return
        self._fleet_btn.config(text="Refresh now")
        t = status["totals"]
        online = ", ".join(status["online"]) or "none"
        self._fleet_status.config(
            foreground=_MUTED,
            text=(
                f"Connected: {t['successes_toward_target']}/{t['target_successes']} successes across"
                f" {t['scenes']} scenes — online: {online} (auto-refreshes every {int(status['interval_s'])} s)"
            ),
        )
        if self._scene_source.get() == "server":
            self._select_needed_btn.config(state="normal")
            self.refresh_table()

    def _select_needed(self) -> None:
        """Tick exactly the server scenes the fleet still needs (under target, not retired)."""
        scenes = self._monitor.scenes if self._monitor is not None else None
        for name in self._scene_rows:
            if scenes and name in scenes:
                self._set_row(name, scene_needed(scenes[name]))

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
        """Launch the teleop with the active scene source's (exclusive) arguments (see :func:`build_launch`)."""
        if self._scene_source.get() == "server" and not self._fleet_connected():
            messagebox.showerror("Not connected", "Connect to the fleet server before starting.")
            return
        settings = self._current_settings()
        try:
            spec = build_launch(settings, self._current_table())
        except ValueError as exc:
            messagebox.showerror("Cannot start", str(exc))
            if "port" in str(exc).lower():
                self._show("params")
            return
        command = spec.command(sys.executable)
        self._save_settings()  # remember everything the run was started with
        shown_env = " ".join(f"{k}={'***' if k == 'FLEET_TOKEN' else v}" for k, v in spec.env.items())
        print("[LAUNCHER] " + " ".join(filter(None, [shown_env, *command])))
        self._proc = subprocess.Popen(command, env={**os.environ, **spec.env})
        self._running_label.config(text=f"Teleop running (pid {self._proc.pid}), {spec.summary} selected.")
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
