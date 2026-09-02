# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Client for the duo fleet server: presence, scene download, episode upload.

Talks to the ``duo-fleet-server`` coordination service (see the fleet section
in README.md). Pure standard library, no simulator imports — usable before
AppLauncher and from helper scripts.

Design (mirrors the server's guarantees):

- **Check-in** at startup returns the fleet-wide status snapshot and wipes any
  presence rows a crashed previous run of this collector left behind.
- **Presence, not locks**: :meth:`FleetClient.declare_scene` tells the server
  "this collector is working on this scene"; any number of collectors may
  share a scene. A heartbeat thread keeps the presence fresh.
- **Outbox**: :meth:`FleetClient.report_episode` only appends to a local
  SQLite journal and returns immediately — the teleop loop never waits on the
  network. A background thread drains the journal: episode file first
  (idempotent PUT keyed by the episode UUID), then the metadata commit; an
  entry is marked sent only after the server acknowledges the metadata, so
  every failure mode ends in a retry of an idempotent operation. Entries that
  could not be sent (server down, network out) survive process restarts and
  are drained by the next session.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

_OUTBOX_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    episode_uuid TEXT PRIMARY KEY,
    scene_id TEXT NOT NULL,
    success INTEGER NOT NULL,
    hdf5_path TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    sent_at REAL
);
"""


class FleetError(RuntimeError):
    """A fleet server request failed (HTTP error status or unreachable)."""


def request(
    server_url: str,
    token: str | None,
    method: str,
    path: str,
    body: dict | None = None,
    raw: bytes | None = None,
    timeout: float = 30.0,
    out_path: str | None = None,
) -> dict:
    """One fleet-server HTTP request; JSON in/out unless ``raw``/``out_path`` is given.

    ``out_path`` streams the response body to that file (atomically, via a temp
    file) instead of parsing it. Raises :class:`FleetError` on any failure.
    """
    headers = {}
    if token:
        headers["X-Fleet-Token"] = token
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif raw is not None:
        data = raw
        headers["Content-Type"] = "application/octet-stream"
    req = urllib.request.Request(server_url.rstrip("/") + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if out_path is not None:
                tmp_path = f"{out_path}.part-{os.getpid()}"
                try:
                    with open(tmp_path, "wb") as f:
                        shutil.copyfileobj(resp, f)
                    os.replace(tmp_path, out_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                return {}
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            detail = exc.read().decode(errors="replace")[:500]
        raise FleetError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise FleetError(f"{method} {path} -> {exc}") from exc


def fetch_status(server_url: str, token: str | None = None, timeout: float = 10.0) -> dict:
    """The fleet status snapshot, read-only: no collector registration, no local state.

    For dashboards/GUIs that only observe the fleet (e.g. the teleop launcher);
    a collecting session uses :meth:`FleetClient.check_in` instead.
    """
    return request(server_url, token or os.environ.get("FLEET_TOKEN"), "GET", "/api/status", timeout=timeout)


class FleetClient:
    """See the module docstring. Construct, :meth:`start`, and :meth:`close` at exit."""

    def __init__(
        self,
        server_url: str,
        collector_id: str | None = None,
        token: str | None = None,
        state_dir: str = "./datasets/duo_teleop",
        heartbeat_s: float = 30.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.collector_id = collector_id or socket.gethostname()
        self.token = token if token is not None else os.environ.get("FLEET_TOKEN")
        self.state_dir = os.path.abspath(state_dir)
        # Local mirror of the server's data layout: scene files reference their
        # assets as ../assets/<path> (the server's convention), so scenes/ and
        # assets/ MUST sit side by side; the loose task-doc JSONs live in
        # scenes/ next to the scene files, where find_task_description looks.
        self.cache_dir = os.path.join(self.state_dir, "fleet_cache")
        self.scene_cache_dir = os.path.join(self.cache_dir, "scenes")
        self.asset_cache_dir = os.path.join(self.cache_dir, "assets")
        os.makedirs(self.scene_cache_dir, exist_ok=True)
        os.makedirs(self.asset_cache_dir, exist_ok=True)
        self._outbox_path = os.path.join(self.state_dir, "fleet_outbox.sqlite3")
        with self._outbox() as conn:
            conn.executescript(_OUTBOX_SCHEMA)
        self.current_scene_id: str | None = None
        self._heartbeat_s = heartbeat_s
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._threads: list[threading.Thread] = []
        self._online = True  # last-known reachability, to log transitions instead of spamming

    # -- http ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        raw: bytes | None = None,
        timeout: float = 30.0,
        out_path: str | None = None,
    ) -> dict:
        return request(
            self.server_url, self.token, method, path, body=body, raw=raw, timeout=timeout, out_path=out_path
        )

    # -- coordination -------------------------------------------------------------

    def check_in(self) -> dict:
        """Startup sync: register this collector and return the fleet status snapshot."""
        path = "/api/checkin"  # codespell:ignore checkin
        return self._request("POST", path, body={"collector_id": self.collector_id})

    def suggest(self, n: int) -> list[dict]:
        """Up to ``n`` under-target scenes the server recommends, best first."""
        path = f"/api/suggest?n={int(n)}&collector_id={urllib.parse.quote(self.collector_id)}"
        return self._request("GET", path)["scenes"]

    def list_scenes(self) -> list[dict]:
        """Every scene row on the server, with progress counts and live worker lists."""
        return self._request("GET", "/api/scenes")["scenes"]

    def declare_scene(self, scene_id: str) -> dict:
        """Declare presence on a scene (informational, never exclusive); returns the scene row."""
        self.current_scene_id = scene_id
        out = self._request("POST", "/api/workers", body={"collector_id": self.collector_id, "scene_id": scene_id})
        return out["scene"]

    def download_scene(self, scene_id: str, sha256: str | None = None) -> str:
        """Fetch a scene file into the local cache (skipped when the hash already matches).

        Returns the local path. The expected ``sha256`` (from the server's scene
        row) pins the scene version: a cached file that does not match is
        re-downloaded, and a downloaded file that does not match raises.
        """
        path = os.path.join(self.scene_cache_dir, os.path.basename(scene_id))
        if os.path.exists(path) and sha256 and self._sha256(path) == sha256:
            return path
        self._request("GET", f"/api/scenes/{urllib.parse.quote(scene_id)}/file", out_path=path, timeout=300.0)
        if sha256:
            got = self._sha256(path)
            if got != sha256:
                raise FleetError(f"scene {scene_id}: downloaded sha256 {got[:12]} != expected {sha256[:12]}")
        return path

    def sync_docs(self) -> list[str]:
        """Fetch every loose JSON document from the server's scenes/ into the scene cache.

        These docs (e.g. ``scene_instruct.json``) carry per-scene task
        descriptions; placed next to the cached scene files they are exactly
        where ``task_display.find_task_description`` looks. They are small and
        ``/api/docs`` reports no content hash, so they are always re-downloaded
        — the local copies are the latest by construction. Returns the names.
        """
        names = []
        for doc in self._request("GET", "/api/docs")["docs"]:
            name = doc["name"]
            out_path = os.path.join(self.scene_cache_dir, os.path.basename(name))
            self._request("GET", f"/api/docs/{urllib.parse.quote(name)}", out_path=out_path, timeout=120.0)
            names.append(name)
        return names

    def scene_assets(self, scene_id: str) -> list[dict]:
        """The asset files one scene references: ``{path, size_bytes, sha256, missing}`` each.

        ``path`` is relative to the server's ``assets/`` tree AND the location
        to store the file at under the local asset cache — the layout is the
        contract that makes the scene's ``../assets/...`` references resolve.
        """
        return self._request("GET", f"/api/scenes/{urllib.parse.quote(scene_id)}/assets")["assets"]

    def download_asset(self, path: str, sha256: str | None = None) -> str:
        """Fetch one asset into the local mirror (skipped when the hash already matches).

        Same freshness contract as :meth:`download_scene`: the server's sha256
        decides whether the cached copy is current, and a downloaded file that
        does not match it raises instead of being used.
        """
        local = os.path.normpath(os.path.join(self.asset_cache_dir, path))
        if not local.startswith(self.asset_cache_dir + os.sep):
            raise FleetError(f"asset path {path!r} escapes the asset cache")
        if os.path.exists(local) and sha256 and self._sha256(local) == sha256:
            return local
        os.makedirs(os.path.dirname(local), exist_ok=True)
        self._request("GET", f"/api/assets/{urllib.parse.quote(path)}", out_path=local, timeout=600.0)
        if sha256:
            got = self._sha256(local)
            if got != sha256:
                raise FleetError(f"asset {path}: downloaded sha256 {got[:12]} != expected {sha256[:12]}")
        return local

    def sync_scene_assets(self, scene_id: str) -> tuple[int, int, list[str]]:
        """Make every asset the scene references present and current in the local mirror.

        Returns ``(current, downloaded, missing)``: how many assets were already
        up to date, how many were (re-)downloaded, and the referenced paths the
        SERVER itself does not have (the scene cannot fully load anywhere until
        those are pushed to the server).
        """
        current, downloaded, missing = 0, 0, []
        for asset in self.scene_assets(scene_id):
            if asset.get("missing"):
                missing.append(asset["path"])
                continue
            local = os.path.normpath(os.path.join(self.asset_cache_dir, asset["path"]))
            if os.path.exists(local) and self._sha256(local) == asset["sha256"]:
                current += 1
            else:
                self.download_asset(asset["path"], asset["sha256"])
                downloaded += 1
        return current, downloaded, missing

    def push_scene(
        self,
        path: str,
        target_successes: int | None = None,
        priority: int | None = None,
        task_description: str | None = None,
    ) -> dict:
        """Upload one scene file (its basename becomes the scene id); returns the scene row.

        ``None`` fields keep the server-side value (or its default for a new scene).
        """
        params = {}
        if target_successes is not None:
            params["target_successes"] = target_successes
        if priority is not None:
            params["priority"] = priority
        if task_description:
            params["task_description"] = task_description
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        scene_id = urllib.parse.quote(os.path.basename(path))
        with open(path, "rb") as f:
            blob = f.read()
        return self._request("PUT", f"/api/scenes/{scene_id}/file{query}", raw=blob, timeout=300.0)["scene"]

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    # -- episode outbox --------------------------------------------------------------

    @contextlib.contextmanager
    def _outbox(self):
        """A fresh connection per operation: trivially safe across our threads."""
        conn = sqlite3.connect(self._outbox_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def report_episode(self, episode_uuid: str, scene_id: str, success: bool, hdf5_path: str, meta: dict | None = None):
        """Queue one labeled episode for upload; returns immediately (never blocks teleop)."""
        with self._outbox() as conn:
            conn.execute(
                "INSERT INTO outbox (episode_uuid, scene_id, success, hdf5_path, meta_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(episode_uuid) DO NOTHING",
                (episode_uuid, scene_id, int(success), os.path.abspath(hdf5_path), json.dumps(meta or {}), time.time()),
            )
        self._wake.set()

    def pending_count(self) -> int:
        with self._outbox() as conn:
            return conn.execute("SELECT COUNT(*) FROM outbox WHERE sent_at IS NULL").fetchone()[0]

    def _send_one(self, row: sqlite3.Row) -> None:
        """Upload one outbox entry: file first, then the metadata commit."""
        with open(row["hdf5_path"], "rb") as f:
            blob = f.read()
        self._request("PUT", f"/api/episodes/{row['episode_uuid']}/file", raw=blob, timeout=600.0)
        meta = json.loads(row["meta_json"])
        out = self._request(
            "POST",
            "/api/episodes",
            body={
                "episode_uuid": row["episode_uuid"],
                "scene_id": row["scene_id"],
                "collector_id": self.collector_id,
                "success": bool(row["success"]),
                "embodiment": meta.get("embodiment"),
                "num_steps": meta.get("num_steps"),
                "meta": meta,
            },
        )
        progress = out.get("progress", {})
        quota = " — AT QUOTA, move on when convenient" if progress.get("at_quota") else ""
        print(
            f"[FLEET] Synced episode {row['episode_uuid'][:8]} ({row['scene_id']}):"
            f" {progress.get('successes')}/{progress.get('target_successes')} successes{quota}"
        )

    def _uploader_loop(self) -> None:
        backoff = 5.0
        while not self._stop.is_set():
            with self._outbox() as conn:
                row = conn.execute("SELECT * FROM outbox WHERE sent_at IS NULL ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                self._wake.wait(timeout=5.0)
                self._wake.clear()
                continue
            try:
                self._send_one(row)
            except FileNotFoundError:
                # The episode file is gone (deleted/moved) — nothing to retry, ever.
                print(f"[FLEET] Episode {row['episode_uuid'][:8]}: file {row['hdf5_path']} is missing; dropped.")
                with self._outbox() as conn:
                    conn.execute(
                        "UPDATE outbox SET sent_at = ?, last_error = 'file missing' WHERE episode_uuid = ?",
                        (time.time(), row["episode_uuid"]),
                    )
                continue
            except FleetError as exc:
                if self._online:
                    self._online = False
                    print(f"[FLEET] Upload failed ({exc}); retrying every {backoff:.0f}s — episodes stay queued.")
                with self._outbox() as conn:
                    conn.execute(
                        "UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE episode_uuid = ?",
                        (str(exc)[:500], row["episode_uuid"]),
                    )
                self._stop.wait(timeout=backoff)
                backoff = min(60.0, backoff * 2)
                continue
            if not self._online:
                self._online = True
                print("[FLEET] Server reachable again; the queued episodes are syncing.")
            backoff = 5.0
            with self._outbox() as conn:
                conn.execute("UPDATE outbox SET sent_at = ? WHERE episode_uuid = ?", (time.time(), row["episode_uuid"]))

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(timeout=self._heartbeat_s):
            with contextlib.suppress(FleetError):
                self._request(
                    "POST",
                    "/api/heartbeat",
                    body={"collector_id": self.collector_id, "scene_id": self.current_scene_id},
                    timeout=10.0,
                )

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        """Start the uploader and heartbeat threads (also drains any previous session's backlog)."""
        pending = self.pending_count()
        if pending:
            print(f"[FLEET] {pending} episode(s) queued from a previous session will sync now.")
        for target in (self._uploader_loop, self._heartbeat_loop):
            thread = threading.Thread(target=target, daemon=True, name=f"fleet-{target.__name__}")
            thread.start()
            self._threads.append(thread)

    def close(self, drain_s: float = 30.0) -> None:
        """Give the outbox up to ``drain_s`` to finish syncing, then stop the threads.

        Unsent episodes are not lost: they stay in the on-disk outbox and the
        next session's :meth:`start` sends them.
        """
        deadline = time.time() + drain_s
        while time.time() < deadline:
            pending = self.pending_count()
            if pending == 0 or not self._online:
                break
            print(f"[FLEET] Waiting for {pending} episode upload(s) to finish ...")
            time.sleep(2.0)
        pending = self.pending_count()
        if pending:
            print(f"[FLEET] {pending} episode(s) still queued; they will sync on the next session.")
        self._stop.set()
        self._wake.set()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads.clear()
