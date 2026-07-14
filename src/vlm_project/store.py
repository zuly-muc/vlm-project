"""SQLite store for scene descriptions, with FTS5 full-text search.

Two tables kept in lock-step by :meth:`SceneStore.upsert`:

* ``descriptions``     -- the structured rows (one per scene_id, the PK).
* ``descriptions_fts`` -- an FTS5 index over the description text.

A standalone (not external-content) FTS5 table is used and maintained manually
in ``upsert``; this keeps re-ingest idempotent (delete-then-insert by scene_id)
without the rowid-churn footguns of external-content triggers.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

from vlm_project.models import SceneDescription

_SCHEMA = """
CREATE TABLE IF NOT EXISTS descriptions (
    scene_id          TEXT PRIMARY KEY,
    clip_id           TEXT NOT NULL,
    segment_idx       INTEGER NOT NULL,
    loader            TEXT NOT NULL,
    camera            TEXT,
    image_path        TEXT NOT NULL,
    frame_start       INTEGER NOT NULL,
    frame_end         INTEGER NOT NULL,
    description       TEXT NOT NULL,
    backend           TEXT NOT NULL,
    inference_seconds REAL NOT NULL,
    metadata          TEXT NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS descriptions_fts
    USING fts5(scene_id UNINDEXED, description);
"""


class SceneStore:
    """A thin, well-typed wrapper around one SQLite database file.

    ``stage_dir`` opts into *staged* writes: the database is built on a
    container-local scratch path and only moved to ``db_path`` when the store is
    finalized (context-manager exit on a clean run). This keeps the many small
    ``ingest`` commits off slow/locking bind mounts (e.g. a Windows host volume
    under Docker Desktop / WSL2) and lands the finished file in a single move.
    Read paths (``query``/``export-json``) open ``db_path`` directly and never
    stage.
    """

    def __init__(self, db_path: str | Path, *, stage_dir: str | Path | None = None):
        self.final_path = str(db_path)
        self._staged = stage_dir is not None

        if self._staged:
            stage = Path(stage_dir)
            stage.mkdir(parents=True, exist_ok=True)
            self.db_path = str(stage / Path(self.final_path).name)
        else:
            self.db_path = self.final_path
            parent = Path(self.db_path).parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # -- lifecycle ------------------------------------------------------------

    def finalize(self) -> None:
        """Close the connection and, if staged, publish the DB to ``final_path``.

        The staging dir (local scratch) and the final path (often a bind mount)
        are usually on different filesystems, so ``os.replace`` across them would
        raise. We copy into the final *directory* first, then ``os.replace`` the
        temp file onto the destination — which is atomic within that directory's
        filesystem, so readers never observe a half-written DB.
        """
        self.conn.close()
        if not self._staged:
            return

        final = Path(self.final_path)
        if final.parent and not final.parent.exists():
            final.parent.mkdir(parents=True, exist_ok=True)

        tmp_final = final.with_name(final.name + ".tmp")
        shutil.copyfile(self.db_path, tmp_final)
        os.replace(tmp_final, final)
        try:
            os.remove(self.db_path)
        except OSError:
            pass  # scratch cleanup is best-effort; the published DB is what matters

    # -- writes ---------------------------------------------------------------

    def upsert(self, rec: SceneDescription) -> None:
        """Insert or replace one row, keeping the FTS index in sync."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM descriptions WHERE scene_id = ?", (rec.scene_id,))
        cur.execute("DELETE FROM descriptions_fts WHERE scene_id = ?", (rec.scene_id,))
        cur.execute(
            """
            INSERT INTO descriptions (
                scene_id, clip_id, segment_idx, loader, camera, image_path,
                frame_start, frame_end, description, backend, inference_seconds, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.scene_id,
                rec.clip_id,
                rec.segment_idx,
                rec.loader,
                rec.camera,
                rec.image_path,
                rec.frame_range[0],
                rec.frame_range[1],
                rec.description,
                rec.backend,
                rec.inference_seconds,
                json.dumps(rec.metadata),
            ),
        )
        cur.execute(
            "INSERT INTO descriptions_fts (scene_id, description) VALUES (?, ?)",
            (rec.scene_id, rec.description),
        )
        self.conn.commit()

    # -- reads ----------------------------------------------------------------

    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM descriptions").fetchone()
        return int(row["n"])

    def all(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM descriptions ORDER BY scene_id"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search over descriptions, best matches first (FTS5 rank)."""
        rows = self.conn.execute(
            """
            SELECT d.*
            FROM descriptions_fts
            JOIN descriptions d ON d.scene_id = descriptions_fts.scene_id
            WHERE descriptions_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def to_json(self) -> list[dict]:
        """All rows in the same shape as :meth:`SceneDescription.to_dict`."""
        return self.all()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["frame_range"] = [d.pop("frame_start"), d.pop("frame_end")]
        d["metadata"] = json.loads(d["metadata"])
        return d

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> SceneStore:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # Failed run: never publish a partial DB over an existing good one.
            self.close()
            return
        self.finalize()
