"""SQLite store for scene descriptions, with FTS5 full-text search.

Two tables kept in lock-step by :meth:`SceneStore.upsert`:

* ``descriptions``:     the structured rows (one per scene_id, the PK).
* ``descriptions_fts``: an FTS5 index over the description text.

A standalone (not external-content) FTS5 table is used and maintained manually
in ``upsert``; this keeps re-ingest idempotent (delete-then-insert by scene_id)
without the rowid-churn footguns of external-content triggers.
"""

from __future__ import annotations

import json
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

    ``ingest`` writes rows straight to ``db_path`` (creating parent directories
    as needed); ``query`` and ``export-json`` open the same path read-only.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        parent = Path(self.db_path).parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

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
        """All rows, each a dict of the ``SceneDescription`` fields."""
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
        self.close()
