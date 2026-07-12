"""SQLite store + FTS5 round-trip and idempotent upsert."""

from __future__ import annotations

from pathlib import Path

from vlm_project.models import SceneDescription
from vlm_project.store import SceneStore


def _rec(scene_id: str, description: str) -> SceneDescription:
    return SceneDescription(
        scene_id=scene_id,
        clip_id=scene_id.split("#")[0],
        segment_idx=0,
        loader="test",
        camera="CAM_FRONT",
        image_path="/x.jpg",
        frame_range=(0, 0),
        description=description,
        backend="fake",
        inference_seconds=0.01,
        metadata={"k": "v"},
    )


def test_search_finds_the_right_row(tmp_path: Path):
    with SceneStore(tmp_path / "s.db") as store:
        store.upsert(_rec("scene-0001", "a busy street with pedestrians crossing"))
        store.upsert(_rec("scene-0002", "an empty highway at night"))

        hits = store.search("pedestrians")
        assert [h["scene_id"] for h in hits] == ["scene-0001"]

        assert store.search("highway")[0]["scene_id"] == "scene-0002"
        assert store.count() == 2


def test_upsert_is_idempotent(tmp_path: Path):
    with SceneStore(tmp_path / "s.db") as store:
        store.upsert(_rec("scene-0001", "old caption"))
        store.upsert(_rec("scene-0001", "new caption with trucks"))

        assert store.count() == 1  # replaced, not duplicated
        assert store.search("old") == []  # FTS updated too
        assert store.search("trucks")[0]["description"] == "new caption with trucks"


def test_row_shape_matches_scene_description(tmp_path: Path):
    with SceneStore(tmp_path / "s.db") as store:
        store.upsert(_rec("scene-0001", "hello"))
        row = store.all()[0]
        assert row["frame_range"] == [0, 0]
        assert row["metadata"] == {"k": "v"}
