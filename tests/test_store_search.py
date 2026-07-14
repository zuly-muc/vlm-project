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


def test_staged_publishes_only_on_finalize(tmp_path: Path):
    final = tmp_path / "out" / "scenes.db"
    stage = tmp_path / "stage"
    store = SceneStore(final, stage_dir=stage)
    store.upsert(_rec("scene-0001", "a staged row"))
    # Nothing at the final path yet, writes go to the local stage dir.
    assert not final.exists()

    store.finalize()

    # After finalize the DB is published to the final path and is queryable.
    assert final.exists()
    with SceneStore(final) as reopened:
        assert reopened.count() == 1
        assert reopened.search("staged")[0]["scene_id"] == "scene-0001"
    # Scratch copy is cleaned up.
    assert not (stage / "scenes.db").exists()


def test_staged_context_manager_finalizes_on_clean_exit(tmp_path: Path):
    final = tmp_path / "out" / "scenes.db"
    with SceneStore(final, stage_dir=tmp_path / "stage") as store:
        store.upsert(_rec("scene-0001", "hello"))
        assert not final.exists()  # not yet published inside the block
    assert final.exists()  # published on clean __exit__


def test_staged_failure_does_not_clobber_existing_db(tmp_path: Path):
    final = tmp_path / "out" / "scenes.db"
    # Seed a good existing DB at the final path.
    with SceneStore(final) as store:
        store.upsert(_rec("scene-0001", "original good row"))

    # A staged run that raises must leave the existing DB untouched.
    try:
        with SceneStore(final, stage_dir=tmp_path / "stage") as store:
            store.upsert(_rec("scene-0002", "partial row that should not land"))
            raise RuntimeError("boom mid-run")
    except RuntimeError:
        pass

    with SceneStore(final) as reopened:
        assert reopened.count() == 1
        assert reopened.search("original")[0]["scene_id"] == "scene-0001"
        assert reopened.search("partial") == []
