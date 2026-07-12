"""Unit tests for the acceptance verifier — the checks that gate production.

Builds synthetic DBs and a fake scene.json so the verification logic itself is
proven correct (both the passing and failing paths) without any real dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from vlm_project.acceptance import scene_names_from_dataset, verify_db
from vlm_project.models import SceneDescription
from vlm_project.store import SceneStore


def _rec(scene_id: str, desc: str, path: str, secs: float = 1.0) -> SceneDescription:
    return SceneDescription(
        scene_id=scene_id, clip_id=scene_id.split("#")[0], segment_idx=0,
        loader="nuscenes", camera="CAM_FRONT", image_path=path,
        frame_range=(0, 0), description=desc, backend="blip-base",
        inference_seconds=secs, metadata={},
    )


def _img(tmp_path: Path, name: str) -> str:
    from PIL import Image

    p = tmp_path / name
    Image.new("RGB", (16, 16), (10, 10, 10)).save(p)
    return str(p)


def _good_db(tmp_path: Path) -> str:
    db = str(tmp_path / "scenes.db")
    with SceneStore(db) as store:
        store.upsert(_rec("scene-0001",
                          "a car driving down a city street with traffic",
                          _img(tmp_path, "a.jpg")))
        store.upsert(_rec("scene-0002",
                          "a busy road with pedestrians and buildings",
                          _img(tmp_path, "b.jpg")))
    return db


def test_passes_on_sound_db(tmp_path: Path):
    report = verify_db(_good_db(tmp_path), expected_scene_names=["scene-0001", "scene-0002"])
    assert report.ok, report.render()
    names = {c.name for c in report.checks}
    assert {"rows_present", "scene_coverage", "descriptions_wellformed",
            "fts_roundtrip", "driving_vocabulary"} <= names


def test_fails_on_missing_scene(tmp_path: Path):
    report = verify_db(_good_db(tmp_path),
                       expected_scene_names=["scene-0001", "scene-0002", "scene-0003"])
    assert not report.ok
    cov = next(c for c in report.checks if c.name == "scene_coverage")
    assert not cov.ok and "scene-0003" in cov.detail


def test_fails_on_empty_description(tmp_path: Path):
    db = str(tmp_path / "s.db")
    with SceneStore(db) as store:
        store.upsert(_rec("scene-0001", "the", _img(tmp_path, "a.jpg")))  # <2 words
    report = verify_db(db, check_files=True, vocab_fraction=0.0)
    assert not report.ok
    assert not next(c for c in report.checks if c.name == "descriptions_wellformed").ok


def test_fails_on_missing_image_file(tmp_path: Path):
    db = str(tmp_path / "s.db")
    with SceneStore(db) as store:
        store.upsert(_rec("scene-0001", "a car on a road",
                          str(tmp_path / "does_not_exist.jpg")))
    report = verify_db(db, vocab_fraction=0.0)
    assert not next(c for c in report.checks if c.name == "image_files_exist").ok


def test_fails_on_low_driving_vocabulary(tmp_path: Path):
    db = str(tmp_path / "s.db")
    with SceneStore(db) as store:
        store.upsert(_rec("scene-0001", "a fluffy orange cat sleeping",
                          _img(tmp_path, "a.jpg")))
    report = verify_db(db, vocab_fraction=0.5)
    assert not next(c for c in report.checks if c.name == "driving_vocabulary").ok


def test_fts_roundtrip_not_truncated_by_common_word(tmp_path: Path):
    # Regression: >20 rows all sharing a common token must still each round-trip,
    # i.e. the check must not be capped by the store's default search limit.
    db = str(tmp_path / "s.db")
    with SceneStore(db) as store:
        for i in range(25):
            store.upsert(_rec(f"scene-{i:04d}",
                              f"a car driving down a street number {i:04d}",
                              _img(tmp_path, f"{i}.jpg")))
    report = verify_db(db, vocab_fraction=0.0)
    assert next(c for c in report.checks if c.name == "fts_roundtrip").ok, report.render()


def test_scene_names_from_dataset(tmp_path: Path):
    version_dir = tmp_path / "v1.0-mini"
    version_dir.mkdir()
    (version_dir / "scene.json").write_text(
        json.dumps([{"name": "scene-0061"}, {"name": "scene-0103"}])
    )
    assert scene_names_from_dataset(str(tmp_path)) == ["scene-0061", "scene-0103"]
