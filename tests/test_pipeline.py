"""End-to-end pipeline orchestration with fakes: loader + selector + fake VLM."""

from __future__ import annotations

from pathlib import Path

from vlm_project.dataset.nuscenes_loader import NuScenesLoader
from vlm_project.models import ClipItem, Frame
from vlm_project.pipeline import run
from vlm_project.selectors.single import SingleKeyframeSelector
from vlm_project.store import SceneStore


def test_single_mode_one_row_per_clip(tmp_path: Path, fake_nusc, fake_backend):
    loader = NuScenesLoader(nusc=fake_nusc)
    selector = SingleKeyframeSelector("middle")
    with SceneStore(tmp_path / "s.db") as store:
        stats = run(loader, selector, fake_backend, store)

        assert stats.clips == 2
        assert stats.scenes == 2
        rows = store.all()
        assert [r["scene_id"] for r in rows] == ["scene-0000", "scene-0001"]
        assert all(r["description"] for r in rows)
        assert all(r["backend"] == "fake" for r in rows)


def test_missing_image_is_skipped_not_fatal(tmp_path: Path, fake_backend):
    class _OneBadLoader:
        name = "test"

        def iter_clips(self):
            yield ClipItem(
                clip_id="good",
                frames=[Frame(image_path=_valid_image(tmp_path), index=0)],
                loader="test",
            )
            yield ClipItem(
                clip_id="bad",
                frames=[Frame(image_path=str(tmp_path / "nope.jpg"), index=0)],
                loader="test",
            )

    with SceneStore(tmp_path / "s.db") as store:
        stats = run(_OneBadLoader(), SingleKeyframeSelector(), fake_backend, store)

        assert stats.scenes == 1
        assert stats.failures == 1
        assert [r["scene_id"] for r in store.all()] == ["good"]


def _valid_image(tmp_path: Path) -> str:
    from PIL import Image

    p = tmp_path / "ok.jpg"
    Image.new("RGB", (16, 16), (10, 200, 10)).save(p, format="JPEG")
    return str(p)
