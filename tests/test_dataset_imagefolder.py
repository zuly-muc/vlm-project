"""ImageFolderLoader: one single-frame clip per image, deterministic order."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vlm_project.dataset.imagefolder_loader import ImageFolderLoader


def _write(dirpath: Path, name: str) -> None:
    Image.new("RGB", (32, 32), (200, 10, 10)).save(dirpath / name, format="JPEG")


def test_one_clip_per_image_sorted(tmp_path: Path):
    _write(tmp_path, "b.jpg")
    _write(tmp_path, "a.jpg")
    (tmp_path / "notes.txt").write_text("ignored")

    clips = list(ImageFolderLoader(str(tmp_path)).iter_clips())

    assert [c.clip_id for c in clips] == ["a", "b"]  # sorted, non-images skipped
    assert all(len(c.frames) == 1 for c in clips)
    assert all(c.loader == "imagefolder" for c in clips)


def test_recursive_discovery(tmp_path: Path):
    sub = tmp_path / "nested"
    sub.mkdir()
    _write(sub, "deep.png")

    clips = list(ImageFolderLoader(str(tmp_path), recursive=True).iter_clips())
    assert [c.clip_id for c in clips] == ["deep"]
