"""Shared fixtures. Everything here is tiny and synthetic — no dataset, no model.

* ``make_image`` writes solid-colour JPEGs so loaders/selectors have real files.
* ``fake_nusc`` is a stand-in for the devkit ``NuScenes`` object exposing only the
  three methods :class:`NuScenesLoader` uses, wired as a short ``next``-linked
  keyframe chain — so the loader is exercised end-to-end on kilobytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from vlm_project.models import ClipItem, Frame
from vlm_project.vlm.fake import FakeBackend


@pytest.fixture
def make_image(tmp_path: Path):
    def _make(name: str, color=(120, 120, 120), size=(64, 48)) -> str:
        path = tmp_path / name
        Image.new("RGB", size, color).save(path, format="JPEG")
        return str(path)

    return _make


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


class _FakeNuScenes:
    """Minimal NuScenes stand-in: two scenes, a few keyframes each."""

    def __init__(self, image_paths: list[str]):
        # Build two scenes sharing a pool of image paths.
        self._samples: dict[str, dict] = {}
        self.scene: list[dict] = []
        self._sample_data: dict[str, str] = {}

        cursor = 0
        for s in range(2):
            tokens = [f"s{s}_k{k}" for k in range(3)]
            for k, tok in enumerate(tokens):
                nxt = tokens[k + 1] if k + 1 < len(tokens) else ""
                cam_tok = f"{tok}_cam"
                self._sample_data[cam_tok] = image_paths[cursor % len(image_paths)]
                cursor += 1
                self._samples[tok] = {
                    "token": tok,
                    "next": nxt,
                    "data": {"CAM_FRONT": cam_tok},
                }
            self.scene.append(
                {
                    "name": f"scene-000{s}",
                    "token": f"scene_tok_{s}",
                    "first_sample_token": tokens[0],
                }
            )

    def get(self, table: str, token: str) -> dict:
        assert table == "sample"
        return self._samples[token]

    def get_sample_data(self, cam_token: str):
        return self._sample_data[cam_token], [], None


@pytest.fixture
def fake_nusc(make_image) -> _FakeNuScenes:
    paths = [make_image(f"kf_{i}.jpg", color=(i * 20 % 255, 40, 80)) for i in range(6)]
    return _FakeNuScenes(paths)


@pytest.fixture
def single_frame_clip(make_image) -> ClipItem:
    return ClipItem(
        clip_id="img-001",
        frames=[Frame(image_path=make_image("one.jpg"), index=0)],
        loader="imagefolder",
    )
