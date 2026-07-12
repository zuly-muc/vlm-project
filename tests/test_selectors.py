"""Selector logic with deterministic, injected embeddings — no BLIP needed."""

from __future__ import annotations

import numpy as np

from vlm_project.models import ClipItem, Frame
from vlm_project.selectors.clusters import ClusterSegmentSelector
from vlm_project.selectors.single import SingleKeyframeSelector


def _clip(make_image, n: int) -> ClipItem:
    frames = [Frame(image_path=make_image(f"f{i}.jpg"), index=i) for i in range(n)]
    return ClipItem(clip_id="clip-x", frames=frames, loader="test")


class _ScriptedEmbedder:
    """Returns pre-set vectors regardless of image content."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def embed(self, images):
        return self.vectors[: len(images)]


def test_single_middle_picks_midpoint(make_image):
    clip = _clip(make_image, 5)
    items = SingleKeyframeSelector("middle").select(clip)
    assert len(items) == 1
    assert items[0].scene_id == "clip-x"  # no segment suffix
    assert items[0].frame_range == (2, 2)


def test_single_empty_clip(make_image):
    empty = ClipItem(clip_id="c", frames=[], loader="test")
    assert SingleKeyframeSelector().select(empty) == []


def test_clusters_split_on_distance(make_image):
    # Two tight groups: frames 0-1 similar, then a jump, then 2-3 similar.
    vectors = np.array(
        [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]], dtype=np.float32
    )
    clip = _clip(make_image, 4)
    selector = ClusterSegmentSelector(_ScriptedEmbedder(vectors), threshold=0.25)

    items = selector.select(clip)

    assert [it.scene_id for it in items] == ["clip-x#0", "clip-x#1"]
    assert items[0].frame_range == (0, 1)
    assert items[1].frame_range == (2, 3)


def test_clusters_single_segment_when_uniform(make_image):
    vectors = np.array([[1.0, 0.0]] * 3, dtype=np.float32)
    clip = _clip(make_image, 3)
    items = ClusterSegmentSelector(_ScriptedEmbedder(vectors), threshold=0.25).select(clip)
    assert len(items) == 1
    assert items[0].frame_range == (0, 2)
