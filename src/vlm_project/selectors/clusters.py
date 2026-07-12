"""Cluster/segment selector: split a clip into a few temporally-distinct scenes.

Pipeline within the selector:

1. Embed every frame (cheap) using an injected ``Embedder`` -- in production this
   is BLIP's own vision encoder, so no extra model is loaded.
2. Walk the frames in order and start a new segment whenever the cosine distance
   to the previous frame exceeds ``threshold``. This keeps segments *contiguous
   in time* (a video property), unlike generic k-means.
3. Pick each segment's medoid (the frame closest to the others) as its
   representative -- only these get captioned by the VLM.

Embeddings are injected, so tests drive this with deterministic vectors and no
model.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vlm_project.models import ClipItem, SceneItem
from vlm_project.vlm.base import Embedder


class ClusterSegmentSelector:
    name = "clusters"

    def __init__(self, embedder: Embedder, threshold: float = 0.25):
        self.embedder = embedder
        self.threshold = threshold

    def select(self, clip: ClipItem) -> list[SceneItem]:
        n = len(clip.frames)
        if n == 0:
            return []
        if n == 1:
            return [self._scene_item(clip, 0, (0, 0), 0)]

        images = [Image.open(f.image_path).convert("RGB") for f in clip.frames]
        emb = _l2_normalize(self.embedder.embed(images))

        segments = _segment_by_distance(emb, self.threshold)
        items: list[SceneItem] = []
        for seg_idx, (start, end) in enumerate(segments):
            medoid = start + _medoid(emb[start : end + 1])
            items.append(self._scene_item(clip, seg_idx, (start, end), medoid))
        return items

    def _scene_item(
        self, clip: ClipItem, seg_idx: int, span: tuple[int, int], medoid: int
    ) -> SceneItem:
        frame = clip.frames[medoid]
        return SceneItem(
            scene_id=f"{clip.clip_id}#{seg_idx}",
            clip_id=clip.clip_id,
            segment_idx=seg_idx,
            image_path=frame.image_path,
            frame_range=(clip.frames[span[0]].index, clip.frames[span[1]].index),
            metadata={"medoid_index": frame.index, **frame.metadata},
        )


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def _segment_by_distance(emb: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Contiguous segments; cut where consecutive cosine distance > threshold."""
    segments: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(emb)):
        distance = 1.0 - float(emb[i] @ emb[i - 1])
        if distance > threshold:
            segments.append((start, i - 1))
            start = i
    segments.append((start, len(emb) - 1))
    return segments


def _medoid(seg_emb: np.ndarray) -> int:
    """Index (within the segment) of the frame most similar to the rest."""
    if len(seg_emb) == 1:
        return 0
    # Sum of cosine similarities to all others; normalized rows -> dot product.
    sims = seg_emb @ seg_emb.T
    scores = sims.sum(axis=1)  # includes self-similarity (constant), harmless
    return int(np.argmax(scores))
