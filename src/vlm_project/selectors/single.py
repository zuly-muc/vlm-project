"""Single-keyframe selector: one representative frame per clip.

This is the default and the core case ("pick a representative keyframe").
Two strategies:

* ``middle``   -- the temporal midpoint. Zero image decoding; robust default.
* ``sharpest`` -- the least-blurry frame by variance-of-Laplacian. Decodes each
  frame once; avoids picking a motion-blurred frame.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from vlm_project.models import ClipItem, SceneItem


class SingleKeyframeSelector:
    name = "single"

    def __init__(self, strategy: str = "middle"):
        if strategy not in ("middle", "sharpest"):
            raise ValueError(f"unknown single strategy: {strategy!r}")
        self.strategy = strategy

    def select(self, clip: ClipItem) -> list[SceneItem]:
        if not clip.frames:
            return []
        idx = self._middle(clip) if self.strategy == "middle" else self._sharpest(clip)
        frame = clip.frames[idx]
        return [
            SceneItem(
                scene_id=clip.clip_id,  # no segment suffix in single mode
                clip_id=clip.clip_id,
                segment_idx=0,
                image_path=frame.image_path,
                frame_range=(frame.index, frame.index),
                metadata={"strategy": self.strategy, **frame.metadata},
            )
        ]

    @staticmethod
    def _middle(clip: ClipItem) -> int:
        return len(clip.frames) // 2

    @staticmethod
    def _sharpest(clip: ClipItem) -> int:
        best_idx, best_score = 0, -1.0
        for i, frame in enumerate(clip.frames):
            score = _laplacian_variance(frame.image_path)
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx


def _laplacian_variance(image_path: str) -> float:
    """Focus measure: variance of the Laplacian of the greyscale image.

    Implemented with numpy (no OpenCV dependency) so ``single`` works on the
    light core install.
    """
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float64)
    # 4-neighbour Laplacian kernel applied via array shifts.
    lap = (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )
    # Drop the wrapped border rows/cols so roll artefacts don't skew the variance.
    return float(lap[1:-1, 1:-1].var()) if arr.size > 4 else 0.0
