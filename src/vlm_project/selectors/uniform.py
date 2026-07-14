"""Uniform-sample selector: a few evenly-spaced frames, no embedding.

The cheap middle ground between the other two selectors:

* ``single``   -- one frame, zero embedding cost, but no temporal coverage.
* ``clusters`` -- temporal coverage, but embeds *every* frame through BLIP's
  vision encoder to find scene cuts (costly on CPU).
* ``uniform``  -- temporal coverage for free: pick N evenly-spaced frames by
  index alone. No model, no image decode, fully deterministic.

``n=1`` collapses to the middle frame (matching ``single``'s no-suffix rule);
``n=3`` gives start / middle / end. It is the right default when you want a
handful of representative frames per clip and the embedding-based scene-cut
detection of ``clusters`` is not worth its CPU cost.
"""

from __future__ import annotations

from vlm_project.models import ClipItem, SceneItem


class UniformSampleSelector:
    name = "uniform"

    def __init__(self, n: int = 3):
        if n < 1:
            raise ValueError(f"uniform n must be >= 1, got {n}")
        self.n = n

    def select(self, clip: ClipItem) -> list[SceneItem]:
        total = len(clip.frames)
        if total == 0:
            return []

        indices = self._even_indices(total, min(self.n, total))
        # A single frame is emitted like ``single``: plain clip_id, no #segment.
        segmented = len(indices) > 1

        items: list[SceneItem] = []
        for seg_idx, frame_pos in enumerate(indices):
            frame = clip.frames[frame_pos]
            scene_id = f"{clip.clip_id}#{seg_idx}" if segmented else clip.clip_id
            items.append(
                SceneItem(
                    scene_id=scene_id,
                    clip_id=clip.clip_id,
                    segment_idx=seg_idx,
                    image_path=frame.image_path,
                    frame_range=(frame.index, frame.index),
                    metadata={"strategy": "uniform", "n": len(indices), **frame.metadata},
                )
            )
        return items

    @staticmethod
    def _even_indices(total: int, k: int) -> list[int]:
        """``k`` frame positions spread across ``range(total)``, endpoints included.

        ``k == 1`` -> the midpoint; ``k >= 2`` -> ``0 .. total-1`` evenly spaced.
        Positions are deduped so short clips (where rounding collides) yield fewer
        but never duplicate frames.
        """
        if k <= 1:
            return [total // 2]
        step = (total - 1) / (k - 1)
        positions: list[int] = []
        for i in range(k):
            pos = round(i * step)
            if pos not in positions:
                positions.append(pos)
        return positions
