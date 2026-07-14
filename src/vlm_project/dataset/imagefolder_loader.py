"""Generic image-folder loader: every image is a one-frame clip.

This is what makes the container dataset-agnostic: point it at any directory of
images and each becomes a clip with a single frame, so ``single`` selection
produces one description per image. (Clustering degenerates to one segment here,
which is the correct behaviour for standalone images.)
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from vlm_project.models import ClipItem, Frame

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageFolderLoader:
    name = "imagefolder"

    def __init__(self, root: str, recursive: bool = True):
        self.root = root
        self.recursive = recursive

    def iter_clips(self) -> Iterable[ClipItem]:
        for path in self._iter_image_paths():
            clip_id = os.path.splitext(os.path.basename(path))[0]
            yield ClipItem(
                clip_id=clip_id,
                frames=[Frame(image_path=path, index=0)],
                loader=self.name,
                camera=None,
                metadata={"source_path": path},
            )

    def _iter_image_paths(self) -> list[str]:
        root = Path(self.root)
        globber = root.rglob("*") if self.recursive else root.glob("*")
        paths = [
            str(p)
            for p in globber
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        ]
        # Stable, deterministic order (important for reproducible scene_ids).
        return sorted(paths)
