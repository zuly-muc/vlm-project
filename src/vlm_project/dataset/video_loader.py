"""Optional video loader: sample frames from ``.mp4`` files into clips.

Each video becomes one clip whose frames are sampled at a fixed stride and
written to a scratch directory as JPEGs (so downstream code only ever deals with
image paths, exactly like the other loaders). The same selectors then apply, so
"video input" reuses the whole keyframe/clustering machinery.

Requires the ``video`` extra (opencv). Imported lazily so the core install stays
light.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from vlm_project.models import ClipItem, Frame

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


class VideoLoader:
    name = "video"

    def __init__(self, path: str, frame_dir: str, every_n: int = 15):
        """``path`` is a video file or a directory of videos.

        ``every_n`` samples one frame per N decoded frames (~2 Hz for 30 fps).
        """
        self.path = path
        self.frame_dir = frame_dir
        self.every_n = max(1, every_n)

    def iter_clips(self) -> Iterable[ClipItem]:
        for video_path in self._iter_video_paths():
            yield self._clip_for_video(video_path)

    def _clip_for_video(self, video_path: str) -> ClipItem:
        import cv2  # lazy: only needed on the video path

        clip_id = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = Path(self.frame_dir) / clip_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frames: list[Frame] = []
        raw_idx = 0
        kept = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if raw_idx % self.every_n == 0:
                    frame_path = str(out_dir / f"{kept:05d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frames.append(
                        Frame(image_path=frame_path, index=kept,
                              metadata={"source_frame": raw_idx})
                    )
                    kept += 1
                raw_idx += 1
        finally:
            cap.release()

        return ClipItem(
            clip_id=clip_id,
            frames=frames,
            loader=self.name,
            camera=None,
            metadata={"source_path": video_path, "sampled_every_n": self.every_n},
        )

    def _iter_video_paths(self) -> list[str]:
        p = Path(self.path)
        if p.is_file():
            return [str(p)]
        return sorted(
            str(f) for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in _VIDEO_EXTS
        )
