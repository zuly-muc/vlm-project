"""nuScenes loader: each scene is a clip of camera keyframes.

A nuScenes "scene" is a ~20 s clip whose keyframes form a ``next``-linked list of
samples. We walk that list, collecting one camera image per keyframe, and hand
the whole frame list to the selector, which then decides whether to describe a
single representative frame or several segments.

The devkit ``NuScenes`` object is injected (see :meth:`from_dataroot`) so tests
can supply a lightweight stand-in without the 4 GB dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vlm_project.models import ClipItem, Frame


class NuScenesLoader:
    name = "nuscenes"

    def __init__(self, nusc: Any, camera: str = "CAM_FRONT"):
        self.nusc = nusc
        self.camera = camera

    @classmethod
    def from_dataroot(
        cls, dataroot: str, version: str = "v1.0-mini", camera: str = "CAM_FRONT"
    ) -> NuScenesLoader:
        # Imported lazily so the rest of the package (and its tests) never need
        # the heavy nuscenes-devkit dependency installed.
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        return cls(nusc=nusc, camera=camera)

    def iter_clips(self) -> Iterable[ClipItem]:
        for scene in self.nusc.scene:
            frames: list[Frame] = []
            token = scene["first_sample_token"]
            idx = 0
            while token:
                sample = self.nusc.get("sample", token)
                cam_token = sample["data"][self.camera]
                image_path = self._image_path(cam_token)
                frames.append(
                    Frame(
                        image_path=image_path,
                        index=idx,
                        metadata={"sample_token": sample["token"]},
                    )
                )
                token = sample["next"]  # '' terminates the clip
                idx += 1

            yield ClipItem(
                clip_id=scene["name"],
                frames=frames,
                loader=self.name,
                camera=self.camera,
                metadata={"scene_token": scene["token"], "n_keyframes": len(frames)},
            )

    def _image_path(self, cam_token: str) -> str:
        """Absolute path to the keyframe image for a sample_data token."""
        # get_sample_data returns (abspath, boxes, camera_intrinsic).
        data_path, _boxes, _intrinsic = self.nusc.get_sample_data(cam_token)
        return data_path
