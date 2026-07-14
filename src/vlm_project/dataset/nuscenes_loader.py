"""nuScenes loader: each scene is a clip of camera keyframes.

A nuScenes "scene" is a ~20 s clip whose keyframes form a ``next``-linked list of
samples. We walk that list, collecting one camera image per keyframe, and hand
the whole frame list to the selector, which then decides whether to describe a
single representative frame or several segments.

The devkit ``NuScenes`` object is injected (see :meth:`from_dataroot`) so tests
can supply a lightweight stand-in without the 4 GB dataset.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from vlm_project.models import ClipItem, Frame

logger = logging.getLogger("vlm_project.nuscenes")


class NuScenesLoader:
    name = "nuscenes"

    def __init__(self, nusc: Any, camera: str = "CAM_FRONT", version: str = "v1.0-mini"):
        self.nusc = nusc
        self.camera = camera
        self.version = version

    @classmethod
    def from_dataroot(
        cls, dataroot: str, version: str = "v1.0-mini", camera: str = "CAM_FRONT"
    ) -> NuScenesLoader:
        # Imported lazily so the rest of the package (and its tests) never need
        # the heavy nuscenes-devkit dependency installed.
        from nuscenes.nuscenes import NuScenes

        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        return cls(nusc=nusc, camera=camera, version=version)

    def iter_clips(self) -> Iterable[ClipItem]:
        for scene in self.nusc.scene:
            name = scene.get("name", "?")
            try:
                frames = self._frames_for_scene(scene)
            except Exception:  # noqa: BLE001 - one bad scene must not kill the run
                logger.exception("skipping unreadable scene %s", name)
                frames = []
            yield ClipItem(
                clip_id=name,
                frames=frames,
                loader=self.name,
                camera=self.camera,
                metadata={
                    "scene_token": scene.get("token"),
                    "n_keyframes": len(frames),
                    "nuscenes_version": self.version,
                },
            )

    def _frames_for_scene(self, scene: dict) -> list[Frame]:
        """Walk a scene's ``next``-linked keyframe chain into a list of frames.

        Resilient to real-world dataset defects: a keyframe missing the requested
        camera channel is skipped (not fatal), and a self-referential/looping
        chain is broken rather than iterated forever. An empty result (no usable
        keyframes) yields a frameless clip, which the pipeline records as a
        failure instead of a silent zero-row scene.
        """
        frames: list[Frame] = []
        token = scene["first_sample_token"]
        idx = 0
        seen: set[str] = set()
        while token:
            if token in seen:
                logger.warning("cycle in sample chain of scene %s at %s; stopping",
                               scene.get("name", "?"), token)
                break
            seen.add(token)
            sample = self.nusc.get("sample", token)
            cam_token = sample.get("data", {}).get(self.camera)
            if cam_token is None:
                logger.warning("scene %s sample %s has no %s channel; skipping keyframe",
                               scene.get("name", "?"), token, self.camera)
            else:
                frames.append(
                    Frame(
                        image_path=self._image_path(cam_token),
                        index=idx,
                        metadata={"sample_token": sample["token"]},
                    )
                )
            token = sample.get("next", "")  # '' terminates the clip
            idx += 1
        return frames

    def _image_path(self, cam_token: str) -> str:
        """Absolute path to the keyframe image for a sample_data token."""
        # get_sample_data returns (abspath, boxes, camera_intrinsic).
        data_path, _boxes, _intrinsic = self.nusc.get_sample_data(cam_token)
        return data_path
