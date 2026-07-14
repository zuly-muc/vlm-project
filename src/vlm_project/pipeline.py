"""The pipeline: clips -> select frame(s) -> describe -> write rows.

``run`` takes its collaborators as arguments (dependency injection). Production
wires the real loader/selector/BLIP backend via :mod:`vlm_project.factory`; tests
wire fakes. This one signature is what makes the whole thing testable without a
dataset or model.

Per-scene failures (a missing image, a decode error) are logged and skipped so
one bad frame never aborts a long ingest.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from PIL import Image

from vlm_project.dataset.base import DatasetLoader
from vlm_project.models import SceneDescription
from vlm_project.selectors.base import FrameSelector
from vlm_project.store import SceneStore
from vlm_project.vlm.base import VLMBackend

logger = logging.getLogger("vlm_project.pipeline")


@dataclass
class RunStats:
    clips: int = 0
    scenes: int = 0
    failures: int = 0


def run(
    loader: DatasetLoader,
    selector: FrameSelector,
    backend: VLMBackend,
    store: SceneStore,
) -> RunStats:
    stats = RunStats()
    for clip in loader.iter_clips():
        stats.clips += 1
        if not clip.frames:
            # An empty clip (e.g. a nuScenes scene whose camera channel was
            # missing on every keyframe) produces no description. Count it as a
            # failure in the run stats rather than dropping it silently.
            logger.warning("clip %s has no usable frames; counting as failure", clip.clip_id)
            stats.failures += 1
            continue
        try:
            scene_items = selector.select(clip)
        except Exception:  # noqa: BLE001 - one clip must not kill the run
            logger.exception("selector failed for clip %s", clip.clip_id)
            stats.failures += 1
            continue

        for item in scene_items:
            try:
                rec = _describe_item(clip, item, backend)
            except Exception:  # noqa: BLE001
                logger.exception("failed to describe scene %s", item.scene_id)
                stats.failures += 1
                continue
            store.upsert(rec)
            stats.scenes += 1
            logger.info("%s -> %s", item.scene_id, rec.description)

    return stats


def _describe_item(clip, item, backend: VLMBackend) -> SceneDescription:
    image = Image.open(item.image_path).convert("RGB")
    started = time.perf_counter()
    description = backend.describe(image)
    elapsed = time.perf_counter() - started
    metadata = {**clip.metadata, **item.metadata}
    # Record model provenance (repo, revision, generation params) alongside the
    # row so a stored caption can be traced to the exact model that produced it.
    provenance = getattr(backend, "provenance", None)
    if provenance:
        metadata["model"] = provenance
    return SceneDescription(
        scene_id=item.scene_id,
        clip_id=item.clip_id,
        segment_idx=item.segment_idx,
        loader=clip.loader,
        camera=clip.camera,
        image_path=item.image_path,
        frame_range=item.frame_range,
        description=description,
        backend=backend.name,
        inference_seconds=round(elapsed, 4),
        metadata=metadata,
    )
