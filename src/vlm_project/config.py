"""Runtime configuration, resolved from environment variables with CLI overrides.

Precedence: CLI flag > environment variable > built-in default. The CLI builds a
``Config`` and hands it to :mod:`vlm_project.factory`, which is the only place
that turns these strings into concrete loader/selector/backend objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # Where to read scenes from and how to interpret them.
    loader: str = "nuscenes"          # nuscenes | imagefolder | video
    dataroot: str = "./data"          # dataset root / image dir / video file
    nuscenes_version: str = "v1.0-mini"
    camera: str = "CAM_FRONT"         # nuScenes sensor channel

    # How to pick the frame(s) that get described.
    selector: str = "single"          # single | uniform | clusters
    single_strategy: str = "middle"   # middle | sharpest  (single selector only)
    # Number of evenly-spaced frames the ``uniform`` selector emits per clip
    # (1 = middle only; 3 = start/middle/end). No embedding cost.
    uniform_samples: int = 3
    # Cosine-distance cut point for clustering. BLIP's pooled vision embeddings
    # are highly concentrated (a hard scene cut measures ~0.08, within-scene
    # frames ~0.00), so the default is deliberately small. Tune per dataset via
    # --cluster-threshold / CLUSTER_THRESHOLD.
    cluster_threshold: float = 0.05

    # Which model produces the description.
    backend: str = "blip"             # blip | moondream | fake
    blip_model: str = "Salesforce/blip-image-captioning-base"
    # Local instruction-following alternative to BLIP (opt-in; ``moondream`` extra).
    moondream_model: str = "vikhyatk/moondream2"

    # Where results land.
    db_path: str = "./out/scenes.db"
    # Optional container-local scratch dir for staged ingest writes. When set,
    # ingest builds the DB here and moves the finished file to ``db_path`` once,
    # keeping per-row commits off slow/locking bind mounts. Unset = write
    # directly to ``db_path`` (the plain default for local dev). The container
    # sets STAGE_DIR so Dockerized runs stage by default.
    stage_dir: str | None = None

    @classmethod
    def from_env(cls) -> Config:
        """Build a Config from environment variables (the container's contract)."""
        return cls(
            loader=_env("LOADER", cls.loader),
            dataroot=_env("DATAROOT", cls.dataroot),
            nuscenes_version=_env("NUSCENES_VERSION", cls.nuscenes_version),
            camera=_env("CAMERA", cls.camera),
            selector=_env("SELECTOR", cls.selector),
            single_strategy=_env("SINGLE_STRATEGY", cls.single_strategy),
            uniform_samples=int(_env("UNIFORM_SAMPLES", str(cls.uniform_samples))),
            cluster_threshold=float(_env("CLUSTER_THRESHOLD", str(cls.cluster_threshold))),
            backend=_env("VLM_BACKEND", cls.backend),
            blip_model=_env("BLIP_MODEL", cls.blip_model),
            moondream_model=_env("MOONDREAM_MODEL", cls.moondream_model),
            db_path=_env("DB", cls.db_path),
            stage_dir=os.environ.get("STAGE_DIR"),  # None when unset
        )
