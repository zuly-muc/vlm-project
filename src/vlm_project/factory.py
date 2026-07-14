"""Wire a :class:`~vlm_project.config.Config` into concrete objects.

This is the *only* module that maps config strings to classes, so the pipeline,
CLI, and tests stay free of ``if backend == "blip"`` branching. Import of heavy
backends/loaders is deferred to the branch that needs them.
"""

from __future__ import annotations

from vlm_project.config import Config
from vlm_project.dataset.base import DatasetLoader
from vlm_project.selectors.base import FrameSelector
from vlm_project.vlm.base import Embedder, VLMBackend


def build_backend(cfg: Config) -> VLMBackend:
    if cfg.backend == "blip":
        from vlm_project.vlm.blip import BlipBackend

        return BlipBackend(model_name=cfg.blip_model)
    if cfg.backend == "fake":
        from vlm_project.vlm.fake import FakeBackend

        return FakeBackend()
    if cfg.backend == "moondream":
        from vlm_project.vlm.moondream import MoondreamBackend

        return MoondreamBackend(model_name=cfg.moondream_model)
    if cfg.backend == "anthropic":
        from vlm_project.vlm.api import AnthropicBackend

        return AnthropicBackend()
    if cfg.backend == "openai":
        from vlm_project.vlm.api import OpenAIBackend

        return OpenAIBackend()
    raise ValueError(f"unknown backend: {cfg.backend!r}")


def build_selector(cfg: Config, backend: VLMBackend) -> FrameSelector:
    if cfg.selector == "single":
        from vlm_project.selectors.single import SingleKeyframeSelector

        return SingleKeyframeSelector(strategy=cfg.single_strategy)
    if cfg.selector == "uniform":
        from vlm_project.selectors.uniform import UniformSampleSelector

        return UniformSampleSelector(n=cfg.uniform_samples)
    if cfg.selector == "clusters":
        from vlm_project.selectors.clusters import ClusterSegmentSelector

        if not isinstance(backend, Embedder):
            raise ValueError(
                f"backend {cfg.backend!r} cannot embed images; "
                "use --selector single with it, or --backend blip/fake for clusters"
            )
        return ClusterSegmentSelector(embedder=backend, threshold=cfg.cluster_threshold)
    raise ValueError(f"unknown selector: {cfg.selector!r}")


def build_loader(cfg: Config) -> DatasetLoader:
    if cfg.loader == "nuscenes":
        from vlm_project.dataset.nuscenes_loader import NuScenesLoader

        return NuScenesLoader.from_dataroot(
            dataroot=cfg.dataroot, version=cfg.nuscenes_version, camera=cfg.camera
        )
    if cfg.loader == "imagefolder":
        from vlm_project.dataset.imagefolder_loader import ImageFolderLoader

        return ImageFolderLoader(root=cfg.dataroot)
    if cfg.loader == "video":
        from vlm_project.dataset.video_loader import VideoLoader

        return VideoLoader(path=cfg.dataroot, frame_dir="./out/_video_frames")
    raise ValueError(f"unknown loader: {cfg.loader!r}")
