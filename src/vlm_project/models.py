"""Core data contracts passed between the loader, selector, VLM backend, and store.

These dataclasses are the single serialization source: the pipeline builds
``SceneDescription`` objects and the store/JSON export read from them, so JSON,
the SQLite rows, and the tests all agree on one shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Frame:
    """A single image within a clip."""

    image_path: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClipItem:
    """A driving clip: an ordered list of frames plus provenance.

    A nuScenes "scene" maps to one ClipItem (its ~40 keyframes). A single image
    or a sampled video also become ClipItems (with one or more frames).
    """

    clip_id: str
    frames: list[Frame]
    loader: str
    camera: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneItem:
    """One selected sub-scene of a clip: the frame(s) that will be described.

    ``scene_id`` is the stable primary key. Single-frame selection uses the clip
    id directly (e.g. ``scene-0061``); clustering appends a segment index
    (e.g. ``scene-0061#0``).
    """

    scene_id: str
    clip_id: str
    segment_idx: int
    image_path: str
    frame_range: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneDescription:
    """The final record written to the store and exported to JSON."""

    scene_id: str
    clip_id: str
    segment_idx: int
    loader: str
    camera: str | None
    image_path: str
    frame_range: tuple[int, int]
    description: str
    backend: str
    inference_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # JSON has no tuple type; represent the inclusive range as a 2-list.
        d["frame_range"] = list(self.frame_range)
        return d
