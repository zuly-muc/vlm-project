"""The frame-selector contract.

Selecting *which* frame(s) of a clip to describe is a concern distinct from
loading the clip. A selector maps one ClipItem to one or more SceneItems:

* ``single``   -> exactly one SceneItem (one row per clip; the literal task).
* ``clusters`` -> a few SceneItems, one per temporally-distinct segment.

Because it operates on ClipItems, the same selector serves nuScenes clips and
raw video alike.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vlm_project.models import ClipItem, SceneItem


@runtime_checkable
class FrameSelector(Protocol):
    name: str

    def select(self, clip: ClipItem) -> list[SceneItem]:
        """Return the sub-scenes of ``clip`` to describe."""
        ...
