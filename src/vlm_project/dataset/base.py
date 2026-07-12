"""The dataset loader contract.

A loader is an iterable of :class:`~vlm_project.models.ClipItem`. It is the only
component that knows the on-disk layout of a particular dataset; everything
downstream works on ClipItems, which is what lets the same container ingest
nuScenes, a plain image folder, or a video with no pipeline changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from vlm_project.models import ClipItem


@runtime_checkable
class DatasetLoader(Protocol):
    #: Recorded on every row so results are traceable to their source.
    name: str

    def iter_clips(self) -> Iterable[ClipItem]:
        """Yield one ClipItem per clip in the dataset."""
        ...

    def __iter__(self) -> Iterator[ClipItem]:  # convenience
        return iter(self.iter_clips())
