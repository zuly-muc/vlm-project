"""The VLM backend contract.

A backend turns an image into a short natural-language description. Optionally it
can also embed images (the clustering selector reuses BLIP's vision encoder for
this), so embedding is a separate, optional capability checked via
``supports_embedding``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class VLMBackend(Protocol):
    #: Human-readable name recorded on every row (e.g. "blip-base", "fake").
    name: str

    def describe(self, image: Image.Image) -> str:
        """Return a short description of a single image."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Optional capability: turn images into vectors for clustering."""

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Return an ``(n, d)`` float array, one row per image."""
        ...
