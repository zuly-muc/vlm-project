"""A deterministic, dependency-free backend used throughout the test suite.

It never loads model weights, so the whole pipeline can be exercised in
milliseconds. ``describe`` is a pure function of the image size, and ``embed``
is a stable hash of pixel bytes, so clustering tests are reproducible.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


class FakeBackend:
    name = "fake"

    def describe(self, image: Image.Image) -> str:
        w, h = image.size
        return f"a fake description of a {w}x{h} image"

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        vectors = []
        for img in images:
            digest = hashlib.sha256(img.tobytes()).digest()[:16]
            vectors.append(np.frombuffer(digest, dtype=np.uint8).astype(np.float32))
        return np.vstack(vectors) if vectors else np.zeros((0, 16), dtype=np.float32)
