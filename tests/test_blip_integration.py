"""Opt-in integration test: load real BLIP weights and caption one image.

Skipped by default (``addopts = -m 'not integration'``). Run explicitly with:

    pytest -m integration

Downloads ~0.5 GB of weights on first run and needs torch/transformers.
"""

from __future__ import annotations

import pytest
from PIL import Image

pytest.importorskip("transformers")
pytest.importorskip("torch")


@pytest.mark.integration
def test_blip_captions_and_embeds():
    from vlm_project.vlm.blip import BlipBackend

    backend = BlipBackend()
    image = Image.new("RGB", (384, 384), (90, 120, 160))

    caption = backend.describe(image)
    assert isinstance(caption, str) and caption.strip()

    emb = backend.embed([image, image])
    assert emb.shape[0] == 2 and emb.shape[1] > 0
