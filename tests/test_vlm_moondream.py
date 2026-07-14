"""Moondream backend wiring — no weights downloaded.

Verifies lazy loading, the ``describe`` contract with a stubbed model, and that
the factory rejects clustering with Moondream (it has no ``embed``), exactly like
the API backends.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from vlm_project.config import Config
from vlm_project.factory import build_selector
from vlm_project.vlm.moondream import MoondreamBackend


def _img() -> Image.Image:
    return Image.new("RGB", (16, 16), (0, 0, 0))


def test_construction_is_lazy_no_weights_loaded():
    backend = MoondreamBackend()
    assert backend.name == "moondream2"
    assert backend._model is None  # nothing loaded until describe()


def test_describe_uses_encode_then_answer(monkeypatch):
    backend = MoondreamBackend()
    # Stub the model/tokenizer so no weights are pulled.
    backend._model = SimpleNamespace(
        encode_image=lambda img: ("ENC", img.size),
        answer_question=lambda enc, prompt, tok: "  cars queued at a red light  ",
    )
    backend._tokenizer = object()

    assert backend.describe(_img()) == "cars queued at a red light"


def test_clusters_rejects_moondream_backend():
    backend = MoondreamBackend()  # no embed()
    cfg = Config(selector="clusters", backend="moondream")
    with pytest.raises(ValueError, match="cannot embed"):
        build_selector(cfg, backend)
