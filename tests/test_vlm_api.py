"""API backends with a mocked client — no network, no key required.

Also covers the factory guard that rejects clustering with a non-embedding
backend.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from vlm_project.config import Config
from vlm_project.factory import build_selector
from vlm_project.vlm.api import AnthropicBackend, OpenAIBackend


def _img() -> Image.Image:
    return Image.new("RGB", (16, 16), (0, 0, 0))


def test_anthropic_backend_parses_response():
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **_: SimpleNamespace(
                content=[SimpleNamespace(text=" a quiet street ")]
            )
        )
    )
    backend = AnthropicBackend(model="claude-haiku-4-5", client=fake_client)
    assert backend.describe(_img()) == "a quiet street"
    assert backend.name == "anthropic:claude-haiku-4-5"


def test_openai_backend_parses_response():
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content="a highway"))
                    ]
                )
            )
        )
    )
    backend = OpenAIBackend(model="gpt-4o-mini", client=fake_client)
    assert backend.describe(_img()) == "a highway"


def test_clusters_rejects_non_embedding_backend():
    backend = AnthropicBackend(client=SimpleNamespace())  # no embed()
    cfg = Config(selector="clusters", backend="anthropic")
    with pytest.raises(ValueError, match="cannot embed"):
        build_selector(cfg, backend)
