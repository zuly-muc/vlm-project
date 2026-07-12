"""Optional cloud VLM backends (Anthropic Claude / OpenAI).

Opt-in via ``VLM_BACKEND=anthropic|openai`` and the ``api`` extra. They implement
the same ``describe`` contract as BLIP, so nothing else in the pipeline changes.
They do not implement ``embed``; pair them with the ``single`` selector (the
factory enforces this).

Networked and billed — never exercised in the default path or in unit tests
(tests mock the client).
"""

from __future__ import annotations

import base64
import io

from PIL import Image

_PROMPT = "Describe this driving scene in one short sentence."


def _encode_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class AnthropicBackend:
    def __init__(self, model: str = "claude-haiku-4-5", client=None):
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client = client

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    def describe(self, image: Image.Image) -> str:
        b64 = _encode_jpeg(image)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        )
        return msg.content[0].text.strip()


class OpenAIBackend:
    def __init__(self, model: str = "gpt-4o-mini", client=None):
        self.model = model
        if client is None:
            import openai

            client = openai.OpenAI()  # reads OPENAI_API_KEY
        self.client = client

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def describe(self, image: Image.Image) -> str:
        b64 = _encode_jpeg(image)
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        return resp.choices[0].message.content.strip()
