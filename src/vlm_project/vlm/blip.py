"""Local BLIP backend (default): CPU image captioning via HuggingFace transformers.

One weight load serves both jobs:

* ``describe`` -- conditional/unconditional caption generation.
* ``embed``    -- pooled features from ``model.vision_model``, reused by the
  clustering selector so no second model is needed.

Weights are loaded lazily on first use, so importing this module (e.g. in the
CLI) is cheap and tests never touch it.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

_DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


class BlipBackend:
    def __init__(self, model_name: str = _DEFAULT_MODEL, max_new_tokens: int = 30,
                 num_beams: int = 3):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._processor = None
        self._model = None

    @property
    def name(self) -> str:
        return "blip-base"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self._model.eval()  # CPU by default; no .to("cuda")

    @torch.no_grad()
    def describe(self, image: Image.Image) -> str:
        self._ensure_loaded()
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        out = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, num_beams=self.num_beams
        )
        return self._processor.decode(out[0], skip_special_tokens=True).strip()

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Pooled vision-encoder features, ``(n, d)`` — used for clustering."""
        self._ensure_loaded()
        rgb = [img.convert("RGB") for img in images]
        inputs = self._processor(images=rgb, return_tensors="pt")
        vision = self._model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = vision.pooler_output  # (n, hidden)
        return pooled.cpu().numpy().astype(np.float32)
