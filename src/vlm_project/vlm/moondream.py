"""Optional local Moondream2 backend: an instruction-following VLM for CPU.

Moondream2 (~1.5B params) is a small vision-language model tuned for edge/CPU
inference. Unlike BLIP -- a pure image *captioner* -- it follows a prompt, so its
descriptions tend to be more scene-aware. The trade is size and speed: a larger
download and slower CPU inference than BLIP-base (~0.5B). BLIP therefore stays
the default; this is an opt-in upgrade.

Opt in with ``VLM_BACKEND=moondream`` and the ``moondream`` extra. Like the API
backends it implements only ``describe`` (no ``embed``), so pair it with the
``single`` or ``uniform`` selector -- the factory rejects ``clusters`` for any
backend that cannot embed.

Weights load lazily on the first ``describe`` so importing this module (e.g. from
the CLI) stays cheap and unit tests never pull the model.
"""

from __future__ import annotations

from PIL import Image

# Pinned model + revision: Moondream2 ships its own remote code, so pinning the
# revision keeps behavior reproducible across image rebuilds.
_DEFAULT_MODEL = "vikhyatk/moondream2"
_DEFAULT_REVISION = "2024-08-26"
_PROMPT = "Describe this driving scene in one short sentence."


class MoondreamBackend:
    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        revision: str = _DEFAULT_REVISION,
        prompt: str = _PROMPT,
    ):
        self.model_name = model_name
        self.revision = revision
        self.prompt = prompt
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "moondream2"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # trust_remote_code: Moondream ships its own modeling code on the Hub.
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name, revision=self.revision, trust_remote_code=True
        )
        self._model.eval()  # CPU by default; no .to("cuda")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.revision
        )

    def describe(self, image: Image.Image) -> str:
        self._ensure_loaded()
        encoded = self._model.encode_image(image.convert("RGB"))
        answer = self._model.answer_question(encoded, self.prompt, self._tokenizer)
        return answer.strip()
