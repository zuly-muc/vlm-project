"""Model-sanity: does the VLM still caption general-domain images well?

A *different axis* from the nuScenes object-grounding gate, that asks "is the
caption right for this driving frame?"; this asks "has the model itself regressed
on ordinary photos?" (e.g. after a torch/transformers/weights bump). It scores the
backend with BLEU-4 against gold COCO reference captions.

The COCO micro-set is fetched from the official source
(``scripts/fetch_coco_sanity.py``), never vendored. The backend is injected, so
this module is unit-tested with a ``FakeBackend`` and a tiny synthetic data dir;
the real BLIP path runs only in the e2e / integration tiers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from vlm_project.bleu import corpus_bleu
from vlm_project.vlm.base import VLMBackend

# Conservative broken-model tripwire, NOT a quality target: a working BLIP-base
# scores well above this on COCO; a broken/mis-loaded model scores ~0.
DEFAULT_MIN_BLEU = 0.10


@dataclass
class SanityReport:
    n: int
    bleu4: float
    min_bleu: float

    @property
    def ok(self) -> bool:
        return self.bleu4 >= self.min_bleu

    def render(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return (f"  [{mark}] model-sanity: BLEU-4 {self.bleu4:.3f} over {self.n} image(s) "
                f"(floor {self.min_bleu:.2f})")


def references_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "references.json"


def run_bleu_sanity(
    data_dir: str | Path,
    backend: VLMBackend,
    min_bleu: float = DEFAULT_MIN_BLEU,
) -> SanityReport:
    """Caption each image in ``data_dir`` and score BLEU-4 vs its references.

    ``data_dir`` holds ``references.json`` ([{image, captions[]}]) and ``images/``.
    Raises ``FileNotFoundError`` if the set has not been fetched, callers decide
    whether that is a skip (offline) or a failure.
    """
    data = Path(data_dir)
    entries = json.loads(references_path(data).read_text(encoding="utf-8"))

    hyps: list[str] = []
    refs: list[list[str]] = []
    for e in entries:
        image = Image.open(data / "images" / e["image"]).convert("RGB")
        hyps.append(backend.describe(image))
        refs.append(e["captions"])

    score = corpus_bleu(hyps, refs) if hyps else 0.0
    return SanityReport(n=len(hyps), bleu4=score, min_bleu=min_bleu)
