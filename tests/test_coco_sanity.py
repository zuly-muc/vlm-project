"""Model-sanity: does BLIP still caption general-domain COCO images well?

A *different axis* from the nuScenes object-grounding gate — it checks the model
itself hasn't regressed (e.g. after a torch/transformers/weights bump), against
gold COCO reference captions using BLEU-4 (shared with the ``bleu-sanity`` CLI and
the acceptance flow via :mod:`vlm_project.sanity`).

Opt-in and self-skipping, mirroring how nuScenes e2e is handled:
  * marked ``integration`` (needs real BLIP weights), and
  * skipped unless the COCO micro-set has been fetched:
        python scripts/fetch_coco_sanity.py

Run with:  pytest -m integration tests/test_coco_sanity.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from vlm_project.sanity import DEFAULT_MIN_BLEU, references_path, run_bleu_sanity  # noqa: E402

_DATA = Path(__file__).parent / "data" / "coco_sanity"


@pytest.mark.integration
def test_blip_bleu_on_coco_micro_set(capsys):
    if not references_path(_DATA).exists():
        pytest.skip(f"COCO micro-set not fetched — run scripts/fetch_coco_sanity.py ({_DATA})")

    from vlm_project.vlm.blip import BlipBackend

    report = run_bleu_sanity(_DATA, BlipBackend(), min_bleu=DEFAULT_MIN_BLEU)
    with capsys.disabled():
        print(f"\n[coco-sanity] {report.render().strip()}")
    assert report.ok, f"BLIP BLEU-4 {report.bleu4:.3f} below floor {report.min_bleu} — regression?"
