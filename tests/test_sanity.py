"""Model-sanity BLEU gate — FakeBackend + synthetic COCO-shaped dir, no network."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from vlm_project.sanity import run_bleu_sanity
from vlm_project.vlm.fake import FakeBackend


def _make_dataset(tmp_path: Path, references: list[list[str]]) -> Path:
    """A coco_sanity-shaped dir; FakeBackend captions each image by its size."""
    (tmp_path / "images").mkdir()
    entries = []
    for i, caps in enumerate(references):
        w, h = 40 + i, 30 + i
        Image.new("RGB", (w, h), (i * 10, 20, 30)).save(tmp_path / "images" / f"{i}.jpg")
        entries.append({"image": f"{i}.jpg", "captions": caps})
    (tmp_path / "references.json").write_text(json.dumps(entries))
    return tmp_path


def test_passes_when_captions_match_references(tmp_path: Path):
    # FakeBackend emits "a fake description of a WxH image"; make that the reference.
    data = _make_dataset(tmp_path, [
        ["a fake description of a 40x30 image"],
        ["a fake description of a 41x31 image"],
    ])
    report = run_bleu_sanity(data, FakeBackend(), min_bleu=0.10)
    assert report.n == 2
    assert report.bleu4 > 0.9 and report.ok


def test_fails_when_model_output_is_unrelated(tmp_path: Path):
    data = _make_dataset(tmp_path, [["a completely different gold caption about cats"]])
    report = run_bleu_sanity(data, FakeBackend(), min_bleu=0.90)
    assert not report.ok  # low BLEU trips the floor
