#!/usr/bin/env python3
"""Fetch a tiny, fixed COCO caption micro-set for the BLIP model-sanity check.

We do NOT vendor COCO images (licensing + size); instead we fetch a deterministic
handful from the **official** COCO 2017 source and write them under
``tests/data/coco_sanity/``: the same "download real data, don't fabricate it"
pattern the project uses for nuScenes. The model-sanity test skips until this has
been run (locally or in CI, where network exists).

Determinism: we select the first ``--limit`` image ids in sorted order from
``captions_val2017.json``, so every run picks the same images.

Outputs:
  tests/data/coco_sanity/images/<id>.jpg
  tests/data/coco_sanity/references.json   # [{"image": "<id>.jpg", "captions": [...]}]

Only the standard library is used.

Usage:
  python scripts/fetch_coco_sanity.py                 # download annotations + images
  python scripts/fetch_coco_sanity.py --captions path/to/captions_val2017.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests" / "data" / "coco_sanity"
ANNOTATIONS_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CAPTIONS_MEMBER = "annotations/captions_val2017.json"

# Set by --insecure: an unverified TLS context, for networks behind a
# TLS-intercepting proxy (common in corporate CI). Default stays secure.
_SSL_CTX: ssl.SSLContext | None = None


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    ctx = _SSL_CTX if url.startswith("https") else None
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:  # noqa: S310 - official host
        with open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
    tmp.replace(dest)


def _load_captions_json(captions_arg: str | None, workdir: Path) -> dict:
    if captions_arg:
        return json.loads(Path(captions_arg).read_text(encoding="utf-8"))
    zip_path = workdir / "ann.zip"
    _download(ANNOTATIONS_ZIP, zip_path)
    print(f"  extracting {CAPTIONS_MEMBER}", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        with z.open(CAPTIONS_MEMBER) as fh:
            return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=18, help="number of images to fetch")
    p.add_argument("--captions", help="path to a local captions_val2017.json (skip zip download)")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--insecure", action="store_true",
                   help="disable TLS verification (for networks behind a TLS-intercepting proxy)")
    args = p.parse_args(argv)

    if args.insecure:
        global _SSL_CTX
        _SSL_CTX = ssl.create_default_context()
        _SSL_CTX.check_hostname = False
        _SSL_CTX.verify_mode = ssl.CERT_NONE
        print("  [insecure] TLS verification disabled", flush=True)

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        data = _load_captions_json(args.captions, Path(tmp))

    url_by_id = {img["id"]: img["coco_url"] for img in data["images"]}
    caps_by_id: dict[int, list[str]] = defaultdict(list)
    for ann in data["annotations"]:
        caps_by_id[ann["image_id"]].append(ann["caption"].strip())

    chosen = sorted(caps_by_id)[: args.limit]  # deterministic selection
    references = []
    for image_id in chosen:
        img_path = out / "images" / f"{image_id}.jpg"
        if not img_path.exists():
            _download(url_by_id[image_id], img_path)
        references.append({"image": f"{image_id}.jpg", "captions": caps_by_id[image_id]})

    (out / "references.json").write_text(json.dumps(references, indent=2), encoding="utf-8")
    print(f"Wrote {len(references)} image(s) + references to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
