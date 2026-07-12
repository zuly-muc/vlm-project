#!/usr/bin/env python3
"""Render a README montage from a REAL produced scenes.db.

Reads scene rows (image_path + BLIP description) straight out of the database an
ingest run produced, and composes a labelled grid of the actual nuScenes
front-camera keyframes with their captions. The output PNG is committed and shown
in the README, so the visuals are genuine artifacts of a real run — not mockups.

Usage:
    python scripts/make_readme_assets.py --db out/scenes.db \
        --out docs/assets/nuscenes_captions.png --cols 2 --rows 3
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vlm_project.store import SceneStore

# Layout constants (pixels).
THUMB_W, THUMB_H = 520, 293          # 16:9 keyframe thumbnail
CAPTION_H = 84                       # caption strip under each thumbnail
PAD = 18
HEADER_H = 76
BG = (17, 19, 24)
STRIP = (28, 31, 38)
FG = (233, 236, 241)
MUTED = (150, 200, 255)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    # matplotlib ships DejaVuSans — a reliable cross-platform fallback.
    try:
        import matplotlib

        candidates.append(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"))
    except Exception:  # noqa: BLE001
        pass
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _cell(row: dict, font_cap, font_id) -> Image.Image:
    cell = Image.new("RGB", (THUMB_W, THUMB_H + CAPTION_H), STRIP)
    try:
        photo = Image.open(row["image_path"]).convert("RGB")
        photo = photo.resize((THUMB_W, THUMB_H))
        cell.paste(photo, (0, 0))
    except OSError:
        ImageDraw.Draw(cell).text((10, 10), "image missing", fill=FG, font=font_cap)

    draw = ImageDraw.Draw(cell)
    draw.text((12, THUMB_H + 8), row["scene_id"], fill=MUTED, font=font_id)
    wrapped = textwrap.fill(f'"{row["description"]}"', width=64)
    draw.multiline_text((12, THUMB_H + 30), wrapped, fill=FG, font=font_cap, spacing=3)
    return cell


def build_montage(rows: list[dict], cols: int, rows_n: int, title: str) -> Image.Image:
    rows = rows[: cols * rows_n]
    font_cap = _font(17)
    font_id = _font(15)
    font_title = _font(30)

    grid_w = cols * THUMB_W + (cols + 1) * PAD
    grid_h = HEADER_H + rows_n * (THUMB_H + CAPTION_H) + (rows_n + 1) * PAD
    canvas = Image.new("RGB", (grid_w, grid_h), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((PAD, 22), title, fill=FG, font=font_title)

    for i, row in enumerate(rows):
        r, c = divmod(i, cols)
        x = PAD + c * (THUMB_W + PAD)
        y = HEADER_H + PAD + r * (THUMB_H + CAPTION_H + PAD)
        canvas.paste(_cell(row, font_cap, font_id), (x, y))
    return canvas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="out/scenes.db")
    ap.add_argument("--out", default="docs/assets/nuscenes_captions.png")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--clip", help="only rows for this clip_id (e.g. a clustering demo)")
    ap.add_argument("--title", help="override the header title")
    args = ap.parse_args(argv)

    with SceneStore(args.db) as store:
        rows = store.all()
    if args.clip:
        rows = [r for r in rows if r["clip_id"] == args.clip]
    if not rows:
        print("No rows in DB — run an ingest first.")
        return 1

    title = args.title or "nuScenes v1.0-mini  ·  CAM_FRONT  ·  BLIP captions"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    montage = build_montage(rows, args.cols, args.rows, title)
    montage.save(out)
    print(f"Wrote {out}  ({montage.size[0]}x{montage.size[1]}) from {len(rows)} scene(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
