"""CLI smoke test: ingest an image folder with the fake backend, then query.

Exercises the real argparse -> factory -> pipeline -> store path end to end with
no dataset and no model download.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from vlm_project.cli import main


def _seed_images(dirpath: Path, n: int = 3) -> None:
    for i in range(n):
        Image.new("RGB", (32, 32), (i * 40, 10, 10)).save(dirpath / f"img{i}.jpg")


def test_ingest_then_query(tmp_path: Path, capsys):
    images = tmp_path / "imgs"
    images.mkdir()
    _seed_images(images)
    db = tmp_path / "out" / "scenes.db"

    rc = main(
        [
            "ingest",
            "--loader", "imagefolder",
            "--backend", "fake",
            "--dataroot", str(images),
            "--db", str(db),
        ]
    )
    assert rc == 0
    assert "Ingested 3 scene(s)" in capsys.readouterr().out
    assert db.exists()

    rc = main(["query", "image", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "img0" in out  # fake caption mentions the image dimensions


def test_verbose_flag_accepted_after_subcommand(tmp_path: Path, capsys):
    # Regression: `-v` must be accepted after the subcommand, not only before it.
    images = tmp_path / "imgs"
    images.mkdir()
    _seed_images(images, n=1)
    db = tmp_path / "s.db"

    rc = main(["ingest", "--loader", "imagefolder", "--backend", "fake",
               "--dataroot", str(images), "--db", str(db), "-v"])
    assert rc == 0

    rc = main(["-v", "ingest", "--loader", "imagefolder", "--backend", "fake",
               "--dataroot", str(images), "--db", str(db)])
    assert rc == 0


def test_ingest_uniform_selector_flags(tmp_path: Path, capsys):
    # Exercise the --selector uniform / --uniform-samples plumbing end to end.
    # (imagefolder yields one-frame clips, so this is one scene per image.)
    images = tmp_path / "imgs"
    images.mkdir()
    _seed_images(images, n=2)
    db = tmp_path / "scenes.db"

    rc = main(["ingest", "--loader", "imagefolder", "--backend", "fake",
               "--selector", "uniform", "--uniform-samples", "3",
               "--dataroot", str(images), "--db", str(db)])
    assert rc == 0
    assert "Ingested 2 scene(s)" in capsys.readouterr().out


def test_nuscenes_version_flag_threads_into_config():
    # Regression: --nuscenes-version must be a real ingest flag and override env.
    from vlm_project.cli import _config_from, build_parser

    args = build_parser().parse_args(
        ["ingest", "--loader", "nuscenes", "--nuscenes-version", "v1.0-trainval"]
    )
    cfg = _config_from(args)
    assert cfg.nuscenes_version == "v1.0-trainval"


def test_export_json(tmp_path: Path, capsys):
    images = tmp_path / "imgs"
    images.mkdir()
    _seed_images(images, n=2)
    db = tmp_path / "scenes.db"
    out = tmp_path / "descriptions.json"

    main(["ingest", "--loader", "imagefolder", "--backend", "fake",
          "--dataroot", str(images), "--db", str(db)])
    capsys.readouterr()

    rc = main(["export-json", "--db", str(db), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    import json

    rows = json.loads(out.read_text())
    assert len(rows) == 2
