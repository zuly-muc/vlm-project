"""Command-line entrypoint: ``vlm-project {ingest,query,export-json}``.

Config comes from environment variables (the container's contract) with CLI flags
taking precedence, so the same image is driven either way.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from vlm_project import factory
from vlm_project.config import Config
from vlm_project.pipeline import run
from vlm_project.store import SceneStore


def _add_common(p: argparse.ArgumentParser) -> None:
    # Accept --verbose after the subcommand too (natural usage: `ingest -v`).
    # SUPPRESS default so it never overrides a top-level `-v` given before it.
    p.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="log each description",
    )
    p.add_argument("--db", help="SQLite database path (env: DB)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlm-project",
        description="Turn driving footage into a searchable database of scene descriptions.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log each description")
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="run the VLM over a dataset and populate the DB")
    _add_common(ing)
    ing.add_argument("--loader", choices=["nuscenes", "imagefolder", "video"])
    ing.add_argument("--dataroot", help="dataset root / image dir / video path (env: DATAROOT)")
    ing.add_argument("--camera", help="nuScenes sensor channel (env: CAMERA)")
    ing.add_argument("--selector", choices=["single", "uniform", "clusters"])
    ing.add_argument("--single-strategy", choices=["middle", "sharpest"])
    ing.add_argument("--uniform-samples", type=int,
                     help="frames per clip for --selector uniform (env: UNIFORM_SAMPLES)")
    ing.add_argument("--cluster-threshold", type=float)
    ing.add_argument("--backend",
                     choices=["blip", "fake", "anthropic", "openai", "moondream"])
    ing.add_argument("--stage-dir",
                     help="build the DB in this local scratch dir, then move it to "
                          "--db on completion (env: STAGE_DIR); avoids per-row writes "
                          "to a bind mount")

    q = sub.add_parser("query", help="full-text search the descriptions")
    _add_common(q)
    q.add_argument("text", help="search query, e.g. \"pedestrian crossing\"")
    q.add_argument("--limit", type=int, default=20)
    q.add_argument("--json", action="store_true", help="print results as JSON")

    ex = sub.add_parser("export-json", help="dump all rows to JSON")
    _add_common(ex)
    ex.add_argument("--out", default="descriptions.json")

    doc = sub.add_parser("doctor", help="verify dependencies & environment health")
    doc.add_argument("--blip-model", default="Salesforce/blip-image-captioning-base")

    ver = sub.add_parser("verify-db", help="assert a produced DB is sound & correct")
    _add_common(ver)
    ver.add_argument("--dataroot", help="dataset root, to cross-check scene coverage")
    ver.add_argument("--version", default="v1.0-mini")
    ver.add_argument("--no-file-check", action="store_true",
                     help="skip image-file existence (e.g. paths not mounted here)")
    ver.add_argument("--vocab-fraction", type=float, default=0.5)

    bs = sub.add_parser("bleu-sanity",
                        help="model-sanity: caption a COCO micro-set and score BLEU-4")
    bs.add_argument("--data", required=True,
                    help="dir with references.json + images/ (see fetch_coco_sanity.py)")
    bs.add_argument("--backend", choices=["blip", "fake", "moondream"], default="blip")
    bs.add_argument("--min-bleu", type=float, default=0.10, help="broken-model floor")

    acc = sub.add_parser("verify-accuracy",
                         help="object-grounding correctness vs nuScenes GT boxes")
    _add_common(acc)
    acc.add_argument("--dataroot", required=True, help="nuScenes dataset root (has GT boxes)")
    acc.add_argument("--version", default="v1.0-mini")
    acc.add_argument("--camera", default="CAM_FRONT")
    acc.add_argument("--out", help="write metrics JSON here (e.g. out/accuracy.json)")
    acc.add_argument("--baseline", help="baseline metrics JSON (default: package baseline)")
    acc.add_argument("--write-baseline", metavar="PATH",
                     help="write this run's metrics as a baseline file and exit 0")

    return parser


def _config_from(args: argparse.Namespace) -> Config:
    """Env defaults, overridden by any CLI flag that was actually provided."""
    cfg = Config.from_env()
    overrides = {
        "loader": getattr(args, "loader", None),
        "dataroot": getattr(args, "dataroot", None),
        "camera": getattr(args, "camera", None),
        "selector": getattr(args, "selector", None),
        "single_strategy": getattr(args, "single_strategy", None),
        "uniform_samples": getattr(args, "uniform_samples", None),
        "cluster_threshold": getattr(args, "cluster_threshold", None),
        "backend": getattr(args, "backend", None),
        "db_path": getattr(args, "db", None),
        "stage_dir": getattr(args, "stage_dir", None),
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _cmd_ingest(cfg: Config) -> int:
    loader = factory.build_loader(cfg)
    backend = factory.build_backend(cfg)
    selector = factory.build_selector(cfg, backend)
    with SceneStore(cfg.db_path, stage_dir=cfg.stage_dir) as store:
        stats = run(loader, selector, backend, store)
        total = store.count()
    print(
        f"Ingested {stats.scenes} scene(s) from {stats.clips} clip(s) "
        f"({stats.failures} failure(s)). DB now holds {total} row(s): {cfg.db_path}"
    )
    return 0


def _cmd_query(cfg: Config, text: str, limit: int, as_json: bool) -> int:
    with SceneStore(cfg.db_path) as store:
        rows = store.search(text, limit=limit)
    if as_json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print(f"No matches for {text!r}.")
        return 0
    for r in rows:
        print(f"{r['scene_id']:<18}  {r['description']}")
    return 0


def _cmd_export(cfg: Config, out_path: str) -> int:
    with SceneStore(cfg.db_path) as store:
        rows = store.to_json()
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"Wrote {len(rows)} row(s) to {out_path}")
    return 0


def _cmd_doctor(model_name: str) -> int:
    from vlm_project import doctor

    return doctor.main(model_name)


def _cmd_verify_db(cfg: Config, args: argparse.Namespace) -> int:
    from vlm_project.acceptance import verify_db

    report = verify_db(
        db_path=cfg.db_path,
        dataroot=args.dataroot,
        version=args.version,
        check_files=not args.no_file_check,
        vocab_fraction=args.vocab_fraction,
    )
    print(report.render())
    return 0 if report.ok else 1


def _cmd_bleu_sanity(cfg: Config, args: argparse.Namespace) -> int:
    from vlm_project import sanity

    backend = factory.build_backend(cfg)
    report = sanity.run_bleu_sanity(args.data, backend, min_bleu=args.min_bleu)
    print(report.render())
    return 0 if report.ok else 1


def _cmd_verify_accuracy(cfg: Config, args: argparse.Namespace) -> int:
    from nuscenes.nuscenes import NuScenes

    from vlm_project import grounding

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    provider = grounding.make_nuscenes_gt_provider(nusc, camera=args.camera)
    with SceneStore(cfg.db_path) as store:
        rows = store.all()
    report = grounding.evaluate(rows, provider)
    print(report.render())

    metrics = report.metrics()
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"metrics": metrics,
                       "scenes": [{"scene_id": s.scene_id, "said": sorted(s.said),
                                   "present": sorted(s.present)} for s in report.scenes]},
                      fh, indent=2)
        print(f"  wrote metrics -> {args.out}")

    if args.write_baseline:
        with open(args.write_baseline, "w", encoding="utf-8") as fh:
            json.dump({"metrics": metrics}, fh, indent=2)
        print(f"  wrote baseline -> {args.write_baseline}")
        return 0

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as fh:
            baseline = json.load(fh)["metrics"]
    else:
        baseline = grounding.load_default_baseline()

    gate = grounding.check_gate(report, baseline)
    print(gate.render())
    return 0 if gate.ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(message)s",
    )

    if args.command == "doctor":
        return _cmd_doctor(args.blip_model)

    cfg = _config_from(args)

    if args.command == "ingest":
        return _cmd_ingest(cfg)
    if args.command == "query":
        return _cmd_query(cfg, args.text, args.limit, args.json)
    if args.command == "export-json":
        return _cmd_export(cfg, args.out)
    if args.command == "verify-db":
        return _cmd_verify_db(cfg, args)
    if args.command == "verify-accuracy":
        return _cmd_verify_accuracy(cfg, args)
    if args.command == "bleu-sanity":
        return _cmd_bleu_sanity(cfg, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
