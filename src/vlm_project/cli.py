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
    ing.add_argument("--selector", choices=["single", "clusters"])
    ing.add_argument("--single-strategy", choices=["middle", "sharpest"])
    ing.add_argument("--cluster-threshold", type=float)
    ing.add_argument("--backend", choices=["blip", "fake", "anthropic", "openai"])

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
        "cluster_threshold": getattr(args, "cluster_threshold", None),
        "backend": getattr(args, "backend", None),
        "db_path": getattr(args, "db", None),
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _cmd_ingest(cfg: Config) -> int:
    loader = factory.build_loader(cfg)
    backend = factory.build_backend(cfg)
    selector = factory.build_selector(cfg, backend)
    with SceneStore(cfg.db_path) as store:
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
    return 1


if __name__ == "__main__":
    sys.exit(main())
