#!/usr/bin/env python3
"""Full production acceptance test for vlm-project.

One command that installs/builds the app, verifies dependencies, obtains the REAL
nuScenes v1.0-mini dataset, ingests real front-camera clips end to end, and then
asserts the results are sound and correct. Exit code 0 == production-ready.

Two modes (``--mode``):
  * ``docker`` (default): builds the image and runs every step inside it, so it
    validates the shipping artifact exactly as deployed.
  * ``local``: runs against the installed package/venv (no Docker),
    useful for fast iteration and CI where Docker isn't available.

Stages:
  1. preflight: docker/python present
  2. build: docker build (docker mode)
  3. doctor: dependency & environment health
  4. dataset: ensure real nuScenes v1.0-mini (download+extract if needed)
  5. ingest: real clips -> descriptions DB (single-frame)
  6. clusters: real clip segmentation on a subset (multi-scene)
  7. verify: soundness assertions on the produced DB (verify-db)
  8. query: full-text search returns matches
  9. accuracy: correctness: object-grounding vs nuScenes GT (verify-accuracy)
 10. model-sanity: fetch official COCO micro-set, BLIP BLEU-4 vs floor (skips offline)

Only the standard library is used here so the orchestrator runs anywhere.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "https://www.nuscenes.org/data/v1.0-mini.tgz"
MINI_SCENE_COUNT = 10


# --------------------------------------------------------------------------- #
# small logging helpers
# --------------------------------------------------------------------------- #
def _hr(title: str) -> None:
    print(f"\n{'=' * 70}\n== {title}\n{'=' * 70}", flush=True)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kw)


def _fail(msg: str) -> None:
    print(f"\n[FATAL] {msg}", flush=True)
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# dataset acquisition
# --------------------------------------------------------------------------- #
def _dataset_valid(dataroot: Path, version: str) -> bool:
    return (dataroot / version / "scene.json").is_file() and (dataroot / "samples").is_dir()


def ensure_dataset(dataroot: Path, version: str, url: str, allow_download: bool) -> None:
    if _dataset_valid(dataroot, version):
        print(f"Dataset present at {dataroot}", flush=True)
        return
    if not allow_download:
        _fail(f"No dataset at {dataroot} and downloading disabled. "
              f"Set --nuscenes-dir to an extracted v1.0-mini.")
    dataroot.mkdir(parents=True, exist_ok=True)
    tgz = dataroot.parent / "v1.0-mini.tgz"
    if not tgz.is_file() or tgz.stat().st_size < 3_000_000_000:
        print(f"Downloading {url} -> {tgz} (~4 GB, one-time)…", flush=True)
        _download(url, tgz)
    print(f"Extracting {tgz} -> {dataroot}…", flush=True)
    with tarfile.open(tgz, "r:gz") as tar:
        tar.extractall(dataroot)  # noqa: S202 - trusted official archive
    if not _dataset_valid(dataroot, version):
        _fail("Dataset extracted but structure invalid (expected v1.0-mini/scene.json).")


def _download(url: str, dest: Path) -> None:
    start = time.time()
    last = [0.0]

    def hook(block: int, block_size: int, total: int) -> None:
        done = block * block_size
        now = time.time()
        if now - last[0] > 3:
            pct = (done / total * 100) if total > 0 else 0
            rate = done / 1e6 / max(now - start, 1)
            print(f"  {done / 1e9:5.2f} GB ({pct:4.1f}%)  {rate:.0f} MB/s", flush=True)
            last[0] = now

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)  # noqa: S310 - https official host
    tmp.replace(dest)


# --------------------------------------------------------------------------- #
# command runners for the two modes
# --------------------------------------------------------------------------- #
class Runner:
    """Runs `vlm-project <args>` either in a container or the local venv."""

    def __init__(self, mode: str, image: str, dataroot: Path, out_dir: Path):
        self.mode = mode
        self.image = image
        self.dataroot = dataroot
        self.out_dir = out_dir

    # Paths as seen by the vlm-project process (container paths in docker mode).
    def data_path(self) -> str:
        return "/data" if self.mode == "docker" else str(self.dataroot)

    def db_path(self) -> str:
        return "/out/scenes.db" if self.mode == "docker" else str(self.out_dir / "scenes.db")

    def db_path_clusters(self) -> str:
        return ("/out/clusters.db" if self.mode == "docker"
                else str(self.out_dir / "clusters.db"))

    def accuracy_path(self) -> str:
        return ("/out/accuracy.json" if self.mode == "docker"
                else str(self.out_dir / "accuracy.json"))

    def accuracy_path_clusters(self) -> str:
        return ("/out/accuracy_clusters.json" if self.mode == "docker"
                else str(self.out_dir / "accuracy_clusters.json"))

    def vlm(self, args: list[str], *, mount_data: bool = False,
            mounts: list[tuple[str, str]] | None = None) -> subprocess.CompletedProcess:
        if self.mode == "docker":
            cmd = ["docker", "run", "--rm"]
            # The image runs as a non-root user; on a POSIX host run it as the
            # invoking user so it can write to the host-owned output mount.
            if hasattr(os, "getuid"):
                cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
            cmd += ["-v", f"{self.out_dir}:/out"]
            if mount_data:
                cmd += ["-v", f"{self.dataroot}:/data:ro"]
            for host, container in (mounts or []):
                cmd += ["-v", f"{host}:{container}:ro"]
            cmd += [self.image] + args
        else:
            # Use the running interpreter's module entrypoint so we don't depend
            # on the venv's Scripts dir being on PATH.
            cmd = [sys.executable, "-m", "vlm_project.cli"] + args
        return _run(cmd)


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_preflight(mode: str) -> None:
    _hr("1/10 preflight")
    if mode == "docker":
        if not shutil.which("docker"):
            _fail("docker not found on PATH.")
        r = _run(["docker", "info", "--format", "{{.ServerVersion}}"],
                 capture_output=True, text=True)
        if r.returncode != 0:
            _fail("Docker daemon not reachable, start Docker Desktop and retry.")
        print(f"Docker daemon OK (server {r.stdout.strip()})")
    else:
        import importlib.util

        if importlib.util.find_spec("vlm_project") is None:
            _fail("vlm_project not importable in this environment (pip install -e .).")
        print("Local vlm_project package importable.")


def stage_build(runner: Runner, skip: bool) -> None:
    _hr("2/10 build image")
    if runner.mode != "docker":
        print("local mode, skipping image build")
        return
    if skip:
        print("--skip-build set, using existing image")
        return
    r = _run(["docker", "build", "-t", runner.image, str(REPO_ROOT)])
    if r.returncode != 0:
        _fail("docker build failed.")


def stage_doctor(runner: Runner) -> None:
    _hr("3/10 dependency & health check (doctor)")
    if runner.vlm(["doctor"]).returncode != 0:
        _fail("doctor reported a critical dependency failure.")


def stage_ingest(runner: Runner, db: str, extra: list[str]) -> None:
    args = ["ingest", "--loader", "nuscenes", "--backend", "blip",
            "--dataroot", runner.data_path(), "--db", db, "-v"] + extra
    if runner.vlm(args, mount_data=True).returncode != 0:
        _fail("ingest failed.")


def stage_verify(runner: Runner, db: str) -> None:
    _hr("7/10 verify correctness (verify-db)")
    args = ["verify-db", "--db", db, "--dataroot", runner.data_path()]
    if runner.vlm(args, mount_data=True).returncode != 0:
        _fail("acceptance checks failed, results are not sound.")


def stage_query(runner: Runner, db: str) -> None:
    _hr("8/10 query smoke")
    # Query a term BLIP reliably emits on driving scenes.
    if runner.vlm(["query", "road", "--db", db]).returncode != 0:
        _fail("query failed.")


def stage_accuracy(runner: Runner) -> None:
    _hr("9/10 correctness, object-grounding vs nuScenes GT")
    # single-frame DB: precision floor + regression vs the committed baseline.
    single = ["verify-accuracy", "--db", runner.db_path(), "--dataroot", runner.data_path(),
              "--out", runner.accuracy_path()]
    if runner.vlm(single, mount_data=True).returncode != 0:
        _fail("correctness gate failed (single), captions not grounded in nuScenes GT.")
    # clusters DB: no committed baseline for this config, so gate on the precision
    # floor only (a slightly lower floor gives the medoid captions margin).
    clusters = [
        "verify-accuracy", "--db", runner.db_path_clusters(), "--dataroot", runner.data_path(),
        "--out", runner.accuracy_path_clusters(), "--no-baseline", "--min-precision", "0.8",
    ]
    if runner.vlm(clusters, mount_data=True).returncode != 0:
        _fail("correctness gate failed (clusters), captions not grounded in nuScenes GT.")


def stage_model_sanity(runner: Runner) -> None:
    """Independent axis: has BLIP itself regressed on general-domain images?

    Fetches a small COCO caption set from the official source (host-side, network
    needed) and scores BLEU-4 against a floor. Offline, the fetch fails and the
    stage is skipped with a warning, it never blocks acceptance for lack of net;
    it only fails if the model is actually present but degraded.
    """
    _hr("10/10 model-sanity, BLIP BLEU-4 on COCO")
    coco_dir = runner.out_dir / "coco_sanity"
    fetch = _run([sys.executable, str(REPO_ROOT / "scripts" / "fetch_coco_sanity.py"),
                  "--out", str(coco_dir)])
    if fetch.returncode != 0:
        print("[WARN] COCO fetch failed (offline?), skipping model-sanity.", flush=True)
        return

    if runner.mode == "docker":
        args = ["bleu-sanity", "--data", "/coco"]
        result = runner.vlm(args, mounts=[(str(coco_dir), "/coco")])
    else:
        result = runner.vlm(["bleu-sanity", "--data", str(coco_dir)])
    if result.returncode != 0:
        _fail("model-sanity BLEU below floor, BLIP may have regressed.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="vlm-project production acceptance test")
    p.add_argument("--mode", choices=["docker", "local"],
                   default=os.environ.get("ACCEPTANCE_MODE", "docker"))
    p.add_argument("--image", default=os.environ.get("IMAGE", "vlm-project"))
    p.add_argument("--nuscenes-dir",
                   default=os.environ.get("NUSCENES_DIR", str(Path.home() / "nuscenes")))
    p.add_argument("--out-dir", default=os.environ.get("OUT_DIR", str(REPO_ROOT / "out")))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--url", default=os.environ.get("NUSCENES_MINI_URL", DEFAULT_URL))
    p.add_argument("--skip-build", action="store_true")
    p.add_argument("--no-download", action="store_true",
                   help="fail instead of downloading if the dataset is absent")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataroot = Path(args.nuscenes_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.mode, args.image, dataroot, out_dir)

    print(f"vlm-project production acceptance, mode={args.mode}, dataset={dataroot}")
    t0 = time.time()

    stage_preflight(args.mode)
    stage_build(runner, args.skip_build)
    stage_doctor(runner)

    _hr("4/10 ensure real nuScenes dataset")
    ensure_dataset(dataroot, args.version, args.url, allow_download=not args.no_download)

    _hr("5/10 ingest real clips (single-frame)")
    stage_ingest(runner, runner.db_path(), extra=[])

    _hr("6/10 ingest real clips (clustering, multi-scene)")
    stage_ingest(runner, runner.db_path_clusters(), extra=["--selector", "clusters"])

    stage_verify(runner, runner.db_path())
    stage_verify(runner, runner.db_path_clusters())

    stage_query(runner, runner.db_path())

    # Correctness (not just soundness): captions grounded in real GT boxes.
    stage_accuracy(runner)

    # Independent model-sanity axis: BLIP quality on general-domain COCO images.
    stage_model_sanity(runner)

    _hr(f"RESULT: PRODUCTION ACCEPTANCE PASSED in {time.time() - t0:.0f}s")
    print(f"Databases in {out_dir}: scenes.db (single), clusters.db (segments)")
    print(f"Accuracy report: {out_dir / 'accuracy.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
