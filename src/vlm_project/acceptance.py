"""Correctness checks for a produced scene-description database.

Pure standard library (sqlite3, json, os, string) so it runs on the host, in the
container, or in CI without torch/transformers/nuscenes-devkit. Used by the
``vlm-project verify-db`` command and the production acceptance script to assert
that an ingest over real data produced *sound* results — not just that it ran.

The checks are deliberately structural + heuristic (a generative caption has no
single "right" answer): row coverage matches the dataset, every description is
well-formed, the search index round-trips, and captions contain plausible
driving vocabulary.
"""

from __future__ import annotations

import json
import os
import string
from dataclasses import dataclass, field

from vlm_project.store import SceneStore

# Words BLIP reliably emits on real driving keyframes. Used only as a soundness
# heuristic (a fraction of scenes must contain at least one), never per-scene.
DRIVING_VOCAB = {
    "road", "roads", "street", "streets", "car", "cars", "truck", "trucks",
    "traffic", "city", "highway", "intersection", "parking", "building",
    "buildings", "bus", "bike", "bicycle", "motorcycle", "pedestrian",
    "pedestrians", "people", "person", "sign", "signs", "lane", "lanes",
    "vehicle", "vehicles", "sidewalk", "crosswalk", "night", "driving", "drive",
    "town", "urban", "van", "bridge", "tunnel", "light", "lights",
}

_PRINTABLE = set(string.printable)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class AcceptanceReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def render(self) -> str:
        lines = []
        for c in self.checks:
            mark = "PASS" if c.ok else "FAIL"
            lines.append(f"  [{mark}] {c.name}" + (f" — {c.detail}" if c.detail else ""))
        lines.append(f"  => {'ALL CHECKS PASSED' if self.ok else 'ACCEPTANCE FAILED'}")
        return "\n".join(lines)


def scene_names_from_dataset(dataroot: str, version: str = "v1.0-mini") -> list[str]:
    """Read the real scene names straight from ``<dataroot>/<version>/scene.json``.

    No devkit needed — just the JSON table — so the host can cross-check coverage.
    """
    path = os.path.join(dataroot, version, "scene.json")
    with open(path, encoding="utf-8") as fh:
        return [s["name"] for s in json.load(fh)]


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(c if c.isalnum() else " " for c in text.lower()).split() if t]


def verify_db(
    db_path: str,
    dataroot: str | None = None,
    version: str = "v1.0-mini",
    expected_scene_names: list[str] | None = None,
    check_files: bool = True,
    vocab_fraction: float = 0.5,
    max_inference_seconds: float = 120.0,
) -> AcceptanceReport:
    """Run all acceptance checks against a produced DB. Returns a report."""
    report = AcceptanceReport()
    store = SceneStore(db_path)
    try:
        rows = store.all()

        # 1) Non-empty.
        report.add("rows_present", bool(rows), f"{len(rows)} row(s)")
        if not rows:
            return report

        # 2) Scene coverage vs. the real dataset.
        if expected_scene_names is None and dataroot is not None:
            try:
                expected_scene_names = scene_names_from_dataset(dataroot, version)
            except OSError as e:
                report.add("scene_coverage", False, f"could not read scene.json: {e}")
                expected_scene_names = None
        if expected_scene_names is not None:
            clip_ids = {r["clip_id"] for r in rows}
            expected = set(expected_scene_names)
            report.add(
                "scene_coverage",
                clip_ids == expected,
                f"{len(clip_ids)} distinct clip(s) vs {len(expected)} scene(s)"
                + ("" if clip_ids == expected else f"; missing={sorted(expected - clip_ids)}"),
            )

        # 3) Every description is well-formed (>=2 words, printable ASCII).
        bad = [
            r["scene_id"]
            for r in rows
            if len(_tokens(r["description"])) < 2
            or any(ch not in _PRINTABLE for ch in r["description"])
        ]
        report.add("descriptions_wellformed", not bad,
                   "all sound" if not bad else f"bad: {bad[:5]}")

        # 4) Positive, plausible inference timings.
        bad_time = [r["scene_id"] for r in rows
                    if not (0.0 < r["inference_seconds"] <= max_inference_seconds)]
        report.add("inference_timings", not bad_time,
                   "all positive & bounded" if not bad_time else f"suspect: {bad_time[:5]}")

        # 5) Image paths exist on disk.
        if check_files:
            missing = [r["scene_id"] for r in rows if not os.path.isfile(r["image_path"])]
            report.add("image_files_exist", not missing,
                       "all present" if not missing else f"missing: {missing[:5]}")

        # 6) FTS round-trip: a token of each caption retrieves its own row. Search
        # with no truncation (limit = row count) — this verifies the index contains
        # the row, not that it ranks in some top-N against rows sharing the word.
        misses = []
        for r in rows:
            toks = sorted((t for t in _tokens(r["description"]) if len(t) >= 4),
                          key=len, reverse=True)
            if not toks:
                continue
            hits = {h["scene_id"] for h in store.search(toks[0], limit=len(rows))}
            if r["scene_id"] not in hits:
                misses.append(r["scene_id"])
        report.add("fts_roundtrip", not misses,
                   "search finds every row" if not misses else f"unfound: {misses[:5]}")

        # 7) Driving-vocabulary soundness across the set.
        with_vocab = sum(1 for r in rows if set(_tokens(r["description"])) & DRIVING_VOCAB)
        frac = with_vocab / len(rows)
        report.add("driving_vocabulary", frac >= vocab_fraction,
                   f"{with_vocab}/{len(rows)} scenes ({frac:.0%}) mention driving terms "
                   f"(threshold {vocab_fraction:.0%})")

        return report
    finally:
        store.close()
