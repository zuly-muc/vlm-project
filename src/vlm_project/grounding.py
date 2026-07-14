"""Object-grounding *correctness* for produced captions vs. nuScenes ground truth.

nuScenes has no gold captions, so BLEU/CIDEr/SPICE do not apply. It does have
ground-truth 3D boxes per keyframe, so we measure whether a caption is
*grounded*: the object families it names must actually be visible in that
CAM_FRONT frame (**precision**), and it should name the frame's dominant family
(**dominant-recall**). Plain recall against *all* visible families is reported but
never gated — a terse captioner names only the most salient object, so a hard
recall floor would be arbitrary.

The scoring core here is **pure** (no dataset, no devkit): it takes, per caption,
the GT families present and computes the metrics. The devkit-backed provider that
turns a ``sample_token`` into visible GT families is **injected**
(:func:`make_nuscenes_gt_provider`), so this module is unit-tested with fakes and
the real nuScenes devkit is touched only in the e2e tier — the same dependency-
injection seam used for loaders and backends.

Caveat: "hallucination" here means a family the caption names that has no
*annotated, visible* box. nuScenes annotates objects only within a range, so a
very distant object BLIP might see could count as a false positive; the gate's
tolerance and the empirically perfect precision make this a non-issue in practice.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

# A GT provider maps a nuScenes sample_token to a {family: visible_box_count}.
GTProvider = Callable[[str], "Counter[str]"]

# nuScenes category prefix -> coarse family.
_FAMILY_PREFIX = {
    "human.pedestrian": "pedestrian",
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.trailer": "truck",
    "vehicle.bus": "bus",
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.construction": "construction",
    "vehicle.emergency": "car",
}

# Caption word (whole word, lowercased) -> family.
_CAPTION_WORDS = {
    "car": "car", "cars": "car", "sedan": "car", "suv": "car",
    "vehicle": "car", "vehicles": "car",
    "truck": "truck", "trucks": "truck", "trailer": "truck", "van": "truck",
    "bus": "bus", "buses": "bus",
    "pedestrian": "pedestrian", "pedestrians": "pedestrian", "person": "pedestrian",
    "people": "pedestrian", "ped": "pedestrian", "peds": "pedestrian",
    "man": "pedestrian", "woman": "pedestrian", "jaywalker": "pedestrian",
    "bicycle": "bicycle", "bike": "bicycle", "bikes": "bicycle",
    "cyclist": "bicycle", "bicyclist": "bicycle",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "scooter": "motorcycle",
    "construction": "construction",
}


def family_of(category: str) -> str | None:
    """Map a nuScenes category name to a coarse family (or None if uncounted)."""
    for prefix, fam in _FAMILY_PREFIX.items():
        if category.startswith(prefix):
            return fam
    return None


def caption_families(text: str) -> set[str]:
    """The object families a caption mentions."""
    return {_CAPTION_WORDS[w] for w in re.findall(r"[a-z]+", text.lower())
            if w in _CAPTION_WORDS}


@dataclass
class SceneScore:
    scene_id: str
    said: set[str]
    present: set[str]
    dominant: str | None

    @property
    def tp(self) -> int:
        return len(self.said & self.present)

    @property
    def fp(self) -> int:
        return len(self.said - self.present)

    @property
    def fn(self) -> int:
        return len(self.present - self.said)


@dataclass
class GroundingReport:
    scenes: list[SceneScore] = field(default_factory=list)

    @property
    def precision(self) -> float:
        tp = sum(s.tp for s in self.scenes)
        fp = sum(s.fp for s in self.scenes)
        return tp / (tp + fp) if (tp + fp) else 1.0

    @property
    def recall(self) -> float:
        tp = sum(s.tp for s in self.scenes)
        fn = sum(s.fn for s in self.scenes)
        return tp / (tp + fn) if (tp + fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def dominant_total(self) -> int:
        return sum(1 for s in self.scenes if s.dominant is not None)

    @property
    def dominant_hits(self) -> int:
        return sum(1 for s in self.scenes if s.dominant is not None and s.dominant in s.said)

    @property
    def dominant_recall(self) -> float:
        return self.dominant_hits / self.dominant_total if self.dominant_total else 1.0

    def metrics(self) -> dict:
        """The gate-relevant numbers, JSON-serializable."""
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "dominant_recall": round(self.dominant_recall, 4),
            "dominant_hits": self.dominant_hits,
            "dominant_total": self.dominant_total,
            "n_captions": len(self.scenes),
        }

    def render(self) -> str:
        lines = [f"  {'scene':14} {'P':>4} {'R':>4}  said / gt"]
        for s in self.scenes:
            sp = s.tp / (s.tp + s.fp) if (s.tp + s.fp) else 1.0
            sr = s.tp / (s.tp + s.fn) if (s.tp + s.fn) else 1.0
            lines.append(f"  {s.scene_id[:14]:14} {sp:4.2f} {sr:4.2f}  "
                         f"said={sorted(s.said) or '—'} gt={sorted(s.present) or '—'}")
        m = self.metrics()
        lines.append(
            f"  => precision={m['precision']:.2f} recall={m['recall']:.2f} "
            f"f1={m['f1']:.2f} dominant_recall={m['dominant_recall']:.2f} "
            f"({m['dominant_hits']}/{m['dominant_total']})"
        )
        return "\n".join(lines)


def evaluate(rows: list[dict], gt: GTProvider) -> GroundingReport:
    """Score each DB row against the GT families visible in its keyframe.

    ``rows`` are :meth:`SceneStore.all` dicts (``metadata`` already parsed). Rows
    without a ``sample_token`` (non-nuScenes loaders) are skipped.
    """
    report = GroundingReport()
    for r in rows:
        token = (r.get("metadata") or {}).get("sample_token")
        if not token:
            continue
        counts = gt(token)
        present = set(counts)
        dominant = counts.most_common(1)[0][0] if counts else None
        report.scenes.append(
            SceneScore(scene_id=r["scene_id"], said=caption_families(r["description"]),
                       present=present, dominant=dominant)
        )
    return report


@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.ok:
            return "  [PASS] correctness gate"
        return "\n".join(f"  [FAIL] {r}" for r in self.reasons)


def check_gate(
    report: GroundingReport,
    baseline: dict | None = None,
    *,
    min_precision: float = 0.90,
    max_precision_drop: float = 0.10,
    max_dominant_recall_drop: float = 0.15,
) -> GateResult:
    """Absolute precision floor + regression-vs-baseline. Recall never gates."""
    m = report.metrics()
    reasons: list[str] = []

    if m["precision"] < min_precision:
        reasons.append(f"precision {m['precision']:.2f} < floor {min_precision:.2f} "
                       "(captions naming absent objects)")

    if baseline:
        p_drop = baseline.get("precision", 0.0) - m["precision"]
        if p_drop > max_precision_drop:
            reasons.append(f"precision regressed {p_drop:.2f} vs baseline "
                           f"{baseline['precision']:.2f} (allowed {max_precision_drop:.2f})")
        d_drop = baseline.get("dominant_recall", 0.0) - m["dominant_recall"]
        if d_drop > max_dominant_recall_drop:
            reasons.append(f"dominant-recall regressed {d_drop:.2f} vs baseline "
                           f"{baseline['dominant_recall']:.2f} "
                           f"(allowed {max_dominant_recall_drop:.2f})")

    return GateResult(ok=not reasons, reasons=reasons)


def make_nuscenes_gt_provider(nusc, camera: str = "CAM_FRONT") -> GTProvider:
    """A GT provider over the real devkit: families with a box visible in the frame."""
    from nuscenes.utils.geometry_utils import BoxVisibility

    def provider(sample_token: str) -> Counter[str]:
        sample = nusc.get("sample", sample_token)
        cam_token = sample["data"][camera]
        # Only boxes whose projection is visible in the image count as "in frame".
        _path, boxes, _k = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
        counts: Counter[str] = Counter()
        for box in boxes:
            fam = family_of(box.name)
            if fam:
                counts[fam] += 1
        return counts

    return provider


def load_default_baseline() -> dict | None:
    """The committed, package-shipped baseline metrics (or None if unavailable)."""
    try:
        from importlib.resources import files

        text = files("vlm_project.baselines").joinpath("accuracy_baseline.json").read_text()
        return json.loads(text)["metrics"]
    except (FileNotFoundError, ModuleNotFoundError, KeyError, ValueError):
        return None
