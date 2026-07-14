"""Object-grounding scorer + gate — pure logic, fake GT provider, no devkit.

Proves the metric math and both gate outcomes (pass / precision-floor /
regression) without nuScenes, exactly like the acceptance verifier is tested.
"""

from __future__ import annotations

from collections import Counter

from vlm_project.grounding import (
    caption_families,
    check_gate,
    evaluate,
    family_of,
    load_default_baseline,
)


def _row(scene_id: str, description: str, token: str | None) -> dict:
    md = {"sample_token": token} if token else {}
    return {"scene_id": scene_id, "description": description, "metadata": md}


def test_family_and_caption_mapping():
    assert family_of("vehicle.car") == "car"
    assert family_of("human.pedestrian.adult") == "pedestrian"
    assert family_of("movable_object.barrier") is None
    assert caption_families("a car and a pedestrian near a truck") == {"car", "pedestrian", "truck"}
    assert caption_families("white lines on the road") == set()  # no object nouns


def test_evaluate_precision_recall_and_dominant():
    gt = {
        "t1": Counter({"car": 3, "pedestrian": 1}),  # dominant = car
        "t2": Counter({"pedestrian": 2}),            # dominant = pedestrian
    }
    rows = [
        _row("s1", "a car driving down the road", "t1"),   # said car -> tp; miss ped
        _row("s2", "a truck on the street", "t2"),         # said truck -> fp; miss ped
    ]
    report = evaluate(rows, lambda tok: gt[tok])
    m = report.metrics()
    # tp=1 (car), fp=1 (truck), fn=2 (ped, ped)
    assert m["precision"] == 0.5      # 1 / (1+1)
    assert m["recall"] == round(1 / 3, 4)
    # dominant: s1 car said -> hit; s2 ped not said -> miss
    assert m["dominant_hits"] == 1 and m["dominant_total"] == 2
    assert m["dominant_recall"] == 0.5


def test_rows_without_sample_token_are_skipped():
    report = evaluate([_row("s1", "a car", None)], lambda tok: Counter({"car": 1}))
    assert report.metrics()["n_captions"] == 0


def test_gate_passes_when_precise_and_no_regression():
    gt = {"t1": Counter({"car": 1})}
    report = evaluate([_row("s1", "a car on the road", "t1")], lambda tok: gt[tok])
    baseline = {"precision": 1.0, "dominant_recall": 1.0}
    assert check_gate(report, baseline).ok


def test_gate_fails_on_precision_floor():
    gt = {"t1": Counter({"car": 1})}
    # says "bus" which isn't present -> precision 0.0
    report = evaluate([_row("s1", "a bus", "t1")], lambda tok: gt[tok])
    result = check_gate(report, baseline=None)
    assert not result.ok
    assert any("precision" in r for r in result.reasons)


def test_gate_fails_on_dominant_recall_regression():
    gt = {"t1": Counter({"car": 1}), "t2": Counter({"car": 1})}
    # precise (no hallucination) but names the dominant car in neither -> dom recall 0.0
    rows = [_row("s1", "the road is empty", "t1"), _row("s2", "white lines", "t2")]
    report = evaluate(rows, lambda tok: gt[tok])
    baseline = {"precision": 1.0, "dominant_recall": 0.5}
    result = check_gate(report, baseline)
    assert not result.ok
    assert any("dominant-recall" in r for r in result.reasons)


def test_default_baseline_is_shipped_and_wellformed():
    baseline = load_default_baseline()
    assert baseline is not None
    assert 0.0 <= baseline["precision"] <= 1.0
    assert "dominant_recall" in baseline
