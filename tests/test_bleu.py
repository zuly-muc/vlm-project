"""Corpus/sentence BLEU-4 — pure math, no model or data."""

from __future__ import annotations

from vlm_project.bleu import corpus_bleu, sentence_bleu


def test_identical_hypothesis_scores_one():
    ref = "a car driving down a city street at night"
    assert sentence_bleu(ref, [ref]) > 0.99


def test_disjoint_hypothesis_scores_near_zero():
    score = sentence_bleu("completely unrelated tokens here", ["a car on a road"])
    assert score < 0.05


def test_partial_overlap_is_between():
    ref = ["a car driving down a city street"]
    full = sentence_bleu("a car driving down a city street", ref)
    part = sentence_bleu("a car parked beside a quiet street", ref)
    assert full > part > 0.0


def test_brevity_penalty_punishes_short_hypotheses():
    ref = ["a car driving down a long city street at night downtown"]
    short = sentence_bleu("a car", ref)      # heavily penalized
    longer = sentence_bleu("a car driving down a long city street", ref)
    assert longer > short


def test_multiple_references_take_the_best():
    hyp = "a dog running in the park"
    two = corpus_bleu([hyp], [["a cat on a sofa", "a dog running in the park"]])
    assert two > 0.99  # matches the second reference exactly
