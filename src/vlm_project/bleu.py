"""Corpus BLEU-4, a small, dependency-free implementation.

Used by the optional model-sanity check (does BLIP still caption general-domain
COCO images at its expected quality?). Hand-rolled rather than pulling in
``nltk``/``sacrebleu`` so the core install stays light, the same reasoning that
keeps the sharpness measure OpenCV-free.

This is standard BLEU: clipped modified n-gram precision (n=1..4), a brevity
penalty against the closest reference length, geometric mean with uniform
weights. A tiny epsilon smooths zero higher-order matches so small corpora don't
collapse to 0 spuriously.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_EPS = 1e-9


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """BLEU-``max_n`` over a corpus.

    ``hypotheses[i]`` is one generated caption; ``references[i]`` is its list of
    reference captions. Returns a score in ``[0, 1]``.
    """
    if not hypotheses:
        return 0.0

    num = [0] * max_n
    den = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, refs in zip(hypotheses, references, strict=True):
        h = _tokens(hyp)
        rs = [_tokens(r) for r in refs] or [[]]
        hyp_len += len(h)
        # Effective reference length: the ref length closest to the hypothesis.
        ref_len += min((len(r) for r in rs), key=lambda rl: (abs(rl - len(h)), rl))

        for n in range(1, max_n + 1):
            h_ng = _ngram_counts(h, n)
            if not h_ng:
                continue
            # Clip each hypothesis n-gram by its max count across references.
            max_ref: Counter = Counter()
            for r in rs:
                for g, c in _ngram_counts(r, n).items():
                    if c > max_ref[g]:
                        max_ref[g] = c
            num[n - 1] += sum(min(c, max_ref[g]) for g, c in h_ng.items())
            den[n - 1] += sum(h_ng.values())

    if hyp_len == 0:
        return 0.0

    log_p = 0.0
    for n in range(max_n):
        if den[n] == 0:
            return 0.0
        numerator = num[n] if num[n] > 0 else _EPS
        log_p += (1.0 / max_n) * math.log(numerator / den[n])

    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)
    return bp * math.exp(log_p)


def sentence_bleu(hypothesis: str, references: list[str], max_n: int = 4) -> float:
    """BLEU for a single hypothesis against its references."""
    return corpus_bleu([hypothesis], [references], max_n=max_n)
