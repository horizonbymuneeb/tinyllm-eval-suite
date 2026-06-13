"""Pluggable metric registry.

Built-in metrics:
  - `exact_match`: case-insensitive string equality after whitespace strip.
  - `token_accuracy`: 1 - (edit_distance / max(len_a, len_b)).
  - `f1`: token-level F1 (precision/recall over whitespace tokens).
  - `bleu`: corpus BLEU-4 with NLTK brevity penalty. Requires `nltk` and
    the `punkt` resource to be available.

Custom metrics: write a function that takes `(predictions, references)` and
returns a float (or dict of floats), then call `register(name, fn)`. They
become available in YAML by name.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

MetricFn = Callable[[Sequence[str], Sequence[str]], float | dict[str, float]]


_REGISTRY: dict[str, MetricFn] = {}


# --------------------------------------------------------------------- #
# Built-ins
# --------------------------------------------------------------------- #


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return 0.0
    correct = sum(_normalize(p) == _normalize(r) for p, r in zip(predictions, references))
    return correct / len(predictions)


def token_accuracy(predictions: Sequence[str], references: Sequence[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return 0.0
    total = 0.0
    for pred, ref in zip(predictions, references):
        p_tokens = _normalize(pred).split()
        r_tokens = _normalize(ref).split()
        if not r_tokens and not p_tokens:
            total += 1.0
            continue
        if not r_tokens or not p_tokens:
            total += 0.0
            continue
        dist = _levenshtein(p_tokens, r_tokens)
        denom = max(len(p_tokens), len(r_tokens))
        total += 1.0 - (dist / denom)
    return total / len(predictions)


def f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Token-level F1 (micro-averaged across the corpus)."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    if not predictions:
        return 0.0
    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        p_tokens = set(_normalize(pred).split())
        r_tokens = set(_normalize(ref).split())
        tp += len(p_tokens & r_tokens)
        fp += len(p_tokens - r_tokens)
        fn += len(r_tokens - p_tokens)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Corpus BLEU-4 with brevity penalty. Returns 0.0 if nltk punkt is missing."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    except ImportError:
        return 0.0
    list_of_refs = [[r.split()] for r in references]
    list_of_hyps = [p.split() for p in predictions]
    smoothing = SmoothingFunction().method1
    try:
        return float(
            corpus_bleu(list_of_refs, list_of_hyps, smoothing_function=smoothing)
        )
    except Exception:
        return 0.0


# --------------------------------------------------------------------- #
# Helpers (internal)
# --------------------------------------------------------------------- #


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Standard DP edit distance over token sequences."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur[j] = min(ins, dele, sub)
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #


def register(name: str, fn: MetricFn) -> None:
    """Register a custom metric under `name`."""
    if not name or not isinstance(name, str):
        raise ValueError("metric name must be a non-empty string")
    if not callable(fn):
        raise ValueError("metric fn must be callable")
    _REGISTRY[name] = fn


def get(name: str) -> MetricFn:
    """Look up a registered metric by name. Raises KeyError if unknown."""
    return _REGISTRY[name]


def available() -> list[str]:
    """Sorted list of all registered metric names."""
    return sorted(_REGISTRY)


def compute_all(
    metric_names: Iterable[str],
    predictions: Sequence[str],
    references: Sequence[str],
) -> dict[str, float]:
    """Compute every requested metric. Unknown names raise KeyError."""
    out: dict[str, float] = {}
    for name in metric_names:
        fn = _REGISTRY[name]
        score = fn(predictions, references)
        if isinstance(score, dict):
            out.update({f"{name}.{k}": float(v) for k, v in score.items()})
        else:
            out[name] = float(score)
    return out


# Register built-ins at import time.
register("exact_match", exact_match)
register("token_accuracy", token_accuracy)
register("f1", f1)
register("bleu", bleu)


__all__ = [
    "MetricFn",
    "register",
    "get",
    "available",
    "compute_all",
    "exact_match",
    "token_accuracy",
    "f1",
    "bleu",
]
