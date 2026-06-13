"""Tests for `tinyllm_eval.metrics` — no model or dataset needed."""

from __future__ import annotations

import pytest

from tinyllm_eval import metrics


# --------------------------------------------------------------------- #
# exact_match
# --------------------------------------------------------------------- #


def test_exact_match_perfect() -> None:
    assert metrics.exact_match(["yes", "no"], ["yes", "no"]) == 1.0


def test_exact_match_case_insensitive() -> None:
    assert metrics.exact_match(["Yes", "NO"], ["yes", "no"]) == 1.0


def test_exact_match_whitespace_normalized() -> None:
    assert metrics.exact_match(["  yes  "], ["yes"]) == 1.0


def test_exact_match_half() -> None:
    assert metrics.exact_match(["yes", "maybe"], ["yes", "no"]) == 0.5


def test_exact_match_empty() -> None:
    assert metrics.exact_match([], []) == 0.0


def test_exact_match_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        metrics.exact_match(["yes"], ["yes", "no"])


# --------------------------------------------------------------------- #
# token_accuracy
# --------------------------------------------------------------------- #


def test_token_accuracy_perfect() -> None:
    assert metrics.token_accuracy(["the cat sat"], ["the cat sat"]) == 1.0


def test_token_accuracy_partial() -> None:
    # 1 edit (cat->bat) over 3 tokens -> 1 - 1/3 ≈ 0.6667
    score = metrics.token_accuracy(["the bat sat"], ["the cat sat"])
    assert 0.6 < score < 0.7


def test_token_accuracy_empty_pred() -> None:
    assert metrics.token_accuracy([""], ["the cat sat"]) == 0.0


def test_token_accuracy_both_empty() -> None:
    assert metrics.token_accuracy([""], [""]) == 1.0


# --------------------------------------------------------------------- #
# f1
# --------------------------------------------------------------------- #


def test_f1_perfect() -> None:
    assert metrics.f1(["the cat sat"], ["the cat sat"]) == 1.0


def test_f1_no_overlap() -> None:
    assert metrics.f1(["apple banana"], ["cat dog"]) == 0.0


def test_f1_partial() -> None:
    # p = {a, b, c}, r = {a, b, d}
    # tp=2, fp=1, fn=1 -> precision=2/3, recall=2/3, f1=2/3
    score = metrics.f1(["a b c"], ["a b d"])
    assert abs(score - 2 / 3) < 1e-6


# --------------------------------------------------------------------- #
# bleu
# --------------------------------------------------------------------- #


def test_bleu_perfect() -> None:
    score = metrics.bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert score > 0.99


def test_bleu_no_overlap() -> None:
    score = metrics.bleu(["apple banana cherry"], ["dog elephant frog"])
    assert 0.0 <= score < 0.1


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #


def test_built_ins_registered() -> None:
    for name in ("exact_match", "token_accuracy", "f1", "bleu"):
        assert name in metrics.available()


def test_custom_metric_registration() -> None:
    def my_metric(preds: list[str], refs: list[str]) -> float:
        return sum(1.0 for p, r in zip(preds, refs) if p == r) / max(len(preds), 1)

    metrics.register("all_or_nothing_test", my_metric)
    try:
        assert "all_or_nothing_test" in metrics.available()
        assert metrics.get("all_or_nothing_test") is my_metric
    finally:
        # Best-effort cleanup; registry is process-global
        from tinyllm_eval.metrics import _REGISTRY

        _REGISTRY.pop("all_or_nothing_test", None)


def test_compute_all_mixed() -> None:
    out = metrics.compute_all(
        ["exact_match", "f1"],
        ["yes", "no"],
        ["yes", "yes"],
    )
    assert "exact_match" in out
    assert "f1" in out
    assert out["exact_match"] == 0.5


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        metrics.get("not_a_real_metric")


def test_register_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        metrics.register("", lambda p, r: 0.0)
    with pytest.raises(ValueError):
        metrics.register("ok", "not callable")  # type: ignore[arg-type]
