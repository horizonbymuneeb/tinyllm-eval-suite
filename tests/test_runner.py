"""Tests for `tinyllm_eval.runner` — uses a deterministic fake model
to avoid loading anything from HuggingFace.

The fake model returns a fixed continuation for any prompt. This lets us
exercise the full eval pipeline (batching, generation, postprocessing,
metric computation) without network or GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from tinyllm_eval.config import (
    DatasetSpec,
    InferenceSpec,
    MetricSpec,
    ModelSpec,
    PromptSpec,
    TaskConfig,
)
from tinyllm_eval.runner import postprocess, run_eval


# --------------------------------------------------------------------- #
# Fake model
# --------------------------------------------------------------------- #


VOCAB_SIZE = 100


class FakeLM(nn.Module):
    """A minimal model that satisfies the runner's interface.

    - `.generate(**enc, max_new_tokens=N, ...)` returns a tensor of shape
      `(batch, prompt_len + N)`.
    - `.eval()` and `.to(device)` are no-ops here.
    - The model has a `.parameters()` method (inherited from nn.Module).
    """

    def __init__(self) -> None:
        super().__init__()
        # A real parameter so `.parameters()` works for the device lookup
        self.dummy = nn.Parameter(torch.zeros(1))

    def generate(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        batch = input_ids.shape[0]
        # Always generate the same token IDs (a tiny "echo" + label-ish)
        new_tokens = torch.full(
            (batch, max_new_tokens),
            fill_value=42,  # arbitrary token id
            dtype=input_ids.dtype,
        )
        return torch.cat([input_ids, new_tokens], dim=1)

    def eval(self) -> "FakeLM":
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        texts: list[str],
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
    ) -> dict[str, torch.Tensor]:
        # Deterministic encoding: length of text, padded to longest
        max_len = max(len(t) for t in texts)
        ids = []
        mask = []
        for t in texts:
            padded = [2] * len(t) + [0] * (max_len - len(t))
            ids.append(padded)
            mask.append([1] * len(t) + [0] * (max_len - len(t)))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        # Return a fixed string so the metric scores are predictable
        return "the answer is yes"


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


class _Row(dict):
    """A dict that also supports integer indexing (HF Dataset row protocol)."""

    pass


def _make_dataset(rows: list[dict[str, Any]]) -> list[_Row]:
    return [_Row(r) for r in rows]


def _make_cfg(
    *,
    limit: int = 5,
    batch_size: int = 2,
    max_new_tokens: int = 4,
    seed: int = 42,
) -> TaskConfig:
    return TaskConfig(
        name="test_task",
        dataset=DatasetSpec(name="fake", split="train", limit=limit),
        model=ModelSpec(name="fake/model", device="cpu", dtype="float32"),
        prompt=PromptSpec(
            template="Q: {q}\nA:",
            input_fields=["q"],
            target_field="a",
        ),
        metrics=[MetricSpec(name="exact_match")],
        inference=InferenceSpec(
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
        ),
    )


# --------------------------------------------------------------------- #
# postprocess
# --------------------------------------------------------------------- #


def test_postprocess_no_regex_strips_whitespace() -> None:
    assert postprocess("  hello  ", None) == "hello"


def test_postprocess_regex_extracts_group() -> None:
    out = postprocess("the answer is yes (label: positive)", r"label:\s*(\w+)")
    assert out == "positive"


def test_postprocess_regex_no_match_returns_text() -> None:
    assert postprocess("nope", r"label:\s*(\w+)") == "nope"


# --------------------------------------------------------------------- #
# run_eval
# --------------------------------------------------------------------- #


def test_run_eval_batches_and_scores() -> None:
    cfg = _make_cfg(limit=5, batch_size=2)
    ds = _make_dataset(
        [
            {"q": "q1", "a": "yes"},
            {"q": "q2", "a": "yes"},
            {"q": "q3", "a": "no"},
            {"q": "q4", "a": "yes"},
            {"q": "q5", "a": "no"},
        ]
    )

    result = run_eval(cfg, FakeLM(), FakeTokenizer(), ds, ["exact_match"])

    assert result.task_name == "test_task"
    assert result.num_examples == 5
    # Fake tokenizer always decodes to "the answer is yes"
    assert all(e.prediction == "the answer is yes" for e in result.examples)
    # "yes" -> exact_match, "no" -> 0 -> 2/5
    assert abs(result.aggregate["exact_match"] - 0.4) < 1e-6
    # Each example should have its own score too
    for ex in result.examples:
        assert "exact_match" in ex.metric_scores
        assert ex.metric_scores["exact_match"] in (0.0, 1.0)


def test_run_eval_config_snapshot_includes_inference() -> None:
    cfg = _make_cfg(limit=2, batch_size=1, max_new_tokens=8, seed=99)
    ds = _make_dataset([{"q": "q1", "a": "yes"}, {"q": "q2", "a": "no"}])
    result = run_eval(cfg, FakeLM(), FakeTokenizer(), ds, ["exact_match"])
    snap = result.config_snapshot
    assert snap["inference"]["batch_size"] == 1
    assert snap["inference"]["max_new_tokens"] == 8
    assert snap["inference"]["seed"] == 99
    assert snap["inference"]["temperature"] == 0.0


def test_run_eval_handles_int_label_field() -> None:
    """GLUE-style datasets store labels as ints. The runner should coerce
    them to strings before passing to metrics.
    """
    cfg = _make_cfg(limit=2, batch_size=2)
    ds = _make_dataset([{"q": "q1", "a": 1}, {"q": "q2", "a": 0}])
    result = run_eval(cfg, FakeLM(), FakeTokenizer(), ds, ["exact_match"])
    # The references should now be strings, not ints
    assert all(isinstance(e.reference, str) for e in result.examples)
