"""Deterministic, batched evaluation runner.

The runner is the only place that calls `model.generate`. It enforces:

  - `temperature=0` (or whatever the task says, but defaults to greedy)
  - fixed RNG seeds per run
  - batched generation (`batch_size` from the task config)
  - left-padded inputs for causal LMs (required for batched generation)
  - extracting only the newly generated tokens (no prompt echo)

The runner does NOT do any post-processing of predictions. The task YAML
can include a `filters.regex` if you want to extract the first match from
the generated text — that logic lives in `runner.postprocess`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from tinyllm_eval.config import InferenceSpec, TaskConfig
from tinyllm_eval.loaders import render_prompt


@dataclass
class EvalExample:
    """One scored example, ready to be written to a report."""
    index: int
    prompt: str
    prediction: str
    reference: str
    metric_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalResult:
    """The full output of a single eval run."""
    task_name: str
    model_name: str
    examples: list[EvalExample]
    aggregate: dict[str, float]
    config_snapshot: dict[str, Any]

    @property
    def num_examples(self) -> int:
        return len(self.examples)


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #


def postprocess(text: str, regex: str | None) -> str:
    """Strip the prompt echo and optionally extract a regex match.

    If `regex` is given, return the first capturing group of the first match
    in `text`, or the whole match if no group. If no match, return `text`
    unchanged.
    """
    text = text.strip()
    if not regex:
        return text
    m = re.search(regex, text)
    if not m:
        return text
    if m.groups():
        return m.group(1).strip()
    return m.group(0).strip()


def run_eval(
    cfg: TaskConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    metric_names: Iterable[str],
) -> EvalResult:
    """Run a full deterministic eval.

    Args:
        cfg: The validated task config.
        model: A causal LM in eval mode (as returned by `load_model_and_tokenizer`).
        tokenizer: The corresponding tokenizer.
        dataset: A HuggingFace `Dataset` whose rows expose the fields in
            `cfg.prompt.input_fields` and `cfg.prompt.target_field`.
        metric_names: Names of metrics to compute (already registered).

    Returns:
        An `EvalResult` with per-example scores and aggregate metrics.
    """
    import torch

    from tinyllm_eval import metrics

    inf = cfg.inference
    target_field = cfg.prompt.target_field or "label"
    examples: list[EvalExample] = []

    model.eval()
    torch.manual_seed(inf.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(inf.seed)

    indices = list(range(len(dataset)))
    for batch_start in range(0, len(indices), inf.batch_size):
        batch_idx = indices[batch_start : batch_start + inf.batch_size]
        batch_rows = [dataset[int(i)] for i in batch_idx]

        prompts = [
            render_prompt(cfg.prompt.template, dict(row)) for row in batch_rows
        ]
        refs = [str(row.get(target_field, "")) for row in batch_rows]

        preds = _generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            inference=inf,
        )

        for i, (prompt, pred_raw, ref, idx) in enumerate(
            zip(prompts, preds, refs, batch_idx)
        ):
            pred_clean = postprocess(pred_raw, cfg.filters.regex)
            # Truncate reference to a string if it's a number/int (common in GLUE label cols)
            ref_str = ref if not isinstance(ref, int | float) else str(ref)
            examples.append(
                EvalExample(
                    index=int(idx),
                    prompt=prompt,
                    prediction=pred_clean,
                    reference=ref_str,
                )
            )

    # Compute metrics across the whole eval
    preds_all = [e.prediction for e in examples]
    refs_all = [e.reference for e in examples]
    aggregate = metrics.compute_all(metric_names, preds_all, refs_all)
    for ex, p, r in zip(examples, preds_all, refs_all):
        ex.metric_scores = metrics.compute_all(metric_names, [p], [r])

    return EvalResult(
        task_name=cfg.name,
        model_name=cfg.model.name or "(default)",
        examples=examples,
        aggregate=aggregate,
        config_snapshot={
            "task_name": cfg.name,
            "task_version": cfg.version,
            "model": cfg.model.name,
            "model_revision": cfg.model.revision,
            "dataset": cfg.dataset.name,
            "dataset_config": cfg.dataset.config,
            "dataset_split": cfg.dataset.split,
            "dataset_limit": cfg.dataset.limit,
            "inference": {
                "max_new_tokens": inf.max_new_tokens,
                "batch_size": inf.batch_size,
                "temperature": inf.temperature,
                "top_p": inf.top_p,
                "seed": inf.seed,
            },
        },
    )


# --------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------- #


def _generate_batch(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    inference: InferenceSpec,
) -> list[str]:
    """Run batched generation. Returns the generated continuation for each
    prompt, with the prompt echo stripped.
    """
    import torch

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    do_sample = inference.temperature > 0.0
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=inference.max_new_tokens,
            do_sample=do_sample,
            temperature=inference.temperature if do_sample else 1.0,
            top_p=inference.top_p if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = enc["input_ids"].shape[1]
    decoded: list[str] = []
    for i in range(out.shape[0]):
        new_tokens = out[i, input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        decoded.append(text)
    return decoded


__all__ = ["EvalExample", "EvalResult", "run_eval", "postprocess"]
