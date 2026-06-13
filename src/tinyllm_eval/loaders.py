"""HuggingFace Hub model and dataset loaders with version pinning.

The loaders are the only place that talks to `transformers` and `datasets`.
Everything else in the suite depends on these functions returning plain
Python objects (tokenizer, model, dataset) that can be reused without
re-downloading.

All network calls are routed through `huggingface_hub.snapshot_download` /
`datasets.load_dataset` so the standard `HF_HOME`, `HF_TOKEN`, and offline-mode
environment variables are respected.
"""

from __future__ import annotations

import os
from typing import Any

from tinyllm_eval.config import DatasetSpec, ModelSpec


# --------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------- #


def _resolve_dtype(name: str) -> Any:
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    spec: ModelSpec,
    default_model: str | None = None,
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer from HuggingFace.

    Args:
        spec: The model spec from the task config. `name` and `revision` are
            both optional; if absent, fall back to `default_model` and the
            latest revision.
        default_model: Used when `spec.name` is None. Typically the
            `--model` CLI flag value.

    Returns:
        (model, tokenizer) tuple. Model is moved to the resolved device and
        set to eval mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = spec.name or default_model
    if not name:
        raise ValueError(
            "No model specified: set `model.name` in the task YAML or pass --model."
        )

    tokenizer = AutoTokenizer.from_pretrained(name, revision=spec.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=spec.revision,
        torch_dtype=_resolve_dtype(spec.dtype),
    )
    model = model.to(_resolve_device(spec.device))
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------- #


def load_eval_dataset(spec: DatasetSpec) -> Any:
    """Load a HuggingFace dataset, optionally with config and limit.

    `limit` is applied as `select(range(limit))` after loading, so it works
    for any dataset size without precomputing the full index.

    Raises:
        ValueError: if dataset cannot be loaded.
    """
    from datasets import load_dataset

    try:
        if spec.config:
            ds = load_dataset(
                spec.name,
                spec.config,
                split=spec.split,
                revision=spec.revision,
            )
        else:
            ds = load_dataset(
                spec.name,
                split=spec.split,
                revision=spec.revision,
            )
    except Exception as e:
        raise ValueError(
            f"Failed to load dataset {spec.name!r} "
            f"(config={spec.config!r}, split={spec.split!r}): {e}"
        ) from e

    if spec.limit is not None and spec.limit > 0:
        n = min(spec.limit, len(ds))
        ds = ds.select(range(n))

    return ds


# --------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------- #


def render_prompt(template: str, row: dict[str, Any]) -> str:
    """Render a prompt template against a single dataset row.

    Supports `{field_name}` placeholders (str.format style). Missing fields
    raise `KeyError`. Unknown extra placeholders are left as-is so the model
    can still see the literal `{{ ... }}` if that was the intent.
    """
    try:
        return template.format(**row)
    except KeyError as e:
        raise KeyError(
            f"Prompt template references missing field {e.args[0]!r}; "
            f"available fields: {sorted(row.keys())}"
        ) from e


# --------------------------------------------------------------------- #
# Offline / cache helpers
# --------------------------------------------------------------------- #


def configure_hf_env(*, offline: bool = False, cache_dir: str | None = None) -> None:
    """Set HF env vars for a given run. No-op unless caller wants offline mode
    or a custom cache directory.

    Intended to be called once at CLI startup, before any HF imports happen
    in worker threads.
    """
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if cache_dir:
        os.environ["HF_HOME"] = cache_dir


__all__ = [
    "load_model_and_tokenizer",
    "load_eval_dataset",
    "render_prompt",
    "configure_hf_env",
]
