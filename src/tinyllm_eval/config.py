"""YAML task-config loader with JSON-schema validation.

A *task* is a single evaluation definition: which dataset, which model fields
(input/output), which prompt template, which metrics, and which post-processing
hooks. Tasks live as YAML files in `examples/tasks/` (or wherever the user
points the CLI at).

This module does three things:
  1. Parse YAML -> dict.
  2. Validate the dict against a JSON schema. Unknown keys are rejected.
  3. Materialize the dict as a typed `TaskConfig` dataclass.

Anything more sophisticated (template rendering, dataset-aware defaults) lives
in `loaders.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


# ----------------------------------------------------------------------- #
# Schema
# ----------------------------------------------------------------------- #

TASK_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TaskConfig",
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "dataset", "prompt", "metrics"],
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 200},
        "description": {"type": "string"},
        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "dataset": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "split"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "config": {"type": "string"},
                "split": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1},
                "revision": {"type": "string"},
            },
        },
        "model": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "revision": {"type": "string"},
                "device": {"type": "string", "enum": ["cpu", "cuda", "mps", "auto"]},
                "dtype": {"type": "string", "enum": ["float16", "bfloat16", "float32"]},
            },
        },
        "prompt": {
            "type": "object",
            "additionalProperties": False,
            "required": ["template"],
            "properties": {
                "template": {"type": "string", "minLength": 1},
                "input_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "target_field": {"type": "string"},
            },
        },
        "metrics": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "args": {"type": "object"},
                },
            },
        },
        "inference": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 2048},
                "batch_size": {"type": "integer", "minimum": 1, "maximum": 64},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "seed": {"type": "integer", "minimum": 0},
            },
        },
        "filters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "max_input_length": {"type": "integer", "minimum": 1},
                "regex": {"type": "string"},
            },
        },
    },
}


# ----------------------------------------------------------------------- #
# Typed dataclasses
# ----------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str
    config: str | None = None
    limit: int | None = None
    revision: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str | None = None
    revision: str | None = None
    device: str = "auto"
    dtype: str = "float32"


@dataclass(frozen=True)
class PromptSpec:
    template: str
    input_fields: list[str] = field(default_factory=list)
    target_field: str | None = None


@dataclass(frozen=True)
class MetricSpec:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceSpec:
    max_new_tokens: int = 32
    batch_size: int = 8
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class FilterSpec:
    max_input_length: int | None = None
    regex: str | None = None


@dataclass(frozen=True)
class TaskConfig:
    name: str
    dataset: DatasetSpec
    prompt: PromptSpec
    metrics: list[MetricSpec]
    model: ModelSpec = field(default_factory=ModelSpec)
    description: str = ""
    version: str = "0.1.0"
    inference: InferenceSpec = field(default_factory=InferenceSpec)
    filters: FilterSpec = field(default_factory=FilterSpec)
    source_path: Path | None = None

    @property
    def metric_names(self) -> list[str]:
        return [m.name for m in self.metrics]


# ----------------------------------------------------------------------- #
# Loader
# ----------------------------------------------------------------------- #


class TaskConfigError(ValueError):
    """Raised when a task YAML is malformed, fails validation, or is missing required fields."""


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise TaskConfigError(f"{ctx}: missing required key '{key}'")
    return d[key]


def load_task_config(path: str | Path) -> TaskConfig:
    """Load and validate a task YAML file. Returns a typed `TaskConfig`.

    Raises:
        FileNotFoundError: if `path` does not exist.
        TaskConfigError: if the file is invalid YAML, fails schema validation,
            or has a structural problem.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Task config not found: {p}")

    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise TaskConfigError(f"{p}: invalid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise TaskConfigError(
            f"{p}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )

    validator = Draft202012Validator(TASK_SCHEMA)
    errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        msgs = []
        for err in errors:
            loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
            msgs.append(f"  at {loc}: {err.message}")
        raise TaskConfigError(f"{p}: schema validation failed:\n" + "\n".join(msgs))

    # Materialize
    ds_raw = _require(raw, "dataset", str(p))
    prompt_raw = _require(raw, "prompt", str(p))
    metrics_raw = _require(raw, "metrics", str(p))
    model_raw = raw.get("model") or {}
    inference_raw = raw.get("inference") or {}
    filters_raw = raw.get("filters") or {}

    dataset = DatasetSpec(
        name=ds_raw["name"],
        split=ds_raw["split"],
        config=ds_raw.get("config"),
        limit=ds_raw.get("limit"),
        revision=ds_raw.get("revision"),
    )
    model = ModelSpec(
        name=model_raw.get("name"),
        revision=model_raw.get("revision"),
        device=model_raw.get("device", "auto"),
        dtype=model_raw.get("dtype", "float32"),
    )
    prompt = PromptSpec(
        template=prompt_raw["template"],
        input_fields=list(prompt_raw.get("input_fields", [])),
        target_field=prompt_raw.get("target_field"),
    )
    metrics = [
        MetricSpec(name=m["name"], args=dict(m.get("args") or {})) for m in metrics_raw
    ]
    inference = InferenceSpec(
        max_new_tokens=inference_raw.get("max_new_tokens", 32),
        batch_size=inference_raw.get("batch_size", 8),
        temperature=inference_raw.get("temperature", 0.0),
        top_p=inference_raw.get("top_p", 1.0),
        seed=inference_raw.get("seed", 42),
    )
    filters = FilterSpec(
        max_input_length=filters_raw.get("max_input_length"),
        regex=filters_raw.get("regex"),
    )

    return TaskConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        version=raw.get("version", "0.1.0"),
        dataset=dataset,
        model=model,
        prompt=prompt,
        metrics=metrics,
        inference=inference,
        filters=filters,
        source_path=p,
    )


__all__ = [
    "TASK_SCHEMA",
    "TaskConfig",
    "TaskConfigError",
    "DatasetSpec",
    "ModelSpec",
    "PromptSpec",
    "MetricSpec",
    "InferenceSpec",
    "FilterSpec",
    "load_task_config",
]
