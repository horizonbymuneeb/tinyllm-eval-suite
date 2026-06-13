"""Tests for `tinyllm_eval.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinyllm_eval.config import (
    DatasetSpec,
    InferenceSpec,
    MetricSpec,
    ModelSpec,
    PromptSpec,
    TaskConfig,
    TaskConfigError,
    load_task_config,
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


VALID_TASK = """\
name: glue_sst2_mini
description: Minimal sentiment classification
version: 0.1.0
dataset:
  name: glue
  config: sst2
  split: validation
  limit: 100
prompt:
  template: "Sentence: {sentence}\\nSentiment:"
  input_fields: [sentence]
  target_field: label
metrics:
  - name: exact_match
inference:
  batch_size: 4
  seed: 7
"""


@pytest.fixture
def tmp_task(tmp_path: Path) -> Path:
    p = tmp_path / "task.yaml"
    p.write_text(VALID_TASK, encoding="utf-8")
    return p


# --------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------- #


def test_load_valid_task(tmp_task: Path) -> None:
    cfg = load_task_config(tmp_task)
    assert isinstance(cfg, TaskConfig)
    assert cfg.name == "glue_sst2_mini"
    assert cfg.description == "Minimal sentiment classification"
    assert cfg.version == "0.1.0"
    assert isinstance(cfg.dataset, DatasetSpec)
    assert cfg.dataset.name == "glue"
    assert cfg.dataset.config == "sst2"
    assert cfg.dataset.split == "validation"
    assert cfg.dataset.limit == 100
    assert isinstance(cfg.prompt, PromptSpec)
    assert cfg.prompt.template.startswith("Sentence:")
    assert cfg.prompt.input_fields == ["sentence"]
    assert cfg.prompt.target_field == "label"
    assert isinstance(cfg.metrics[0], MetricSpec)
    assert cfg.metrics[0].name == "exact_match"
    assert isinstance(cfg.model, ModelSpec)
    assert cfg.model.device == "auto"  # default
    assert isinstance(cfg.inference, InferenceSpec)
    assert cfg.inference.batch_size == 4
    assert cfg.inference.seed == 7
    assert cfg.inference.temperature == 0.0  # default
    assert cfg.source_path == tmp_task.resolve()


def test_metric_names_property(tmp_task: Path) -> None:
    cfg = load_task_config(tmp_task)
    assert cfg.metric_names == ["exact_match"]


# --------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------- #


def test_minimal_task_uses_defaults(tmp_path: Path) -> None:
    """Only the required keys; everything else defaults."""
    p = tmp_path / "minimal.yaml"
    p.write_text(
        """\
name: minimal
dataset:
  name: glue
  split: train
prompt:
  template: "{x}:"
  input_fields: [x]
metrics:
  - name: exact_match
""",
        encoding="utf-8",
    )
    cfg = load_task_config(p)
    assert cfg.version == "0.1.0"
    assert cfg.inference.batch_size == 8
    assert cfg.inference.seed == 42
    assert cfg.model.device == "auto"
    assert cfg.model.dtype == "float32"
    assert cfg.filters.max_input_length is None


# --------------------------------------------------------------------- #
# Failure cases
# --------------------------------------------------------------------- #


def test_missing_required_top_level_key(tmp_path: Path) -> None:
    p = tmp_path / "no_metrics.yaml"
    p.write_text(
        """\
name: nope
dataset:
  name: glue
  split: train
prompt:
  template: "{x}:"
  input_fields: [x]
""",
        encoding="utf-8",
    )
    with pytest.raises(TaskConfigError, match="schema validation failed"):
        load_task_config(p)


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = tmp_path / "extra.yaml"
    p.write_text(
        """\
name: nope
not_a_real_field: 1
dataset:
  name: glue
  split: train
prompt:
  template: "{x}:"
  input_fields: [x]
metrics:
  - name: exact_match
""",
        encoding="utf-8",
    )
    with pytest.raises(TaskConfigError, match="schema validation failed"):
        load_task_config(p)


def test_invalid_batch_size(tmp_path: Path) -> None:
    p = tmp_path / "bad_batch.yaml"
    p.write_text(
        """\
name: bad
dataset:
  name: glue
  split: train
prompt:
  template: "{x}:"
  input_fields: [x]
metrics:
  - name: exact_match
inference:
  batch_size: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(TaskConfigError, match="schema validation failed"):
        load_task_config(p)


def test_bad_dataset_split_type(tmp_path: Path) -> None:
    p = tmp_path / "bad_split.yaml"
    p.write_text(
        """\
name: bad
dataset:
  name: glue
  split: 42
prompt:
  template: "{x}:"
  input_fields: [x]
metrics:
  - name: exact_match
""",
        encoding="utf-8",
    )
    with pytest.raises(TaskConfigError, match="schema validation failed"):
        load_task_config(p)


def test_metrics_min_items_violated(tmp_path: Path) -> None:
    p = tmp_path / "no_metrics_list.yaml"
    p.write_text(
        """\
name: bad
dataset:
  name: glue
  split: train
prompt:
  template: "{x}:"
  input_fields: [x]
metrics: []
""",
        encoding="utf-8",
    )
    with pytest.raises(TaskConfigError, match="schema validation failed"):
        load_task_config(p)


def test_metric_with_extra_args_preserved(tmp_path: Path) -> None:
    p = tmp_path / "metric_args.yaml"
    p.write_text(
        """\
name: ok
dataset:
  name: glue
  split: train
prompt:
  template: "{x}:"
  input_fields: [x]
metrics:
  - name: f1
    args:
      average: macro
""",
        encoding="utf-8",
    )
    cfg = load_task_config(p)
    assert cfg.metrics[0].name == "f1"
    assert cfg.metrics[0].args == {"average": "macro"}


# --------------------------------------------------------------------- #
# File-level errors
# --------------------------------------------------------------------- #


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_task_config(tmp_path / "does_not_exist.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "broken.yaml"
    p.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(TaskConfigError, match="invalid YAML"):
        load_task_config(p)


def test_top_level_not_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(TaskConfigError, match="must be a mapping"):
        load_task_config(p)
