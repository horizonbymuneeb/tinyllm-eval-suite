# Usage guide

This is a worked walkthrough of the suite, from installing it to reading
the report it produces.

## 1. Install

```bash
git clone https://github.com/horizonbymuneeb/tinyllm-eval-suite.git
cd tinyllm-eval-suite
pip install -r requirements.txt
pip install -e .
```

Python 3.11+ is required (see `pyproject.toml`). GPU is optional; CPU works
fine for the example task.

## 2. Validate a task YAML

Before running anything, you can check that a task file is well-formed:

```bash
python -m tinyllm_eval validate --task examples/tasks/glue_sst2.yaml
```

A valid task prints its name, dataset, metrics, and prompt input fields. An
invalid one prints the schema error and exits non-zero — useful in CI.

## 3. Run an eval

The bundled example runs the GLUE SST-2 validation split (100 examples)
through `sshleifer/tiny-distilbert-base-cased`:

```bash
python -m tinyllm_eval run \
  --task examples/tasks/glue_sst2.yaml \
  --model sshleifer/tiny-distilbert-base-cased \
  --output reports/
```

The first run downloads the model and dataset, so give it a minute. Subsequent
runs use the HF cache and start instantly.

The CLI prints progress and, on completion, a summary table:

```
        Aggregate metrics
┏━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric       ┃   Score  ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ exact_match  │   0.4500 │
└──────────────┴──────────┘
```

Two files land in `reports/`:

- `glue_sst2_tiny-distilbert_<timestamp>.md` — human-readable report with
  config snapshot, metric table, and the first 10 examples.
- `glue_sst2_tiny-distilbert_<timestamp>.csv` — one row per example with
  prompt, prediction, reference, and per-metric scores. Diff this across
  runs to spot regressions.

## 4. Writing your own task

A task is a single YAML file. Here is the full schema, annotated:

```yaml
# Required: a short identifier; used in report filenames
name: my_task

# Optional
description: Free-text description
version: 0.1.0

# Required: which dataset to evaluate on
dataset:
  name: glue                 # HF dataset name
  config: sst2               # optional HF config name
  split: validation          # required split
  limit: 100                 # optional; cap on examples evaluated
  revision: pinned-rev       # optional; pin the dataset revision

# Optional: model defaults; can be overridden by --model on the CLI
model:
  name: sshleifer/tiny-distilbert-base-cased
  revision: main             # optional
  device: auto               # cpu | cuda | mps | auto
  dtype: float32             # float16 | bfloat16 | float32

# Required: how to build the prompt from a row
prompt:
  template: "Sentence: {sentence}\nSentiment:"
  input_fields: [sentence]   # fields read from each row
  target_field: label        # the field holding the gold answer

# Required: one or more registered metrics
metrics:
  - name: exact_match
  - name: f1
    args:
      average: macro         # metric-specific kwargs

# Optional: inference knobs
inference:
  max_new_tokens: 32
  batch_size: 8
  temperature: 0.0           # 0.0 = greedy
  top_p: 1.0
  seed: 42

# Optional: input filtering and output postprocessing
filters:
  max_input_length: 1024
  regex: "label:\\s*(\\w+)"   # extract a capture group from the model's output
```

The schema is enforced at load time — unknown keys are rejected, missing
required keys fail with a path to the offending field. See
`src/tinyllm_eval/config.py` for the canonical JSON schema.

## 5. Adding a custom metric

```python
# in my_metrics.py
from tinyllm_eval import metrics

@metrics.register  # or call register("name", fn) directly
def char_overlap(preds, refs):
    if not preds:
        return 0.0
    total = sum(len(set(p) & set(r)) for p, r in zip(preds, refs))
    return total / sum(max(len(p), len(r)) for p, r in zip(preds, refs))
```

Then reference it in your task YAML:

```yaml
metrics:
  - name: char_overlap
```

(You'll need to import your module before running the CLI; one common
pattern is to import it from `tinyllm_eval.cli`.)

## 6. Reproducibility

Two runs of the same task YAML on the same hardware should produce the
same scores, provided:

- `inference.temperature` is 0.0 (greedy decoding)
- `inference.seed` is set
- `model.revision` and `dataset.revision` are pinned
- `requirements.txt` versions are unchanged

If you see different scores across runs, check the device: CPU runs are
deterministic, but CUDA runs can vary slightly unless you also set
`torch.use_deterministic_algorithms(True)`. We don't enable that by default
because it can crash on some operations.

## 7. CI

The bundled `.github/workflows/ci.yml` runs `pytest -q` and `ruff check`
on every push. It does not run the example eval (that would download a
model in CI), only the unit tests.
