# tinyllm-eval-suite

> Reproducible evaluation harness for small (sub-7B) language models on task-specific benchmarks.

`tinyllm-eval-suite` is a CLI for running deterministic, version-pinned, batched evaluation of
small HuggingFace causal language models against user-defined tasks. It produces per-task
Markdown and CSV reports that can be diffed across model versions to detect regressions.

## Why it exists

Evaluating tiny LLMs (TinyLlama, Phi-2, Qwen-1.5B, Gemma-2B, Mistral-7B) on a custom task
should not require spinning up `lm-eval-harness`, a leaderboard server, or a notebook. It
should be one command and one YAML file.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

The second line is only needed if you want the `tinyllm-eval` console script and
`python -m tinyllm_eval` to work without setting `PYTHONPATH`.

## Quick start

```bash
python -m tinyllm_eval run \
  --task examples/tasks/glue_sst2.yaml \
  --model sshleifer/tiny-distilbert-base-cased \
  --output reports/
```

See [`docs/usage.md`](./docs/usage.md) for the full task-config reference and a worked
walkthrough.

## What it does

- **Loads a task definition** from a YAML file. The schema is strict — unknown keys fail
  validation. See `src/tinyllm_eval/config.py`.
- **Loads the model and dataset** from HuggingFace Hub with pinned revisions.
  See `src/tinyllm_eval/loaders.py`.
- **Runs the eval** in batches with `temperature=0`, fixed seeds, and deterministic token
  IDs. See `src/tinyllm_eval/runner.py`.
- **Scores outputs** with a pluggable metric registry: `exact_match`, `f1`, `bleu`,
  `token_accuracy`, and any custom scorer you register. See `src/tinyllm_eval/metrics.py`.
- **Writes reports** as Markdown (human-readable summary) and CSV (per-row scores for
  diffing across runs). See `src/tinyllm_eval/reports.py`.

## What it deliberately does not do

- Not a training framework. Use `transformers` or `trl` for that.
- Not a model hub or leaderboard. Use HuggingFace Hub for hosting.
- Not a replacement for `lm-eval-harness`. This is the smaller, single-machine, no-telemetry
  alternative for when you have one model and one task and want results in 30 seconds.
- Not a multi-GPU orchestrator. Run multiple invocations if you want to fan out.

## Module map

- `tinyllm_eval.config` — YAML task loader with JSON-schema validation.
- `tinyllm_eval.loaders` — HuggingFace Hub model + dataset loaders with version pinning.
- `tinyllm_eval.runner` — Deterministic eval loop (batched, seed-fixed, temperature=0).
- `tinyllm_eval.metrics` — Pluggable metric registry.
- `tinyllm_eval.reports` — Markdown and CSV report writer.
- `tinyllm_eval.cli` — `typer` CLI.

## Development

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

CI runs `pytest` and `ruff` on every push (see `.github/workflows/ci.yml`).

## License

MIT — see [`LICENSE`](./LICENSE).
