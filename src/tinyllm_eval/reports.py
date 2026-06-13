"""Markdown and CSV report writer.

Writes:
  - `<output>/<task>_<model>_<timestamp>.md`: human-readable summary with
    per-metric table, config snapshot, and the first N examples.
  - `<output>/<task>_<model>_<timestamp>.csv`: one row per example, with
    columns `index, prompt, prediction, reference, <each metric>...`. This
    is the file you diff across runs to detect regressions.
"""

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path

from tinyllm_eval.runner import EvalResult


_PREVIEW_N = 10


def _timestamp() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def write_report(result: EvalResult, output_dir: str | Path) -> tuple[Path, Path]:
    """Write Markdown and CSV reports. Returns the (md_path, csv_path) tuple."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    stem = f"{_safe_slug(result.task_name)}_{_safe_slug(result.model_name)}_{_timestamp()}"
    md_path = out / f"{stem}.md"
    csv_path = out / f"{stem}.csv"

    _write_markdown(result, md_path)
    _write_csv(result, csv_path)
    return md_path, csv_path


def _write_markdown(result: EvalResult, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Eval report: `{result.task_name}`")
    lines.append("")
    lines.append(f"**Model:** `{result.model_name}`  ")
    lines.append(f"**Examples:** {result.num_examples}  ")
    lines.append(f"**Generated:** {_timestamp()}  ")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    if result.aggregate:
        lines.append("| Metric | Score |")
        lines.append("|:-------|------:|")
        for k, v in sorted(result.aggregate.items()):
            lines.append(f"| `{k}` | {v:.4f} |")
    else:
        lines.append("_(no metrics computed)_")
    lines.append("")
    lines.append("## Config snapshot")
    lines.append("")
    lines.append("```yaml")
    import json

    lines.append(json.dumps(result.config_snapshot, indent=2))
    lines.append("```")
    lines.append("")
    n = min(_PREVIEW_N, result.num_examples)
    if n:
        lines.append(f"## First {n} examples")
        lines.append("")
        for ex in result.examples[:n]:
            lines.append(f"### Example {ex.index}")
            lines.append("")
            lines.append("**Prompt:**")
            lines.append("")
            lines.append("```")
            lines.append(ex.prompt)
            lines.append("```")
            lines.append("")
            lines.append(f"**Prediction:** `{ex.prediction}`")
            lines.append("")
            lines.append(f"**Reference:** `{ex.reference}`")
            lines.append("")
            if ex.metric_scores:
                lines.append("**Scores:**")
                for k, v in ex.metric_scores.items():
                    lines.append(f"- `{k}`: {v:.4f}")
                lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(result: EvalResult, path: Path) -> None:
    # Collect all metric names in stable order (union across examples)
    metric_names: list[str] = []
    seen: set[str] = set()
    for ex in result.examples:
        for k in ex.metric_scores:
            if k not in seen:
                metric_names.append(k)
                seen.add(k)

    fieldnames = ["index", "prompt", "prediction", "reference", *metric_names]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ex in result.examples:
            row: dict[str, str | int | float] = {
                "index": ex.index,
                "prompt": ex.prompt,
                "prediction": ex.prediction,
                "reference": ex.reference,
            }
            for m in metric_names:
                row[m] = ex.metric_scores.get(m, "")
            writer.writerow(row)


__all__ = ["write_report"]
