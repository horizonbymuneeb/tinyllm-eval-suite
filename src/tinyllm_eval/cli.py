"""`typer`-based CLI for `tinyllm-eval-suite`.

Usage:
    python -m tinyllm_eval run --task PATH --model NAME [--output DIR] [--offline]
    python -m tinyllm_eval metrics                       # list registered metrics
    python -m tinyllm_eval validate --task PATH          # validate a YAML without running

The CLI is intentionally thin: all the real logic lives in the library
modules. Keep it that way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tinyllm_eval import metrics as _metrics
from tinyllm_eval.config import TaskConfigError, load_task_config
from tinyllm_eval.loaders import configure_hf_env, load_eval_dataset, load_model_and_tokenizer
from tinyllm_eval.reports import write_report
from tinyllm_eval.runner import run_eval

app = typer.Typer(
    name="tinyllm-eval",
    help="Reproducible evaluation harness for small (sub-7B) language models.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command()
def run(
    task: Path = typer.Option(..., "--task", "-t", help="Path to task YAML."),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="HF model id. Overrides task config."
    ),
    output: Path = typer.Option(
        Path("reports/"), "--output", "-o", help="Output directory for reports."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Run in offline mode (no HF Hub network calls)."
    ),
) -> None:
    """Run a full eval: load model + dataset, generate, score, write report."""
    configure_hf_env(offline=offline)
    cfg = load_task_config(task)
    if model:
        # Override model name; keep the rest of the model spec
        from dataclasses import replace
        from tinyllm_eval.config import ModelSpec

        cfg = replace(cfg, model=ModelSpec(name=model, device=cfg.model.device, dtype=cfg.model.dtype))
    console.print(f"[bold]Task:[/bold]   {cfg.name}")
    console.print(f"[bold]Model:[/bold]  {cfg.model.name or model}")
    console.print(f"[bold]Metrics:[/bold] {', '.join(cfg.metric_names)}")
    console.print(f"[bold]Limit:[/bold]  {cfg.dataset.limit or 'all'}")
    console.print("")

    with console.status("Loading model..."):
        mdl, tok = load_model_and_tokenizer(cfg.model, default_model=model)
    with console.status("Loading dataset..."):
        ds = load_eval_dataset(cfg.dataset)
    console.print(f"[green]Loaded {len(ds)} examples.[/green]")

    with console.status("Running eval..."):
        result = run_eval(cfg, mdl, tok, ds, cfg.metric_names)

    md_path, csv_path = write_report(result, output)
    console.print(f"[bold green]Done.[/bold green]")
    console.print(f"  Markdown: {md_path}")
    console.print(f"  CSV:      {csv_path}")
    console.print("")

    table = Table(title="Aggregate metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    for k, v in sorted(result.aggregate.items()):
        table.add_row(k, f"{v:.4f}")
    console.print(table)


@app.command()
def validate(
    task: Path = typer.Option(..., "--task", "-t", help="Path to task YAML."),
) -> None:
    """Validate a task YAML without running anything."""
    try:
        cfg = load_task_config(task)
    except (FileNotFoundError, TaskConfigError) as e:
        console.print(f"[bold red]Invalid:[/bold red] {e}")
        raise typer.Exit(code=1)
    console.print(f"[bold green]Valid:[/bold green] {cfg.name}")
    console.print(f"  dataset: {cfg.dataset.name} / {cfg.dataset.split}")
    console.print(f"  metrics: {', '.join(cfg.metric_names)}")
    console.print(f"  prompt input fields: {', '.join(cfg.prompt.input_fields)}")


@app.command()
def metrics() -> None:
    """List all registered metric names."""
    table = Table(title="Registered metrics", show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    for name in _metrics.available():
        table.add_row(name)
    console.print(table)


if __name__ == "__main__":
    app()
