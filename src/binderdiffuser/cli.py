"""Command-line interface for BinderDiffuser.

Subcommands:

    binderdiffuser run <config.yaml>
        Execute the full pipeline (RFdiffusion + MPNN + AF + filter+rank).

    binderdiffuser show-contigs <config.yaml> [--n 5]
        Print the first N contig strings the run *would* feed to RFdiffusion,
        without invoking the diffuser. Handy for sanity-checking motif setup.

    binderdiffuser report <run_dir>
        Re-emit the README-facing figures from a finished run's designs.csv.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from binderdiffuser.config import load_config_from_yaml
from binderdiffuser.diffusion.rfdiff_wrapper import RFDiffusionRunner
from binderdiffuser.pipeline import build_motif_spec, run_pipeline
from binderdiffuser.viz import metrics_violin, self_consistency_scatter

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="DEBUG-level logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """BinderDiffuser — de novo protein binder design pipeline."""
    ctx.ensure_object(dict)
    _configure_logging(verbose)


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def run(config_path: Path) -> None:
    """Execute the full pipeline from a YAML config."""
    cfg = load_config_from_yaml(config_path)
    console.rule(f"[bold]BinderDiffuser run: {cfg.run_name}")
    result = run_pipeline(cfg)

    table = Table(title="run summary", show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("backbones generated", str(result.n_backbones))
    table.add_row("sequences designed", str(result.n_sequences))
    table.add_row("predictions folded", str(result.n_predictions))
    table.add_row("records kept (filtered)", str(len(result.filtered_records)))
    table.add_row("top-k ranked", str(len(result.ranked_records)))
    table.add_row("summary csv", str(result.summary_csv))
    console.print(table)


@main.command(name="show-contigs")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-n", "--num", default=5, show_default=True, help="How many contigs to preview.")
def show_contigs(config_path: Path, num: int) -> None:
    """Print sample contig strings without running RFdiffusion."""
    cfg = load_config_from_yaml(config_path)
    spec = build_motif_spec(cfg)
    runner = RFDiffusionRunner(check_executable=False)
    contigs = runner.sample_contigs(spec, num_designs=num, seed=cfg.seed)
    console.print(f"[bold]first {num} contigs for {cfg.run_name}:")
    for i, (contig, seed) in enumerate(contigs):
        console.print(f"  [{i:02d}] seed={seed:<10} {contig}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def report(run_dir: Path) -> None:
    """Re-emit figures from an existing run directory."""
    csv_path = run_dir / "designs.csv"
    if not csv_path.exists():
        raise click.UsageError(f"no designs.csv at {csv_path}")
    df = pd.read_csv(csv_path)
    figures_dir = run_dir / "figures"
    scatter = self_consistency_scatter(df, figures_dir / "sc_scatter.png")
    violin = metrics_violin(df, figures_dir / "metrics_violin.png")
    console.print(f"[green]wrote[/green] {scatter}")
    console.print(f"[green]wrote[/green] {violin}")


if __name__ == "__main__":  # pragma: no cover
    # Seed Python's RNG so the CLI itself is reproducible across invocations
    # of show-contigs even when the user forgets to set seed in YAML.
    random.seed(0)
    main()
