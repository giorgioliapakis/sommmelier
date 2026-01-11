"""Main CLI entry point for Sommmelier."""

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="sommmelier",
    help="AI-driven Marketing Mix Modeling powered by Google Meridian",
    no_args_is_help=True,
)
console = Console()


def find_latest_results(outputs_dir: Path = Path("outputs")) -> Path | None:
    """Find the most recent results file."""
    results_files = list(outputs_dir.glob("full_results_*.json"))
    if not results_files:
        results_files = list(outputs_dir.glob("results_*.json"))
    if not results_files:
        return None
    results_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return results_files[0]


@app.command()
def validate(
    data_path: Annotated[Path, typer.Argument(help="Path to MMM data CSV")],
    kpi_column: Annotated[str, typer.Option(help="KPI column name")] = "conversions",
    date_column: Annotated[str, typer.Option(help="Date column name")] = "date",
    geo_column: Annotated[str, typer.Option(help="Geography column name")] = "geo",
):
    """Validate a dataset for MMM readiness."""
    from mmm.data import load_mmm_data, validate_dataset
    from mmm.data.schema import DataConfig

    console.print(f"\n[bold]Validating:[/bold] {data_path}\n")

    try:
        config = DataConfig(
            kpi_column=kpi_column,
            date_column=date_column,
            geo_column=geo_column,
        )
        dataset = load_mmm_data(data_path, config)
        console.print(dataset.summary())

        report = validate_dataset(dataset)
        console.print(report.summary())

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def run(
    data_path: Annotated[Path, typer.Argument(help="Path to MMM data CSV")],
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("outputs"),
    kpi_column: Annotated[str, typer.Option(help="KPI column name")] = "conversions",
    n_chains: Annotated[int, typer.Option(help="Number of MCMC chains")] = 4,
    n_keep: Annotated[int, typer.Option(help="Samples to keep per chain")] = 500,
):
    """Run the MMM model on a dataset (local, no GPU)."""
    from mmm.data import load_mmm_data
    from mmm.data.schema import DataConfig
    from mmm.model import AutoMMM, ModelConfig
    from mmm.analysis.reports import generate_report

    console.print(f"\n[bold]Running Sommmelier[/bold]\n")
    console.print(f"Data: {data_path}")
    console.print(f"Output: {output_dir}\n")

    with console.status("Loading data..."):
        config = DataConfig(kpi_column=kpi_column)
        dataset = load_mmm_data(data_path, config)
        console.print(f"[green]✓[/green] Loaded {dataset.n_time_periods} time periods, {dataset.n_geos} geos")

    model_config = ModelConfig(
        n_chains=n_chains,
        n_keep=n_keep,
        output_dir=output_dir,
    )

    mmm = AutoMMM(dataset, model_config)

    with console.status("Preparing model..."):
        mmm.prepare()
        console.print("[green]✓[/green] Model prepared")

    console.print("\n[bold]Fitting model...[/bold] (this may take 10-30 minutes)\n")
    results = mmm.fit()

    console.print("\n[bold]Results[/bold]\n")
    console.print(results.summary())

    output_dir.mkdir(parents=True, exist_ok=True)
    mmm.save(output_dir / "model.pkl")
    console.print(f"\n[green]✓[/green] Model saved to {output_dir / 'model.pkl'}")

    report = generate_report(mmm, output_dir / "report.md")
    console.print(f"[green]✓[/green] Report saved to {output_dir / 'report.md'}")


@app.command()
def analyze(
    results_path: Annotated[Optional[Path], typer.Argument(help="Path to results JSON (uses latest if not specified)")] = None,
    output_json: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
):
    """Analyze MMM results and generate recommendations."""
    from mmm.recommendations import generate_analysis, format_report_for_claude

    if results_path is None:
        results_path = find_latest_results()

    if not results_path or not results_path.exists():
        console.print("[red]Error:[/red] No results file found.")
        console.print("Run: modal run modal_mmm_full.py --data <your_data.csv>")
        raise typer.Exit(1)

    console.print(f"Analyzing: {results_path}")
    report = generate_analysis(results_path)

    if output_json:
        output = {
            "timestamp": report.timestamp,
            "summary": report.summary,
            "recommendations": [
                {
                    "category": r.category,
                    "priority": r.priority,
                    "title": r.title,
                    "detail": r.detail,
                    "action": r.action,
                    "impact": r.impact,
                }
                for r in report.recommendations
            ],
            "budget_reallocation": report.budget_reallocation,
            "model_health": report.model_health,
            "week_over_week": report.week_over_week,
        }
        console.print(json.dumps(output, indent=2))
    else:
        console.print(format_report_for_claude(report))

        analysis_path = results_path.with_name(
            results_path.stem.replace("full_results", "analysis") + ".txt"
        )
        analysis_path.write_text(format_report_for_claude(report))
        console.print(f"\nAnalysis saved to: {analysis_path}")


@app.command()
def report(
    results_path: Annotated[Path, typer.Argument(help="Path to results JSON")],
    open_browser: Annotated[bool, typer.Option("--open", help="Open in browser")] = False,
):
    """Generate HTML report from MMM results."""
    from mmm.analysis.visualize import generate_html_report

    if not results_path.exists():
        console.print(f"[red]Error:[/red] File not found: {results_path}")
        raise typer.Exit(1)

    console.print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        results = json.load(f)

    report_path = results_path.with_suffix(".html")
    generate_html_report(results, report_path)
    console.print(f"[green]✓[/green] Report generated: {report_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{report_path.absolute()}")
        console.print("Opened in browser")


@app.command()
def quality(
    show_history: Annotated[bool, typer.Option("--history", help="Show full history")] = False,
):
    """Check model quality history and trends."""
    from mmm.tracking import ModelQualityTracker

    tracker = ModelQualityTracker()

    if show_history:
        console.print("[bold]MODEL QUALITY HISTORY[/bold]")
        console.print("=" * 60)
        for i, run in enumerate(tracker.history, 1):
            console.print(f"\nRun {i}: {run.get('timestamp', 'Unknown')}")
            console.print(f"  Data: {run.get('data_file', 'N/A')}")
            console.print(f"  Periods: {run.get('n_time_periods', 'N/A')}")
            console.print(f"  R-squared: {run.get('r_squared', 'N/A')}")
            console.print(f"  MAPE: {run.get('mape', 'N/A')}")
            console.print(f"  Convergence: {'OK' if run.get('convergence_ok') else 'WARNING'}")
    else:
        console.print(tracker.generate_quality_report())


@app.command()
def optimize(
    model_path: Annotated[Path, typer.Argument(help="Path to saved model")],
    budget: Annotated[Optional[float], typer.Option(help="Total budget to optimize")] = None,
):
    """Run budget optimization on a fitted model."""
    from mmm.model import AutoMMM

    console.print(f"\n[bold]Budget Optimization[/bold]\n")

    with console.status("Loading model..."):
        mmm = AutoMMM.load(model_path)

    current_spend = mmm.dataset.total_spend
    target_budget = budget or current_spend

    console.print(f"Current total spend: ${current_spend:,.2f}")
    console.print(f"Optimizing for budget: ${target_budget:,.2f}\n")

    with console.status("Running optimization..."):
        allocation = mmm.optimize_budget(budget=target_budget)

    table = Table(title="Optimal Budget Allocation")
    table.add_column("Channel", style="cyan")
    table.add_column("Current", justify="right")
    table.add_column("Optimal", justify="right")
    table.add_column("Change", justify="right")

    config = mmm.dataset.config
    df = mmm.dataset.df
    current_allocation = {}
    for ch in config.media_channels:
        spend_col = ch["spend_column"] if isinstance(ch, dict) else ch.spend_column
        channel_name = ch["name"] if isinstance(ch, dict) else ch.name
        current_allocation[channel_name] = df[spend_col].sum()

    for channel, optimal in allocation.items():
        current = current_allocation.get(channel, 0)
        change = ((optimal - current) / current * 100) if current > 0 else 0
        change_str = f"{change:+.1f}%" if current > 0 else "N/A"

        table.add_row(
            channel,
            f"${current:,.0f}",
            f"${optimal:,.0f}",
            change_str,
        )

    console.print(table)


@app.command()
def insights(
    model_path: Annotated[Path, typer.Argument(help="Path to saved model")],
):
    """Generate insights from a fitted model."""
    from mmm.model import AutoMMM
    from mmm.analysis import generate_insights

    with console.status("Loading model..."):
        mmm = AutoMMM.load(model_path)

    if not mmm.results:
        console.print("[red]Error:[/red] Model has no results. Ensure it was fitted.")
        raise typer.Exit(1)

    config = mmm.dataset.config
    df = mmm.dataset.df
    channel_spend = {}
    for ch in config.media_channels:
        spend_col = ch["spend_column"] if isinstance(ch, dict) else ch.spend_column
        channel_name = ch["name"] if isinstance(ch, dict) else ch.name
        channel_spend[channel_name] = df[spend_col].sum()

    insights_list = generate_insights(mmm.results, channel_spend)

    console.print("\n[bold]MMM Insights[/bold]\n")

    for insight in insights_list:
        priority_color = {
            "high": "red",
            "medium": "yellow",
            "low": "blue",
        }[insight.priority.value]

        console.print(f"[{priority_color}][{insight.priority.value.upper()}][/{priority_color}] {insight.title}")
        console.print(f"  {insight.description}")
        console.print(f"  [green]→[/green] {insight.recommendation}")
        console.print()


if __name__ == "__main__":
    app()
