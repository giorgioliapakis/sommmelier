"""
Model quality tracking for Sommmelier.

Tracks model performance metrics over time to measure improvement
and identify when the model needs retuning.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ModelMetrics:
    """Metrics for a single model run."""
    timestamp: str
    data_file: str

    # Predictive accuracy
    r_squared: Optional[float] = None
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    wmape: Optional[float] = None  # Weighted MAPE

    # Data quality
    n_time_periods: int = 0
    n_geos: int = 0
    n_channels: int = 0

    # Convergence
    convergence_ok: bool = True
    rhat_warnings: int = 0

    # Confidence (avg CI width as % of mean)
    avg_roi_ci_width: Optional[float] = None


def extract_metrics_from_results(results: dict, data_file: str = "") -> ModelMetrics:
    """Extract model metrics from results JSON."""
    metadata = results.get("metadata", {})
    diagnostics = results.get("diagnostics", {})
    model_fit = results.get("model_fit", {})
    roi_data = results.get("roi", {})

    # Calculate average CI width
    ci_widths = []
    for ch, data in roi_data.items():
        if isinstance(data, dict):
            mean = data.get("mean", 0)
            ci_lo = data.get("ci_lower", 0)
            ci_hi = data.get("ci_upper", 0)
            if mean > 0:
                ci_widths.append((ci_hi - ci_lo) / mean)

    avg_ci_width = sum(ci_widths) / len(ci_widths) if ci_widths else None

    return ModelMetrics(
        timestamp=results.get("timestamp", datetime.now().isoformat()),
        data_file=data_file,
        r_squared=model_fit.get("r_squared"),
        mape=model_fit.get("mape"),
        wmape=model_fit.get("wmape"),
        n_time_periods=metadata.get("n_time_periods", 0),
        n_geos=metadata.get("n_geos", 0),
        n_channels=len(metadata.get("channels", [])),
        convergence_ok=diagnostics.get("convergence_ok", True),
        rhat_warnings=diagnostics.get("rhat_warnings", 0),
        avg_roi_ci_width=avg_ci_width,
    )


class ModelQualityTracker:
    """Tracks model quality metrics over time."""

    def __init__(self, tracking_file: Path | str = "outputs/model_quality_history.json"):
        self.tracking_file = Path(tracking_file)
        self.history: list[dict] = []
        self._load()

    def _load(self):
        """Load history from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file) as f:
                    self.history = json.load(f)
            except Exception:
                self.history = []

    def _save(self):
        """Save history to file."""
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracking_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def add_run(self, metrics: ModelMetrics):
        """Add a new run to history."""
        self.history.append(asdict(metrics))
        self._save()

    def get_trend(self, metric: str, n_runs: int = 5) -> dict:
        """Get trend for a metric over recent runs."""
        recent = self.history[-n_runs:] if len(self.history) >= n_runs else self.history

        values = [r.get(metric) for r in recent if r.get(metric) is not None]

        if len(values) < 2:
            return {"trend": "insufficient_data", "values": values}

        # Calculate trend
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if metric in ["mape", "wmape", "rhat_warnings", "avg_roi_ci_width"]:
            # Lower is better
            if second_avg < first_avg * 0.9:
                trend = "improving"
            elif second_avg > first_avg * 1.1:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            # Higher is better (r_squared, n_time_periods)
            if second_avg > first_avg * 1.1:
                trend = "improving"
            elif second_avg < first_avg * 0.9:
                trend = "degrading"
            else:
                trend = "stable"

        return {
            "trend": trend,
            "values": values,
            "first_avg": first_avg,
            "latest": values[-1] if values else None,
        }

    def generate_quality_report(self) -> str:
        """Generate a quality report for Claude to review."""
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL QUALITY TRACKING REPORT")
        lines.append("=" * 60)

        if not self.history:
            lines.append("\nNo historical data yet. Run the model to start tracking.")
            return "\n".join(lines)

        latest = self.history[-1]
        lines.append(f"\nLatest run: {latest.get('timestamp', 'Unknown')}")
        lines.append(f"Data periods: {latest.get('n_time_periods', 'N/A')}")

        # Current metrics
        lines.append("\n" + "-" * 40)
        lines.append("CURRENT MODEL METRICS")
        lines.append("-" * 40)

        r2 = latest.get("r_squared")
        mape = latest.get("mape")
        wmape = latest.get("wmape")

        if r2 is not None:
            quality = "Excellent" if r2 > 0.9 else "Good" if r2 > 0.7 else "Fair" if r2 > 0.5 else "Poor"
            lines.append(f"  R-squared: {r2:.3f} ({quality})")
        else:
            lines.append("  R-squared: Not available")

        if mape is not None:
            quality = "Excellent" if mape < 0.1 else "Good" if mape < 0.2 else "Fair" if mape < 0.3 else "Poor"
            lines.append(f"  MAPE: {mape:.1%} ({quality})")
        else:
            lines.append("  MAPE: Not available")

        if wmape is not None:
            lines.append(f"  Weighted MAPE: {wmape:.1%}")

        ci_width = latest.get("avg_roi_ci_width")
        if ci_width is not None:
            quality = "Tight" if ci_width < 0.5 else "Moderate" if ci_width < 1.0 else "Wide"
            lines.append(f"  Avg ROI CI Width: {ci_width:.1%} ({quality})")

        # Convergence
        if latest.get("convergence_ok"):
            lines.append("  Convergence: OK")
        else:
            lines.append(f"  Convergence: WARNING - {latest.get('rhat_warnings', 0)} parameters with high R-hat")

        # Trends (if enough history)
        if len(self.history) >= 3:
            lines.append("\n" + "-" * 40)
            lines.append("TRENDS (last 5 runs)")
            lines.append("-" * 40)

            for metric, label in [
                ("r_squared", "R-squared"),
                ("mape", "MAPE"),
                ("n_time_periods", "Data periods"),
                ("avg_roi_ci_width", "CI Width"),
            ]:
                trend_data = self.get_trend(metric)
                if trend_data["trend"] != "insufficient_data":
                    icon = {"improving": "↑", "degrading": "↓", "stable": "→"}.get(trend_data["trend"], "?")
                    lines.append(f"  {label}: {icon} {trend_data['trend']}")

        # Recommendations for model improvement
        lines.append("\n" + "-" * 40)
        lines.append("MODEL IMPROVEMENT RECOMMENDATIONS")
        lines.append("-" * 40)

        recommendations = []

        if latest.get("n_time_periods", 0) < 26:
            recommendations.append("- Accumulate more time periods (currently {}, need 26+)".format(
                latest.get("n_time_periods", 0)))

        if mape is not None and mape > 0.25:
            recommendations.append("- High MAPE suggests model fit issues. Consider:")
            recommendations.append("  - Adding control variables (seasonality, promotions)")
            recommendations.append("  - Adjusting priors if you have domain knowledge")

        if ci_width is not None and ci_width > 1.0:
            recommendations.append("- Wide confidence intervals. Consider:")
            recommendations.append("  - More data")
            recommendations.append("  - Informative priors based on industry benchmarks")

        if not latest.get("convergence_ok"):
            recommendations.append("- Convergence issues detected. Try:")
            recommendations.append("  - Increasing n_keep to 1000")
            recommendations.append("  - Checking for data quality issues")

        if recommendations:
            lines.extend(recommendations)
        else:
            lines.append("- Model quality looks good! No immediate improvements needed.")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def update_tracking(results_path: Path | str, data_file: str = "") -> str:
    """
    Update model quality tracking with new results.

    Returns the quality report for Claude to review.
    """
    results_path = Path(results_path)

    with open(results_path) as f:
        results = json.load(f)

    metrics = extract_metrics_from_results(results, data_file)

    tracker = ModelQualityTracker()
    tracker.add_run(metrics)

    return tracker.generate_quality_report()
