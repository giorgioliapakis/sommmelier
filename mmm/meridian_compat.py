"""Compatibility helpers for Meridian result objects across supported releases."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def extract_rhat_diagnostics(rhat_summary: Any, threshold: float = 1.1) -> dict[str, Any]:
    """Normalize Meridian's R-hat summary without assuming one release's columns."""
    if rhat_summary is None or len(rhat_summary) == 0:
        return {
            "convergence_ok": False,
            "rhat_warnings": 0,
            "max_rhat": None,
            "diagnostics_available": False,
        }

    if "max_rhat" in rhat_summary.columns:
        values = rhat_summary["max_rhat"]
    elif "rhat" in rhat_summary.columns:
        values = rhat_summary["rhat"]
    else:
        return {
            "convergence_ok": False,
            "rhat_warnings": 0,
            "max_rhat": None,
            "diagnostics_available": False,
        }

    finite_values = values.dropna()
    max_rhat = float(finite_values.max()) if len(finite_values) else None
    warnings = int((finite_values > threshold).sum())
    return {
        "convergence_ok": warnings == 0 and max_rhat is not None,
        "rhat_warnings": warnings,
        "max_rhat": max_rhat,
        "diagnostics_available": True,
        "threshold": threshold,
    }


def serialize_model_review(review: Any) -> dict[str, Any]:
    """Turn Meridian's ReviewSummary dataclass into stable JSON-compatible fields."""
    if hasattr(review, "overall_status") and hasattr(review, "results"):
        checks = []
        for result in review.results:
            name = result.__class__.__name__.removesuffix("CheckResult")
            status = getattr(getattr(result, "case", None), "status", None)
            checks.append(
                {
                    "name": name,
                    "status": getattr(status, "name", str(status)),
                    "recommendation": str(getattr(result, "recommendation", "")),
                    "details": getattr(result, "details", {}),
                }
            )
        overall_status = getattr(review.overall_status, "name", str(review.overall_status))
        return {
            "overall_status": overall_status,
            "passed": overall_status == "PASS",
            "summary": str(getattr(review, "summary_message", "")),
            "checks": checks,
        }

    if isinstance(review, Mapping):
        return {"raw_mapping": dict(review)}
    return {"raw": str(review)}


def extract_optimization_result(result: Any) -> tuple[dict[str, float], float | None]:
    """Extract channel spend and expected outcome from OptimizationResults."""
    if hasattr(result, "optimized_data"):
        dataset = result.optimized_data
        channels = [str(channel) for channel in dataset.coords["channel"].values.tolist()]
        spend = dataset["spend"].values.tolist()
        allocation = {channel: float(value) for channel, value in zip(channels, spend)}
        outcome = dataset.attrs.get("total_incremental_outcome")
        return allocation, float(outcome) if outcome is not None else None

    if hasattr(result, "optimal_spend"):
        values = result.optimal_spend
        return {str(index): float(value) for index, value in enumerate(values)}, None

    return {}, None


def save_chart(chart: Any, output_path: str | Path) -> None:
    """Persist Altair, chart mappings, or Matplotlib outputs as a real PNG."""
    output_path = Path(output_path)
    if isinstance(chart, Mapping):
        charts = list(chart.values())
        if not charts:
            raise ValueError("Chart mapping was empty")
        if len(charts) == 1:
            chart = charts[0]
        else:
            import altair as alt

            chart = alt.vconcat(*charts)

    if hasattr(chart, "save"):
        chart.save(str(output_path), scale_factor=1.5)
    elif hasattr(chart, "savefig"):
        chart.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        raise TypeError(f"Unsupported chart type: {type(chart).__name__}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Chart output was not created: {output_path}")
