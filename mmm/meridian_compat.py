"""Compatibility helpers for Meridian result objects across supported releases."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def summarize_channel_tensor(tensor: Any, channels: list[str]) -> dict[str, dict[str, float]]:
    """Summarize a Meridian posterior tensor across chain and draw dimensions."""
    import numpy as np

    values = np.asarray(tensor.numpy() if hasattr(tensor, "numpy") else tensor)
    if values.ndim < 3 or values.shape[-1] != len(channels):
        raise ValueError(
            "Expected a posterior tensor ending in one value per channel; "
            f"got shape {values.shape} for {len(channels)} channels"
        )

    return {
        channel: {
            "mean": float(values[..., index].mean()),
            "std": float(values[..., index].std()),
            "ci_lower": float(np.percentile(values[..., index], 5)),
            "ci_upper": float(np.percentile(values[..., index], 95)),
        }
        for index, channel in enumerate(channels)
    }


def extract_channel_contributions(tensor: Any, channels: list[str]) -> dict[str, dict[str, float]]:
    """Aggregate incremental outcome and normalize channel contributions."""
    import numpy as np

    values = np.asarray(tensor.numpy() if hasattr(tensor, "numpy") else tensor)
    if values.ndim < 3 or values.shape[-1] != len(channels):
        raise ValueError(
            "Expected incremental outcome ending in one value per channel; "
            f"got shape {values.shape} for {len(channels)} channels"
        )

    posterior_mean = values.mean(axis=(0, 1))
    if posterior_mean.ndim > 1:
        channel_totals = posterior_mean.sum(axis=tuple(range(posterior_mean.ndim - 1)))
    else:
        channel_totals = posterior_mean
    total = float(channel_totals.sum())

    return {
        channel: {
            "absolute": float(channel_totals[index]),
            "percentage": float(channel_totals[index] / total * 100) if total > 0 else 0.0,
        }
        for index, channel in enumerate(channels)
    }


def extract_non_paid_contributions(
    dataset: Any, channels: list[str]
) -> dict[str, dict[str, float]]:
    """Extract organic and treatment effects from Meridian summary metrics."""
    import numpy as np

    required_variables = {"incremental_outcome", "pct_of_contribution"}
    if (
        dataset is None
        or "channel" not in dataset.coords
        or not required_variables.issubset(dataset.data_vars)
    ):
        return {}

    available_channels = {str(channel) for channel in dataset.coords["channel"].values}
    contributions = {}
    for channel in channels:
        if channel not in available_channels:
            continue

        def metric(variable: str, name: str) -> float:
            selected = dataset[variable].sel(
                channel=channel,
                distribution="posterior",
                metric=name,
            )
            return float(np.asarray(selected.values, dtype=float).mean())

        contributions[channel] = {
            "absolute": metric("incremental_outcome", "mean"),
            "percentage": metric("pct_of_contribution", "mean"),
            "ci_lower": metric("incremental_outcome", "ci_lo"),
            "ci_upper": metric("incremental_outcome", "ci_hi"),
        }
    return contributions


def extract_predictive_accuracy(dataset: Any) -> dict[str, float]:
    """Normalize Meridian's predictive-accuracy xarray dataset."""
    import numpy as np

    if dataset is None or "metric" not in dataset.coords or "value" not in dataset.data_vars:
        return {}

    aliases = {
        "rsquared": "r_squared",
        "mape": "mape",
        "wmape": "wmape",
    }
    metrics: dict[str, float] = {}
    for raw_name in dataset.coords["metric"].values:
        normalized = str(raw_name).lower().replace("_", "")
        output_name = aliases.get(normalized)
        if output_name is None:
            continue
        values = np.asarray(dataset.sel(metric=raw_name)["value"].values, dtype=float)
        metrics[output_name] = float(values.mean())
    return metrics


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

            config_free_charts = []
            for child in charts:
                if hasattr(child, "config") and hasattr(child, "copy"):
                    child = child.copy(deep=True)
                    child.config = alt.Undefined
                config_free_charts.append(child)
            chart = alt.vconcat(*config_free_charts)

    if hasattr(chart, "save"):
        chart.save(str(output_path), scale_factor=1.5)
    elif hasattr(chart, "savefig"):
        chart.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        raise TypeError(f"Unsupported chart type: {type(chart).__name__}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Chart output was not created: {output_path}")
