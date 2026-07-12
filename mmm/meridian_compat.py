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


def extract_holdout_accuracy(dataset: Any, holdout_weeks: int) -> dict[str, Any]:
    """Extract out-of-sample metrics without averaging Train, Test, and All Data."""
    import numpy as np

    required_coords = {"metric", "evaluation_set"}
    if (
        dataset is None
        or not required_coords.issubset(dataset.coords)
        or "value" not in dataset.data_vars
    ):
        raise ValueError("Predictive accuracy omitted holdout evaluation coordinates")

    evaluation_sets = [str(value) for value in dataset.coords["evaluation_set"].values]
    test_set = next((value for value in evaluation_sets if value.lower() == "test"), None)
    if test_set is None:
        raise ValueError("Predictive accuracy omitted the Test evaluation set")

    granularities = (
        [str(value) for value in dataset.coords["geo_granularity"].values]
        if "geo_granularity" in dataset.coords
        else [None]
    )
    aliases = {"rsquared": "r_squared", "mape": "mape", "wmape": "wmape"}
    metrics: dict[str, dict[str, float]] = {}
    for granularity in granularities:
        output_name = granularity.lower().replace(" ", "_") if granularity else "all"
        output_metrics: dict[str, float] = {}
        for raw_metric in dataset.coords["metric"].values:
            metric_name = aliases.get(str(raw_metric).lower().replace("_", ""))
            if metric_name is None:
                continue
            selectors = {"evaluation_set": test_set, "metric": raw_metric}
            if granularity is not None:
                selectors["geo_granularity"] = granularity
            values = np.asarray(dataset["value"].sel(selectors).values, dtype=float)
            output_metrics[metric_name] = float(values.mean())
        metrics[output_name] = output_metrics

    return {
        "holdout_weeks": holdout_weeks,
        "evaluation_set": test_set,
        "metrics": metrics,
    }


def extract_response_curves(
    dataset: Any,
    channels: list[str],
    spend_multipliers: list[float] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Normalize Meridian response curves and select the named mean metric."""
    import numpy as np

    if (
        dataset is None
        or "channel" not in dataset.coords
        or "incremental_outcome" not in dataset.data_vars
    ):
        return {}

    available_channels = {str(channel) for channel in dataset.coords["channel"].values}
    multipliers = (
        np.asarray(dataset.coords["spend_multiplier"].values, dtype=float).tolist()
        if "spend_multiplier" in dataset.coords
        else list(spend_multipliers or [])
    )
    curves: dict[str, dict[str, list[float]]] = {}
    for channel in channels:
        if channel not in available_channels:
            continue
        values = dataset["incremental_outcome"].sel(channel=channel)
        values = _select_mean_metric(values)
        response = np.asarray(values.values, dtype=float).reshape(-1).tolist()
        if multipliers and len(response) != len(multipliers):
            raise ValueError(
                f"Response curve for '{channel}' has {len(response)} values for "
                f"{len(multipliers)} spend multipliers"
            )
        curves[channel] = {
            "spend_multiplier": [float(value) for value in multipliers],
            "response": [float(value) for value in response],
        }
    return curves


def extract_adstock_decay(data: Any, channels: list[str]) -> dict[str, dict[str, float]]:
    """Extract one-period retention or a stable mean from adstock summary rows."""
    if data is None or not hasattr(data, "columns") or "mean" not in data.columns:
        return {}

    decay: dict[str, dict[str, float]] = {}
    for channel in channels:
        channel_data = data[data["channel"] == channel] if "channel" in data.columns else data
        if len(channel_data) == 0:
            continue
        integer_data = channel_data
        if "is_int_time_unit" in channel_data.columns:
            integer_data = channel_data[channel_data["is_int_time_unit"].astype(bool)]
        one_period = (
            integer_data[integer_data["time_units"] == 1.0]
            if "time_units" in integer_data.columns
            else None
        )
        if one_period is not None and len(one_period) > 0:
            decay[channel] = {"retention_at_1_period": float(one_period["mean"].iloc[0])}
        else:
            decay[channel] = {"mean_decay": float(channel_data["mean"].mean())}
    return decay


def extract_optimal_frequency(result: Any, channels: list[str]) -> dict[str, float]:
    """Normalize tensor and xarray optimal-frequency results by channel."""
    import numpy as np

    if result is None:
        return {}
    if hasattr(result, "data_vars"):
        if "optimal_frequency" not in result.data_vars:
            return {}
        channel_coord = "rf_channel" if "rf_channel" in result.coords else "channel"
        if channel_coord not in result.coords:
            return {}
        available_channels = {str(channel) for channel in result.coords[channel_coord].values}
        frequencies = {}
        for channel in channels:
            if channel not in available_channels:
                continue
            values = result["optimal_frequency"].sel({channel_coord: channel})
            values = _select_mean_metric(values)
            frequencies[channel] = float(np.asarray(values.values, dtype=float).mean())
        return frequencies

    values = np.asarray(result.numpy() if hasattr(result, "numpy") else result, dtype=float)
    if values.ndim == 0 or values.shape[-1] != len(channels):
        raise ValueError(
            "Expected optimal frequency ending in one value per R&F channel; "
            f"got shape {values.shape} for {len(channels)} channels"
        )
    means = values.mean(axis=tuple(range(values.ndim - 1)))
    return {channel: float(means[index]) for index, channel in enumerate(channels)}


def _select_mean_metric(values: Any) -> Any:
    if "metric" not in values.dims:
        return values
    metrics = [str(metric) for metric in values.coords["metric"].values]
    mean_metric = next((metric for metric in metrics if metric.lower() == "mean"), metrics[0])
    return values.sel(metric=mean_metric)


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
