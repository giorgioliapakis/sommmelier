"""Generate synthetic weekly panels and explicit simulator ground truth."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCENARIOS = (
    "baseline",
    "kpi_only",
    "national",
    "zero_effect",
    "correlated",
    "structural_break",
    "missing_week",
    "duplicate",
    "non_finite",
)
INVALID_SCENARIOS = {"missing_week", "duplicate", "non_finite"}


def generate_scenario(
    scenario: str = "baseline", *, seed: int = 42, n_geos: int = 4, n_weeks: int = 104
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate finite-lag media effects; these are not fitted Meridian outputs."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if n_geos < 1 or n_weeks < 26:
        raise ValueError("Require at least one geo and 26 weeks")
    if scenario == "national":
        n_geos = 1
    rng = np.random.default_rng(seed)
    channels = ["meta", "search", "video"]
    spend = rng.uniform(200, 1600, size=(n_geos, n_weeks, 3))
    if scenario == "correlated":
        spend[:, :, 1] = spend[:, :, 0]
    amplitudes = np.array([130.0, 100.0, 80.0])
    if scenario == "zero_effect":
        amplitudes[2] = 0.0
    decays = np.array([0.5, 0.2, 0.7])
    execution = np.zeros_like(spend)
    weights = sum(decays**lag for lag in range(9))
    for lag in range(9):
        if lag == 0:
            execution += spend
        else:
            execution[:, lag:] += spend[:, :-lag] * decays**lag
    execution /= weights
    effects = amplitudes * execution / (execution + 700.0)
    if scenario == "structural_break":
        effects[:, n_weeks // 2 :, 0] *= 0.25
    weeks = np.arange(n_weeks)
    seasonality = np.sin(2 * np.pi * weeks / 52)
    baseline = 400 + np.arange(n_geos)[:, None] * 40 + 35 * seasonality
    noise = rng.normal(0, 5, size=(n_geos, n_weeks))
    kpi = baseline + effects.sum(axis=2) + noise
    dates = pd.date_range("2023-01-02", periods=n_weeks, freq="7D")
    frame = pd.DataFrame(
        {
            "time": np.tile(dates, n_geos),
            "geo": np.repeat([f"Geo{i}" for i in range(n_geos)], n_weeks),
            "population": np.repeat(100_000 + np.arange(n_geos) * 20_000, n_weeks),
            "conversions": kpi.ravel(),
            "seasonality_control": np.tile(seasonality, n_geos),
        }
    )
    for i, channel in enumerate(channels):
        frame[f"{channel}_spend"] = spend[:, :, i].ravel()
        frame[f"{channel}_impressions"] = spend[:, :, i].ravel() * 100
    monetary = scenario != "kpi_only"
    if monetary:
        frame["revenue_per_conversion"] = 20.0
    totals = spend.sum(axis=(0, 1))
    increments = effects.sum(axis=(0, 1))
    truth = {
        "schema_version": 1,
        "scenario": scenario,
        "seed": seed,
        "n_geos": n_geos,
        "n_weeks": n_weeks,
        "expected_validation": "reject" if scenario in INVALID_SCENARIOS else "pass",
        "roi_is_monetary": monetary,
        "true_roi": dict(
            zip(channels, (increments * (20 if monetary else 1) / totals).tolist(), strict=True)
        ),
        "true_incremental_kpi": dict(zip(channels, increments.tolist(), strict=True)),
        "simulator": {
            "max_lag": 8,
            "decays": decays.tolist(),
            "half_saturation_spend": 700,
            "noise_sd": 5,
        },
        "limitations": "Controlled simulator; passing does not prove causal validity on customer data. Correlated and structural-break cases are stress tests, not recovery acceptance cases.",
    }
    if scenario == "missing_week":
        frame = frame.drop(index=10).reset_index(drop=True)
    elif scenario == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif scenario == "non_finite":
        frame.loc[0, "meta_spend"] = np.inf
    return frame, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS if args.scenario == "all" else (args.scenario,):
        frame, truth = generate_scenario(scenario, seed=args.seed)
        frame.to_csv(args.output_dir / f"{scenario}.csv", index=False)
        (args.output_dir / f"{scenario}.truth.json").write_text(
            json.dumps(truth, indent=2, allow_nan=False)
        )
        print(f"{scenario}: {len(frame)} rows; expected validation={truth['expected_validation']}")


if __name__ == "__main__":
    main()
