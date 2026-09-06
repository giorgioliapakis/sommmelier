"""Measure synthetic effect recovery without declaring a model decision-ready."""

import argparse
import json
from pathlib import Path
from typing import Any

from mmm.result_manifest import decision_readiness, finite_number


def score_recovery(results: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Score channel ROI and interval coverage against one simulator realization."""
    if truth.get("expected_validation") != "pass":
        raise ValueError("Invalid-input scenarios must be rejected, not fitted")
    if results.get("metadata", {}).get("roi_is_monetary") is not truth["roi_is_monetary"]:
        raise ValueError("Result and ground-truth outcome units do not match")
    actual = results.get("roi", {})
    expected = truth["true_roi"]
    if set(actual) != set(expected):
        raise ValueError("Result and ground-truth channels do not match")
    channels = {}
    for channel, true_value in expected.items():
        estimate = actual[channel]
        mean, lower, upper = (estimate.get(key) for key in ("mean", "ci_lower", "ci_upper"))
        if not all(finite_number(v) for v in (mean, lower, upper, true_value)):
            raise ValueError(f"Missing or non-finite recovery metrics for {channel}")
        if lower > upper:
            raise ValueError(f"Inverted interval for {channel}")
        channels[channel] = {
            "truth": true_value,
            "estimate": mean,
            "absolute_error": abs(mean - true_value),
            "interval_covers_truth": lower <= true_value <= upper,
            "interval_width": upper - lower,
        }
    ready, reason = decision_readiness(results)
    return {
        "scenario": truth["scenario"],
        "seed": truth["seed"],
        "decision_ready": ready,
        "readiness_reason": reason,
        "channels": channels,
        "mean_absolute_error": sum(c["absolute_error"] for c in channels.values()) / len(channels),
        "interval_coverage": sum(c["interval_covers_truth"] for c in channels.values())
        / len(channels),
        "interpretation": "Descriptive single-run scores; acceptance requires multiple seeds and predefined tolerances. These scores do not override readiness gates.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("truth", type=Path)
    args = parser.parse_args()
    scores = score_recovery(
        json.loads(args.results.read_text()), json.loads(args.truth.read_text())
    )
    print(json.dumps(scores, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
