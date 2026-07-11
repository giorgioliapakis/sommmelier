"""Tests for release-sensitive Meridian result adapters."""

from enum import Enum, auto
from types import SimpleNamespace

import pandas as pd

from mmm.meridian_compat import (
    extract_optimization_result,
    extract_rhat_diagnostics,
    serialize_model_review,
)


def test_extracts_max_rhat_summary():
    summary = pd.DataFrame({"max_rhat": [1.02, 5.46], "n_params": [4, 12]})

    diagnostics = extract_rhat_diagnostics(summary)

    assert diagnostics["convergence_ok"] is False
    assert diagnostics["rhat_warnings"] == 1
    assert diagnostics["max_rhat"] == 5.46


def test_missing_rhat_shape_is_conservatively_unavailable():
    diagnostics = extract_rhat_diagnostics(pd.DataFrame({"average": [1.0]}))

    assert diagnostics["convergence_ok"] is False
    assert diagnostics["diagnostics_available"] is False


def test_serializes_review_summary():
    class Status(Enum):
        PASS = auto()
        FAIL = auto()

    class ConvergenceCheckResult:
        case = SimpleNamespace(status=Status.FAIL)
        recommendation = "Increase sampling."
        details = {"rhat": 5.46}

    review = SimpleNamespace(
        overall_status=Status.FAIL,
        summary_message="Model did not converge.",
        results=[ConvergenceCheckResult()],
    )

    serialized = serialize_model_review(review)

    assert serialized["passed"] is False
    assert serialized["checks"][0]["name"] == "Convergence"
    assert serialized["checks"][0]["status"] == "FAIL"


def test_extracts_optimizer_dataset():
    dataset = SimpleNamespace(
        coords={"channel": SimpleNamespace(values=pd.Series(["meta", "google"]))},
        __getitem__=None,
    )
    dataset = _FakeDataset(dataset.coords, {"spend": [80.0, 120.0]}, {"total_incremental_outcome": 450.0})
    result = SimpleNamespace(optimized_data=dataset)

    allocation, outcome = extract_optimization_result(result)

    assert allocation == {"meta": 80.0, "google": 120.0}
    assert outcome == 450.0


class _FakeDataset:
    def __init__(self, coords, values, attrs):
        self.coords = coords
        self._values = values
        self.attrs = attrs

    def __getitem__(self, key):
        return SimpleNamespace(values=pd.Series(self._values[key]))
