"""Tests for release-sensitive Meridian result adapters."""

from enum import Enum, auto
from types import SimpleNamespace

import pandas as pd

from mmm.meridian_compat import (
    extract_channel_contributions,
    extract_optimization_result,
    extract_predictive_accuracy,
    extract_rhat_diagnostics,
    serialize_model_review,
    summarize_channel_tensor,
)


def test_summarizes_channel_posterior_tensor():
    tensor = _Array([[[1.0, 4.0], [3.0, 6.0]]])

    summary = summarize_channel_tensor(tensor, ["meta", "search"])

    assert summary["meta"]["mean"] == 2.0
    assert summary["search"]["mean"] == 5.0
    assert summary["meta"]["ci_lower"] < summary["meta"]["ci_upper"]


def test_aggregates_channel_contributions():
    tensor = _Array([[[[1.0, 3.0], [1.0, 5.0]]]])

    contributions = extract_channel_contributions(tensor, ["meta", "search"])

    assert contributions["meta"]["absolute"] == 2.0
    assert contributions["search"]["absolute"] == 8.0
    assert contributions["meta"]["percentage"] == 20.0


def test_extracts_predictive_accuracy_metrics():
    dataset = _AccuracyDataset(
        metrics=["R_Squared", "MAPE", "wMAPE"],
        values={
            "R_Squared": [0.7, 0.9],
            "MAPE": [0.1, 0.2],
            "wMAPE": [0.08, 0.12],
        },
    )

    metrics = extract_predictive_accuracy(dataset)

    assert metrics == {"r_squared": 0.8, "mape": 0.15000000000000002, "wmape": 0.1}


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
    dataset = _FakeDataset(
        dataset.coords, {"spend": [80.0, 120.0]}, {"total_incremental_outcome": 450.0}
    )
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


class _Array:
    def __init__(self, values):
        self._values = values

    def numpy(self):
        import numpy as np

        return np.asarray(self._values)


class _AccuracyDataset:
    def __init__(self, metrics, values):
        self.coords = {"metric": SimpleNamespace(values=metrics)}
        self.data_vars = {"value": object()}
        self._values = values
        self._selected = None

    def sel(self, *, metric):
        selected = _AccuracyDataset(self.coords["metric"].values, self._values)
        selected._selected = metric
        return selected

    def __getitem__(self, key):
        assert key == "value"
        return SimpleNamespace(values=self._values[self._selected])
