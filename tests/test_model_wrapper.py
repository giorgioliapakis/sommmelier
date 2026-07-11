"""Tests for the stable local wrapper around release-sensitive Meridian APIs."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from mmm.model.mmm import AutoMMM, ModelResults


def test_summary_preserves_zero_metrics_and_convergence_status():
    summary = ModelResults(
        r_squared=0.0,
        mape=0.0,
        convergence_passed=False,
        r_hat_max=0.0,
    ).summary()

    assert "R-squared: 0.000" in summary
    assert "MAPE: 0.0%" in summary
    assert "Passed: no" in summary
    assert "Max R-hat: 0.000" in summary


def test_optimize_budget_uses_current_meridian_contract():
    captured = {}

    class FakeBudgetOptimizer:
        def __init__(self, model):
            captured["model"] = model

        def optimize(self, **kwargs):
            captured["kwargs"] = kwargs
            dataset = _FakeDataset(
                ["meta", "search"],
                [40.0, 60.0],
                {"total_incremental_outcome": 12.0},
            )
            return SimpleNamespace(optimized_data=dataset)

    optimizer = ModuleType("meridian.analysis.optimizer")
    optimizer.BudgetOptimizer = FakeBudgetOptimizer
    analysis = ModuleType("meridian.analysis")
    analysis.optimizer = optimizer
    meridian = ModuleType("meridian")
    meridian.analysis = analysis

    wrapper = AutoMMM.__new__(AutoMMM)
    wrapper._meridian = object()
    wrapper.dataset = SimpleNamespace(total_spend=100.0)

    modules = {
        "meridian": meridian,
        "meridian.analysis": analysis,
        "meridian.analysis.optimizer": optimizer,
    }
    with patch.dict(sys.modules, modules):
        allocation = wrapper.optimize_budget()

    assert captured["kwargs"] == {"fixed_budget": True, "budget": 100.0}
    assert allocation == {"meta": 40.0, "search": 60.0}


class _FakeDataset:
    def __init__(self, channels, spend, attrs):
        self.coords = {"channel": SimpleNamespace(values=_Values(channels))}
        self._spend = spend
        self.attrs = attrs

    def __getitem__(self, key):
        assert key == "spend"
        return SimpleNamespace(values=_Values(self._spend))


class _Values:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)
