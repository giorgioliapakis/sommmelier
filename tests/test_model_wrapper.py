"""Tests for the stable local wrapper around release-sensitive Meridian APIs."""

import sys
from datetime import date
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mmm.data.schema import DataConfig, MediaChannel, MMMDataset
from mmm.model.mmm import AutoMMM, ModelConfig, ModelResults


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


def test_summary_labels_non_monetary_efficiency_without_roi_multiplier():
    summary = ModelResults(channel_roi={"meta": 0.8}, roi_is_monetary=False).summary()

    assert "Channel KPI Efficiency" in summary
    assert "0.80 KPI/currency" in summary
    assert "0.80x" not in summary


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


def test_optimize_budget_rejects_silently_ignored_constraints():
    wrapper = AutoMMM.__new__(AutoMMM)
    wrapper._meridian = object()
    wrapper.dataset = SimpleNamespace(total_spend=100.0)

    with pytest.raises(NotImplementedError, match="constraints are not supported"):
        wrapper.optimize_budget(constraints={"meta": (20.0, 80.0)})


def test_extract_results_uses_shared_meridian_contract():
    class FakeAnalyzer:
        def __init__(self, model):
            assert model is fitted_model

        def roi(self, *, use_posterior):
            assert use_posterior is True
            return _Tensor([[[1.0, 3.0], [3.0, 5.0]]])

        def incremental_outcome(self, *, use_posterior):
            assert use_posterior is True
            return _Tensor([[[[1.0, 3.0], [1.0, 5.0]]]])

        def predictive_accuracy(self):
            return xr.Dataset(
                {"value": (("metric", "geo_granularity"), [[0.7, 0.9], [0.1, 0.2]])},
                coords={
                    "metric": ["R_Squared", "MAPE"],
                    "geo_granularity": ["geo", "national"],
                },
            )

        def rhat_summary(self):
            return pd.DataFrame({"max_rhat": [1.01, 1.05]})

    analyzer = ModuleType("meridian.analysis.analyzer")
    analyzer.Analyzer = FakeAnalyzer
    analysis = ModuleType("meridian.analysis")
    analysis.analyzer = analyzer
    meridian = ModuleType("meridian")
    meridian.analysis = analysis

    fitted_model = object()
    wrapper = AutoMMM.__new__(AutoMMM)
    wrapper._meridian = fitted_model
    wrapper.dataset = SimpleNamespace(
        media_channels=["meta", "search"],
        config=SimpleNamespace(
            kpi_type="non_revenue",
            revenue_per_kpi_column=None,
            revenue_column=None,
        ),
    )

    modules = {
        "meridian": meridian,
        "meridian.analysis": analysis,
        "meridian.analysis.analyzer": analyzer,
    }
    with patch.dict(sys.modules, modules):
        results = wrapper._extract_results()

    assert results.channel_roi == {"meta": 2.0, "search": 4.0}
    assert results.channel_contributions == {"meta": 2.0, "search": 8.0}
    assert results.r_squared == pytest.approx(0.8)
    assert results.mape == pytest.approx(0.15)
    assert results.convergence_passed is True
    assert results.r_hat_max == 1.05


def test_model_bundle_round_trip_uses_safe_formats(tmp_path):
    saved_model = object()
    loaded_model = object()

    meridian_serde = ModuleType("meridian.schema.serde.meridian_serde")

    def save_meridian(model, path):
        assert model is saved_model
        with open(path, "wb") as file:
            file.write(b"protobuf")

    def load_meridian(path):
        with open(path, "rb") as file:
            assert file.read() == b"protobuf"
        return loaded_model

    meridian_serde.save_meridian = save_meridian
    meridian_serde.load_meridian = load_meridian
    serde = ModuleType("meridian.schema.serde")
    serde.meridian_serde = meridian_serde
    schema = ModuleType("meridian.schema")
    schema.serde = serde
    meridian = ModuleType("meridian")
    meridian.schema = schema

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-12"]),
            "geo": ["AU", "AU"],
            "conversions": [10.0, 12.0],
            "meta_spend": [100.0, 120.0],
        }
    )
    dataset = MMMDataset(
        df=frame,
        config=DataConfig(media_channels=[MediaChannel(name="meta", spend_column="meta_spend")]),
        date_range=(date(2026, 1, 5), date(2026, 1, 12)),
        geos=["AU"],
        n_time_periods=2,
        n_geos=1,
        media_channels=["meta"],
        total_spend=220.0,
        total_kpi=22.0,
    )
    wrapper = AutoMMM(dataset, ModelConfig(n_keep=10))
    wrapper._meridian = saved_model
    wrapper._results = ModelResults(channel_roi={"meta": 1.2}, meridian_model=saved_model)
    bundle = tmp_path / "model"

    modules = {
        "meridian": meridian,
        "meridian.schema": schema,
        "meridian.schema.serde": serde,
        "meridian.schema.serde.meridian_serde": meridian_serde,
    }
    with patch.dict(sys.modules, modules):
        wrapper.save(bundle)
        restored = AutoMMM.load(bundle)

    assert sorted(path.name for path in bundle.iterdir()) == [
        "dataset.parquet",
        "metadata.json",
        "model.binpb",
    ]
    assert restored._meridian is loaded_model
    assert restored.results.channel_roi == {"meta": 1.2}
    assert restored.dataset.total_spend == 220.0
    assert restored.config.n_keep == 10


def test_model_bundle_refuses_to_overwrite(tmp_path):
    wrapper = AutoMMM.__new__(AutoMMM)
    wrapper._meridian = object()
    wrapper._results = ModelResults()
    existing = tmp_path / "model"
    existing.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        wrapper.save(existing)


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


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def numpy(self):
        return self._values
