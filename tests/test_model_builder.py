"""Tests for safe Meridian input construction without importing the heavy runtime."""

import sys
from datetime import date
from types import ModuleType
from unittest.mock import patch

import pandas as pd
import pytest

from mmm.data.schema import DataConfig, MediaChannel, MMMDataset
from mmm.model.builder import build_meridian_input


def _dataset(
    *,
    kpi_column="conversions",
    kpi_type="non_revenue",
    include_population=True,
    include_impressions=True,
    allow_population_estimates=False,
    allow_impression_estimates=False,
):
    data = {
        "date": pd.to_datetime(["2026-01-05", "2026-01-12"]),
        "geo": ["AU", "AU"],
        kpi_column: [10.0, 12.0],
        "meta_spend": [100.0, 120.0],
    }
    if include_population:
        data["population"] = [26_000_000, 26_000_000]
    if include_impressions:
        data["meta_impressions"] = [10_000, 12_000]
    frame = pd.DataFrame(data)
    return MMMDataset(
        df=frame,
        config=DataConfig(
            kpi_column=kpi_column,
            kpi_type=kpi_type,
            population_column="population" if include_population else None,
            allow_population_estimates=allow_population_estimates,
            allow_impression_estimates=allow_impression_estimates,
            revenue_column="revenue" if "revenue" in frame else None,
            media_channels=[
                MediaChannel(
                    name="meta",
                    spend_column="meta_spend",
                    impressions_column="meta_impressions" if include_impressions else None,
                )
            ],
        ),
        date_range=(date(2026, 1, 5), date(2026, 1, 12)),
        geos=["AU"],
        n_time_periods=2,
        n_geos=1,
        media_channels=["meta"],
        total_spend=220.0,
        total_kpi=22.0,
    )


def _builder_modules(calls):
    class FakeBuilder:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def __getattr__(self, name):
            if name.startswith("with_"):

                def method(*args, **kwargs):
                    calls.append((name, kwargs))
                    return self

                return method
            raise AttributeError(name)

        def build(self):
            calls.append(("build", {}))
            return "input-data"

    builder_module = ModuleType("meridian.data.data_frame_input_data_builder")
    builder_module.DataFrameInputDataBuilder = FakeBuilder
    data_module = ModuleType("meridian.data")
    data_module.data_frame_input_data_builder = builder_module
    meridian = ModuleType("meridian")
    meridian.data = data_module
    return {
        "meridian": meridian,
        "meridian.data": data_module,
        "meridian.data.data_frame_input_data_builder": builder_module,
    }


def test_revenue_kpi_does_not_add_revenue_per_kpi():
    calls = []
    dataset = _dataset(kpi_column="revenue", kpi_type="revenue")

    with patch.dict(sys.modules, _builder_modules(calls)):
        assert build_meridian_input(dataset) == "input-data"

    assert "with_revenue_per_kpi" not in [name for name, _ in calls]


def test_missing_population_requires_explicit_estimate():
    with patch.dict(sys.modules, _builder_modules([])):
        with pytest.raises(ValueError, match="Population data is required"):
            build_meridian_input(_dataset(include_population=False))


def test_missing_impressions_requires_explicit_estimate():
    with patch.dict(sys.modules, _builder_modules([])):
        with pytest.raises(ValueError, match="needs impressions"):
            build_meridian_input(_dataset(include_impressions=False))


def test_explicit_estimates_are_visible_in_builder_inputs():
    calls = []
    dataset = _dataset(
        include_population=False,
        include_impressions=False,
        allow_population_estimates=True,
        allow_impression_estimates=True,
    )

    with patch.dict(sys.modules, _builder_modules(calls)):
        assert build_meridian_input(dataset) == "input-data"

    media_call = next(kwargs for name, kwargs in calls if name == "with_media")
    assert media_call["media_cols"] == ["meta_impressions_est"]
    assert any(name == "with_population" for name, _ in calls)
