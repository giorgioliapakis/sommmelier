"""Tests for safe Meridian input construction without importing the heavy runtime."""

import sys
from datetime import date
from types import ModuleType
from unittest.mock import patch

import pandas as pd
import pytest

from mmm.data.schema import DataConfig, MediaChannel, MMMDataset, OrganicMediaChannel
from mmm.model.builder import build_meridian_input, prepare_dataframe_for_meridian


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


def test_builder_routes_reach_organic_treatment_and_time_varying_controls():
    calls = []
    dataset = _dataset()
    dataset.df["meta_reach"] = [5_000, 6_000]
    dataset.df["meta_frequency"] = [2.0, 2.1]
    dataset.df["newsletter_organic"] = [100, 120]
    dataset.df["promotion_treatment"] = [0.0, 0.2]
    dataset.df["holiday_control"] = [0, 1]
    dataset.df["constant_control"] = [1, 1]
    dataset.config.media_channels = [
        MediaChannel(
            name="meta",
            spend_column="meta_spend",
            reach_column="meta_reach",
            frequency_column="meta_frequency",
        )
    ]
    dataset.config.organic_channels = [
        OrganicMediaChannel(name="newsletter", column="newsletter_organic")
    ]
    dataset.config.treatment_columns = ["promotion_treatment"]
    dataset.config.control_columns = ["holiday_control", "constant_control"]

    with patch.dict(sys.modules, _builder_modules(calls)):
        assert build_meridian_input(dataset) == "input-data"

    by_name = {name: kwargs for name, kwargs in calls}
    assert "with_media" not in by_name
    assert by_name["with_reach"]["rf_channels"] == ["meta"]
    assert by_name["with_organic_media"]["organic_media_channels"] == ["newsletter"]
    assert by_name["with_non_media_treatments"]["non_media_treatment_cols"] == [
        "promotion_treatment"
    ]
    assert by_name["with_controls"]["control_cols"] == ["holiday_control"]


def test_total_revenue_is_converted_to_revenue_per_kpi():
    calls = []
    dataset = _dataset()
    dataset.df["revenue"] = [100.0, 180.0]
    dataset.config.revenue_column = "revenue"

    with patch.dict(sys.modules, _builder_modules(calls)):
        build_meridian_input(dataset)

    revenue_call = next(kwargs for name, kwargs in calls if name == "with_revenue_per_kpi")
    assert revenue_call["revenue_per_kpi_col"] == "_revenue_per_kpi"


def test_nonzero_revenue_with_zero_kpi_is_rejected():
    dataset = _dataset()
    dataset.df.loc[0, "conversions"] = 0
    dataset.df["revenue"] = [100.0, 180.0]
    dataset.config.revenue_column = "revenue"

    with patch.dict(sys.modules, _builder_modules([])):
        with pytest.raises(ValueError, match="Revenue cannot be non-zero"):
            build_meridian_input(dataset)


def test_prepare_dataframe_normalizes_types_and_order():
    frame = pd.DataFrame(
        {
            "week": ["2026-01-12", "2026-01-05"],
            "market": [2, 1],
            "value": [20, 10],
        }
    )

    prepared = prepare_dataframe_for_meridian(frame, "week", "market")

    assert list(prepared.columns) == ["time", "geo", "value"]
    assert prepared["geo"].tolist() == ["1", "2"]
    assert prepared["time"].tolist() == list(pd.to_datetime(["2026-01-05", "2026-01-12"]))
