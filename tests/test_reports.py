"""Tests for local wrapper reports and outcome-unit semantics."""

from datetime import date
from types import SimpleNamespace

import pandas as pd

from mmm.analysis.insights import generate_insights
from mmm.analysis.reports import generate_report
from mmm.data.schema import DataConfig, MediaChannel, MMMDataset
from mmm.model.mmm import ModelResults


def _dataset():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-12"]),
            "geo": ["AU", "AU"],
            "conversions": [10.0, 12.0],
            "population": [26_000_000, 26_000_000],
            "meta_spend": [100.0, 120.0],
            "meta_impressions": [10_000, 12_000],
        }
    )
    return MMMDataset(
        df=frame,
        config=DataConfig(
            population_column="population",
            media_channels=[
                MediaChannel(
                    name="meta",
                    spend_column="meta_spend",
                    impressions_column="meta_impressions",
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


def test_non_monetary_local_report_avoids_profitability_claims():
    results = ModelResults(
        channel_roi={"meta": 1.8},
        channel_contributions={"meta": 12.0},
        roi_is_monetary=False,
    )
    wrapper = SimpleNamespace(
        is_fitted=True,
        results=results,
        dataset=_dataset(),
        _meridian=object(),
    )

    report = generate_report(wrapper)
    insights = generate_insights(results)

    assert "Channel KPI Efficiency" in report
    assert "1.8000 KPI/currency" in report
    assert "Break-even" not in report
    assert insights[0].title == "meta has the highest KPI efficiency"
    assert "profitability claims" in insights[0].recommendation
