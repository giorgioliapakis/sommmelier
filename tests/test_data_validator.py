"""Tests for validation checks that protect paid model runs."""

from pathlib import Path

import pandas as pd

from mmm.data.loader import load_mmm_data
from mmm.data.validator import validate_dataset


def _write_dataset(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _valid_rows() -> list[dict]:
    rows = []
    for week in range(26):
        date = pd.Timestamp("2025-01-06") + pd.Timedelta(weeks=week)
        for geo_index, geo in enumerate(("AU", "US")):
            rows.append(
                {
                    "date": date,
                    "geo": geo,
                    "conversions": 100 + week * 10 + geo_index * 20,
                    "meta_spend": 1000 + week,
                    "google_spend": 800 + geo_index,
                    "tiktok_spend": 400 + week,
                }
            )
    return rows


def test_valid_panel_passes_with_only_data_length_warning(tmp_path: Path):
    report = validate_dataset(load_mmm_data(_write_dataset(tmp_path, _valid_rows())))

    assert report.passed
    assert report.errors == 0
    assert report.warnings == 1


def test_duplicate_geo_time_rows_fail(tmp_path: Path):
    rows = _valid_rows()
    rows.append(dict(rows[0]))

    report = validate_dataset(load_mmm_data(_write_dataset(tmp_path, rows)))

    duplicate_check = next(r for r in report.results if r.check_name == "Unique Geo-Time Rows")
    assert not report.passed
    assert not duplicate_check.passed


def test_missing_panel_coordinate_fails(tmp_path: Path):
    rows = _valid_rows()
    rows.pop()

    report = validate_dataset(load_mmm_data(_write_dataset(tmp_path, rows)))

    panel_check = next(r for r in report.results if r.check_name == "Balanced Geo-Time Panel")
    assert not report.passed
    assert not panel_check.passed


def test_zero_spend_channel_fails(tmp_path: Path):
    rows = _valid_rows()
    for row in rows:
        row["tiktok_spend"] = 0

    report = validate_dataset(load_mmm_data(_write_dataset(tmp_path, rows)))

    spend_check = next(r for r in report.results if r.check_name == "Channel Spend Totals")
    assert not report.passed
    assert not spend_check.passed


def test_revenue_with_zero_kpi_fails(tmp_path: Path):
    rows = _valid_rows()
    rows[0]["conversions"] = 0
    rows[0]["revenue"] = 100

    report = validate_dataset(load_mmm_data(_write_dataset(tmp_path, rows)))

    revenue_check = next(
        r for r in report.results if r.check_name == "Revenue-to-KPI Consistency"
    )
    assert not report.passed
    assert not revenue_check.passed


def test_non_finite_control_fails(tmp_path: Path):
    rows = _valid_rows()
    for row in rows:
        row["seasonality_control"] = 1.0
    rows[0]["seasonality_control"] = float("inf")

    report = validate_dataset(load_mmm_data(_write_dataset(tmp_path, rows)))

    finite_check = next(
        r for r in report.results if r.check_name == "Additional Input Finite Values"
    )
    assert not report.passed
    assert not finite_check.passed


def test_negative_revenue_fails(tmp_path: Path):
    rows = _valid_rows()
    for row in rows:
        row["revenue"] = row["conversions"] * 10
    rows[0]["revenue"] = -10

    report = validate_dataset(load_mmm_data(_write_dataset(tmp_path, rows)))

    nonnegative_check = next(
        r for r in report.results if r.check_name == "Additional Input Non-Negativity"
    )
    assert not report.passed
    assert not nonnegative_check.passed
