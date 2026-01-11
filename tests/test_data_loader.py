"""Tests for data loading functionality."""

from pathlib import Path

import pandas as pd
import pytest

from mmm.data.loader import load_mmm_data
from mmm.data.schema import DataConfig


@pytest.fixture
def sample_data_path() -> Path:
    """Path to sample data file."""
    return Path(__file__).parent.parent / "data" / "examples" / "sample_data.csv"


def test_load_mmm_data(sample_data_path: Path):
    """Test loading MMM data from CSV."""
    dataset = load_mmm_data(sample_data_path)

    assert dataset.n_geos == 3
    assert dataset.n_time_periods > 0
    assert len(dataset.media_channels) >= 3
    assert dataset.total_spend > 0
    assert dataset.total_kpi > 0


def test_load_with_custom_config(sample_data_path: Path):
    """Test loading with custom configuration."""
    config = DataConfig(
        kpi_column="conversions",
        date_column="date",
        geo_column="geo",
    )
    dataset = load_mmm_data(sample_data_path, config)

    assert "conversions" in str(dataset.config.kpi_column)


def test_auto_detect_channels(sample_data_path: Path):
    """Test auto-detection of media channels."""
    dataset = load_mmm_data(sample_data_path)

    # Should detect meta, google, tiktok from _spend columns
    assert "meta" in dataset.media_channels
    assert "google" in dataset.media_channels
    assert "tiktok" in dataset.media_channels


def test_date_range_extraction(sample_data_path: Path):
    """Test date range is correctly extracted."""
    dataset = load_mmm_data(sample_data_path)

    assert dataset.date_range[0] < dataset.date_range[1]
