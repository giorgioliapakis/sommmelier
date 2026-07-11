"""Tests for data loading functionality."""

from pathlib import Path

import pytest

from mmm.data.loader import load_mmm_data
from mmm.data.schema import DataConfig


@pytest.fixture
def sample_data_path() -> Path:
    """Path to sample data file."""
    return Path(__file__).parent.parent / "data" / "examples" / "sample_data.csv"


@pytest.fixture
def extended_data_path() -> Path:
    return Path(__file__).parent.parent / "data" / "examples" / "sample_data_extended.csv"


@pytest.fixture
def meridian_data_path() -> Path:
    return Path(__file__).parent.parent / "data" / "examples" / "meridian_sample.csv"


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


def test_detects_extended_model_inputs(extended_data_path: Path):
    dataset = load_mmm_data(extended_data_path)

    assert [channel.name for channel in dataset.config.organic_channels] == [
        "newsletter",
        "blog",
    ]
    assert dataset.config.treatment_columns == ["promotion_discount_treatment"]
    assert dataset.config.control_columns == ["is_holiday", "product_launch"]
    assert dataset.config.revenue_column == "revenue"
    tiktok = next(channel for channel in dataset.config.media_channels if channel.name == "tiktok")
    assert tiktok.reach_column == "tiktok_reach"
    assert tiktok.frequency_column == "tiktok_frequency"


def test_loading_does_not_mutate_caller_config(sample_data_path: Path):
    config = DataConfig()

    dataset = load_mmm_data(sample_data_path, config)

    assert config.media_channels == []
    assert dataset.config.media_channels


def test_time_column_is_accepted_as_documented(meridian_data_path: Path):
    dataset = load_mmm_data(meridian_data_path)

    assert dataset.config.date_column == "time"
    assert dataset.n_time_periods >= 52
