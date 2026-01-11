"""Data schemas for Sommmelier using Pydantic."""

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class MediaChannel(BaseModel):
    """Configuration for a single media channel."""

    name: str = Field(..., description="Channel identifier (e.g., 'meta', 'google')")
    spend_column: str = Field(..., description="Column name for spend data")
    impressions_column: str | None = Field(None, description="Column name for impressions")
    reach_column: str | None = Field(None, description="Column name for reach data")
    frequency_column: str | None = Field(None, description="Column name for frequency data")


class DataConfig(BaseModel):
    """Configuration for loading MMM data."""

    # Required columns
    date_column: str = Field(default="date", description="Date column name")
    geo_column: str = Field(default="geo", description="Geography column name")
    kpi_column: str = Field(default="conversions", description="Primary KPI column")

    # Optional columns
    revenue_column: str | None = Field(None, description="Revenue column if applicable")
    population_column: str | None = Field(None, description="Population column for geo scaling")

    # Media channels
    media_channels: list[MediaChannel] = Field(
        default_factory=list, description="List of media channel configurations"
    )

    # Control variables
    control_columns: list[str] = Field(
        default_factory=list, description="Control variable column names"
    )

    # KPI type
    kpi_type: Literal["revenue", "non_revenue"] = Field(
        default="non_revenue", description="Whether KPI is revenue-based"
    )


class MediaData(BaseModel):
    """Validated media spend data."""

    model_config = {"arbitrary_types_allowed": True}

    df: pd.DataFrame
    channels: list[str]
    date_range: tuple[date, date]
    geos: list[str]

    @field_validator("df", mode="before")
    @classmethod
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        if v.empty:
            raise ValueError("Media data cannot be empty")
        return v


class KPIData(BaseModel):
    """Validated KPI/conversion data."""

    model_config = {"arbitrary_types_allowed": True}

    df: pd.DataFrame
    kpi_column: str
    date_range: tuple[date, date]
    geos: list[str]
    total_kpi: float

    @field_validator("df", mode="before")
    @classmethod
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        if v.empty:
            raise ValueError("KPI data cannot be empty")
        return v


class MMMDataset(BaseModel):
    """Complete validated dataset ready for Meridian."""

    model_config = {"arbitrary_types_allowed": True}

    df: pd.DataFrame = Field(..., description="Merged dataset")
    config: DataConfig = Field(..., description="Data configuration used")
    date_range: tuple[date, date] = Field(..., description="Date range of data")
    geos: list[str] = Field(..., description="List of geographies")
    n_time_periods: int = Field(..., description="Number of time periods")
    n_geos: int = Field(..., description="Number of geographies")
    media_channels: list[str] = Field(..., description="List of media channel names")
    total_spend: float = Field(..., description="Total media spend")
    total_kpi: float = Field(..., description="Total KPI value")

    def summary(self) -> str:
        """Return a human-readable summary of the dataset."""
        return f"""
MMM Dataset Summary
==================
Date Range: {self.date_range[0]} to {self.date_range[1]}
Time Periods: {self.n_time_periods}
Geographies: {self.n_geos} ({', '.join(self.geos[:5])}{'...' if len(self.geos) > 5 else ''})
Media Channels: {len(self.media_channels)} ({', '.join(self.media_channels)})
Total Spend: ${self.total_spend:,.2f}
Total KPI: {self.total_kpi:,.0f}
"""
