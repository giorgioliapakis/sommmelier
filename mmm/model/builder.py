"""Build Meridian InputData from Sommmelier datasets."""

from typing import TYPE_CHECKING

import pandas as pd

from mmm.data.schema import MMMDataset

if TYPE_CHECKING:
    from meridian.data import input_data


def build_meridian_input(dataset: MMMDataset) -> "input_data.InputData":
    """
    Convert an MMMDataset to Meridian's InputData format.

    This bridges our data loading layer with Meridian's expected format.

    Args:
        dataset: Validated MMMDataset

    Returns:
        Meridian InputData object ready for modeling
    """
    from meridian.data import data_frame_input_data_builder

    df = dataset.df.copy()
    config = dataset.config

    # Meridian expects 'time' column, not 'date'
    if config.date_column in df.columns and config.date_column != "time":
        df = df.rename(columns={config.date_column: "time"})

    # Meridian expects 'geo' column
    if config.geo_column in df.columns and config.geo_column != "geo":
        df = df.rename(columns={config.geo_column: "geo"})

    # Initialize the builder (Meridian 1.4+ API)
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=config.kpi_type,
        default_kpi_column=config.kpi_column,
    )

    # Add KPI data
    builder = builder.with_kpi(df)

    # Add population (REQUIRED for geo-level models)
    if config.population_column and config.population_column in df.columns:
        builder = builder.with_population(df, population_column=config.population_column)
    elif "population" in df.columns:
        builder = builder.with_population(df)
    else:
        # Add default population estimates if not provided
        # This is a fallback - real data should include population
        geo_populations = _estimate_population(df["geo"].unique())
        df["population"] = df["geo"].map(geo_populations)
        builder = builder.with_population(df)

    # Add revenue per KPI if available
    if config.revenue_column and config.revenue_column in df.columns:
        builder = builder.with_revenue_per_kpi(df, revenue_per_kpi_column=config.revenue_column)

    # Build media channel lists
    media_channel_names = []
    media_impression_cols = []
    media_spend_cols = []

    for channel in config.media_channels:
        if isinstance(channel, dict):
            name = channel["name"]
            spend_col = channel["spend_column"]
            impressions_col = channel.get("impressions_column")
        else:
            name = channel.name
            spend_col = channel.spend_column
            impressions_col = channel.impressions_column

        media_channel_names.append(name)
        media_spend_cols.append(spend_col)

        if impressions_col and impressions_col in df.columns:
            media_impression_cols.append(impressions_col)
        else:
            # Estimate impressions from spend if not available (assume $10 CPM)
            est_col = f"{name}_impressions_est"
            df[est_col] = df[spend_col] * 100  # $10 CPM = 100 impressions per $1
            media_impression_cols.append(est_col)

    # Add media (Meridian 1.4+ requires media_channels parameter)
    builder = builder.with_media(
        df,
        media_channels=media_channel_names,
        media_cols=media_impression_cols,
        media_spend_cols=media_spend_cols,
    )

    # Add control variables (only if they vary by geo)
    if config.control_columns:
        valid_controls = []
        for col in config.control_columns:
            if col in df.columns:
                # Check if control varies by geo (required by Meridian)
                geo_variation = df.groupby("time")[col].nunique()
                if (geo_variation > 1).any():
                    valid_controls.append(col)

        if valid_controls:
            builder = builder.with_controls(df, control_cols=valid_controls)

    return builder.build()


def _estimate_population(geos: list[str]) -> dict[str, int]:
    """
    Estimate population for common geo codes.

    This is a fallback - real data should include actual population figures.
    """
    # Common country populations (approximate)
    known_populations = {
        "US": 330_000_000,
        "USA": 330_000_000,
        "UK": 67_000_000,
        "GB": 67_000_000,
        "AU": 26_000_000,
        "AUS": 26_000_000,
        "CA": 40_000_000,
        "CAN": 40_000_000,
        "DE": 83_000_000,
        "FR": 67_000_000,
        "JP": 125_000_000,
        "BR": 215_000_000,
        "IN": 1_400_000_000,
        "MX": 130_000_000,
    }

    result = {}
    for geo in geos:
        geo_upper = str(geo).upper()
        if geo_upper in known_populations:
            result[geo] = known_populations[geo_upper]
        else:
            # Default fallback
            result[geo] = 10_000_000

    return result


def prepare_dataframe_for_meridian(
    df: pd.DataFrame,
    date_column: str = "date",
    geo_column: str = "geo",
) -> pd.DataFrame:
    """
    Prepare a DataFrame for Meridian by ensuring correct types and format.

    Args:
        df: Raw DataFrame
        date_column: Date column name
        geo_column: Geo column name

    Returns:
        Cleaned DataFrame ready for Meridian
    """
    df = df.copy()

    # Rename to Meridian's expected column names
    if date_column != "time":
        df = df.rename(columns={date_column: "time"})
    if geo_column != "geo":
        df = df.rename(columns={geo_column: "geo"})

    # Ensure time is datetime
    df["time"] = pd.to_datetime(df["time"])

    # Ensure geo is string
    df["geo"] = df["geo"].astype(str)

    # Sort by geo and time
    df = df.sort_values(["geo", "time"]).reset_index(drop=True)

    # Fill any missing numeric values with 0
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df
