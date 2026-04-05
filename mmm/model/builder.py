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

    # Separate media channels into spend+impressions vs reach+frequency
    si_channel_names = []
    si_impression_cols = []
    si_spend_cols = []
    rf_channel_names = []
    rf_reach_cols = []
    rf_frequency_cols = []
    rf_spend_cols = []

    for channel in config.media_channels:
        if isinstance(channel, dict):
            name = channel["name"]
            spend_col = channel["spend_column"]
            impressions_col = channel.get("impressions_column")
            reach_col = channel.get("reach_column")
            frequency_col = channel.get("frequency_column")
        else:
            name = channel.name
            spend_col = channel.spend_column
            impressions_col = channel.impressions_column
            reach_col = channel.reach_column
            frequency_col = channel.frequency_column

        # Determine channel type: R&F if both reach and frequency are present
        if reach_col and frequency_col and reach_col in df.columns and frequency_col in df.columns:
            rf_channel_names.append(name)
            rf_reach_cols.append(reach_col)
            rf_frequency_cols.append(frequency_col)
            rf_spend_cols.append(spend_col)
        else:
            si_channel_names.append(name)
            si_spend_cols.append(spend_col)

            if impressions_col and impressions_col in df.columns:
                si_impression_cols.append(impressions_col)
            else:
                est_col = f"{name}_impressions_est"
                df[est_col] = df[spend_col] * 100  # $10 CPM
                si_impression_cols.append(est_col)

    # Add spend+impressions media channels
    if si_channel_names:
        builder = builder.with_media(
            df,
            media_channels=si_channel_names,
            media_cols=si_impression_cols,
            media_spend_cols=si_spend_cols,
        )

    # Add reach+frequency media channels
    if rf_channel_names:
        builder = builder.with_media_rf(
            df,
            media_channels=rf_channel_names,
            reach_cols=rf_reach_cols,
            frequency_cols=rf_frequency_cols,
            spend_cols=rf_spend_cols,
        )

    # Add organic media channels
    if hasattr(config, 'organic_channels') and config.organic_channels:
        organic_names = [ch.name if hasattr(ch, 'name') else ch["name"] for ch in config.organic_channels]
        organic_cols = [ch.column if hasattr(ch, 'column') else ch["column"] for ch in config.organic_channels]
        valid_organic = [(n, c) for n, c in zip(organic_names, organic_cols) if c in df.columns]
        if valid_organic:
            builder = builder.with_organic_media(
                df,
                organic_channels=[n for n, _ in valid_organic],
                organic_cols=[c for _, c in valid_organic],
            )

    # Add non-media treatment variables
    if hasattr(config, 'treatment_columns') and config.treatment_columns:
        treatment_cols = [c for c in config.treatment_columns if c in df.columns]
        if treatment_cols:
            builder = builder.with_non_media_treatments(df, treatment_cols=treatment_cols)

    # Add control variables (only if they vary by geo)
    if config.control_columns:
        valid_controls = []
        for col in config.control_columns:
            if col in df.columns:
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
