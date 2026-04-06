"""Data loading utilities for Sommmelier."""

from datetime import date
from pathlib import Path

import pandas as pd

from mmm.data.schema import DataConfig, KPIData, MediaData, MMMDataset


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


def load_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(path, **kwargs)


def load_media_data(
    path: str | Path,
    date_column: str = "date",
    geo_column: str = "geo",
    spend_columns: list[str] | None = None,
) -> MediaData:
    """
    Load media spend data from CSV.

    Args:
        path: Path to media data CSV
        date_column: Name of date column
        geo_column: Name of geography column
        spend_columns: List of spend column names (auto-detected if None)

    Returns:
        MediaData object with validated data
    """
    df = load_csv(path, parse_dates=[date_column])

    # Auto-detect spend columns if not provided
    if spend_columns is None:
        spend_columns = [col for col in df.columns if "_spend" in col.lower()]

    if not spend_columns:
        raise ValueError("No spend columns found. Provide spend_columns or use '_spend' suffix.")

    # Extract metadata
    df[date_column] = pd.to_datetime(df[date_column])
    date_range = (df[date_column].min().date(), df[date_column].max().date())
    geos = df[geo_column].unique().tolist()

    # Derive channel names from spend columns
    channels = [col.replace("_spend", "").replace("_Spend", "") for col in spend_columns]

    return MediaData(
        df=df,
        channels=channels,
        date_range=date_range,
        geos=geos,
    )


def load_kpi_data(
    path: str | Path,
    date_column: str = "date",
    geo_column: str = "geo",
    kpi_column: str = "conversions",
) -> KPIData:
    """
    Load KPI/conversion data from CSV.

    Args:
        path: Path to KPI data CSV
        date_column: Name of date column
        geo_column: Name of geography column
        kpi_column: Name of KPI column

    Returns:
        KPIData object with validated data
    """
    df = load_csv(path, parse_dates=[date_column])

    if kpi_column not in df.columns:
        raise ValueError(f"KPI column '{kpi_column}' not found in data")

    # Extract metadata
    df[date_column] = pd.to_datetime(df[date_column])
    date_range = (df[date_column].min().date(), df[date_column].max().date())
    geos = df[geo_column].unique().tolist()
    total_kpi = df[kpi_column].sum()

    return KPIData(
        df=df,
        kpi_column=kpi_column,
        date_range=date_range,
        geos=geos,
        total_kpi=total_kpi,
    )


def load_mmm_data(
    path: str | Path,
    config: DataConfig | None = None,
) -> MMMDataset:
    """
    Load a complete MMM dataset from a single CSV.

    This is the main entry point for loading data. The CSV should contain:
    - date: Date column
    - geo: Geography column
    - KPI column (e.g., conversions, revenue)
    - Media spend columns (e.g., meta_spend, google_spend)
    - Optional: impression columns, control variables

    Args:
        path: Path to the MMM data CSV
        config: Optional DataConfig for column mappings

    Returns:
        MMMDataset object ready for Meridian
    """
    if config is None:
        config = DataConfig()

    df = load_csv(path, parse_dates=[config.date_column])
    df[config.date_column] = pd.to_datetime(df[config.date_column])

    # Validate required columns exist
    required_cols = [config.date_column, config.geo_column, config.kpi_column]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Auto-detect media channels if not provided
    if not config.media_channels:
        spend_cols = [col for col in df.columns if "_spend" in col.lower()]
        for spend_col in spend_cols:
            channel_name = spend_col.replace("_spend", "").replace("_Spend", "")

            # Check for reach+frequency columns
            reach_col = None
            frequency_col = None
            for suffix in ["_reach"]:
                potential = channel_name + suffix
                if potential in df.columns:
                    reach_col = potential
            for suffix in ["_frequency"]:
                potential = channel_name + suffix
                if potential in df.columns:
                    frequency_col = potential

            # Check for impressions column
            impressions_col = None
            for suffix in ["_impressions", "_imps", "_Impressions"]:
                potential_col = channel_name + suffix
                if potential_col in df.columns:
                    impressions_col = potential_col
                    break

            config.media_channels.append(
                {
                    "name": channel_name,
                    "spend_column": spend_col,
                    "impressions_column": impressions_col,
                    "reach_column": reach_col,
                    "frequency_column": frequency_col,
                }
            )

    # Auto-detect organic media columns (suffix: _organic)
    if not config.control_columns:
        # Also populate control columns if not already set
        control_candidates = [col for col in df.columns if "_control" in col.lower()]
        config.control_columns = control_candidates

    # Extract metadata
    date_range = (df[config.date_column].min().date(), df[config.date_column].max().date())
    geos = df[config.geo_column].unique().tolist()
    n_time_periods = df[config.date_column].nunique()
    media_channel_names = [ch["name"] if isinstance(ch, dict) else ch.name for ch in config.media_channels]

    # Calculate totals
    spend_cols = [
        ch["spend_column"] if isinstance(ch, dict) else ch.spend_column
        for ch in config.media_channels
    ]
    total_spend = df[spend_cols].sum().sum()
    total_kpi = df[config.kpi_column].sum()

    return MMMDataset(
        df=df,
        config=config,
        date_range=date_range,
        geos=geos,
        n_time_periods=n_time_periods,
        n_geos=len(geos),
        media_channels=media_channel_names,
        total_spend=total_spend,
        total_kpi=total_kpi,
    )


def merge_media_and_kpi(
    media_data: MediaData,
    kpi_data: KPIData,
    date_column: str = "date",
    geo_column: str = "geo",
) -> pd.DataFrame:
    """
    Merge separate media and KPI dataframes.

    Use this when you have separate exports from Windsor and Snowflake.

    Args:
        media_data: MediaData from Windsor export
        kpi_data: KPIData from Snowflake export
        date_column: Name of date column for joining
        geo_column: Name of geography column for joining

    Returns:
        Merged DataFrame ready for load_mmm_data
    """
    merged = pd.merge(
        media_data.df,
        kpi_data.df,
        on=[date_column, geo_column],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "Merge resulted in empty dataset. Check date/geo alignment between sources."
        )

    return merged
