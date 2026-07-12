"""Build deterministic, decision-safe sample fixtures for Meridian compatibility tests."""

import argparse
from pathlib import Path

import pandas as pd


def build_model_shape_fixture(
    source: Path,
    destination: Path,
    *,
    n_geos: int = 8,
    n_periods: int = 52,
    national: bool = False,
) -> pd.DataFrame:
    """Derive an extended R&F/organic/treatment fixture from Meridian's sample."""
    if n_geos < 1 or n_periods < 26:
        raise ValueError("Fixtures require at least 1 geo and 26 weekly periods")

    source_frame = pd.read_csv(source)
    required = {
        "time",
        "geo",
        "conversions",
        "revenue_per_conversion",
        "population",
        "Channel0_impression",
        "Channel0_spend",
        "Channel1_impression",
        "Channel1_spend",
        "Channel2_impression",
        "Channel2_spend",
        "Organic_channel0_impression",
        "Promo",
        "competitor_sales_control",
        "sentiment_score_control",
    }
    missing = required.difference(source_frame.columns)
    if missing:
        raise ValueError(f"Source sample is missing columns: {', '.join(sorted(missing))}")

    source_frame["time"] = pd.to_datetime(source_frame["time"])
    selected_geos = source_frame["geo"].drop_duplicates().iloc[:n_geos]
    selected_times = source_frame["time"].drop_duplicates().sort_values().iloc[:n_periods]
    frame = source_frame[
        source_frame["geo"].isin(selected_geos) & source_frame["time"].isin(selected_times)
    ].copy()
    frame = frame.sort_values(["geo", "time"]).reset_index(drop=True)
    if frame["geo"].nunique() != n_geos or frame["time"].nunique() != n_periods:
        raise ValueError("Source sample does not contain the requested fixture dimensions")

    week_index = frame.groupby("geo").cumcount()
    geo_index = pd.Categorical(frame["geo"], categories=selected_geos).codes
    frame["video_frequency"] = 1.4 + ((week_index + geo_index) % 8) * 0.15
    frame["video_reach"] = frame["Channel2_impression"] / frame["video_frequency"]
    frame["_video_impressions"] = frame["Channel2_impression"]

    fixture = frame.rename(
        columns={
            "Channel0_impression": "meta_impressions",
            "Channel0_spend": "meta_spend",
            "Channel1_impression": "search_impressions",
            "Channel1_spend": "search_spend",
            "Channel2_spend": "video_spend",
            "Organic_channel0_impression": "newsletter_organic",
            "Promo": "promotion_treatment",
        }
    )[
        [
            "time",
            "geo",
            "conversions",
            "revenue_per_conversion",
            "population",
            "meta_spend",
            "meta_impressions",
            "search_spend",
            "search_impressions",
            "video_spend",
            "video_reach",
            "video_frequency",
            "_video_impressions",
            "newsletter_organic",
            "promotion_treatment",
            "competitor_sales_control",
            "sentiment_score_control",
        ]
    ]

    if national:
        weighted_revenue = fixture["conversions"] * fixture["revenue_per_conversion"]
        fixture = fixture.assign(_weighted_revenue=weighted_revenue)
        totals = fixture.groupby("time", as_index=False).agg(
            {
                "conversions": "sum",
                "_weighted_revenue": "sum",
                "population": "sum",
                "meta_spend": "sum",
                "meta_impressions": "sum",
                "search_spend": "sum",
                "search_impressions": "sum",
                "video_spend": "sum",
                "video_reach": "sum",
                "video_frequency": "mean",
                "_video_impressions": "sum",
                "newsletter_organic": "sum",
                "promotion_treatment": "mean",
                "competitor_sales_control": "mean",
                "sentiment_score_control": "mean",
            }
        )
        totals["geo"] = "national"
        totals["revenue_per_conversion"] = totals["_weighted_revenue"] / totals["conversions"]
        aggregate_frequency = totals["_video_impressions"] / totals["video_reach"]
        totals["video_frequency"] = aggregate_frequency.where(
            totals["video_reach"] > 0, totals["video_frequency"]
        )
        fixture = totals

    fixture = fixture.drop(columns=["_video_impressions", "_weighted_revenue"], errors="ignore")
    fixture = fixture.sort_values(["time", "geo"]).reset_index(drop=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixture.to_csv(destination, index=False)
    return fixture


def main() -> None:
    """Build a fixture from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--geos", type=int, default=8)
    parser.add_argument("--periods", type=int, default=52)
    parser.add_argument("--national", action="store_true")
    args = parser.parse_args()
    fixture = build_model_shape_fixture(
        args.source,
        args.destination,
        n_geos=args.geos,
        n_periods=args.periods,
        national=args.national,
    )
    print(
        f"Wrote {args.destination}: {len(fixture)} rows, "
        f"{fixture['geo'].nunique()} geos, {fixture['time'].nunique()} periods"
    )


if __name__ == "__main__":
    main()
