"""Column detection shared by local validation and the Modal runner."""

from collections.abc import Iterable
from dataclasses import dataclass

COMMON_CONTROL_COLUMNS = frozenset(
    {
        "is_holiday",
        "product_launch",
        "is_promotion",
    }
)


@dataclass(frozen=True)
class DetectedColumns:
    """Detected MMM roles for a flat collection of column names."""

    media_channels: tuple[dict[str, str | None], ...]
    organic_columns: tuple[str, ...]
    treatment_columns: tuple[str, ...]
    control_columns: tuple[str, ...]


def detect_columns(columns: Iterable[str]) -> DetectedColumns:
    """Detect supported MMM columns using case-insensitive suffix conventions."""
    column_names = tuple(columns)
    by_lower = {column.lower(): column for column in column_names}
    media_channels: list[dict[str, str | None]] = []

    for spend_column in column_names:
        if not spend_column.lower().endswith("_spend"):
            continue

        channel_name = spend_column[: -len("_spend")]
        channel_key = channel_name.lower()
        impressions_column = _first_match(
            by_lower,
            f"{channel_key}_impressions",
            f"{channel_key}_impression",
            f"{channel_key}_imps",
        )
        reach_column = _first_match(by_lower, f"{channel_key}_reach")
        frequency_column = _first_match(by_lower, f"{channel_key}_frequency")

        media_channels.append(
            {
                "name": channel_name,
                "spend_column": spend_column,
                "impressions_column": impressions_column,
                "reach_column": reach_column,
                "frequency_column": frequency_column,
            }
        )

    organic_columns = tuple(
        column for column in column_names if column.lower().endswith("_organic")
    )
    treatment_columns = tuple(
        column for column in column_names if column.lower().endswith("_treatment")
    )
    control_columns = tuple(
        column
        for column in column_names
        if column.lower().endswith("_control") or column.lower() in COMMON_CONTROL_COLUMNS
    )

    return DetectedColumns(
        media_channels=tuple(media_channels),
        organic_columns=organic_columns,
        treatment_columns=treatment_columns,
        control_columns=control_columns,
    )


def _first_match(by_lower: dict[str, str], *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    return None
