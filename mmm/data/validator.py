"""Data validation utilities for Sommmelier."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmm.data.schema import MMMDataset


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    check_name: str
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""

    results: list[ValidationResult]
    passed: bool
    errors: int
    warnings: int

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = ["Data Validation Report", "=" * 40]

        for result in self.results:
            marker = "PASS" if result.passed else result.severity.upper()
            lines.append(f"[{marker}] {result.check_name}: {result.message}")

        lines.append("")
        lines.append(f"Result: {'PASSED' if self.passed else 'FAILED'}")
        lines.append(f"Errors: {self.errors}, Warnings: {self.warnings}")

        return "\n".join(lines)


def validate_dataset(dataset: MMMDataset) -> ValidationReport:
    """
    Validate an MMM dataset for common issues.

    Checks performed:
    - Minimum data requirements (time periods, geos)
    - Missing values
    - Negative spend values
    - Date continuity
    - KPI reasonableness

    Args:
        dataset: MMMDataset to validate

    Returns:
        ValidationReport with all check results
    """
    results: list[ValidationResult] = []
    df = dataset.df
    config = dataset.config

    def add_result(
        passed: bool,
        check_name: str,
        message: str,
        severity: str = "error",
    ) -> None:
        results.append(
            ValidationResult(
                passed=passed,
                check_name=check_name,
                message=message,
                severity=severity,
            )
        )

    # Check 1: Minimum time periods (one year recommended; six months required)
    min_periods = 52  # ~1 year of weekly data
    add_result(
        dataset.n_time_periods >= min_periods,
        "Minimum Time Periods",
        f"{dataset.n_time_periods} periods (minimum: 26; recommended: {min_periods})",
        "warning" if dataset.n_time_periods >= 26 else "error",
    )

    # Check 2: At least 2 geos for geo-level modeling
    add_result(
        dataset.n_geos >= 2,
        "Geographic Coverage",
        f"{dataset.n_geos} geos found",
        "warning" if dataset.n_geos == 1 else "error" if dataset.n_geos == 0 else "info",
    )

    # Check 3: Every geo/time coordinate must identify exactly one observation.
    duplicate_count = int(df.duplicated([config.geo_column, config.date_column]).sum())
    add_result(
        duplicate_count == 0,
        "Unique Geo-Time Rows",
        "No duplicate geo/time rows"
        if duplicate_count == 0
        else f"{duplicate_count} duplicate geo/time rows found",
    )

    expected_rows = dataset.n_geos * dataset.n_time_periods
    missing_panel_rows = max(expected_rows - len(df.drop_duplicates([config.geo_column, config.date_column])), 0)
    add_result(
        missing_panel_rows == 0,
        "Balanced Geo-Time Panel",
        "Every geography has every time period"
        if missing_panel_rows == 0
        else f"{missing_panel_rows} geo/time combinations are missing",
    )

    # Check 4: KPI must be complete, numeric, finite, and non-negative.
    kpi_missing = df[config.kpi_column].isna().sum()
    add_result(
        kpi_missing == 0,
        "KPI Completeness",
        f"{kpi_missing} missing KPI values" if kpi_missing > 0 else "No missing values",
    )

    kpi_numeric = pd.api.types.is_numeric_dtype(df[config.kpi_column])
    add_result(
        kpi_numeric,
        "KPI Type",
        "KPI is numeric" if kpi_numeric else "KPI must be numeric",
    )

    if kpi_numeric:
        kpi_values = df[config.kpi_column]
        non_finite_kpi = int((~np.isfinite(kpi_values.dropna())).sum())
        negative_kpi = int((kpi_values < 0).sum())
        add_result(
            non_finite_kpi == 0,
            "KPI Finite Values",
            "All KPI values are finite"
            if non_finite_kpi == 0
            else f"{non_finite_kpi} non-finite KPI values found",
        )
        add_result(
            negative_kpi == 0,
            "KPI Non-Negativity",
            "All KPI values are non-negative"
            if negative_kpi == 0
            else f"{negative_kpi} negative KPI values found",
        )

    # Check 5: Paid media is required and every spend series must be usable.
    spend_cols = [channel.spend_column for channel in config.media_channels]
    add_result(
        bool(spend_cols),
        "Paid Media Presence",
        f"{len(spend_cols)} paid media channels found"
        if spend_cols
        else "No columns ending in '_spend' were found",
    )

    if spend_cols:
        missing_spend = int(df[spend_cols].isna().sum().sum())
        numeric_spend_cols = [
            column for column in spend_cols if pd.api.types.is_numeric_dtype(df[column])
        ]
        add_result(
            len(numeric_spend_cols) == len(spend_cols),
            "Spend Types",
            "All spend columns are numeric"
            if len(numeric_spend_cols) == len(spend_cols)
            else "Non-numeric spend columns: "
            + ", ".join(sorted(set(spend_cols) - set(numeric_spend_cols))),
        )
        add_result(
            missing_spend == 0,
            "Spend Completeness",
            "No missing spend values"
            if missing_spend == 0
            else f"{missing_spend} missing spend values found",
        )

        if numeric_spend_cols:
            numeric_spend = df[numeric_spend_cols]
            non_finite_spend = int((~np.isfinite(numeric_spend.dropna())).sum().sum())
            negative_spend = int((numeric_spend < 0).sum().sum())
            zero_spend_channels = [
                column for column in numeric_spend_cols if numeric_spend[column].sum() <= 0
            ]
            add_result(
                non_finite_spend == 0,
                "Spend Finite Values",
                "All spend values are finite"
                if non_finite_spend == 0
                else f"{non_finite_spend} non-finite spend values found",
            )
            add_result(
                negative_spend == 0,
                "Spend Non-Negativity",
                "All spend values are non-negative"
                if negative_spend == 0
                else f"{negative_spend} negative spend values found",
            )
            add_result(
                not zero_spend_channels,
                "Channel Spend Totals",
                "Every channel has positive total spend"
                if not zero_spend_channels
                else "Channels with zero total spend: " + ", ".join(zero_spend_channels),
            )

    # Check 6: Every additional model input must be complete, numeric, and finite.
    auxiliary_columns = [
        column
        for channel in config.media_channels
        for column in (
            channel.impressions_column,
            channel.reach_column,
            channel.frequency_column,
        )
        if column
    ]
    auxiliary_columns.extend(channel.column for channel in config.organic_channels)
    auxiliary_columns.extend(config.treatment_columns)
    auxiliary_columns.extend(config.control_columns)
    auxiliary_columns.extend(
        column
        for column in (
            config.population_column,
            config.revenue_column,
            config.revenue_per_kpi_column,
        )
        if column and column in df.columns
    )
    auxiliary_columns = list(dict.fromkeys(auxiliary_columns))

    if auxiliary_columns:
        missing_auxiliary = int(df[auxiliary_columns].isna().sum().sum())
        numeric_auxiliary = [
            column
            for column in auxiliary_columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]
        add_result(
            len(numeric_auxiliary) == len(auxiliary_columns),
            "Additional Input Types",
            "All additional model inputs are numeric"
            if len(numeric_auxiliary) == len(auxiliary_columns)
            else "Non-numeric model inputs: "
            + ", ".join(sorted(set(auxiliary_columns) - set(numeric_auxiliary))),
        )
        add_result(
            missing_auxiliary == 0,
            "Additional Input Completeness",
            "No missing values in additional model inputs"
            if missing_auxiliary == 0
            else f"{missing_auxiliary} missing values found in additional model inputs",
        )
        if numeric_auxiliary:
            non_finite_auxiliary = int(
                (~np.isfinite(df[numeric_auxiliary].dropna())).sum().sum()
            )
            add_result(
                non_finite_auxiliary == 0,
                "Additional Input Finite Values",
                "All additional model inputs are finite"
                if non_finite_auxiliary == 0
                else f"{non_finite_auxiliary} non-finite additional input values found",
            )

    nonnegative_auxiliary = [
        column
        for channel in config.media_channels
        for column in (
            channel.impressions_column,
            channel.reach_column,
            channel.frequency_column,
        )
        if column and pd.api.types.is_numeric_dtype(df[column])
    ]
    nonnegative_auxiliary.extend(
        channel.column
        for channel in config.organic_channels
        if pd.api.types.is_numeric_dtype(df[channel.column])
    )
    nonnegative_auxiliary.extend(
        column
        for column in (
            config.population_column,
            config.revenue_column,
            config.revenue_per_kpi_column,
        )
        if column and column in df.columns and pd.api.types.is_numeric_dtype(df[column])
    )
    nonnegative_auxiliary = list(dict.fromkeys(nonnegative_auxiliary))
    if nonnegative_auxiliary:
        negative_auxiliary = int((df[nonnegative_auxiliary] < 0).sum().sum())
        add_result(
            negative_auxiliary == 0,
            "Additional Input Non-Negativity",
            "Media, population, organic, and revenue inputs are non-negative"
            if negative_auxiliary == 0
            else f"{negative_auxiliary} negative media/population/revenue values found",
        )

    # Check 7: Fewer than three channels is allowed, but weakens separation.
    add_result(
        len(dataset.media_channels) >= 3,
        "Media Channel Count",
        f"{len(dataset.media_channels)} channels",
        "warning" if 0 < len(dataset.media_channels) < 3 else "info",
    )

    # Check 8: Date continuity (no large gaps)
    df_sorted = df.sort_values(config.date_column)
    date_diffs = df_sorted.groupby(config.geo_column)[config.date_column].diff()
    max_gap = date_diffs.max()
    has_gaps = pd.notna(max_gap) and max_gap.days > 14  # More than 2 weeks

    add_result(
        not has_gaps,
        "Date Continuity",
        f"Max gap: {max_gap.days if pd.notna(max_gap) else 0} days",
        "warning" if has_gaps else "info",
    )

    # Check 9: KPI variance (should have some variation)
    if kpi_numeric:
        kpi_mean = df[config.kpi_column].mean()
        kpi_cv = df[config.kpi_column].std() / kpi_mean if kpi_mean else np.nan
        low_variance = not np.isfinite(kpi_cv) or kpi_cv < 0.1
        add_result(
            not low_variance,
            "KPI Variance",
            f"Coefficient of variation: {kpi_cv:.2%}"
            if np.isfinite(kpi_cv)
            else "KPI mean is zero; variance is not usable",
            "warning" if low_variance else "info",
        )

    if config.revenue_column and config.revenue_column in df.columns and kpi_numeric:
        invalid_revenue_rows = int(
            ((df[config.kpi_column] == 0) & (df[config.revenue_column] != 0)).sum()
        )
        add_result(
            invalid_revenue_rows == 0,
            "Revenue-to-KPI Consistency",
            "Revenue can be converted to revenue per KPI"
            if invalid_revenue_rows == 0
            else f"{invalid_revenue_rows} rows have revenue but zero KPI",
        )

    # Compile report
    errors = sum(1 for r in results if not r.passed and r.severity == "error")
    warnings = sum(1 for r in results if not r.passed and r.severity == "warning")
    passed = errors == 0

    return ValidationReport(
        results=results,
        passed=passed,
        errors=errors,
        warnings=warnings,
    )


def check_meridian_compatibility(dataset: MMMDataset) -> list[str]:
    """
    Check if dataset is compatible with Meridian's InputData requirements.

    Returns:
        List of issues (empty if compatible)
    """
    issues = []
    df = dataset.df
    config = dataset.config

    # Meridian requires specific column types
    if not pd.api.types.is_datetime64_any_dtype(df[config.date_column]):
        issues.append(f"Date column '{config.date_column}' must be datetime type")

    # Check for required numeric types on spend columns
    for ch in config.media_channels:
        spend_col = ch.spend_column
        if not pd.api.types.is_numeric_dtype(df[spend_col]):
            issues.append(f"Spend column '{spend_col}' must be numeric")

    # KPI must be numeric
    if not pd.api.types.is_numeric_dtype(df[config.kpi_column]):
        issues.append(f"KPI column '{config.kpi_column}' must be numeric")

    return issues
