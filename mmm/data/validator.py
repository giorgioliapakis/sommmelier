"""Data validation utilities for Sommmelier."""

from dataclasses import dataclass

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
            icon = "" if result.passed else "" if result.severity == "error" else ""
            lines.append(f"{icon} {result.check_name}: {result.message}")

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

    # Check 1: Minimum time periods (Meridian recommends 2+ years)
    min_periods = 52  # ~1 year of weekly data
    results.append(
        ValidationResult(
            passed=dataset.n_time_periods >= min_periods,
            check_name="Minimum Time Periods",
            message=f"{dataset.n_time_periods} periods (min recommended: {min_periods})",
            severity="warning" if dataset.n_time_periods >= 26 else "error",
        )
    )

    # Check 2: At least 2 geos for geo-level modeling
    results.append(
        ValidationResult(
            passed=dataset.n_geos >= 2,
            check_name="Geographic Coverage",
            message=f"{dataset.n_geos} geos found",
            severity="warning" if dataset.n_geos == 1 else "error" if dataset.n_geos == 0 else "info",
        )
    )

    # Check 3: No missing KPI values
    kpi_missing = df[config.kpi_column].isna().sum()
    results.append(
        ValidationResult(
            passed=kpi_missing == 0,
            check_name="KPI Completeness",
            message=f"{kpi_missing} missing KPI values" if kpi_missing > 0 else "No missing values",
            severity="error",
        )
    )

    # Check 4: No negative spend values
    spend_cols = [
        ch["spend_column"] if isinstance(ch, dict) else ch.spend_column
        for ch in config.media_channels
    ]
    negative_spend = (df[spend_cols] < 0).any().any()
    results.append(
        ValidationResult(
            passed=not negative_spend,
            check_name="Spend Values",
            message="Negative spend values found" if negative_spend else "All spend values >= 0",
            severity="error",
        )
    )

    # Check 5: At least 3 media channels
    results.append(
        ValidationResult(
            passed=len(dataset.media_channels) >= 3,
            check_name="Media Channel Count",
            message=f"{len(dataset.media_channels)} channels",
            severity="warning" if len(dataset.media_channels) < 3 else "info",
        )
    )

    # Check 6: Date continuity (no large gaps)
    df_sorted = df.sort_values(config.date_column)
    date_diffs = df_sorted.groupby(config.geo_column)[config.date_column].diff()
    max_gap = date_diffs.max()
    has_gaps = pd.notna(max_gap) and max_gap.days > 14  # More than 2 weeks

    results.append(
        ValidationResult(
            passed=not has_gaps,
            check_name="Date Continuity",
            message=f"Max gap: {max_gap.days if pd.notna(max_gap) else 0} days",
            severity="warning" if has_gaps else "info",
        )
    )

    # Check 7: KPI variance (should have some variation)
    kpi_cv = df[config.kpi_column].std() / df[config.kpi_column].mean()
    low_variance = kpi_cv < 0.1
    results.append(
        ValidationResult(
            passed=not low_variance,
            check_name="KPI Variance",
            message=f"Coefficient of variation: {kpi_cv:.2%}",
            severity="warning" if low_variance else "info",
        )
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
        spend_col = ch["spend_column"] if isinstance(ch, dict) else ch.spend_column
        if not pd.api.types.is_numeric_dtype(df[spend_col]):
            issues.append(f"Spend column '{spend_col}' must be numeric")

    # KPI must be numeric
    if not pd.api.types.is_numeric_dtype(df[config.kpi_column]):
        issues.append(f"KPI column '{config.kpi_column}' must be numeric")

    return issues
