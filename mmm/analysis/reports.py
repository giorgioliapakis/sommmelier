"""Report generation for Sommmelier."""

from datetime import datetime
from pathlib import Path

from mmm.analysis.insights import generate_insights, insights_to_markdown
from mmm.data.schema import MMMDataset
from mmm.model.mmm import AutoMMM, ModelResults


def generate_report(
    mmm: AutoMMM,
    output_path: str | Path | None = None,
    include_meridian_summary: bool = True,
) -> str:
    """
    Generate a comprehensive MMM report.

    Args:
        mmm: Fitted AutoMMM instance
        output_path: Optional path to save HTML report
        include_meridian_summary: Whether to include Meridian's native summary

    Returns:
        Report content as string (markdown)
    """
    if not mmm.is_fitted:
        raise ValueError("Model must be fitted before generating report")

    results = mmm.results
    dataset = mmm.dataset

    # Build report sections
    sections = []

    # Header
    sections.append(f"""# Sommmelier Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

---
""")

    # Data Summary
    sections.append(f"""## Data Summary

| Metric | Value |
|--------|-------|
| Date Range | {dataset.date_range[0]} to {dataset.date_range[1]} |
| Time Periods | {dataset.n_time_periods} |
| Geographies | {dataset.n_geos} |
| Media Channels | {len(dataset.media_channels)} |
| Total Spend | ${dataset.total_spend:,.2f} |
| Total KPI | {dataset.total_kpi:,.0f} |

### Media Channels
{', '.join(dataset.media_channels)}

---
""")

    # Model Results
    sections.append("""## Model Results

### Channel ROI
| Channel | ROI | Interpretation |
|---------|-----|----------------|
""")

    if results and results.channel_roi:
        for channel, roi in sorted(results.channel_roi.items(), key=lambda x: -x[1]):
            interpretation = (
                "Strong performer" if roi > 1.5
                else "Good" if roi > 1.0
                else "Break-even" if roi > 0.8
                else "Underperforming"
            )
            sections.append(f"| {channel} | {roi:.2f}x | {interpretation} |\n")

    sections.append("\n")

    # Contribution breakdown
    if results and results.channel_contributions:
        sections.append("""### Channel Contribution to KPI
| Channel | Contribution % |
|---------|---------------|
""")
        total = sum(results.channel_contributions.values())
        for channel, contrib in sorted(results.channel_contributions.items(), key=lambda x: -x[1]):
            pct = contrib / total * 100 if total > 0 else 0
            sections.append(f"| {channel} | {pct:.1f}% |\n")

        sections.append("\n---\n\n")

    # AI Insights
    if results:
        # Get spend per channel for efficiency analysis
        config = dataset.config
        df = dataset.df
        channel_spend = {}
        for ch in config.media_channels:
            spend_col = ch["spend_column"] if isinstance(ch, dict) else ch.spend_column
            channel_name = ch["name"] if isinstance(ch, dict) else ch.name
            channel_spend[channel_name] = df[spend_col].sum()

        insights = generate_insights(results, channel_spend)
        sections.append(insights_to_markdown(insights))

    # Meridian native summary
    if include_meridian_summary and mmm._meridian is not None:
        try:
            from meridian.analysis import summarizer

            mmm_summarizer = summarizer.Summarizer(mmm._meridian)

            sections.append("""
---

## Technical Details

*See Meridian's native summary output for detailed statistical results.*
""")
        except Exception:
            pass

    report_content = "".join(sections)

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_content)

    return report_content


def generate_quick_summary(results: ModelResults, dataset: MMMDataset) -> str:
    """
    Generate a quick one-paragraph summary of MMM results.

    Useful for Slack/email updates or AI-generated summaries.
    """
    if not results.channel_roi:
        return "Model results not available. Please ensure the model has been fitted."

    # Find best and worst channels
    sorted_roi = sorted(results.channel_roi.items(), key=lambda x: x[1], reverse=True)
    best_channel, best_roi = sorted_roi[0]
    worst_channel, worst_roi = sorted_roi[-1]

    # Calculate total efficiency
    total_roi = sum(results.channel_roi.values()) / len(results.channel_roi)

    summary = f"""**MMM Summary ({dataset.date_range[0]} to {dataset.date_range[1]})**

Analyzed ${dataset.total_spend:,.0f} in spend across {len(dataset.media_channels)} channels driving {dataset.total_kpi:,.0f} conversions.

Top performer: **{best_channel}** with {best_roi:.2f}x ROI.
{"Opportunity: " + worst_channel + " at " + f"{worst_roi:.2f}x ROI needs review." if worst_roi < 1.0 else ""}

Average channel ROI: {total_roi:.2f}x.
"""

    return summary
