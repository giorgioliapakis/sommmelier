"""AI-ready insights generation from MMM results."""

from dataclasses import dataclass
from enum import Enum

from mmm.model.mmm import ModelResults


class InsightType(Enum):
    """Types of insights that can be generated."""

    HIGH_ROI = "high_roi"
    LOW_ROI = "low_roi"
    OVER_INVESTED = "over_invested"
    UNDER_INVESTED = "under_invested"
    SATURATION = "saturation"
    EFFICIENCY = "efficiency"


class InsightPriority(Enum):
    """Priority level for insights."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Insight:
    """A single actionable insight from MMM results."""

    type: InsightType
    priority: InsightPriority
    channel: str | None
    title: str
    description: str
    recommendation: str
    potential_impact: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "priority": self.priority.value,
            "channel": self.channel,
            "title": self.title,
            "description": self.description,
            "recommendation": self.recommendation,
            "potential_impact": self.potential_impact,
        }


def generate_insights(
    results: ModelResults,
    channel_spend: dict[str, float] | None = None,
) -> list[Insight]:
    """
    Generate actionable insights from MMM results.

    This is the core "AI layer" - turning model outputs into
    plain-language recommendations.

    Args:
        results: ModelResults from a fitted AutoMMM
        channel_spend: Optional dict of channel -> spend for efficiency analysis

    Returns:
        List of prioritized Insight objects
    """
    insights: list[Insight] = []

    if not results.channel_roi:
        return insights

    # Sort channels by ROI
    sorted_roi = sorted(results.channel_roi.items(), key=lambda x: x[1], reverse=True)

    # Insight 1: Identify high ROI channels
    high_roi_threshold = 1.5
    for channel, roi in sorted_roi:
        if roi >= high_roi_threshold:
            insights.append(
                Insight(
                    type=InsightType.HIGH_ROI,
                    priority=InsightPriority.HIGH,
                    channel=channel,
                    title=f"{channel} shows strong ROI",
                    description=f"{channel} has an ROI of {roi:.2f}x, meaning every $1 spent returns ${roi:.2f} in value.",
                    recommendation=f"Consider increasing {channel} budget if not already at saturation.",
                    potential_impact=f"Potential {((roi - 1) * 100):.0f}% return on incremental spend",
                )
            )

    # Insight 2: Identify low/negative ROI channels
    low_roi_threshold = 0.8
    for channel, roi in sorted_roi:
        if roi < low_roi_threshold:
            insights.append(
                Insight(
                    type=InsightType.LOW_ROI,
                    priority=InsightPriority.HIGH,
                    channel=channel,
                    title=f"{channel} underperforming",
                    description=f"{channel} has an ROI of {roi:.2f}x, returning less than $1 for every $1 spent.",
                    recommendation=f"Review {channel} strategy. Consider reallocating budget to higher-performing channels.",
                    potential_impact=f"Currently losing ${(1 - roi):.2f} per dollar spent",
                )
            )

    # Insight 3: Contribution vs Spend efficiency (if spend data provided)
    if channel_spend and results.channel_contributions:
        total_spend = sum(channel_spend.values())
        total_contrib = sum(results.channel_contributions.values())

        for channel in results.channel_contributions:
            if channel not in channel_spend:
                continue

            spend_share = channel_spend[channel] / total_spend if total_spend > 0 else 0
            contrib_share = (
                results.channel_contributions[channel] / total_contrib if total_contrib > 0 else 0
            )

            efficiency_ratio = contrib_share / spend_share if spend_share > 0 else 0

            if efficiency_ratio > 1.3:  # Contributing more than fair share
                insights.append(
                    Insight(
                        type=InsightType.UNDER_INVESTED,
                        priority=InsightPriority.MEDIUM,
                        channel=channel,
                        title=f"{channel} may be under-invested",
                        description=f"{channel} receives {spend_share:.1%} of budget but drives {contrib_share:.1%} of results.",
                        recommendation=f"Test increasing {channel} budget allocation.",
                        potential_impact=f"Efficiency ratio: {efficiency_ratio:.2f}x",
                    )
                )
            elif efficiency_ratio < 0.7:  # Contributing less than fair share
                insights.append(
                    Insight(
                        type=InsightType.OVER_INVESTED,
                        priority=InsightPriority.MEDIUM,
                        channel=channel,
                        title=f"{channel} may be over-invested",
                        description=f"{channel} receives {spend_share:.1%} of budget but only drives {contrib_share:.1%} of results.",
                        recommendation=f"Consider reducing {channel} allocation or optimizing creative/targeting.",
                        potential_impact=f"Efficiency ratio: {efficiency_ratio:.2f}x",
                    )
                )

    # Insight 4: Overall model quality
    if results.convergence_passed:
        insights.append(
            Insight(
                type=InsightType.EFFICIENCY,
                priority=InsightPriority.LOW,
                channel=None,
                title="Model converged successfully",
                description="The model passed convergence diagnostics, indicating reliable estimates.",
                recommendation="Results can be used for budget planning with reasonable confidence.",
            )
        )
    elif results.r_hat_max and results.r_hat_max > 1.2:
        insights.append(
            Insight(
                type=InsightType.EFFICIENCY,
                priority=InsightPriority.HIGH,
                channel=None,
                title="Model convergence issues detected",
                description=f"R-hat of {results.r_hat_max:.2f} exceeds 1.2 threshold, indicating potential issues.",
                recommendation="Consider running model with more iterations or reviewing data quality.",
            )
        )

    # Sort by priority
    priority_order = {InsightPriority.HIGH: 0, InsightPriority.MEDIUM: 1, InsightPriority.LOW: 2}
    insights.sort(key=lambda x: priority_order[x.priority])

    return insights


def insights_to_markdown(insights: list[Insight]) -> str:
    """Convert insights to markdown format for reports."""
    if not insights:
        return "No insights generated. Ensure model has been fitted successfully."

    lines = ["# MMM Insights & Recommendations", ""]

    # Group by priority
    high_priority = [i for i in insights if i.priority == InsightPriority.HIGH]
    medium_priority = [i for i in insights if i.priority == InsightPriority.MEDIUM]
    low_priority = [i for i in insights if i.priority == InsightPriority.LOW]

    if high_priority:
        lines.extend(["## High Priority", ""])
        for insight in high_priority:
            lines.extend([
                f"### {insight.title}",
                "",
                insight.description,
                "",
                f"**Recommendation:** {insight.recommendation}",
                "",
                f"*Impact: {insight.potential_impact}*" if insight.potential_impact else "",
                "",
            ])

    if medium_priority:
        lines.extend(["## Medium Priority", ""])
        for insight in medium_priority:
            lines.extend([
                f"### {insight.title}",
                "",
                insight.description,
                "",
                f"**Recommendation:** {insight.recommendation}",
                "",
            ])

    if low_priority:
        lines.extend(["## Notes", ""])
        for insight in low_priority:
            lines.extend([
                f"- **{insight.title}**: {insight.description}",
                "",
            ])

    return "\n".join(lines)
