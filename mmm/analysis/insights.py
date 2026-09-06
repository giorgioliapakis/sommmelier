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

    def to_dict(self) -> dict[str, str | None]:
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
    from mmm.recommendations.engine import analyze_roi
    from mmm.result_manifest import decision_readiness

    payload = results.to_result_payload()
    ready, reason = decision_readiness(payload)
    if not ready:
        return [
            Insight(
                type=InsightType.EFFICIENCY,
                priority=InsightPriority.HIGH,
                channel=None,
                title="Recommendations blocked",
                description=reason,
                recommendation="Review diagnostics and refit before making budget decisions.",
            )
        ]

    return [
        Insight(
            type=InsightType.EFFICIENCY,
            priority=InsightPriority.MEDIUM,
            channel=None,
            title=recommendation.title,
            description=recommendation.detail,
            recommendation=recommendation.action,
        )
        for recommendation in analyze_roi(payload)
    ]


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
            lines.extend(
                [
                    f"### {insight.title}",
                    "",
                    insight.description,
                    "",
                    f"**Recommendation:** {insight.recommendation}",
                    "",
                    f"*Impact: {insight.potential_impact}*" if insight.potential_impact else "",
                    "",
                ]
            )

    if medium_priority:
        lines.extend(["## Medium Priority", ""])
        for insight in medium_priority:
            lines.extend(
                [
                    f"### {insight.title}",
                    "",
                    insight.description,
                    "",
                    f"**Recommendation:** {insight.recommendation}",
                    "",
                ]
            )

    if low_priority:
        lines.extend(["## Notes", ""])
        for insight in low_priority:
            lines.extend(
                [
                    f"- **{insight.title}**: {insight.description}",
                    "",
                ]
            )

    return "\n".join(lines)
