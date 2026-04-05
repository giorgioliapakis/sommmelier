"""
Recommendation engine for Sommmelier.

Analyzes MMM results and generates actionable recommendations
for both marketing strategy and model improvements.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .improvement_advisor import (
    ImprovementQuestion,
    generate_improvement_questions,
    format_questions_for_user,
    format_questions_as_checklist,
)


@dataclass
class Recommendation:
    """A single recommendation."""
    category: str  # "budget", "channel", "model", "data"
    priority: str  # "high", "medium", "low"
    title: str
    detail: str
    action: str
    impact: Optional[str] = None  # Estimated impact if known


@dataclass
class AnalysisReport:
    """Complete analysis report with recommendations."""
    timestamp: str
    summary: str
    recommendations: list[Recommendation] = field(default_factory=list)
    improvement_questions: list[ImprovementQuestion] = field(default_factory=list)
    budget_reallocation: dict = field(default_factory=dict)
    model_health: dict = field(default_factory=dict)
    week_over_week: dict = field(default_factory=dict)


def load_results(path: Path | str) -> dict:
    """Load results from JSON file."""
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def load_historical_results(outputs_dir: Path | str) -> list[dict]:
    """Load all historical results, sorted by timestamp."""
    outputs_dir = Path(outputs_dir)
    results = []

    for f in outputs_dir.glob("full_results_*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                data["_file"] = str(f)
                results.append(data)
        except Exception:
            continue

    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""))
    return results


def analyze_roi(results: dict) -> list[Recommendation]:
    """Analyze ROI and generate recommendations."""
    recommendations = []
    roi_data = results.get("roi", {})

    if not roi_data:
        # Handle simple format
        roi_data = {
            ch: {"mean": val}
            for ch, val in results.get("channel_roi", {}).items()
        }

    if not roi_data:
        return recommendations

    # Sort channels by ROI
    sorted_channels = sorted(
        roi_data.items(),
        key=lambda x: x[1].get("mean", 0) if isinstance(x[1], dict) else x[1],
        reverse=True
    )

    # Check for underperforming channels (ROI < 0.5)
    for ch, data in sorted_channels:
        roi = data.get("mean", data) if isinstance(data, dict) else data

        if roi < 0.3:
            recommendations.append(Recommendation(
                category="channel",
                priority="high",
                title=f"Consider pausing {ch.title()}",
                detail=f"{ch.title()} has an ROI of {roi:.2f}x, meaning you lose ${1-roi:.2f} for every $1 spent.",
                action=f"Reduce or pause {ch} spend and reallocate to higher-performing channels.",
                impact=f"Could save ~${results.get('metadata', {}).get('total_spend', {}).get(ch, 0) * (1-roi):,.0f}"
            ))
        elif roi < 0.5:
            recommendations.append(Recommendation(
                category="channel",
                priority="medium",
                title=f"{ch.title()} is underperforming",
                detail=f"ROI of {roi:.2f}x is below breakeven. Review targeting and creative.",
                action="Test new audiences or creative before cutting budget.",
                impact=None
            ))

    # Check for high-performing channels that could scale
    best_ch, best_data = sorted_channels[0]
    best_roi = best_data.get("mean", best_data) if isinstance(best_data, dict) else best_data

    if best_roi > 1.0:
        recommendations.append(Recommendation(
            category="channel",
            priority="high",
            title=f"{best_ch.title()} is highly profitable",
            detail=f"ROI of {best_roi:.2f}x means every $1 returns ${best_roi:.2f}.",
            action="Test increasing budget if marginal ROI supports it.",
            impact=None
        ))

    return recommendations


def analyze_marginal_roi(results: dict) -> list[Recommendation]:
    """Analyze marginal ROI to find saturation and growth opportunities."""
    recommendations = []

    roi_data = results.get("roi", {})
    mroi_data = results.get("marginal_roi", {})

    if not roi_data:
        roi_data = {
            ch: {"mean": val}
            for ch, val in results.get("channel_roi", {}).items()
        }

    if not mroi_data:
        return recommendations

    for ch, mroi in mroi_data.items():
        avg_roi = roi_data.get(ch, {})
        avg_roi = avg_roi.get("mean", avg_roi) if isinstance(avg_roi, dict) else avg_roi

        if avg_roi == 0:
            continue

        ratio = mroi / avg_roi if avg_roi > 0 else 0

        if ratio < 0.5:
            # Saturated - marginal ROI much lower than average
            recommendations.append(Recommendation(
                category="budget",
                priority="high",
                title=f"{ch.title()} is saturated",
                detail=f"Marginal ROI ({mroi:.2f}x) is {(1-ratio)*100:.0f}% lower than average ROI ({avg_roi:.2f}x).",
                action=f"Reduce {ch} budget - you're past the point of diminishing returns.",
                impact="Reallocating budget could improve overall ROAS"
            ))
        elif ratio > 1.2:
            # Room to grow - marginal ROI higher than average
            recommendations.append(Recommendation(
                category="budget",
                priority="medium",
                title=f"{ch.title()} has room to scale",
                detail=f"Marginal ROI ({mroi:.2f}x) exceeds average ROI ({avg_roi:.2f}x).",
                action=f"Test increasing {ch} budget - additional spend should be efficient.",
                impact=None
            ))

    return recommendations


def analyze_contributions(results: dict) -> list[Recommendation]:
    """Analyze contribution distribution."""
    recommendations = []

    contrib_data = results.get("contributions", {})
    if not contrib_data:
        contrib_data = {
            ch: {"percentage": val * 100}
            for ch, val in results.get("channel_contributions", {}).items()
        }

    if not contrib_data:
        return recommendations

    # Normalize to ensure we have percentage
    def get_pct(data):
        if isinstance(data, dict):
            return data.get("percentage", 0)
        return data * 100 if data < 1 else data

    # Check for over-concentration
    sorted_contrib = sorted(
        contrib_data.items(),
        key=lambda x: get_pct(x[1]),
        reverse=True
    )

    top_ch, top_data = sorted_contrib[0]
    top_pct = get_pct(top_data)

    if top_pct > 70:
        recommendations.append(Recommendation(
            category="budget",
            priority="medium",
            title="High channel concentration risk",
            detail=f"{top_ch.title()} drives {top_pct:.0f}% of conversions. This creates platform dependency.",
            action="Consider diversifying budget across channels to reduce risk.",
            impact=None
        ))

    return recommendations


def analyze_model_quality(results: dict) -> tuple[dict, list[Recommendation]]:
    """Assess model quality and suggest improvements."""
    recommendations = []
    health = {
        "convergence": "unknown",
        "data_sufficiency": "unknown",
        "confidence": "unknown",
    }

    metadata = results.get("metadata", {})
    diagnostics = results.get("diagnostics", {})

    # Check convergence
    if diagnostics.get("convergence_ok"):
        health["convergence"] = "good"
    elif diagnostics.get("rhat_warnings", 0) > 0:
        health["convergence"] = "warning"
        recommendations.append(Recommendation(
            category="model",
            priority="high",
            title="Model convergence issues detected",
            detail=f"{diagnostics['rhat_warnings']} parameters have R-hat > 1.1, indicating MCMC didn't converge.",
            action="Run with more samples (n_keep=1000) or check for data issues.",
            impact="Results may be unreliable"
        ))

    # Check data sufficiency
    n_periods = metadata.get("n_time_periods", 0)
    if n_periods < 26:
        health["data_sufficiency"] = "insufficient"
        recommendations.append(Recommendation(
            category="data",
            priority="high" if n_periods < 13 else "medium",
            title="Insufficient time periods",
            detail=f"Only {n_periods} periods. Meridian recommends 26+ for reliable estimates.",
            action="Accumulate more weekly data before making major budget decisions.",
            impact="Wide confidence intervals, uncertain estimates"
        ))
    elif n_periods < 52:
        health["data_sufficiency"] = "adequate"
    else:
        health["data_sufficiency"] = "good"

    # Check confidence intervals
    roi_data = results.get("roi", {})
    wide_ci_channels = []
    for ch, data in roi_data.items():
        if isinstance(data, dict):
            ci_lo = data.get("ci_lower", 0)
            ci_hi = data.get("ci_upper", 0)
            mean = data.get("mean", 0)
            if mean > 0:
                ci_width = (ci_hi - ci_lo) / mean
                if ci_width > 1.5:  # CI wider than 150% of mean
                    wide_ci_channels.append(ch)

    if wide_ci_channels:
        health["confidence"] = "low"
        recommendations.append(Recommendation(
            category="model",
            priority="medium",
            title="High uncertainty in estimates",
            detail=f"Channels with wide confidence intervals: {', '.join(wide_ci_channels)}",
            action="Need more data or adjust priors for these channels.",
            impact="Recommendations for these channels are less certain"
        ))
    else:
        health["confidence"] = "good"

    return health, recommendations


def calculate_budget_reallocation(results: dict) -> dict:
    """Calculate suggested budget reallocation based on marginal ROI."""
    reallocation = {
        "current": {},
        "suggested": {},
        "change": {},
    }

    metadata = results.get("metadata", {})
    mroi_data = results.get("marginal_roi", {})
    current_spend = metadata.get("total_spend", {})

    if not mroi_data or not current_spend:
        return reallocation

    reallocation["current"] = dict(current_spend)

    # Simple reallocation: shift budget proportional to marginal ROI
    total_spend = sum(current_spend.values())
    total_mroi = sum(mroi_data.values())

    if total_mroi == 0:
        return reallocation

    for ch in current_spend:
        mroi = mroi_data.get(ch, 0)
        # Allocate proportional to marginal ROI
        suggested = total_spend * (mroi / total_mroi)
        reallocation["suggested"][ch] = round(suggested, 2)
        reallocation["change"][ch] = round(suggested - current_spend[ch], 2)

    return reallocation


def compare_to_previous(current: dict, previous: dict) -> dict:
    """Compare current results to previous run."""
    comparison = {
        "has_previous": previous is not None,
        "roi_changes": {},
        "contribution_changes": {},
    }

    if not previous:
        return comparison

    # Compare ROI
    curr_roi = current.get("roi", current.get("channel_roi", {}))
    prev_roi = previous.get("roi", previous.get("channel_roi", {}))

    for ch in curr_roi:
        curr_val = curr_roi[ch].get("mean", curr_roi[ch]) if isinstance(curr_roi[ch], dict) else curr_roi[ch]
        prev_val = prev_roi.get(ch, {})
        prev_val = prev_val.get("mean", prev_val) if isinstance(prev_val, dict) else prev_val

        if prev_val:
            change = (curr_val - prev_val) / prev_val * 100
            comparison["roi_changes"][ch] = {
                "previous": prev_val,
                "current": curr_val,
                "change_pct": change,
            }

    return comparison


def generate_analysis(results_path: Path | str, outputs_dir: Path | str = "outputs") -> AnalysisReport:
    """
    Generate complete analysis report with recommendations.

    This is the main entry point for Claude Code to analyze results.
    """
    results = load_results(results_path)

    # Load historical results for comparison
    historical = load_historical_results(outputs_dir)
    previous = historical[-2] if len(historical) >= 2 else None

    # Generate recommendations from different analyzers
    all_recommendations = []
    all_recommendations.extend(analyze_roi(results))
    all_recommendations.extend(analyze_marginal_roi(results))
    all_recommendations.extend(analyze_contributions(results))

    # Analyze model quality
    model_health, model_recs = analyze_model_quality(results)
    all_recommendations.extend(model_recs)

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_recommendations.sort(key=lambda x: priority_order.get(x.priority, 99))

    # Calculate budget reallocation
    budget_reallocation = calculate_budget_reallocation(results)

    # Compare to previous
    week_over_week = compare_to_previous(results, previous)

    # Generate improvement questions (the self-improving part)
    improvement_questions = generate_improvement_questions(results)

    # Generate summary
    high_priority = [r for r in all_recommendations if r.priority == "high"]
    high_priority_questions = [q for q in improvement_questions if q.priority == "high"]

    if high_priority:
        summary = f"Found {len(high_priority)} high-priority issues requiring attention."
    else:
        summary = "No critical issues. Model is healthy and channels are performing as expected."

    if high_priority_questions:
        summary += f" {len(high_priority_questions)} high-impact improvements available."

    return AnalysisReport(
        timestamp=datetime.now().isoformat(),
        summary=summary,
        recommendations=all_recommendations,
        improvement_questions=improvement_questions,
        budget_reallocation=budget_reallocation,
        model_health=model_health,
        week_over_week=week_over_week,
    )


def format_report_for_claude(report: AnalysisReport) -> str:
    """Format the analysis report as readable text for Claude to process."""
    lines = []
    lines.append("=" * 60)
    lines.append("SOMMMELIER ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"\nGenerated: {report.timestamp}")
    lines.append(f"\nSUMMARY: {report.summary}")

    # Recommendations
    lines.append("\n" + "-" * 40)
    lines.append("RECOMMENDATIONS")
    lines.append("-" * 40)

    for i, rec in enumerate(report.recommendations, 1):
        priority_marker = {"high": "[!]", "medium": "[*]", "low": "[-]"}.get(rec.priority, "[ ]")
        lines.append(f"\n{i}. {priority_marker} {rec.title}")
        lines.append(f"   Category: {rec.category}")
        lines.append(f"   Detail: {rec.detail}")
        lines.append(f"   Action: {rec.action}")
        if rec.impact:
            lines.append(f"   Impact: {rec.impact}")

    # Budget reallocation
    if report.budget_reallocation.get("suggested"):
        lines.append("\n" + "-" * 40)
        lines.append("SUGGESTED BUDGET REALLOCATION")
        lines.append("-" * 40)

        for ch in report.budget_reallocation["current"]:
            current = report.budget_reallocation["current"][ch]
            suggested = report.budget_reallocation["suggested"].get(ch, current)
            change = report.budget_reallocation["change"].get(ch, 0)
            direction = "+" if change > 0 else ""
            lines.append(f"  {ch:12s}: ${current:>10,.0f} → ${suggested:>10,.0f} ({direction}{change:,.0f})")

    # Model health
    lines.append("\n" + "-" * 40)
    lines.append("MODEL HEALTH")
    lines.append("-" * 40)
    for metric, status in report.model_health.items():
        status_icon = {"good": "[OK]", "warning": "[!]", "adequate": "[~]"}.get(status, "[?]")
        lines.append(f"  {metric:20s}: {status_icon} {status}")

    # Week over week (if available)
    if report.week_over_week.get("has_previous"):
        lines.append("\n" + "-" * 40)
        lines.append("WEEK OVER WEEK CHANGES")
        lines.append("-" * 40)
        for ch, data in report.week_over_week.get("roi_changes", {}).items():
            change = data["change_pct"]
            direction = "+" if change > 0 else ""
            lines.append(f"  {ch:12s} ROI: {data['previous']:.2f}x → {data['current']:.2f}x ({direction}{change:.1f}%)")

    # Improvement questions (the self-improving part)
    if report.improvement_questions:
        lines.append("\n" + "-" * 40)
        lines.append("HOW TO IMPROVE THIS MODEL")
        lines.append("-" * 40)
        lines.append("\nProvide any of this data to improve accuracy:\n")

        for i, q in enumerate(report.improvement_questions[:5], 1):
            priority_marker = {"high": "[HIGH IMPACT]", "medium": "[MEDIUM]", "low": "[NICE TO HAVE]"}.get(q.priority, "")
            lines.append(f"{i}. {priority_marker} {q.question}")
            lines.append(f"   Why: {q.why_it_helps[:150]}...")
            if q.impact_estimate:
                lines.append(f"   Impact: {q.impact_estimate}")
            lines.append("")

        remaining = len(report.improvement_questions) - 5
        if remaining > 0:
            lines.append(f"({remaining} more suggestions available)")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
