"""Tests for recommendation safety and optimizer handoff."""

import json

import pytest

from mmm.recommendations.engine import (
    analyze_contributions,
    analyze_marginal_roi,
    analyze_model_quality,
    analyze_roi,
    calculate_budget_reallocation,
    compare_to_previous,
    format_report_for_claude,
    generate_analysis,
    load_historical_results,
    load_results,
)
from mmm.recommendations.improvement_advisor import generate_improvement_questions


def test_non_monetary_roi_is_not_compared_to_breakeven():
    recommendations = analyze_roi(
        {
            "metadata": {"roi_is_monetary": False},
            "roi": {"meta": {"mean": 0.2}, "google": {"mean": 0.05}},
        }
    )

    assert len(recommendations) == 1
    assert "not a monetary return" in recommendations[0].title
    assert "pause" not in recommendations[0].action.lower()


def test_budget_reallocation_uses_meridian_optimizer_output():
    result = calculate_budget_reallocation(
        {
            "metadata": {"total_spend": {"meta": 100.0, "google": 100.0}},
            "marginal_roi": {"meta": 100.0, "google": 1.0},
            "optimization": {
                "current": {
                    "optimal_allocation": {"meta": 80.0, "google": 120.0},
                }
            },
        }
    )

    assert result["suggested"] == {"meta": 80.0, "google": 120.0}
    assert result["change"] == {"meta": -20.0, "google": 20.0}


def test_budget_reallocation_does_not_invent_allocation_without_optimizer():
    result = calculate_budget_reallocation(
        {
            "metadata": {"total_spend": {"meta": 100.0, "google": 100.0}},
            "marginal_roi": {"meta": 2.0, "google": 1.0},
        }
    )

    assert result["suggested"] == {}


def test_library_analysis_blocks_non_decision_ready_results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "run_manifest": {
                    "status": "complete",
                    "quality_status": "failed",
                }
            }
        )
    )

    with pytest.raises(ValueError, match="Recommendations blocked"):
        generate_analysis(path, tmp_path)


def test_impression_question_only_appears_when_fallback_was_used():
    base = {"metadata": {"estimated_impression_channels": []}}
    with_real_execution = generate_improvement_questions(base)
    with_estimate = generate_improvement_questions(
        {"metadata": {"estimated_impression_channels": ["meta"]}}
    )

    assert not any("impression data" in question.question for question in with_real_execution)
    assert any("impression data" in question.question for question in with_estimate)


def test_monetary_roi_flags_losses_and_scaling_opportunities():
    recommendations = analyze_roi(
        {
            "metadata": {
                "roi_is_monetary": True,
                "total_spend": {"display": 1000},
            },
            "roi": {
                "search": {"mean": 1.5},
                "social": {"mean": 0.4},
                "display": {"mean": 0.2},
            },
        }
    )

    assert {recommendation.title for recommendation in recommendations} == {
        "Search is highly profitable",
        "Social is underperforming",
        "Consider pausing Display",
    }
    display = next(item for item in recommendations if "Display" in item.title)
    assert display.impact == "Could save ~$800"


def test_marginal_roi_and_concentration_surface_budget_risks():
    marginal = analyze_marginal_roi(
        {
            "roi": {"meta": {"mean": 2.0}, "search": {"mean": 1.0}},
            "marginal_roi": {"meta": 0.5, "search": 1.5},
        }
    )
    concentration = analyze_contributions({"channel_contributions": {"meta": 0.8, "search": 0.2}})

    assert [item.title for item in marginal] == [
        "Meta is saturated",
        "Search has room to scale",
    ]
    assert concentration[0].title == "High channel concentration risk"


def test_model_quality_reports_convergence_data_and_uncertainty():
    health, recommendations = analyze_model_quality(
        {
            "metadata": {"n_time_periods": 12},
            "diagnostics": {"convergence_ok": False, "rhat_warnings": 3},
            "roi": {"meta": {"mean": 1.0, "ci_lower": 0.0, "ci_upper": 2.0}},
        }
    )

    assert health == {
        "convergence": "warning",
        "data_sufficiency": "insufficient",
        "confidence": "low",
    }
    assert {item.title for item in recommendations} == {
        "Model convergence issues detected",
        "Insufficient time periods",
        "High uncertainty in estimates",
    }


def test_previous_comparison_includes_roi_and_contribution_changes():
    comparison = compare_to_previous(
        {
            "roi": {"meta": {"mean": 1.5}},
            "contributions": {"meta": {"percentage": 60}},
        },
        {
            "roi": {"meta": {"mean": 1.0}},
            "channel_contributions": {"meta": 0.5},
        },
    )

    assert comparison["roi_changes"]["meta"]["change_pct"] == pytest.approx(50)
    assert comparison["contribution_changes"]["meta"] == {
        "previous": 50.0,
        "current": 60.0,
        "change_percentage_points": 10.0,
    }


def test_generate_analysis_selects_latest_earlier_run(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manifest = {"status": "complete", "quality_status": "passed"}
    for name, timestamp, roi in (
        ("full_results_old.json", "2026-01-01T00:00:00Z", 1.0),
        ("full_results_current.json", "2026-01-03T00:00:00Z", 1.5),
        ("full_results_future.json", "2026-01-04T00:00:00Z", 2.0),
    ):
        (outputs / name).write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "run_manifest": manifest,
                    "metadata": {"roi_is_monetary": True, "n_time_periods": 52},
                    "roi": {"meta": {"mean": roi}},
                    "contributions": {"meta": {"percentage": 50}},
                    "diagnostics": {"convergence_ok": True},
                    "model_fit": {"r_squared": 0.8, "mape": 0.1},
                }
            )
        )

    report = generate_analysis(outputs / "full_results_current.json", outputs)

    assert report.week_over_week["roi_changes"]["meta"]["previous"] == 1.0
    assert report.week_over_week["roi_changes"]["meta"]["current"] == 1.5
    assert "contribution:" in format_report_for_claude(report)


def test_result_loaders_reject_non_objects_and_skip_invalid_history(tmp_path):
    invalid = tmp_path / "full_results_invalid.json"
    invalid.write_text("[]")
    broken = tmp_path / "full_results_broken.json"
    broken.write_text("not-json")
    valid = tmp_path / "full_results_valid.json"
    valid.write_text('{"timestamp": "2026-01-01"}')

    with pytest.raises(ValueError, match="must contain an object"):
        load_results(invalid)

    historical = load_historical_results(tmp_path)
    assert len(historical) == 1
    assert historical[0]["_file"] == str(valid)
