"""Report-generation safety tests."""

from mmm.analysis.visualize import generate_html_report, generate_roi_chart_svg


def test_html_report_escapes_channel_names(tmp_path):
    output = tmp_path / "report.html"
    results = {
        "timestamp": "2026-07-11T00:00:00",
        "metadata": {
            "channels": ["<script>alert(1)</script>"],
            "n_time_periods": 52,
            "n_geos": 2,
            "total_spend": {"<script>alert(1)</script>": 100},
            "total_kpi": 200,
            "roi_is_monetary": True,
        },
        "roi": {"<script>alert(1)</script>": {"mean": 1.2}},
        "contributions": {
            "<script>alert(1)</script>": {"percentage": 100, "absolute": 200}
        },
        "run_manifest": {"status": "complete", "quality_status": "failed"},
    }

    generate_html_report(results, output)
    html = output.read_text()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Model quality failed — do not use for decisions" in html


def test_non_monetary_efficiency_chart_does_not_label_values_as_roi():
    chart = generate_roi_chart_svg(
        {
            "metadata": {"roi_is_monetary": False},
            "roi": {"meta": {"mean": 0.8, "ci_lower": 0.5, "ci_upper": 1.1}},
        }
    )

    assert "0.80 KPI/currency" in chart
    assert "0.80x" not in chart
