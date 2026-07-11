"""Report-generation safety tests."""

from mmm.analysis.visualize import generate_html_report


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
    }

    generate_html_report(results, output)
    html = output.read_text()

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
