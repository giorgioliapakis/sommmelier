"""Validate artifacts produced by the paid Modal compatibility smoke test."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_CHARTS = {
    "adstock_decay",
    "contribution_pie",
    "cpik_chart",
    "hill_curves",
    "model_fit",
    "prior_posterior",
    "response_curves",
    "rhat_boxplot",
    "roi_bar_chart",
    "roi_vs_mroi",
}


def latest_result(outputs_dir: Path) -> Path:
    """Return the newest full result produced in an output directory."""
    candidates = list(outputs_dir.glob("full_results_*.json"))
    if not candidates:
        raise ValueError(f"No full_results_*.json files found in {outputs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def verify_modal_smoke(result_path: Path) -> dict[str, Any]:
    """Assert that a smoke run exercised the release-sensitive output contracts."""
    with result_path.open() as handle:
        results = json.load(handle)
    if not isinstance(results, dict):
        raise ValueError("Smoke result must be a JSON object")

    manifest = results.get("run_manifest", {})
    if manifest.get("status") != "complete":
        raise ValueError(f"Technical run status is {manifest.get('status', 'missing')}")
    if manifest.get("quality_status") not in {"passed", "failed"}:
        raise ValueError("Smoke run did not produce a definitive model-quality status")
    incomplete = [
        section
        for section in manifest.get("required_sections", [])
        if manifest.get("sections", {}).get(section) != "complete"
    ]
    if incomplete:
        raise ValueError(f"Required result sections are incomplete: {', '.join(incomplete)}")
    if manifest.get("errors"):
        raise ValueError(f"Result extraction recorded errors: {manifest['errors']}")

    charts = results.get("charts", {})
    missing_charts = EXPECTED_CHARTS.difference(charts)
    if missing_charts:
        raise ValueError(f"Missing compatibility charts: {', '.join(sorted(missing_charts))}")

    chart_hashes = set()
    for chart_name in EXPECTED_CHARTS:
        chart_path = Path(charts[chart_name])
        if not chart_path.is_absolute():
            chart_path = result_path.parent.parent / chart_path
        chart_bytes = chart_path.read_bytes()
        if not chart_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Chart is not a valid PNG: {chart_name}")
        chart_hashes.add(hashlib.sha256(chart_bytes).digest())
    if len(chart_hashes) != len(EXPECTED_CHARTS):
        raise ValueError("Smoke run produced duplicate chart images")

    allocation = results.get("optimization", {}).get("current", {}).get("optimal_allocation", {})
    if not allocation:
        raise ValueError("Current-budget optimizer allocation is empty")

    report_path = result_path.with_suffix(".html")
    if not report_path.exists() or report_path.stat().st_size == 0:
        raise ValueError(f"HTML report is missing or empty: {report_path}")

    return results


def main() -> None:
    """Validate the newest smoke artifact and print a compact summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", nargs="?", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    result_path = latest_result(args.outputs_dir)
    results = verify_modal_smoke(result_path)
    manifest = results["run_manifest"]
    print(
        f"Verified {result_path}: run={manifest['status']}, "
        f"quality={manifest['quality_status']}, charts={len(results['charts'])}"
    )


if __name__ == "__main__":
    main()
