#!/usr/bin/env python3
"""
Sommmelier Weekly Run

Main entry point for running the full weekly MMM workflow:
1. Fit the MMM model on Modal GPU
2. Generate HTML report
3. Analyze results and generate recommendations
4. Track model quality over time

Usage:
    python run_weekly.py data/raw/your_data.csv
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        output = result.stdout + result.stderr
        print(output)
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def find_latest_results() -> Path | None:
    """Find the most recent results file."""
    outputs = Path("outputs")
    results = list(outputs.glob("full_results_*.json"))
    if not results:
        results = list(outputs.glob("results_*.json"))
    if results:
        return max(results, key=lambda f: f.stat().st_mtime)
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_weekly.py <data_file.csv>")
        print("Example: python run_weekly.py data/raw/your_data.csv")
        sys.exit(1)

    data_file = Path(sys.argv[1])
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("SOMMMELIER - Weekly Marketing Mix Analysis")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Data: {data_file}")
    print("=" * 60)

    results = {
        "started": datetime.now().isoformat(),
        "data_file": str(data_file),
        "steps": {},
    }

    # Step 1: Run MMM on Modal
    success, output = run_command(
        ["modal", "run", "modal_mmm_full.py", "--data", str(data_file)],
        "Running MMM model on Modal GPU"
    )
    results["steps"]["mmm_fit"] = {"success": success}

    if not success:
        print("\nERROR: MMM fitting failed. Check the output above.")
        results["steps"]["mmm_fit"]["error"] = output
        sys.exit(1)

    # Find the results file
    results_file = find_latest_results()
    if not results_file:
        print("\nERROR: No results file found after MMM run.")
        sys.exit(1)

    results["results_file"] = str(results_file)
    print(f"\nResults saved to: {results_file}")

    # Step 2: Generate HTML report (via CLI)
    success, output = run_command(
        ["python", "-m", "mmm.cli.main", "report", str(results_file)],
        "Generating HTML report"
    )
    results["steps"]["report"] = {"success": success}

    report_file = results_file.with_suffix(".html")
    if report_file.exists():
        results["report_file"] = str(report_file)
        print(f"Report saved to: {report_file}")

    # Step 3: Run analysis and recommendations (via CLI)
    success, output = run_command(
        ["python", "-m", "mmm.cli.main", "analyze", str(results_file)],
        "Analyzing results and generating recommendations"
    )
    results["steps"]["analysis"] = {"success": success, "output": output}

    analysis_file = results_file.with_name(
        results_file.stem.replace("full_results", "analysis") + ".txt"
    )
    if analysis_file.exists():
        results["analysis_file"] = str(analysis_file)

    # Step 4: Update model quality tracking
    print(f"\n{'='*60}")
    print("STEP: Updating model quality tracking")
    print(f"{'='*60}")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from mmm.tracking import update_tracking
        quality_report = update_tracking(results_file, str(data_file))
        print(quality_report)

        quality_file = Path("outputs") / "model_quality_report.txt"
        quality_file.write_text(quality_report)
        results["quality_report_file"] = str(quality_file)
    except Exception as e:
        print(f"Warning: Model quality tracking failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("WEEKLY RUN COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Results JSON:   {results.get('results_file', 'N/A')}")
    print(f"  HTML Report:    {results.get('report_file', 'N/A')}")
    print(f"  Analysis:       {results.get('analysis_file', 'N/A')}")
    print(f"  Quality Report: {results.get('quality_report_file', 'N/A')}")

    # Save run metadata
    run_log = Path("outputs") / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    run_log.write_text(json.dumps(results, indent=2))
    print(f"  Run Log:        {run_log}")

    print("\n" + "-" * 60)
    print("NEXT STEPS FOR CLAUDE:")
    print("-" * 60)
    print("1. Read the analysis file for recommendations")
    print("2. Review the HTML report for visualizations")
    print("3. Make strategic recommendations based on findings")
    print("4. Suggest model improvements if needed")
    print("-" * 60)

    results["completed"] = datetime.now().isoformat()
    return results


if __name__ == "__main__":
    main()
