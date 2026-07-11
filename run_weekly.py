#!/usr/bin/env python3
"""
Sommmelier Weekly Run

Main entry point for running the full weekly MMM workflow:
1. Validate data locally (catches errors before GPU spend)
2. Fit the MMM model on Modal GPU
3. Generate HTML report
4. Analyze results and generate recommendations
5. Track model quality over time

Usage:
    python run_weekly.py data/raw/your_data.csv
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from mmm.result_manifest import decision_readiness as evaluate_decision_readiness


def run_command(
    cmd: list[str], description: str, timeout_seconds: int = 7200
) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = result.stdout + result.stderr
        print(output)
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def find_latest_results(exclude: set[Path] | None = None) -> Path | None:
    """Find the most recent results file, optionally excluding pre-run files."""
    outputs = Path("outputs")
    results = list(outputs.glob("full_results_*.json")) + list(outputs.glob("results_*.json"))
    if exclude:
        excluded = {path.resolve() for path in exclude}
        results = [path for path in results if path.resolve() not in excluded]
    if results:
        return max(results, key=lambda f: f.stat().st_mtime)
    return None


def decision_readiness(results_file: Path) -> tuple[bool, str]:
    """Return whether a result is safe to pass into recommendation generation."""
    return evaluate_decision_readiness(json.loads(results_file.read_text()))


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

    # Step 1: Validate data locally
    print(f"\n{'='*60}")
    print("STEP: Validating data")
    print(f"{'='*60}")

    try:
        from mmm.data import load_mmm_data, validate_dataset
        from mmm.data.validator import check_meridian_compatibility

        dataset = load_mmm_data(data_file)
        validation = validate_dataset(dataset)
        compatibility_issues = check_meridian_compatibility(dataset)

        print(validation.summary())

        if compatibility_issues:
            print("\nMeridian Compatibility Issues:")
            for issue in compatibility_issues:
                print(f"  ✗ {issue}")

        results["steps"]["validation"] = {
            "passed": validation.passed,
            "errors": validation.errors,
            "warnings": validation.warnings,
        }

        if not validation.passed or compatibility_issues:
            print("\n" + "=" * 60)
            print("VALIDATION FAILED - Aborting before GPU run")
            print("=" * 60)
            print("Fix the errors above and try again.")
            sys.exit(1)

        if validation.warnings > 0:
            print(f"\n⚠ {validation.warnings} warning(s) - proceeding anyway")
        else:
            print("\n✓ Validation passed")

    except Exception as e:
        print(f"✗ Validation error: {e}")
        results["steps"]["validation"] = {"error": str(e)}
        sys.exit(1)

    # Step 2: Check for calibration data
    calibration_file = Path("data/calibration.json")
    calibration_args = []
    if calibration_file.exists():
        print(f"\n✓ Found calibration data: {calibration_file}")
        print("  Model will use informative priors from experiments/platform data")
        calibration_args = ["--calibration", str(calibration_file)]
        results["calibration_file"] = str(calibration_file)
    else:
        print(f"\n⚠ No calibration data found at {calibration_file}")
        print("  Model will use default priors (wider uncertainty)")
        print("  To improve accuracy, create data/calibration.json with experiment results")

    # Step 3: Run MMM on Modal. This must remain attached so downstream steps
    # cannot accidentally consume a stale result from a previous run.
    outputs_dir = Path("outputs")
    existing_results = set(outputs_dir.glob("full_results_*.json")) | set(
        outputs_dir.glob("results_*.json")
    )
    modal_cmd = ["modal", "run", "modal_mmm_full.py", "--data", str(data_file)] + calibration_args
    success, output = run_command(
        modal_cmd,
        "Running MMM model on Modal GPU"
    )
    results["steps"]["mmm_fit"] = {"success": success}

    if not success:
        print("\nERROR: MMM fitting failed. Check the output above.")
        results["steps"]["mmm_fit"]["error"] = output
        sys.exit(1)

    # Find the results file
    results_file = find_latest_results(exclude=existing_results)
    if not results_file:
        print("\nERROR: Modal completed but did not create a new local results file.")
        sys.exit(1)

    results["results_file"] = str(results_file)
    print(f"\nResults saved to: {results_file}")
    decision_ready, readiness_reason = decision_readiness(results_file)
    results["decision_ready"] = decision_ready
    results["readiness_reason"] = readiness_reason

    # Step 4: Generate HTML report (via CLI)
    success, output = run_command(
        [sys.executable, "-m", "mmm.cli.main", "report", str(results_file)],
        "Generating HTML report"
    )
    results["steps"]["report"] = {"success": success}

    report_file = results_file.with_suffix(".html")
    if report_file.exists():
        results["report_file"] = str(report_file)
        print(f"Report saved to: {report_file}")

    # Step 5: Generate recommendations only for a technically complete model
    # whose quality checks passed.
    if decision_ready:
        success, output = run_command(
            [sys.executable, "-m", "mmm.cli.main", "analyze", str(results_file)],
            "Analyzing results and generating recommendations",
        )
        results["steps"]["analysis"] = {"success": success, "output": output}
    else:
        print("\nRECOMMENDATIONS SKIPPED - result is not decision-ready")
        print(f"Reason: {readiness_reason}")
        results["steps"]["analysis"] = {
            "success": False,
            "skipped": True,
            "reason": readiness_reason,
        }

    analysis_file = results_file.with_name(
        results_file.stem.replace("full_results", "analysis") + ".md"
    )
    if analysis_file.exists():
        results["analysis_file"] = str(analysis_file)

    # Step 6: Update model quality tracking
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
    print("\nOutputs:")
    print(f"  Results JSON:   {results.get('results_file', 'N/A')}")
    print(f"  HTML Report:    {results.get('report_file', 'N/A')}")
    print(f"  Analysis:       {results.get('analysis_file', 'N/A')}")
    print(f"  Quality Report: {results.get('quality_report_file', 'N/A')}")

    # Save run metadata
    results["completed"] = datetime.now().isoformat()
    run_log = Path("outputs") / f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    run_log.write_text(json.dumps(results, indent=2))
    print(f"  Run Log:        {run_log}")

    print("\n" + "-" * 60)
    print("NEXT STEPS FOR CLAUDE:")
    print("-" * 60)
    if decision_ready:
        print("1. Read the analysis file for recommendations")
        print("2. Review the HTML report for visualizations")
        print("3. Make strategic recommendations based on findings")
    else:
        print("1. Review the HTML report's model-quality warning")
        print("2. Resolve convergence or extraction failures before acting")
        print("3. Refit the model; do not make strategic recommendations yet")
    print("4. Suggest model improvements if needed")
    print("-" * 60)

    if not decision_ready:
        raise SystemExit(2)
    return results


if __name__ == "__main__":
    main()
