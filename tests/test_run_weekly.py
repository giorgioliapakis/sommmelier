"""Weekly orchestration regression tests."""

from pathlib import Path

from run_weekly import decision_readiness, find_latest_results


def test_find_latest_results_can_exclude_preexisting_files(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    stale = outputs / "full_results_old.json"
    fresh = outputs / "full_results_new.json"
    stale.write_text("{}")
    fresh.write_text("{}")
    monkeypatch.chdir(tmp_path)

    assert find_latest_results(exclude={stale}).resolve() == fresh.resolve()


def test_find_latest_results_returns_none_when_only_stale_files_exist(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    stale = outputs / "full_results_old.json"
    stale.write_text("{}")
    monkeypatch.chdir(tmp_path)

    assert find_latest_results(exclude={stale}) is None


def test_find_latest_results_finds_simple_result_when_stale_full_result_exists(
    tmp_path: Path, monkeypatch
):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    stale = outputs / "full_results_old.json"
    fresh = outputs / "results_new.json"
    stale.write_text("{}")
    fresh.write_text("{}")
    monkeypatch.chdir(tmp_path)

    assert find_latest_results(exclude={stale}).resolve() == fresh.resolve()


def test_decision_readiness_requires_complete_run_and_passed_quality(tmp_path: Path):
    result = tmp_path / "result.json"
    result.write_text('{"run_manifest": {"status": "complete", "quality_status": "passed"}}')

    assert decision_readiness(result) == (
        True,
        "run complete and model quality passed",
    )


def test_decision_readiness_blocks_failed_quality_and_legacy_results(tmp_path: Path):
    failed = tmp_path / "failed.json"
    failed.write_text('{"run_manifest": {"status": "complete", "quality_status": "failed"}}')
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}")

    assert decision_readiness(failed) == (False, "model quality status is failed")
    assert decision_readiness(legacy) == (False, "result has no run manifest")
