"""Weekly orchestration regression tests."""

from pathlib import Path

from run_weekly import find_latest_results


def test_find_latest_results_can_exclude_preexisting_files(tmp_path: Path, monkeypatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    stale = outputs / "full_results_old.json"
    fresh = outputs / "full_results_new.json"
    stale.write_text("{}")
    fresh.write_text("{}")
    monkeypatch.chdir(tmp_path)

    assert find_latest_results(exclude={stale}).resolve() == fresh.resolve()


def test_find_latest_results_returns_none_when_only_stale_files_exist(
    tmp_path: Path, monkeypatch
):
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
