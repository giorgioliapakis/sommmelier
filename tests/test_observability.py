"""Tests for stable, correlated structured run events."""

import io
import json

from mmm.observability import configure_run_logger, log_event, new_run_id


def test_json_event_contains_correlation_and_context():
    stream = io.StringIO()
    logger = configure_run_logger("run-123", "modal", stream=stream)

    log_event(logger, "run_completed", status="complete", errors=0)

    payload = json.loads(stream.getvalue())
    assert payload["run_id"] == "run-123"
    assert payload["component"] == "modal"
    assert payload["event"] == "run_completed"
    assert payload["status"] == "complete"
    assert payload["errors"] == 0
    assert payload["level"] == "INFO"


def test_new_run_ids_are_unique():
    assert new_run_id() != new_run_id()
