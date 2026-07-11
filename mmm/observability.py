"""Structured JSON events for correlating local and remote model stages."""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO
from uuid import uuid4


def new_run_id() -> str:
    """Return a globally unique run identifier."""
    return str(uuid4())


class JsonEventFormatter(logging.Formatter):
    """Format a deliberately small, stable set of log fields as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", "unknown"),
            "run_id": getattr(record, "run_id", "unknown"),
            "event": getattr(record, "event", record.getMessage()),
        }
        fields = getattr(record, "event_fields", {})
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_run_logger(
    run_id: str,
    component: str,
    *,
    stream: TextIO | None = None,
) -> logging.LoggerAdapter[logging.Logger]:
    """Create an isolated logger carrying run and component context."""
    logger = logging.getLogger(f"sommmelier.{component}.{run_id}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonEventFormatter())
    logger.addHandler(handler)
    return logging.LoggerAdapter(logger, {"run_id": run_id, "component": component})


def log_event(logger: logging.LoggerAdapter[logging.Logger], event: str, **fields: Any) -> None:
    """Emit one structured event with arbitrary JSON-compatible context."""
    logger.logger.info(
        event,
        extra={**dict(logger.extra or {}), "event": event, "event_fields": fields},
    )
