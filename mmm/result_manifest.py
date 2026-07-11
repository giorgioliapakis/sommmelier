"""Machine-readable completeness and quality state for model runs."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

REQUIRED_SECTIONS = ("roi", "contributions", "model_fit", "diagnostics")
OPTIONAL_SECTIONS = (
    "cpik",
    "response_curves",
    "adstock_decay",
    "marginal_roi",
    "optimal_frequency",
    "organic_contributions",
    "treatment_effects",
    "holdout_validation",
    "model_review",
    "charts",
    "optimization",
)


def create_run_manifest(started_at: str | None = None) -> dict[str, Any]:
    """Create the initial state attached to every result payload."""
    return {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "started_at": started_at or datetime.now(UTC).isoformat(),
        "completed_at": None,
        "status": "running",
        "quality_status": "pending",
        "required_sections": list(REQUIRED_SECTIONS),
        "optional_sections": list(OPTIONAL_SECTIONS),
        "sections": {},
        "errors": [],
    }


def record_section_error(
    results: dict[str, Any],
    section: str,
    error: BaseException,
    *,
    required: bool = False,
) -> None:
    """Record an extraction failure without losing its section or severity."""
    manifest: dict[str, Any] = results["run_manifest"]
    manifest["errors"].append(
        {
            "section": section,
            "required": required,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )


def finalize_run_manifest(
    results: dict[str, Any], completed_at: str | None = None
) -> dict[str, Any]:
    """Finalize technical completeness independently from statistical quality."""
    manifest: dict[str, Any] = results["run_manifest"]
    failed_sections = {error["section"] for error in manifest["errors"]}

    section_presence = {
        section: _section_is_present(section, results.get(section))
        for section in (*REQUIRED_SECTIONS, *OPTIONAL_SECTIONS)
    }
    manifest["sections"] = {
        section: (
            "error"
            if section in failed_sections
            else "complete"
            if present
            else "missing"
            if section in REQUIRED_SECTIONS
            else "not_generated"
        )
        for section, present in section_presence.items()
    }

    required_failures = [
        section
        for section in REQUIRED_SECTIONS
        if manifest["sections"][section] != "complete"
    ]
    if required_failures:
        manifest["status"] = "failed"
    elif manifest["errors"]:
        manifest["status"] = "degraded"
    else:
        manifest["status"] = "complete"

    diagnostics = results.get("diagnostics", {})
    review = results.get("model_review", {})
    if diagnostics.get("convergence_ok") is False or review.get("passed") is False:
        manifest["quality_status"] = "failed"
    elif diagnostics.get("convergence_ok") is True and review.get("passed", True) is True:
        manifest["quality_status"] = "passed"
    else:
        manifest["quality_status"] = "unknown"

    manifest["completed_at"] = completed_at or datetime.now(UTC).isoformat()
    return manifest


def decision_readiness(results: dict[str, Any]) -> tuple[bool, str]:
    """Return whether recommendations may safely consume a result payload."""
    manifest = results.get("run_manifest")
    if not isinstance(manifest, dict):
        return False, "result has no run manifest"
    if manifest.get("status") != "complete":
        return False, f"technical run status is {manifest.get('status', 'unknown')}"
    if manifest.get("quality_status") != "passed":
        return False, f"model quality status is {manifest.get('quality_status', 'unknown')}"
    return True, "run complete and model quality passed"


def _section_is_present(section: str, value: Any) -> bool:
    if section == "model_fit":
        return isinstance(value, dict) and "r_squared" in value and "mape" in value
    if section == "diagnostics":
        return isinstance(value, dict) and value.get("diagnostics_available") is True
    return bool(value)
