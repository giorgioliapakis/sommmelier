"""Backward-compatible alias for the supported full Modal runner.

New integrations should invoke ``modal_mmm_full.py`` directly. Keeping this
small adapter avoids maintaining a second model configuration and result schema.
"""

from modal_mmm_full import app, fit_mmm_full, main

fit_mmm = fit_mmm_full

__all__ = ["app", "fit_mmm", "fit_mmm_full", "main"]
