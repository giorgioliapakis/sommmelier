---
name: sommmelier-walkthrough
description: Guide a beginner through Sommmelier setup, bundled sample validation, an optional paid Modal fit, and safe MMM interpretation. Use whenever a user asks for a tutorial, demo, walkthrough, first run, or help learning the project end to end.
---

# Sommmelier walkthrough

Use a beginner-friendly tone and pause before any paid or account-changing step.

## 1. Verify prerequisites

Run:

```bash
python --version
uv --version
uv run --frozen modal token info
```

Python must be 3.11 or 3.12. If dependencies are missing, use `uv sync --frozen --extra dev`. Explain that Modal provides paid GPU compute and guide the user through `uv run modal setup` only when authentication is missing.

## 2. Explain the sample

Use `data/examples/meridian_sample.csv` for modeling. Show a few rows and explain exact weekly dates, geographies, KPI, population, channel execution, spend, controls, organic media, and treatments as applicable. The smaller `sample_data.csv` is illustrative and intentionally insufficient for fitting.

Validate with:

```bash
uv run --frozen sommmelier validate data/examples/meridian_sample.csv
```

Explain each failed or passed readiness check in plain language.

## 3. Offer the paid fit

State that the next step incurs a small Modal GPU charge and ask for explicit confirmation. If approved, run:

```bash
uv run --frozen modal run modal_mmm_full.py --data data/examples/meridian_sample.csv --report
```

While it runs, explain validation, calibration, Bayesian sampling, diagnostics, reports, and quality tracking. If declined, use an existing result if available and clearly label it as prior output.

## 4. Interpret safely

Walk through ROI or KPI efficiency, CPIK, marginal returns, contributions, intervals, fit, and convergence. Check the run manifest first. Block recommendations if technical completeness or model quality failed. Never call a non-monetary KPI-efficiency value profitable.

Point to the generated JSON and HTML report, then summarize how to prepare real data, onboard brand context, run a real analysis, and improve the model over time.
