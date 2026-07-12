---
name: sommmelier-analysis
description: Run or interpret a Sommmelier MMM result and produce safe, context-aware recommendations. Use whenever a user asks to analyze results, run a model, diagnose model quality, compare runs, generate an MMM report, or decide what to improve next.
---

# Sommmelier analysis

Analyze the latest result by default. If the user supplies a CSV, validate it and run a new fit only with explicit authorization for Modal GPU cost.

## 1. Load context and artifacts

Read relevant `context/*.md`, the newest `outputs/full_results_*.json`, and model-quality history when present. Flag context older than 90 days. Adapt depth to `Experience level`, but let result diagnostics and current user constraints take precedence.

If no result exists and no data path was supplied, explain what is missing and provide the exact next command.

## 2. Validate and fit when requested

```bash
uv run --frozen sommmelier validate <data.csv>
uv run --frozen modal run modal_mmm_full.py --data <data.csv> --report
```

Modal is paid. State the expected boundary and obtain confirmation unless the user already authorized it. Run at most one baseline plus two hypothesis-driven variations. Do not rerun merely because intervals are wide or fit is low; those usually require better data, controls, or priors.

Reasonable variations are:

- R-hat warnings: more kept/adaptation samples.
- Suspected overfit: an authorized holdout run.
- Domain-inappropriate carryover: a justified `max_lag` test.

Record why each variation ran and whether it improved the targeted issue.

## 3. Enforce decision readiness

Before recommendations, require:

- `run_manifest.status == "complete"`
- `run_manifest.quality_status == "passed"`

If either fails, block budget recommendations and focus on diagnostics and remediation. The HTML report may still be generated for inspection.

Only interpret ROI as monetary profitability when `metadata.roi_is_monetary` is true. Otherwise call it KPI efficiency, use KPI/currency units, and prefer CPIK. Never manufacture an optimizer allocation from average or marginal ROI; use populated Meridian `optimization` output.

## 4. Diagnose

Review ROI or KPI efficiency with intervals, CPIK, marginal returns, contributions, model fit, convergence, reviewer checks, optimizer output, and any R&F/organic/treatment/holdout sections. Compare with the genuinely previous run and flag material instability.

Separate problems into:

- Agent-testable: convergence sampling, holdout, or justified carryover hypotheses.
- Human action: more weeks/geos, better execution data, controls, calibration experiments, or corrected source data.

## 5. Deliver and compound

Write `outputs/analysis_<date>.md` with:

1. Executive summary.
2. What was tested and selected.
3. Key findings and run-over-run changes.
4. Prioritized actions tied to goals and constraints.
5. Improvements split into agent-testable and human-required work.
6. Confidence, caveats, and model health.

Update `context/model-learnings.md` and `context/improvement-backlog.md` only when a real analysis or experiment produced durable new evidence. Do not repeat closed advice.
