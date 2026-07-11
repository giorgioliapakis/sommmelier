# Sommmelier Project Audit

Audit date: 2026-07-11

## Executive assessment

Sommmelier has a compelling product shape: local data checks, GPU fitting, model diagnostics, reporting, recommendations, calibration, and longitudinal quality tracking are all present. The project is still alpha-quality, however. Before this audit, several paths could produce confident-looking but incorrect output: an asynchronous weekly run could select an old result, non-monetary KPI efficiency could be described as profit, malformed panel data could reach paid GPU fitting, and calibration parameters mixed linear and log scales.

This update fixes those highest-risk issues and raises measured line coverage from 8% to 41%. The project is safer to experiment with, but it should not yet be treated as an unattended production decision system.

## What was fixed

| Area | Flaw | Resolution |
|---|---|---|
| Weekly orchestration | `modal run --detach` returned before fitting finished, after which the script could select a stale prior result | The run now remains attached and explicitly excludes all pre-run result files |
| KPI semantics | Conversion-only models described KPI-per-dollar as monetary ROI and applied profitability thresholds | Results now record `roi_is_monetary`; reports, recommendations, and agent instructions distinguish monetary ROI from KPI efficiency |
| Revenue handling | A total `revenue` column was treated inconsistently and was not converted to revenue per KPI | Revenue-per-KPI columns are detected; total revenue is safely divided by KPI with zero-KPI validation |
| Data validation | Duplicate coordinates, incomplete geo/time panels, infinities, missing spend, all-zero channels, and negative KPI values could pass | These are now blocking checks in both local validation and the direct Modal entry path |
| Column detection | The loader, Modal runner, and tests each implemented different rules; tests exercised copied test logic | A production detector now handles paid, R&F, organic, treatment, and documented control columns case-insensitively; tests import it directly |
| Controls | `is_holiday` and `product_launch` were documented as automatic but ignored by the local and GPU paths | Both are now included in shared detection |
| Calibration | Platform-derived values in linear space were averaged with LogNormal location parameters in log space | All prior locations are now combined in log space; lift-only experiments are no longer treated as ROI |
| Budget advice | The recommendation layer invented an allocation proportional to marginal ROI | It now uses only Meridian's fitted same-budget optimizer result and returns no allocation when none exists |
| HTML reporting | Channel names were inserted into HTML/SVG without escaping | Dynamic report text is escaped and covered by a regression test |
| Native charts | Remote chart keys did not match report keys, remote paths were not downloaded, and `--report` did not generate a report | Chart keys are aligned, Volume charts are downloaded locally, and `--report` calls the generator |
| Quality history | Corrupt JSON was silently replaced by an empty history; writes were non-atomic | Corruption is reported and history is replaced atomically |
| CLI contract | `sommmelier validate` printed `FAILED` but exited with status 0 | Invalid data now returns status 1, making scripts and CI reliable |
| Packaging | The classifier said Apache while the project and `LICENSE` say MIT | Metadata now consistently says MIT |
| Test setup | A plain test run required an optional coverage plugin | Coverage is now opt-in at the command line; a normal `pytest` run works |

## Remaining risks and opportunities

### P1 — address before production use

1. **Upgrade Meridian in an isolated compatibility branch.** The GPU image is pinned to Meridian 1.4.0 while the official changelog lists 1.6.2. The newer releases add EDA/data-adequacy checks, input guardrails, serialization, JAX support, and bug fixes. This code reaches into several version-sensitive analyzer and visualizer APIs, so a version bump should be followed by a real T4 smoke run and result-schema comparison rather than changed blindly. See the [official Meridian changelog](https://github.com/google/meridian/blob/main/CHANGELOG.md).

2. **Consolidate the three modeling paths.** `mmm/model/mmm.py`, `modal_mmm.py`, and `modal_mmm_full.py` have diverged. The full Modal runner is the de facto product, while the local wrapper exposes a different extraction and optimization contract. Move model preparation and result extraction into versioned package modules, leaving Modal as a thin compute adapter.

3. **Test the paid GPU boundary.** Current tests cannot verify Meridian tensor dimensions, analyzer return schemas, R&F behavior, ModelReviewer parsing, optimizer results, or chart download against the live service. Add a manually triggered, budget-capped smoke workflow using the included Meridian sample.

4. **Make calibration units explicit.** Experiment and platform conversion records are KPI-based, while user `PriorBelief` values are described as ROI. Add an explicit metric/unit field (`monetary_roi`, `incremental_kpi_per_currency`, or `cpik`) and reject calibration that does not match the fitted outcome scale.

5. **Stop swallowing extraction failures.** The full runner catches broad exceptions around most result sections and can publish a partial JSON file without a machine-readable degraded status. Add a result manifest with required/optional sections and fail when required outputs such as fit, convergence, or primary channel effects are absent.

### P2 — next hardening pass

1. Add CI for Python 3.11–3.13 with tests, Ruff, package build, and a minimal CLI smoke test.
2. Add a lock or constraints file for reproducible local and Modal environments. `requirements.txt`, `pyproject.toml`, and the Modal image currently express different constraints.
3. Raise coverage around `mmm/model/builder.py`, the recommendation/advisor modules, CLI commands, calibration serialization, and trend reporting. Overall coverage is now 41%, but the model builder is only 6% because Meridian is not installed in the local test environment.
4. Replace pickle-based model persistence or clearly reject untrusted model files. `AutoMMM.load()` executes Python pickle deserialization and must never be used on files from another party.
5. Add structured logging and run IDs across local and remote stages. The current print-based logs are hard to correlate and partial failures are difficult to query.
6. Validate weekly cadence explicitly, not just gaps larger than two weeks. Irregular but gap-free dates can still violate the model's intended weekly granularity.
7. Make population fallback opt-in. Assigning every unknown geography a population of 10 million can materially change geo scaling while looking valid.
8. Make the configured quality gates real: the repository-wide Ruff check currently reports 200 issues, and strict mypy reports 97 errors. Changed files pass semantic Ruff checks, but CI should not claim full lint/type safety until the baseline is cleaned up.

## Verification performed

- `python3 -m pytest -q`: 43 passed
- `python3 -m pytest -q --cov=mmm --cov-report=term-missing`: 41% total coverage
- Ruff semantic checks on every changed Python file: passed (line-length excluded because the pre-existing codebase does not satisfy its configured 100-character limit)
- `python3 -m compileall`: passed
- `git diff --check`: passed
- CLI failure contract tested through Typer's runner
- No paid Modal GPU run was launched during this audit

## Recommended build sequence

1. Land this correctness and reliability hardening.
2. Consolidate the modeling paths behind one result schema.
3. Upgrade Meridian and run the budget-capped GPU compatibility suite.
4. Add CI and dependency locking.
5. Build the next product layer: a run manifest, explicit degraded-state reporting, and scenario planning that consumes only optimizer-backed allocations.
