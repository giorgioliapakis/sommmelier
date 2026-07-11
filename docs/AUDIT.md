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

1. **Keep Meridian compatibility tested.** The GPU image and package constraints now target Meridian 1.6.2. Every dependency bump should be followed by a real T4 smoke run and result-schema comparison because analyzer, visualizer, optimizer, and diagnostics objects are version-sensitive. See the [official Meridian changelog](https://github.com/google/meridian/blob/main/CHANGELOG.md).

2. **Finish consolidating the modeling paths.** Release-sensitive diagnostics, review, chart, and optimizer adapters are now shared, and the local optimizer uses the current Meridian contract. `modal_mmm.py` remains a divergent legacy runner and result extraction should move fully out of `modal_mmm_full.py`, leaving Modal as a thin compute adapter.

3. **Automate the paid GPU boundary.** Manual T4 smoke tests now verify Meridian 1.6.2 tensor dimensions, analyzer schemas, ModelReviewer parsing, optimizer allocations, chart download, and HTML rendering. Add a manually triggered, budget-capped GitHub workflow using the included sample; keep it out of ordinary pull requests to avoid surprise spend.

4. **Make calibration units explicit.** Experiment and platform conversion records are KPI-based, while user `PriorBelief` values are described as ROI. Add an explicit metric/unit field (`monetary_roi`, `incremental_kpi_per_currency`, or `cpik`) and reject calibration that does not match the fitted outcome scale.

5. **Stop swallowing extraction failures.** The full runner catches broad exceptions around most result sections and can publish a partial JSON file without a machine-readable degraded status. Add a result manifest with required/optional sections and fail when required outputs such as fit, convergence, or primary channel effects are absent.

### P2 — next hardening pass

1. Raise coverage around `mmm/model/builder.py`, the recommendation/advisor modules, CLI commands, calibration serialization, and trend reporting. Overall coverage is now 41%, but the model builder is only 6% because Meridian is not installed in the local test environment.
2. Replace pickle-based model persistence with Meridian's supported `save_mmm` and `load_mmm` APIs, and continue to reject untrusted model files because the underlying format is not a safe interchange format.
3. Add structured logging and run IDs across local and remote stages. The current print-based logs are hard to correlate and partial failures are difficult to query.
4. Validate weekly cadence explicitly, not just gaps larger than two weeks. Irregular but gap-free dates can still violate the model's intended weekly granularity.
5. Make population fallback opt-in. Assigning every unknown geography a population of 10 million can materially change geo scaling while looking valid.
6. Finish the configured quality gates. Repository-wide semantic Ruff checks now pass and run in CI, but 181 pre-existing line-length violations remain excluded and strict mypy still reports 97 errors.

## Verification performed

- `python3 -m pytest -q`: 49 passed
- `python3 -m pytest -q --cov=mmm --cov-report=term-missing`: 41% total coverage
- Repository-wide Ruff semantic checks: passed (line-length is temporarily excluded because 181 pre-existing violations remain)
- `python3 -m compileall`: passed
- `git diff --check`: passed
- CLI failure contract tested through Typer's runner
- Live Meridian 1.6.2 T4 smoke: 10 distinct valid PNG charts, 3 populated optimizer scenarios, structured six-check ModelReviewer output, correct non-convergence status, local JSON, and HTML report
- Locked dependency graph generated with uv; CI covers Python 3.11 and 3.12, tests, semantic Ruff, package build, and CLI startup

## Recommended build sequence

1. Land the Meridian 1.6.2 compatibility adapters, lockfile, and CI.
2. Retire the legacy Modal runner and consolidate model preparation and extraction behind one result schema.
3. Build a run manifest with explicit failed/degraded/complete states.
4. Make calibration units explicit and reject priors incompatible with the modeled outcome.
5. Replace raw model persistence, tighten cadence/population safety, and continue the lint/type/coverage cleanup.
