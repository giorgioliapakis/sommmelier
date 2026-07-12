# Sommmelier Project Audit

Audit date: 2026-07-11

## Executive assessment

Sommmelier has a compelling product shape: local data checks, GPU fitting, model diagnostics, reporting, recommendations, calibration, and longitudinal quality tracking are all present. The project is still alpha-quality, however. Before this audit, several paths could produce confident-looking but incorrect output: an asynchronous weekly run could select an old result, non-monetary KPI efficiency could be described as profit, malformed panel data could reach paid GPU fitting, and calibration parameters mixed linear and log scales.

This work fixes those highest-risk issues and raises measured line coverage from 8% to 82%. The project is substantially safer to experiment with, but still needs more real-data coverage before it should be treated as an unattended production decision system.

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
| Calibration units | Monetary ROI and KPI-per-currency records could be mixed or applied to the wrong outcome | Every record declares its metric; mixed or incompatible calibration is rejected before GPU spend |
| Budget advice | The recommendation layer invented an allocation proportional to marginal ROI | It now uses only Meridian's fitted same-budget optimizer result and returns no allocation when none exists |
| HTML reporting | Channel names were inserted into HTML/SVG without escaping | Dynamic report text is escaped and covered by a regression test |
| Native charts | Remote chart keys did not match report keys, remote paths were not downloaded, and `--report` did not generate a report | Chart keys are aligned, Volume charts are downloaded locally, and `--report` calls the generator |
| Run state | Partial extraction and failed convergence could still flow into strategic recommendations | Results carry independent technical and quality status; reports show a prominent warning and recommendation generation is blocked unless both pass |
| Runtime divergence | A second Modal runner remained pinned to Meridian 1.4 with a different schema | `modal_mmm.py` is now a thin compatibility alias; release-sensitive extraction is shared and tested |
| Model persistence | Raw pickle deserialization could execute code from an untrusted file | Model bundles now use Meridian protobuf, Parquet, and JSON with an atomic directory write |
| Time and population | Irregular dates and a silent 10-million-person fallback could change model scaling | Exact seven-day cadence and population data are required; coarse estimates require an explicit flag |
| Media execution | Missing impressions silently became spend multiplied by 100, imposing a fabricated $10 CPM | Impressions or reach/frequency are required; the coarse estimate is available only through an explicit flag |
| Quality history | Corrupt JSON was silently replaced by an empty history; writes were non-atomic | Corruption is reported and history is replaced atomically |
| CLI contract | `sommmelier validate` printed `FAILED` but exited with status 0 | Invalid data now returns status 1, making scripts and CI reliable |
| Packaging | The classifier said Apache while the project and `LICENSE` say MIT | Metadata now consistently says MIT |
| Test setup | A plain test run required an optional coverage plugin | Coverage is now opt-in at the command line; a normal `pytest` run works |
| Quality gates | Ruff reported 200 issues and strict mypy reported 97 errors while neither ran in CI | The repository is formatted, Ruff and strict mypy pass, and both are enforced on Python 3.11 and 3.12 |
| Observability | Local and remote print logs could not be reliably correlated | Structured JSON events now carry one run ID across submission, sampling, completion, downloads, artifacts, and quality history |
| Historical comparisons | Analysis assumed the current result occupied a specific position in `outputs/`, so external or future-dated files could select the wrong baseline; contribution deltas were never populated | Prior-run selection now excludes the current run by path and ID, respects timestamps, and reports contribution percentage-point changes |
| Time-varying controls | Controls were retained only when they varied across geographies at the same time, dropping valid national and time-varying controls | Any non-constant configured control is now passed to Meridian; constant controls remain excluded |
| Agent tooling | A 210-line Claude-only instruction file and 731 lines of obsolete slash commands duplicated workflows | `.agents` is canonical, four intent-triggered skills replace slash commands, and root/Claude/Codex entry points are relative symlinks to the same files |
| CLI output contract | `--json` mixed a human preamble into machine output, malformed result files surfaced raw exceptions, and a zero optimization budget silently became the current budget | JSON mode is parseable, file errors are concise, non-positive budgets are rejected before model loading, file URIs are safe, and the CLI is exercised end-to-end |
| Extended Meridian shapes | Meridian 1.6.2 changed paid-contribution defaults, non-paid interval coordinates, treatment channel names, and configured Altair chart composition | Paid and non-paid extraction is explicit and quantified, treatment provenance is preserved, configured chart mappings compose safely, and a live R&F/organic/treatment/control run verifies the contracts |
| Direct paid submission | `modal run` performed several readiness checks only after invoking the remote function, so an obviously invalid dataset could start billable GPU compute | Direct runs now execute the shared full local preflight first; a live expected-failure check rejects the 8-week fixture without remote submission |
| Paid-run options | Explicit population/impression fallback flags were ignored by local preflight, while invalid holdouts were silently skipped only after GPU submission | Preflight now preserves authorized fallback flags and validates holdout bounds before invoking Modal; the GPU path shares the same strict holdout-mask implementation |
| Modal input construction | The GPU runtime duplicated local column detection, validation, fallback, revenue, control, and Meridian builder logic across roughly 240 lines | File and in-memory inputs now share one loader, preflight contract, and package-level `InputData` builder; Modal is only the compute adapter for this stage |
| Holdout validity | Holdouts used one contiguous trailing window for every geo, which Meridian explicitly discourages, then averaged Train, Test, and All Data metrics into a mislabeled holdout score | Masks are deterministically balanced across geo and time, and reporting selects the `Test` evaluation set with separate geo and national metrics, following [Meridian's holdout guidance](https://developers.google.com/meridian/docs/advanced-modeling/holdout-observations) |
| Optional result schemas | Response curves assumed the first metric was the mean, optimal frequency selected arbitrary first coordinates, and adstock parsing lived only inside Modal | Release-sensitive response, adstock, optimal-frequency, and holdout shapes now use isolated compatibility adapters with explicit tensor and xarray contracts |

## Remaining risks and opportunities

### P1 — address before production use

1. **Keep Meridian compatibility tested.** The GPU image and package constraints now target Meridian 1.6.2. Every dependency bump should be followed by a real T4 smoke run and result-schema comparison because analyzer, visualizer, optimizer, and diagnostics objects are version-sensitive. See the [official Meridian changelog](https://github.com/google/meridian/blob/main/CHANGELOG.md).

2. **Finish extracting the Modal implementation into package modules.** The divergent legacy runner is retired; input construction and release-sensitive tensor, diagnostics, review, chart, and optimizer adapters are shared. Several optional result sections still live in `modal_mmm_full.py`; move them into testable package modules until Modal is only a compute adapter.

3. **Keep the paid GPU boundary deliberate.** Manual T4 smoke tests verify Meridian 1.6.2 tensor dimensions, analyzer schemas, ModelReviewer parsing, optimizer allocations, chart download, and HTML rendering. The `Paid Modal compatibility smoke` workflow now provides an explicit confirmation gate, reduced sampling, a 20-minute job timeout, artifact-contract verification, and seven-day artifact retention. It remains outside ordinary pull requests to avoid surprise spend; repository secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` are required.

4. **Exercise every live model shape after dependency changes.** The manually authorized workflow can run standard, extended, national, balanced-holdout, and monetary-calibration fixtures. Standard and extended shapes have been exercised on Meridian 1.6.2; the new national, holdout, and calibration choices still need their first paid run before those contracts are considered proven.

### P2 — next hardening pass

1. Raise coverage around the improvement advisor, tracking, and model wrapper. Overall coverage is now 82%; the recommendation engine is at 95%, CLI 94%, model builder 92%, visualization 82%, local reports 75%, insights 80%, tracking 65%, improvement advisor 65%, and model wrapper 68%.
2. Add model-bundle compatibility tests against a small genuinely fitted Meridian artifact, not only the isolated serializer contract.

## Verification performed

- `uv run --frozen pytest -q`: 142 passed
- `uv run --frozen pytest -q --cov=mmm --cov-report=term`: 82% total coverage
- `uv run --frozen ruff check .` and `ruff format --check .`: passed repository-wide
- `uv run --frozen mypy mmm`: strict typing passed for all 30 source files
- `python3 -m compileall`: passed
- `git diff --check`: passed
- CLI failure contract tested through Typer's runner
- Live Meridian 1.6.2 T4 smoke: 10 distinct valid PNG charts, 3 populated optimizer scenarios, structured six-check ModelReviewer output, correct non-convergence status, local JSON, and HTML report
- Follow-up manifest smoke: technical status `complete`, quality status `failed` for the deliberately under-sampled fit, visible HTML warning, and CLI recommendation blocking
- Calibration mismatch preflight: rejected incompatible units before the remote GPU function was invoked
- Structured preflight logs: valid JSON event with the same run ID used by the manifest and remote call
- Local smoke verifier: required manifest sections, 10 distinct PNGs, optimizer allocation, and HTML report validated against the live artifact
- Live extended-shape smoke: R&F optimal frequency, quantified organic and treatment effects with intervals, controls, paid contributions, 10 charts, and 3 optimizer scenarios; technical status `complete` with zero extraction errors
- Live expected-failure smoke: the short fixture exits locally with `MMM data preflight failed`; no remote-submission or model-start event occurs
- CLI contracts: valid pure-JSON output, concise malformed-file errors, local model orchestration, report generation, quality history, optimizer rendering, and insight rendering covered through Typer
- Agent layout: all four instruction/skills symlinks resolve, legacy `.claude/commands` is absent, and every skill has valid discovery metadata
- Locked dependency graph generated with uv; CI covers Python 3.11 and 3.12, tests, Ruff lint/format, strict mypy, package build, and CLI startup

## Recommended build sequence

1. Move the remaining Modal result logic into package modules with isolated contract tests.
2. Run the manually dispatched national, holdout, and calibration fixtures after explicitly authorizing Modal spend.
3. Continue CLI and recommendation-engine coverage cleanup.
4. Build the next product layer only after decision-readiness gates hold across real customer-shaped data.
