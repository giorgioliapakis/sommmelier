# Sommmelier agent instructions

Sommmelier is a Python 3.11/3.12 marketing-mix-modeling toolkit built around Google Meridian and Modal GPU execution.

## Commands

```bash
uv sync --frozen --extra dev
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen mypy mmm
uv build
uv run --frozen sommmelier validate data/examples/meridian_sample.csv
```

Run a paid, reduced-sampling compatibility fit only after the user explicitly authorizes Modal spend:

```bash
uv run --frozen modal run modal_mmm_full.py --data data/examples/meridian_sample.csv --n-chains 2 --n-keep 100 --n-adapt 200 --n-burnin 100 --report
uv run --frozen python -m mmm.smoke outputs
```

## Project constraints

- Never commit client data, `data/calibration.json`, model outputs, credentials, or generated reports. The ignore rules are intentional.
- Treat recommendations as unsafe unless `run_manifest.status` is `complete` and `run_manifest.quality_status` is `passed`.
- Call ROI profitable only when `metadata.roi_is_monetary` is true. Otherwise describe it as KPI efficiency and prefer CPIK.
- Do not silently estimate population or impressions. The explicit fallback flags are for coarse exploratory runs only.
- Preserve the exact weekly, complete geo/time grid required by Meridian.
- Keep `modal_mmm.py` as a compatibility alias; the maintained runtime is `modal_mmm_full.py`.
- Use `uv.lock` and frozen installs for verification. Dependency changes require updating the lockfile.
- Use `origin/main` as the PR base. Do not rewrite or discard unrelated worktree changes.

## Agent workflows

Project workflows live in `.agents/skills/` and should trigger from user intent rather than legacy slash commands:

- `sommmelier-onboarding`: capture brand, KPI, channel, data, and calibration context.
- `sommmelier-analysis`: fit or analyze a model and produce decision-gated recommendations.
- `sommmelier-scenario`: explore optimizer-backed budget scenarios without inventing allocations.
- `sommmelier-walkthrough`: guide a beginner through validation, an authorized sample fit, and interpretation.

Read `context/*.md` only for brand-facing analysis and planning. Do not let stale context override result manifests, diagnostics, or explicit user constraints.
