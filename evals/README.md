# Synthetic evaluation

Generate reproducible dummy datasets without using client data or paid compute:

```bash
uv run --frozen python -m evals.synthetic outputs/evals --seed 42
uv run --frozen sommmelier validate outputs/evals/baseline.csv
uv run --frozen pytest -q
```

The suite covers ordinary geo panels, national panels, KPI-only outcomes,
zero-effect media, perfectly correlated channels, a structural break, missing
coordinates, duplicate coordinates, and non-finite spend. Each CSV has a companion
truth JSON with expected validation behavior, simulator assumptions, units, and
true channel effects. Invalid scenarios must fail before remote submission.

All paid media execution is explicitly simulated at 100 impressions per spend
unit. This is a documented simulator assumption, not a fallback for customer data.
The data-generating process uses an eight-week finite carryover window, saturation,
seasonality, geo baselines, and seeded observation noise. Correlated channels and
time-varying effects deliberately challenge the model's assumptions.

## Paid compatibility run

Requires Modal authentication (`uv run --frozen modal setup`) and explicit
authorization for GPU cost. Run only after both are available:

```bash
uv run --frozen modal run modal_mmm_full.py \
  --data outputs/evals/baseline.csv \
  --n-chains 2 --n-keep 100 --n-adapt 200 --n-burnin 100 --report
uv run --frozen python -m mmm.smoke outputs
```

This reduced-sampling run checks compatibility and may fail statistical quality.
It is not a recovery acceptance benchmark. For adequately sampled runs, compare
the corresponding result file with the exact scenario/seed truth file:

```bash
uv run --frozen python -m evals.score outputs/full_results_RUN.json \
  outputs/evals/baseline.truth.json
```

The scorer reports per-channel absolute error, interval width/coverage, and the
independent decision-readiness status. It rejects mismatched units/channels and
invalid metrics. The caller must pair the exact input run and truth file; channel
and unit checks cannot establish dataset identity. Zero-effect cases use absolute
error because percentage error at zero is undefined.

## Continuous evaluation

The deterministic scenarios run in ordinary pytest/CI without remote compute.
Before adopting statistical acceptance thresholds, establish adequately sampled
multi-seed baselines and record the commit, input/truth files, dependency versions,
priors, sampling configuration, diagnostics, runtime, and cost. Keep development
seeds separate from acceptance seeds. Do not auto-tune to one dataset or equate
good prediction with causal accuracy on customer data.

Future evaluation work includes optimizer regret against simulator outcomes,
explicit agent-prompt evaluations, and an authorized multi-seed GPU benchmark.
