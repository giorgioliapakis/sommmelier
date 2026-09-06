# Sommmelier

**An MMM data scientist in your terminal.**

> **Early stage.** Treat model outputs as directional, not as budget decisions.

Sommmelier is an agent-driven MMM that fits your model, diagnoses what's limiting it, and tells you what to change before the next run.

Runs [Google Meridian](https://github.com/google/meridian) on GPU via [Modal](https://modal.com) (~$0.30/run). Shared `.agents` skills let compatible coding agents act as the analyst layer.

## What it does

1. **Fits MMM models on cloud GPU** via Modal.com (~$0.30/run)
2. **Diagnoses model quality** and re-runs with fixes if needed (convergence, overfitting, carryover windows)
3. **Generates visual reports** with 10 native Meridian charts
4. **Tracks model quality** over time across runs
5. **Tells you what to improve** - data gaps, missing controls, calibration opportunities

After each run, the system analyzes diagnostics and tells you specifically what to change:

- "Meta's CI is too wide. Run a 4-week geo holdout in 3 states to calibrate."
- "Add a holiday control variable. The model is attributing promo lifts to ad spend."
- "Brand search shows 8x ROI, but it's probably capturing demand other channels created."
- "You have 5 channels but only 2 geos. More geographic granularity would help."
- "R-squared dropped from 0.78 to 0.65. Investigate a structural break in the data."

Suggestions are tracked across runs. Act on one, re-run, and see whether it helped.

## Quick start

### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for reproducible installs
- A [Modal](https://modal.com) account (free tier available, this is where the model runs on GPU)
- A coding agent that supports project skills, such as Claude Code or Codex (recommended for personalized analysis)

### Install

```bash
git clone https://github.com/giorgioliapakis/sommmelier.git
cd sommmelier
uv sync --frozen

# Set up Modal for GPU access
uv run modal setup
```

The committed `uv.lock` reproduces the tested dependency set. If you do not use
uv, `pip install -e .` remains supported but may resolve newer transitive dependencies.

### Option A: Guided agent experience

Open a compatible coding agent in this project and ask naturally for the workflow you need. The canonical skills live in `.agents/skills`; compatibility symlinks expose the same files to Claude Code and Codex.

```bash
# Start Claude Code, for example
claude
```

Example prompts:

- “Onboard my brand for Sommmelier.”
- “Walk me through the bundled example.”
- “Analyze my latest MMM result.”
- “Run Sommmelier on `data/raw/your_data.csv`.”
- “Explore moving $5,000 from Meta to Search.”

The onboarding skill asks about your brand, channels, KPIs, and goals. The analysis skill validates decision readiness, reads brand context, and writes recommendations tied to specific goals and constraints.

### Option B: CLI only

The pipeline works without Claude Code:

```bash
# Validate your data
sommmelier validate data/raw/your_data.csv

# Run the full pipeline (validate → fit on GPU → report → analyze → track)
python run_weekly.py data/raw/your_data.csv

# View results
sommmelier analyze                    # Latest analysis
sommmelier report results.json        # Generate HTML report
sommmelier quality --history          # Model quality over time
```

You get the same model results, reports, and automated recommendations, just without the brand-context personalization that Claude Code adds.

## What you provide

### The dataset (required)

A CSV with weekly marketing data. At minimum:

| Column | Required | Example |
|--------|----------|---------|
| `date` or `time` | Yes | `2024-01-01` |
| `geo` | Yes | `US`, `UK`, `AU` |
| `population` | Yes | `330000000` |
| `conversions` | Yes | `1523` |
| `{channel}_spend` | Yes | `meta_spend`, `google_spend` |
| `{channel}_impressions` or reach/frequency | Yes | `meta_impressions` |

To get monetary ROI, also provide either `revenue`, `revenue_per_kpi`, or `revenue_per_conversion`. Without one of those columns, Sommmelier reports incremental KPI units per currency unit spent; a value below 1.0 does not mean the channel is unprofitable.

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Time periods | 26 exact weekly periods | 52+ exact weekly periods |
| Geographies | 1 | 5+ |
| Media channels | 2 | 3-7 |

### Things that improve the model (optional but recommended)

The system tells you which of these matter most for your situation. You don't need all of them upfront. Start with what you have.

**Impression data** (`{channel}_impression` or `{channel}_impressions` columns). A coarse $10 CPM estimate is available only through the explicit `--allow-impression-estimates` flag; fabricated execution data is never used silently.

**Reach & frequency data** (`{channel}_reach` and `{channel}_frequency` columns). For channels where you have reach and frequency data (e.g., YouTube, TV), the model uses frequency-based saturation instead of spend-based. Produces optimal frequency recommendations.

**Organic media** (`{name}_organic` columns, e.g., `newsletter_organic`, `blog_organic`). Organic channels get adstock and saturation treatment like paid channels, but without ROI calculation (since there's no spend).

**Non-media treatments** (`{name}_treatment` columns, e.g., `promotion_discount_treatment`, `pricing_treatment`). For things you can control that aren't media - pricing changes, promotions, distribution changes. Modeled separately from controls.

**Control variables** (columns ending in `_control`, or named `is_holiday`, `product_launch`, etc.). These help the model separate marketing effects from other things that drive conversions:

| Control | What it captures | Format |
|---------|-----------------|--------|
| `is_holiday` | Holiday periods (Black Friday, Christmas, etc.) | 0 or 1 |
| `product_launch` | New product/feature releases | 0 or 1 |
| `is_promotion` | Sale events and discount periods | 0 or 1 |
| `seasonality_control` | Business seasonality index | 0.0 to 1.0 |
| `competitor_control` | Major competitor activity | 0 or 1 |

These are NOT auto-detected. You add them as columns to your CSV. The model picks them up automatically if they follow the naming convention above.

**Calibration data** (`data/calibration.json`). This tells the model what you already know about channel performance. Three sources, in order of value:

1. **Incrementality experiments** (strongest). Geo-lift tests, holdout experiments, or platform lift studies. These dramatically tighten confidence intervals. The system will recommend which channels to test and how.

2. **Platform-reported outcomes** (useful as a ceiling). Attributed conversion or revenue outcomes from Meta Ads Manager or Google Ads. The model treats these as a soft upper bound since platforms tend to overclaim by 2-5x. Provide these during agent-guided onboarding or by editing `data/calibration.json` directly.

3. **Your team's beliefs** (better than nothing). "We think Meta returns about 1-2x" with a confidence level. Even rough estimates beat the model's default (wide-open priors centered around 1x).

See [`data/calibration_example.json`](data/calibration_example.json) for the format.
Every calibration record declares either `monetary_roi` or
`incremental_kpi_per_currency`; the run is rejected before GPU spend if units are
mixed or do not match the model outcome.

### What the system recommends you add

After each model run, the system runs 11 diagnostic checks and tells you what would help most:

| What it checks | Example |
|----------------|---------|
| Wide confidence intervals | "Meta's CI is too wide. Run a 4-week geo holdout in 3 states." |
| Poor model fit | "R-squared is 0.55. Add holiday and promotion columns to explain the missing variance." |
| Missing calibration | "What does Google Ads report as your conversion count? This sets an upper bound." |
| Short data history | "You have 30 weeks of data. Each additional quarter improves estimates by 10-20%." |
| Aggregated channels | "Your 'social' channel should be split into Meta and TikTok." |
| Budget concentration | "Meta is 80% of your spend. The model can barely measure Google and TikTok." |
| Brand search inflation | "Brand search shows 8x ROI, but it may be capturing demand that other channels created." |
| Organic baseline sanity | "The model says 85% of conversions are organic. Does that match your intuition?" |
| Adstock misspecification | "TikTok's ad effect decays instantly in the model. Is your product really an impulse buy?" |
| Geographic signal | "You have 5 channels but only 2 geos. Add more regions so the model can separate effects." |
| Prior-posterior divergence | "Google's ROI estimate is near zero despite platform data showing 2x. Re-examine priors." |

These are prioritized by impact and tracked across runs. Act on a suggestion, re-run the model, and the system compares before and after.

See [`data/examples/sample_data.csv`](data/examples/sample_data.csv) for a complete example with impression data, holiday flags, and product launch controls.

## How it works

```
    FIRST RUN                          ONGOING
    ─────────                          ───────

    Onboard brand                     Analyze latest result
     │                                  │
     ├─ Brand context                   ├─ Baseline run on GPU
     ├─ Data assessment      ┌─────>    ├─ Assess diagnostics
     ├─ Prior calibration    │          ├─ Try parameter variations (auto)
     └─ Ready to run ────────┘          ├─ Compare runs, pick best config
                                        ├─ Write analysis + recommendations
                                        ├─ Log what worked ───────────────┐
                                        │                                 │
                                        ├─ "Things I'll try next run"     │
                                        │   (model params, config)        │
                                        │                                 │
                                        └─ "Things you need to do"        │
                                            (data, experiments)           │
                                                    │                     │
                                                    │  you act on these   │
                                                    │                     │
                                                    └─────> analyze again ─┘
                                                            (next run)
```

Each run, the agent handles what it can (parameter tuning, holdout checks) and tells you what it can't (collect data, run experiments). The loop compounds - learnings from each run inform the next.

## Understanding results

### Channel ROI
```
Channel ROI:
  meta   : 0.85x  (90% CI: 0.52 - 1.21)
  google : 1.42x  (90% CI: 0.89 - 2.05)
```
- Monetary ROI measures incremental revenue per currency unit spent, not profit.
- Compare uncertainty intervals and business margins before considering changes.
- Use constrained optimizer scenarios to explore allocations; average ROI alone
  does not establish where the next dollar should go.

### CPIK (Cost per Incremental KPI)
```
CPIK (Cost per Incremental KPI):
  meta   : $3.21
  google : $6.30
  tiktok : $13.84
```
CPIK is spend per incremental KPI unit: the inverse of KPI-per-currency efficiency.
It is not generally the inverse of monetary ROI. Compare it with an economically
justified target cost per KPI; profitability also depends on margins and other costs.

### Marginal vs average ROI
```
Marginal ROI (at current spend):
  meta   : 0.45x  <- saturated (marginal < average)
  google : 1.65x  <- room to grow (marginal > average)
```
Marginal ROI tells you where your next dollar is best spent. If it's lower than the average ROI for that channel, you're hitting diminishing returns.

### Model quality
```
R-squared: 0.72 (Good)     MAPE: 12.3% (Good)     Convergence: OK
```
- **R-squared > 0.6**: Model explains the data well
- **MAPE < 20%**: Predictions are accurate
- **Convergence OK**: Bayesian sampling worked correctly

## Output files

Each run produces:

```
outputs/
├── full_results_YYYYMMDD.json    # Raw results (ROI, contributions, model fit)
├── full_results_YYYYMMDD.html    # Visual report for stakeholders
├── analysis_YYYYMMDD.md          # AI-generated recommendations
├── model_quality_history.json    # Quality metrics across all runs
└── model_quality_report.txt      # Latest quality assessment
```

## Project structure

```
sommmelier/
├── .agents/                   # Canonical cross-agent instructions
│   ├── AGENTS.md              #   Concise execution contract
│   └── skills/                #   Onboarding, analysis, scenario, walkthrough
├── .claude/skills -> ../.agents/skills
├── .codex/skills -> ../.agents/skills
├── AGENTS.md -> .agents/AGENTS.md
├── CLAUDE.md -> .agents/AGENTS.md
├── context/                   # Brand-specific knowledge from onboarding
├── mmm/                       # Core Python package
│   ├── cli/                   #   CLI commands
│   ├── data/                  #   Data loading, validation, schemas
│   ├── model/                 #   Meridian model wrapper
│   ├── analysis/              #   Visualization and report generation
│   ├── recommendations/       #   Recommendation engine + improvement advisor
│   ├── calibration/           #   Prior calibration (experiments, platform, beliefs)
│   └── tracking/              #   Model quality tracking over time
├── data/
│   ├── raw/                   #   Your data (gitignored)
│   └── examples/              #   Sample datasets
├── outputs/                   #   Model results (gitignored)
├── run_weekly.py              #   Full pipeline: validate → fit → report → analyze
└── modal_mmm_full.py          #   GPU model fitting (runs on Modal)
```

## Cost

~$0.30-0.50 per model run (T4 GPU, 10-15 minutes). No subscriptions, pay only for compute you use.

## Limitations

- Requires a [Modal](https://modal.com) account for GPU access (no local GPU support yet)
- Needs 26+ weeks of weekly data for reliable estimates (52+ recommended)
- Confidence intervals widen with sparse data or few geos
- Currently supports weekly granularity only

## Contributing

Pull requests welcome. For major changes, open an issue first.

Run the locked test suite with `uv run --frozen pytest -q --cov=mmm`.
See [synthetic evaluations](evals/README.md) for reproducible dummy datasets,
ground-truth scoring, and the separately authorized GPU compatibility run.

## License

MIT

## Acknowledgments

- [Google Meridian](https://github.com/google/meridian) for Bayesian MMM
- [Modal](https://modal.com) for serverless GPU compute
- [Claude Code](https://code.claude.com/docs/en/overview) for the AI analyst layer
