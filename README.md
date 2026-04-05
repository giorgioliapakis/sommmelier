# Sommmelier

**An MMM data scientist in your terminal.**

Sommmelier runs Bayesian Marketing Mix Models on GPU, interprets the results in your business context, and tells you what to change to make the model more accurate next time.

Built on [Google Meridian](https://github.com/google/meridian). Runs on GPU via [Modal](https://modal.com) (~$0.30/run). Designed to work with [Claude Code](https://claude.ai/download).

## What it does

1. **Fits MMM models on cloud GPU** via Modal.com (~$0.30/run)
2. **Generates visual reports** with charts for stakeholders
3. **Tracks model quality** (R-squared, MAPE, convergence) over time
4. **Coaches you through improvements** like a data scientist would

After each run, the system analyzes 11 diagnostics and tells you specifically what to change:

- "Meta's CI is too wide. Run a 4-week geo holdout in 3 states to calibrate."
- "Add a holiday control variable. The model is attributing promo lifts to ad spend."
- "Brand search shows 8x ROI, but it's probably capturing demand other channels created."
- "You have 5 channels but only 2 geos. More geographic granularity would help."
- "R-squared dropped from 0.78 to 0.65. Investigate a structural break in the data."

Suggestions are tracked across runs. Act on one, re-run, and see whether it helped.

## Quick start

### Prerequisites

- Python 3.11 or 3.12
- A [Modal](https://modal.com) account (free tier available, this is where the model runs on GPU)
- [Claude Code](https://claude.ai/download) (recommended, acts as your MMM analyst)

### Install

```bash
git clone https://github.com/giorgioliapakis/sommmelier.git
cd sommmelier
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Set up Modal for GPU access
pip install modal
modal setup
```

### Option A: Guided experience (Claude Code)

If you have [Claude Code](https://claude.ai/download), open it in this project directory:

```bash
# 1. Start Claude Code in the project
claude

# 2. Set up your brand context (guided conversation)
/init

# 3. Try with example data first
/walkthrough

# 4. Or run on your own data
/sommmelier data/raw/your_data.csv
```

`/init` asks about your brand, channels, KPIs, and goals. It saves everything to `context/` files that make future analysis specific to your situation, and adjusts how technical or hand-holdy it is based on your experience level.

`/sommmelier` runs the model (or analyzes existing results), reads your brand context, and writes recommendations that reference your specific goals and constraints.

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
| `conversions` | Yes | `1523` |
| `{channel}_spend` | Yes | `meta_spend`, `google_spend` |

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Time periods | 26 weeks | 52+ weeks |
| Geographies | 1 | 5+ |
| Media channels | 2 | 3-7 |

### Things that improve the model (optional but recommended)

The system tells you which of these matter most for your situation. You don't need all of them upfront. Start with what you have.

**Impression data** (`{channel}_impression` columns). Without this, the model estimates impressions from spend at $10 CPM. Real impressions are better.

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

2. **Platform-reported metrics** (useful as a ceiling). What Meta Ads Manager or Google Ads reports as your ROAS or CPA. The model treats these as a soft upper bound since platforms tend to overclaim by 2-5x. You provide these numbers during `/init` or by editing `data/calibration.json` directly.

3. **Your team's beliefs** (better than nothing). "We think Meta returns about 1-2x" with a confidence level. Even rough estimates beat the model's default (wide-open priors centered around 1x).

See [`data/calibration_example.json`](data/calibration_example.json) for the format.

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

    /init                              /sommmelier
     │                                  │
     ├─ Brand context                   ├─ Fit model on GPU
     ├─ Data assessment      ┌─────>    ├─ Interpret results
     ├─ Prior calibration    │          ├─ Compare to last run
     └─ Ready to run ────────┘          ├─ Write recommendations
                                        ├─ Suggest improvements ──┐
                                        └─ Update learnings       │
                                                                  │
                                        ┌─────────────────────────┘
                                        │
                                        v
                              "Add holiday controls"
                              "Run geo holdout for Meta"
                              "Split social → Meta + TikTok"
                              "Prior-posterior divergence on Google"
                                        │
                                        │  you act on suggestions
                                        │
                                        └─────> /sommmelier (next run)
                                                model gets more accurate
```

Each run produces reports and recommendations. The useful part is the loop: change something, re-run, see if it helped.

## Understanding results

### Channel ROI
```
Channel ROI:
  meta   : 0.85x  (90% CI: 0.52 - 1.21)
  google : 1.42x  (90% CI: 0.89 - 2.05)
```
- **> 1.5x**: Strong performer, consider scaling
- **1.0 - 1.5x**: Profitable, maintain or test scaling
- **< 1.0x**: Underperforming, needs investigation

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
├── .claude/commands/          # Claude Code slash commands
│   ├── init.md                #   /init (onboarding)
│   ├── sommmelier.md          #   /sommmelier (analysis)
│   └── walkthrough.md         #   /walkthrough (guided demo)
├── context/                   # Brand-specific knowledge (created by /init)
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
├── CLAUDE.md                  #   Instructions for the AI analyst
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

## License

MIT

## Acknowledgments

- [Google Meridian](https://github.com/google/meridian) for Bayesian MMM
- [Modal](https://modal.com) for serverless GPU compute
- [Claude Code](https://claude.ai/download) for the AI analyst layer
