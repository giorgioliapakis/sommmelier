# Sommmelier

**AI-driven Marketing Mix Modeling powered by Google Meridian.**

Built for [Claude Code](https://github.com/anthropics/claude-code). Run MMM models, get automated reports, then let Claude interpret results and write strategic recommendations.

## Claude Code Workflow

```bash
# In Claude Code, just run:
/sommmelier

# Or with new data:
/sommmelier data/raw/your_data.csv
```

That's it. Claude handles the rest:
1. Runs MMM on Modal GPU
2. Reads results and historical context
3. Writes analysis with strategic recommendations

See [CLAUDE.md](CLAUDE.md) for the full workflow and analysis template.

## What It Does

1. **Fits MMM models on cloud GPU** - Modal.com serverless compute (~$0.30/run)
2. **Generates visual reports** - HTML reports with charts for stakeholders
3. **Tracks model quality** - R-squared, MAPE, convergence over time
4. **Claude interprets** - Automated data collection + Claude's analysis layer

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- [Modal](https://modal.com) account (free tier available)

### Installation

```bash
# Clone the repo
git clone https://github.com/giorgioliapakis/sommmelier.git
cd sommmelier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Authenticate with Modal
pip install modal
modal setup
```

### Run Your First Model

```bash
# Test with sample data
python run_weekly.py data/examples/meridian_sample.csv
```

This will:
1. Validate your data locally (catches errors before GPU spend)
2. Fit the MMM on a T4 GPU (~10 minutes)
3. Generate `outputs/full_results_*.json` with ROI and contributions
4. Generate `outputs/full_results_*.html` visual report
5. Generate `outputs/analysis_*.md` with recommendations
6. Update `outputs/model_quality_history.json` for tracking

### Use Your Own Data

```bash
# Put your data in data/raw/
cp your_marketing_data.csv data/raw/

# Run the model
python run_weekly.py data/raw/your_marketing_data.csv
```

## Data Format

Your CSV needs these columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `date` or `time` | Yes | Time period | `2024-01-01` |
| `geo` | Yes | Geographic region | `US`, `UK`, `AU` |
| `conversions` | Yes | Your KPI | `1523` |
| `{channel}_spend` | Yes | Spend per channel | `meta_spend`, `google_spend` |
| `{channel}_impression` | No | Impressions (estimated from spend if missing) | `meta_impression` |
| `population` | No | Geo population (uses defaults if missing) | `330000000` |
| `{name}_control` | No | Control variables | `seasonality_control` |

### Minimum Data Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Time periods | 26 weeks | 52+ weeks |
| Geographies | 1 | 5+ |
| Media channels | 2 | 3-7 |

## Output Files

After each run:

```
outputs/
├── full_results_YYYYMMDD.json    # Raw results (ROI, contributions, metrics)
├── full_results_YYYYMMDD.html    # Visual report for stakeholders
├── analysis_YYYYMMDD.md          # AI recommendations
├── model_quality_history.json    # Quality tracking across runs
└── model_quality_report.txt      # Latest quality assessment
```

## Understanding the Results

### ROI (Return on Investment)
```
Channel ROI:
  meta   : 0.85x  (90% CI: 0.52 - 1.21)
  google : 1.42x  (90% CI: 0.89 - 2.05)
```
- ROI > 1.0 = profitable
- 90% CI = confidence interval (narrower = more certain)

### Marginal ROI
```
Marginal ROI (at current spend):
  meta   : 0.45x  <- Lower than avg = saturated
  google : 1.65x  <- Higher than avg = room to grow
```
- If marginal ROI < average ROI, you're hitting diminishing returns
- If marginal ROI > average ROI, you can scale spend efficiently

### Model Quality Metrics
```
Model Health:
  R-squared: 0.72 (Good)
  MAPE: 12.3% (Good)
  Convergence: OK
```
- R-squared > 0.6 = model explains variance well
- MAPE < 20% = predictions are accurate
- Convergence OK = MCMC sampling worked

## Commands

### Claude Code (Recommended)

```bash
/sommmelier                           # Analyze latest results
/sommmelier data/raw/your_data.csv    # Run full pipeline on new data
```

### CLI

After installing (`pip install -e .`):

```bash
sommmelier analyze                    # Analyze latest results
sommmelier report results.json        # Generate HTML report
sommmelier quality                    # Show model quality summary
sommmelier quality --history          # Show full quality history
sommmelier validate data.csv          # Validate dataset
```

### Scripts

| Script | Purpose |
|--------|---------|
| `run_weekly.py` | Full pipeline: fit model -> report -> analyze -> track |
| `modal_mmm_full.py` | Modal function for GPU model fitting |

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Your Data     │────>│   Modal GPU     │────>│    Results      │
│   (CSV)         │     │   (Meridian)    │     │    (JSON)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
              ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
              │                              AUTOMATED LAYER                                       │
              │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐               │
              │  │  HTML Report    │    │ Quality Metrics │    │ History Tracking │               │
              │  │  (Charts)       │    │ (R², MAPE)      │    │ (Week/Week)      │               │
              │  └─────────────────┘    └─────────────────┘    └─────────────────┘               │
              └───────────────────────────────────────────────────────────────────────────────────┘
                                                        │
                                                        v
              ┌───────────────────────────────────────────────────────────────────────────────────┐
              │                              CLAUDE LAYER                                          │
              │  - Reads results + history                                                         │
              │  - Compares to previous runs                                                       │
              │  - Writes strategic recommendations                                                │
              │  - Identifies model health issues                                                  │
              └───────────────────────────────────────────────────────────────────────────────────┘
```

The key insight: automated systems collect data and produce charts. Claude interprets what it means and what to do about it.

## Weekly Workflow

Each week when you run `/sommmelier`:

1. **Fit** - New model runs on Modal GPU with latest data
2. **Report** - HTML report generated with charts and metrics
3. **Track** - Quality metrics logged to history
4. **Analyze** - Claude reads everything, writes recommendations

Claude compares week-over-week changes, identifies issues, and writes strategic recommendations you can act on.

## Cost

- **Modal GPU**: ~$0.30-0.50 per run (T4 GPU for 10-15 minutes)
- **No subscription fees** - pay only for compute you use

## Project Structure

```
sommmelier/
├── .claude/
│   └── commands/
│       └── sommmelier.md  # /sommmelier slash command
├── mmm/
│   ├── cli/               # CLI commands
│   ├── data/              # Data loading & validation
│   ├── model/             # Meridian wrapper
│   ├── analysis/          # Insights & visualization
│   ├── recommendations/   # Recommendation engine
│   └── tracking/          # Model quality tracking
├── data/
│   ├── raw/               # Your data (gitignored)
│   └── examples/          # Sample datasets
├── outputs/               # Results (gitignored)
├── CLAUDE.md              # Claude Code instructions
├── run_weekly.py          # Main entry point
└── modal_mmm_full.py      # GPU model fitting
```

## Limitations

- Requires Modal account for GPU access
- Meridian requires 26+ weeks of data for reliable estimates
- Wide confidence intervals with sparse data

## Contributing

Pull requests welcome. For major changes, open an issue first.

## License

MIT

## Acknowledgments

- [Claude Code](https://github.com/anthropics/claude-code) - The AI that interprets results and writes recommendations
- [Google Meridian](https://github.com/google/meridian) - Bayesian MMM framework
- [Modal](https://modal.com) - Serverless GPU compute
