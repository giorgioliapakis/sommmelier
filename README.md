# Sommmelier

**AI-driven Marketing Mix Modeling powered by Google Meridian.**

Run Marketing Mix Models autonomously from the command line. GPU-accelerated model fitting, automated analysis, and self-improving quality tracking.

Built on Google's [Meridian](https://github.com/google/meridian) framework. Designed for autonomous operation via [Claude Code](https://github.com/anthropics/claude-code).

## What It Does

1. **Fits MMM models on cloud GPU** - Uses Modal.com for serverless GPU compute (~$0.30/run)
2. **Generates visual reports** - HTML reports with charts that stakeholders can understand
3. **Tracks model quality over time** - R-squared, MAPE, and convergence metrics
4. **Claude-first design** - Claude Code interprets results and writes recommendations

See [CLAUDE.md](CLAUDE.md) for Claude Code workflow instructions.

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
1. Upload your data to Modal
2. Fit the MMM on a T4 GPU (~10 minutes)
3. Generate `outputs/full_results_*.json` with ROI and contributions
4. Generate `outputs/full_results_*.html` visual report
5. Generate `outputs/analysis_*.txt` with recommendations
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
├── analysis_YYYYMMDD.txt         # AI recommendations
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

## CLI Commands

After installing (`pip install -e .`), you can use the `sommmelier` CLI:

```bash
sommmelier analyze                    # Analyze latest results
sommmelier analyze results.json       # Analyze specific file
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
                        ┌───────────────────────────────┼───────────────────────────────┐
                        v                               v                               v
                ┌─────────────────┐             ┌─────────────────┐             ┌─────────────────┐
                │  HTML Report    │             │  Recommendations │             │ Quality Tracking │
                │  (Visualize)    │             │  (Analyze)       │             │ (Improve)        │
                └─────────────────┘             └─────────────────┘             └─────────────────┘
```

## Self-Improving Loop

Each week when you run the model:

1. **Fit** - New model with latest data
2. **Compare** - Week-over-week ROI changes
3. **Track** - R-squared and MAPE trends
4. **Recommend** - If model quality degrades, get suggestions to fix it

The system tracks whether the model is improving or degrading over time, so you can trust the recommendations.

## Cost

- **Modal GPU**: ~$0.30-0.50 per run (T4 GPU for 10-15 minutes)
- **No subscription fees** - pay only for compute you use

## Project Structure

```
sommmelier/
├── mmm/
│   ├── cli/               # CLI commands
│   ├── data/              # Data loading & validation
│   ├── model/             # Meridian wrapper
│   ├── analysis/          # Insights & visualization
│   ├── recommendations/   # AI recommendation engine
│   └── tracking/          # Model quality tracking
├── data/
│   ├── raw/               # Your data (gitignored)
│   └── examples/          # Sample datasets
├── outputs/               # Results (gitignored)
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

- [Google Meridian](https://github.com/google/meridian) - The underlying Bayesian MMM framework
- [Modal](https://modal.com) - Serverless GPU compute
- Built with [Claude Code](https://github.com/anthropics/claude-code)
