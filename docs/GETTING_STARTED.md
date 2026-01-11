# Getting Started with Sommmelier

This guide walks you through setting up and running your first Marketing Mix Model.

## Prerequisites

1. **Python 3.11 or 3.12** - Meridian doesn't support Python 3.13 yet
2. **Modal account** - Sign up at [modal.com](https://modal.com) (free tier available)

## Step 1: Clone and Install

```bash
git clone https://github.com/giorgioliapakis/sommmelier.git
cd sommmelier

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e .
```

## Step 2: Set Up Modal

```bash
pip install modal
modal setup
```

This opens a browser to authenticate with Modal. Follow the prompts.

## Step 3: Test with Sample Data

```bash
python run_weekly.py data/examples/meridian_sample.csv
```

This runs a full MMM on Google's sample dataset. Takes ~10-15 minutes.

## Step 4: Prepare Your Data

Create a CSV with your marketing data:

```csv
date,geo,conversions,meta_spend,google_spend,tiktok_spend
2024-01-01,US,1523,15000,8500,5000
2024-01-01,UK,412,6000,3500,2000
2024-01-08,US,1687,16000,9000,5500
2024-01-08,UK,398,5500,3200,1800
...
```

### Required Columns

- `date` - Weekly dates (YYYY-MM-DD)
- `geo` - Geographic region codes
- `conversions` - Your KPI (conversions, signups, revenue, etc.)
- `{channel}_spend` - Spend for each marketing channel

### Optional Columns

- `{channel}_impression` - Impressions (estimated from spend if missing)
- `population` - Geographic population (uses defaults if missing)
- `{name}_control` - Control variables (seasonality, promotions, etc.)

### Data Requirements

- **Minimum 26 weeks** of data (52+ recommended)
- At least **2 marketing channels**
- Consistent weekly granularity

## Step 5: Run Your Model

```bash
# Place your data in data/raw/
cp your_data.csv data/raw/

# Run the full pipeline
python run_weekly.py data/raw/your_data.csv
```

## Step 6: Review Results

After the run completes, check the `outputs/` folder:

1. **HTML Report** (`full_results_*.html`) - Open in browser for visual summary
2. **Analysis** (`analysis_*.txt`) - AI-generated recommendations
3. **Quality Report** (`model_quality_report.txt`) - Model health assessment

## Understanding the Output

### ROI Interpretation

| ROI | Meaning |
|-----|---------|
| > 2.0x | Excellent - scale this channel |
| 1.0 - 2.0x | Good - profitable |
| 0.5 - 1.0x | Moderate - optimize or reduce |
| < 0.5x | Poor - consider pausing |

### Model Quality

| Metric | Good | Needs Attention |
|--------|------|-----------------|
| R-squared | > 0.6 | < 0.5 |
| MAPE | < 20% | > 30% |
| Convergence | OK | Warnings |

## Weekly Workflow

For ongoing use:

1. **Update your data CSV** with the latest week
2. **Run the pipeline**: `python run_weekly.py data/raw/your_data.csv`
3. **Review the HTML report** and recommendations
4. **Check quality trends** via `python check_quality.py`

## Troubleshooting

### "Module not found" errors

Make sure you activated the virtual environment:

```bash
source .venv/bin/activate
```

### Modal authentication issues

Re-run `modal setup` to refresh credentials.

### "Insufficient time periods" warning

You need at least 26 weeks of data. The model will still run but confidence intervals will be wide.

### High MAPE or low R-squared

This usually means:

- Not enough data
- Missing important control variables
- High noise in your KPI

## Next Steps

- Read the [full README](../README.md) for advanced configuration
- Check [model quality tracking](../outputs/model_quality_history.json) over time
- Integrate with your data pipeline for automated weekly runs
