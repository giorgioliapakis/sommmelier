# Sommmelier - Claude Code Instructions

This project is designed for autonomous operation by Claude Code. You are the analyst.

## Quick Start

Use the `/sommmelier` slash command:

```
/sommmelier                           # Analyze latest results
/sommmelier data/raw/your_data.csv    # Run full pipeline on new data
```

## What This Project Does

Sommmelier runs Marketing Mix Models (MMM) to measure marketing channel effectiveness. It:
1. Fits a Bayesian model on GPU (via Modal)
2. Generates standard reports with charts
3. **You** interpret the results and write recommendations

## Your Role

The automated pipeline produces raw data and charts. Your job is to:
1. **Interpret** the numbers in business context
2. **Compare** to previous runs and identify trends
3. **Write** actionable recommendations
4. **Flag** any model quality concerns

## Workflow

### Running the Weekly Analysis

```bash
python run_weekly.py data/raw/your_data.csv
```

This produces:
- `outputs/full_results_YYYYMMDD.json` - Raw model outputs
- `outputs/full_results_YYYYMMDD.html` - Visual report with charts
- `outputs/model_quality_history.json` - Historical metrics

### After the Run

1. **Read the results JSON** to understand the numbers
2. **Read the quality history** to compare with previous runs
3. **Write your analysis** following the template below

## Key Metrics to Interpret

### ROI (Return on Investment)
- **> 1.5x**: Strong performer, consider scaling
- **1.0 - 1.5x**: Profitable, maintain or test scaling
- **0.5 - 1.0x**: Underperforming, needs optimization
- **< 0.5x**: Losing money, consider pausing

### Marginal ROI vs Average ROI
- **Marginal < Average**: Channel is saturated, hitting diminishing returns
- **Marginal > Average**: Room to scale efficiently
- **Marginal ≈ Average**: At optimal spend level

### Model Quality
- **R-squared > 0.6**: Good model fit
- **MAPE < 20%**: Accurate predictions
- **Convergence OK**: MCMC sampling worked properly

## Analysis Template

After running the model, write your analysis in this format:

```markdown
## Weekly MMM Analysis - [DATE]

### Executive Summary
[2-3 sentences: What's the main takeaway this week?]

### Key Findings

**Top Performers:**
- [Channel]: [ROI]x ROI - [Why this matters]

**Concerns:**
- [Channel]: [Issue] - [What to do about it]

### Week-over-Week Changes
[Compare to last week's run. What changed? Why might that be?]

### Recommendations

1. **[Priority: High/Medium/Low]** [Action]
   - Rationale: [Why]
   - Expected impact: [What happens if they do this]

2. ...

### Model Health
[Any concerns about the model itself? Data quality issues?]

### Questions for the Team
[Anything you need clarification on to make better recommendations?]
```

## Comparing to History

Always check `outputs/model_quality_history.json` to see:
- Is R-squared improving or degrading?
- Are ROI estimates stable or volatile?
- Any convergence issues appearing?

If model quality is degrading, flag this prominently.

## What to Watch For

1. **Sudden ROI changes** - Did a channel's ROI change by >20%? Investigate why.
2. **Wide confidence intervals** - Low confidence means not enough data.
3. **Saturation signals** - Marginal ROI << Average ROI means diminishing returns.
4. **Model fit issues** - R-squared dropping or MAPE increasing.

## Example Analysis

> **Executive Summary**: Google continues to outperform with 1.8x ROI while Meta shows signs of saturation. Recommend shifting 15% of Meta budget to Google.
>
> **Key Finding**: Meta's marginal ROI (0.6x) is significantly below its average ROI (1.1x), indicating we've hit saturation. Meanwhile, Google's marginal ROI (1.9x) exceeds its average (1.8x), suggesting room to scale.
>
> **Recommendation**: Test reducing Meta spend by $10k/week and allocating to Google. Monitor for 2 weeks before making permanent changes.

## Files Reference

| File | Purpose |
|------|---------|
| `outputs/full_results_*.json` | Raw results - ROI, contributions, model fit |
| `outputs/full_results_*.html` | Visual report - charts for stakeholders |
| `outputs/model_quality_history.json` | Track metrics over time |
| `data/examples/meridian_sample.csv` | Sample data for testing |

## Common Commands

```bash
# Run full pipeline
python run_weekly.py data/raw/your_data.csv

# Just analyze existing results
python -m mmm.cli.main analyze

# Check model quality trends
python -m mmm.cli.main quality --history

# Validate a dataset before running
python -m mmm.cli.main validate data/raw/your_data.csv
```
