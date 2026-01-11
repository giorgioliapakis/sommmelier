---
description: Run MMM analysis and generate recommendations
allowed-tools: Bash, Read, Write, Glob, Grep
---

You are running the Sommmelier Marketing Mix Model analysis workflow.

## Your Task

$ARGUMENTS

If no arguments provided, analyze the latest results.

## Workflow

### Step 1: Check for Data

First, check what data and results exist:
- Look for the latest results in `outputs/full_results_*.json`
- Check `outputs/model_quality_history.json` for historical context

### Step 2: Run the Model (if requested)

If the user provided a data file path, run the full pipeline:
```bash
python run_weekly.py [data_file]
```

This takes ~10-15 minutes on Modal GPU.

### Step 3: Read the Results

Read the latest results JSON file. Key fields to examine:
- `roi` - ROI by channel with confidence intervals
- `marginal_roi` - Current marginal returns
- `contributions` - Channel contribution to KPI
- `model_fit` - R-squared, MAPE, convergence

### Step 4: Compare to History

Read `outputs/model_quality_history.json` and compare:
- Is R-squared improving or degrading?
- How have channel ROIs changed?
- Any convergence issues?

### Step 5: Write Your Analysis

The pipeline creates `outputs/analysis_[DATE].md`. Review and enhance it with:

```markdown
# Weekly MMM Analysis - [DATE]

## Executive Summary
[2-3 sentences: Main takeaway]

## Key Findings

### Top Performers
- [Channel]: [ROI]x - [Interpretation]

### Concerns
- [Channel]: [Issue] - [Recommendation]

## Week-over-Week Changes
[What changed from last run? Why?]

## Recommendations

1. **[High/Medium/Low]** [Action]
   - Rationale: [Why]
   - Expected impact: [What happens]

## Model Health
[R-squared: X, MAPE: Y%, Convergence: OK/Warning]
[Any concerns about the model itself?]
```

## Interpretation Guide

**ROI Thresholds:**
- > 1.5x: Strong - consider scaling
- 1.0-1.5x: Good - maintain
- 0.5-1.0x: Weak - optimize
- < 0.5x: Poor - consider pausing

**Marginal vs Average ROI:**
- Marginal < Average: Saturated, diminishing returns
- Marginal > Average: Room to scale
- Marginal ≈ Average: Optimal level

**Model Quality:**
- R-squared > 0.6: Good fit
- MAPE < 20%: Accurate predictions
- Check for convergence warnings

## Output

After analysis, tell the user:
1. Where to find the HTML report (for stakeholders)
2. Key recommendations (top 2-3)
3. Any model health concerns
