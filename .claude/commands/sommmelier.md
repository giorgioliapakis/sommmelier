---
description: Run MMM analysis and generate recommendations
allowed-tools: Bash, Read, Write, Glob, Grep
---

You are running the Sommmelier Marketing Mix Model analysis workflow.

## Your Task

$ARGUMENTS

If no arguments provided, analyze the latest results.

## Step 0: Read Context

Check for brand context to personalize your analysis:

1. Glob for `context/*.md` files (excluding README.md)
2. If **no context files exist**: tell the user "Tip: Run `/init` to set up your brand context -- it makes analysis much more specific to your situation." Then continue with generic analysis.
3. If **context files exist**, read:
   - `context/brand-profile.md` -- for experience level and company info
   - `context/goals-and-kpis.md` -- for KPI targets and constraints
   - `context/channels.md` -- for channel details and budgets
   - `context/calibration-rationale.md` -- for prior provenance
   - `context/model-learnings.md` -- for findings from previous runs (if exists)
   - `context/improvement-backlog.md` -- for outstanding improvement suggestions (if exists)
4. Note the experience level and adapt your communication accordingly throughout.
5. Check `Last updated:` dates -- if any file is over 90 days old, mention it may need refreshing.

## Step 1: Check for Data

Check what data and results exist:
- Look for the latest results in `outputs/full_results_*.json`
- Check `outputs/model_quality_history.json` for historical context

## Step 2: Run the Model (if requested)

If the user provided a data file path, run the full pipeline:
```bash
python run_weekly.py [data_file]
```

This takes ~10-15 minutes on Modal GPU.

## Step 3: Read the Results

Read the latest results JSON file. Key fields to examine:
- `roi` or `channel_roi` -- ROI by channel with confidence intervals
- `marginal_roi` -- Current marginal returns
- `contributions` -- Channel contribution to KPI
- `model_fit` -- R-squared, MAPE, convergence

## Step 4: Compare to History

Read `outputs/model_quality_history.json` and compare:
- Is R-squared improving or degrading?
- How have channel ROIs changed?
- Any convergence issues?
- Flag any channel with >20% ROI change from previous run

## Step 5: Write Your Analysis

Write the analysis to `outputs/analysis_[DATE].md`. Adapt based on context:

**If context exists**, your analysis should:
- Use the company name and reference their specific KPI targets
- Frame recommendations around their stated goals ("To hit your target CPA of $25...")
- Flag when a recommendation conflicts with a stated constraint (e.g., "Note: This suggests increasing Meta spend, but your context indicates Meta budget is fixed at $50k/month")
- Reference prior model learnings ("Last run showed Google saturation -- this run confirms that trend")

**Always include these sections:**

```markdown
# Weekly MMM Analysis - [DATE]

## Executive Summary
[2-3 sentences: Main takeaway, referencing brand goals if context exists]

## Key Findings

### Top Performers
- [Channel]: [ROI]x ROI - [Why this matters for THIS brand]

### Concerns
- [Channel]: [Issue] - [Recommendation]

## Week-over-Week Changes
[What changed from last run? Why? Flag changes >20%]

## Recommendations

1. **[High/Medium/Low]** [Action]
   - Rationale: [Why, referencing brand context]
   - Expected impact: [What happens]
   - Constraints: [Any conflicts with stated constraints]

## How to Improve This Model

This is the most important section. Think like a data scientist advising a client. Be specific.

For each suggestion, explain: what to do, why it helps, and how much impact to expect.

Examples of what to recommend:
- "Meta's 90% CI spans 0.3-1.8x -- that's too wide to act on. **Run a 4-week geo holdout** in 3 states to calibrate. This would narrow the CI significantly."
- "The model has no holiday control variable, but your brand profile mentions strong Q4 seasonality. **Add an is_holiday column** -- the model is likely attributing holiday lifts to ad spend."
- "You're running Meta and TikTok as one 'social' channel. **Split them** -- they have very different audiences and response curves."
- "R-squared dropped from 0.78 to 0.65 since last run. **Check for a structural change** -- new product launch? pricing change? market shift?"
- "Google's platform reports 2.1x ROAS but the model estimates 0.9x ROI. **This gap is normal** (platform over-attributes), but the divergence is larger than typical -- investigate if Google's attribution window changed."
- "You only have 28 weeks of TikTok data. **Wait 6 more months** before trusting that channel's ROI estimate, or run an incrementality test to calibrate."

Check the improvement backlog from previous runs. If a suggestion was already made and not acted on, note it. If it was acted on, compare before/after.

## Confidence and Caveats

[Calibrate to experience level:]
- **Beginner**: Plain language. "The model is less certain about TikTok because we only have 6 months of data. Take TikTok recommendations with a grain of salt."
- **Intermediate**: Mix of plain and technical. "TikTok's wide confidence interval (0.3-1.8x) means the model needs more data to be sure about its effectiveness."
- **Advanced**: Statistical context. "TikTok posterior CI width of 1.5 with R-hat 1.03 suggests adequate convergence but low posterior concentration. Consider informative priors from platform data or an incrementality test."

[Always state explicitly what the model CAN and CANNOT tell them.]

## Model Health
[R-squared: X, MAPE: Y%, Convergence: OK/Warning]
```

## Step 6: Update Context

After writing the analysis, update context files:

### model-learnings.md
Append a dated entry to `context/model-learnings.md` (create the file if it doesn't exist):
```markdown
## [TODAY'S DATE]
- **Model fit:** R-squared X, MAPE Y%
- **Top performer:** [Channel] at [ROI]x
- **Key changes:** [What shifted from last run]
- **Notable:** [Any surprising findings or concerns]
```
Keep entries concise -- summaries, not full results. If the file is getting long (>50 entries), summarize older entries.

### improvement-backlog.md
Write `context/improvement-backlog.md` with current improvement suggestions:
```markdown
Last updated: [TODAY'S DATE]

# Improvement Backlog

## Active Suggestions
- [ ] [Suggestion from improvement advisor output]
- [ ] [Suggestion]

## Previously Addressed
- [x] [Any items that were acted on, with before/after impact noted]
```

If a previous backlog exists, preserve checked items and note which suggestions are new.

## Output

After analysis, tell the user:
1. Where to find the HTML report (for stakeholders)
2. Key recommendations (top 2-3)
3. Any model health concerns
4. Top improvement suggestion

## Interpretation Guide

**ROI Thresholds:**
- > 1.5x: Strong - consider scaling
- 1.0-1.5x: Good - maintain
- 0.5-1.0x: Weak - optimize
- < 0.5x: Poor - consider pausing

**Marginal vs Average ROI:**
- Marginal < Average: Saturated, diminishing returns
- Marginal > Average: Room to scale
- Marginal ~ Average: Optimal level

**Model Quality:**
- R-squared > 0.6: Good fit
- MAPE < 20%: Accurate predictions
- Check for convergence warnings
