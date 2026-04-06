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

## Step 2: Baseline Run

If the user provided a data file path, run the model:
```bash
modal run modal_mmm_full.py --data [data_file]
```

This is the baseline run. Read the results JSON when it completes.

## Step 3: Diagnose and Decide

After the baseline run completes, review the results like a data scientist would. Your job is to figure out what's limiting this model and whether you can fix it, or whether it needs human action.

**Look at the baseline and ask:**
- Is R-squared suspiciously high (>0.99)? Probably overfit. Run holdout to check.
- Is R-squared low (<0.5)? The model is underspecified. Likely needs more control variables or better priors - that's a recommendation for the human.
- Are confidence intervals very wide on important channels? Usually means not enough data or weak priors - again, a human problem.
- Does the adstock decay look wrong? (e.g., a brand/video channel showing instant decay when it should have a longer carryover). Try a different `max_lag`.
- Are there convergence warnings (R-hat > 1.1)? Bump up `--n-keep` and `--n-adapt` and re-run.

**Most model quality issues are about data and priors, not parameters.** Don't run extra model fits hoping to find a magic config. The defaults are sensible. Only re-run if you have a specific reason:

| Reason to re-run | What to change |
|-------------------|---------------|
| Convergence warnings | `--n-keep 1000 --n-adapt 3000` (just needs more samples) |
| Suspect overfitting | `--holdout-weeks 8` (check out-of-sample fit) |
| Wrong carryover window | `--max-lag 4` or `--max-lag 12` (domain-dependent) |

**Don't re-run for:** wide CIs (need more data/priors), low R-squared (need control variables), weird ROI estimates (need calibration). These are recommendations for the human.

### Budget cap

You may run **up to 2 additional model runs** beyond the baseline (3 total max). Most of the time, one run is enough. Tell the user if and why you're running another.

### Comparing runs

If you did run a variation, compare to baseline:
- Did the specific thing you were testing improve?
- Log what you tried and what happened in model-learnings.md.

Pick the best run's results for the analysis.

## Step 4: Read Final Results

Read the best results JSON. Key fields to examine:
- `roi` -- ROI by channel with confidence intervals
- `cpik` -- Cost per incremental KPI
- `marginal_roi` -- Current marginal returns
- `contributions` -- Channel contribution to KPI
- `model_fit` -- R-squared, MAPE, convergence
- `metadata.config` -- What parameters produced these results
- `model_review` -- Diagnostic check results
- `optimal_frequency` -- For R&F channels
- `organic_contributions` -- For organic media channels
- `treatment_effects` -- For non-media treatments

## Step 5: Compare to History

Read `outputs/model_quality_history.json` and compare:
- Is R-squared improving or degrading?
- How have channel ROIs changed?
- Any convergence issues?
- Flag any channel with >20% ROI change from previous run

## Step 6: Write Your Analysis

Write the analysis to `outputs/analysis_[DATE].md`. Adapt based on context:

**If context exists**, your analysis should:
- Use the company name and reference their specific KPI targets
- Frame recommendations around their stated goals ("To hit your target CPA of $25...")
- Flag when a recommendation conflicts with a stated constraint
- Reference prior model learnings

**Always include these sections:**

```markdown
# MMM Analysis - [DATE]

## Executive Summary
[2-3 sentences: Main takeaway, referencing brand goals if context exists]

## What We Tested
[If you ran multiple variations, summarize what you tried and why]
- Baseline: default config → R²=X, MAPE=Y%
- Variation 1: [what you changed] → R²=X, MAPE=Y% → [kept/discarded, why]
- ...
[Selected config: ...]

## Key Findings

### Top Performers
- [Channel]: [ROI]x ROI, CPIK $[X] - [Why this matters for THIS brand]

### Concerns
- [Channel]: [Issue] - [Recommendation]

## Run-over-Run Changes
[What changed from last run? Why? Flag changes >20%]

## Recommendations

1. **[High/Medium/Low]** [Action]
   - Rationale: [Why, referencing brand context]
   - Expected impact: [What happens]
   - Constraints: [Any conflicts with stated constraints]

## What Would Improve This Model

Split into two categories:

### Things we can test next run (model parameters)
[Only if you have specific hypotheses you didn't get to test within the budget cap]

### Things that need human action
[Data collection, experiments, new columns — things Claude can't do]

Be specific. Not "add more data" but "add a binary is_holiday column for weeks containing Black Friday, Christmas, and Easter — the model is likely attributing seasonal lifts to ad spend."

## Confidence and Caveats

[Calibrate to experience level:]
- **Beginner**: Plain language.
- **Intermediate**: Mix of plain and technical.
- **Advanced**: Statistical context.

[Always state explicitly what the model CAN and CANNOT tell them.]

## Model Health
[R-squared, MAPE, convergence status, config used]
```

## Step 7: Update Context

After writing the analysis, update context files:

### model-learnings.md
Append a dated entry to `context/model-learnings.md` (create if needed):
```markdown
## [TODAY'S DATE]
- **Config:** [key params that differ from defaults, e.g. max_lag=12, n_keep=1000]
- **Model fit:** R-squared X, MAPE Y%
- **Top performer:** [Channel] at [ROI]x
- **Variations tested:** [What you tried and what happened]
- **Key insight:** [The most important thing learned this run]
- **Open questions:** [What would help next time]
```

### improvement-backlog.md
Write `context/improvement-backlog.md` with current suggestions:
```markdown
Last updated: [TODAY'S DATE]

# Improvement Backlog

## Model Parameter Ideas (for next auto-research)
- [ ] [Things to try with different config next run]

## Needs Human Action
- [ ] [Data/experiment suggestions the agent can't do itself]

## Previously Addressed
- [x] [Items acted on, with before/after impact]
```

## Output

After analysis, tell the user:
1. How many runs you did and why (or why just one was enough)
2. Key recommendations (top 2-3)
3. What would improve the model — split into "things I can try next time" vs "things you need to do"
4. Any model health concerns

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
