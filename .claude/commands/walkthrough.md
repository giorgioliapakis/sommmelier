---
description: Guided first-run walkthrough with example data
allowed-tools: Bash, Read, Write, Glob, Grep
---

You are running a guided walkthrough of Sommmelier using example data. Walk the user through the entire pipeline step by step, explaining what's happening and why at each stage.

Use a beginner-friendly tone throughout -- this is a tutorial experience.

## Step 1: Check Prerequisites

Verify the environment is ready:

```bash
python --version
```

Check Python is 3.11 or 3.12. If not, let the user know.

Check if Modal is set up:
```bash
modal --version
```

If Modal isn't installed or authenticated:
- Tell the user: "Modal provides the GPU for model fitting. You'll need a free account."
- Provide: `pip install modal && modal setup`
- Wait for them to complete setup before proceeding

Check dependencies:
```bash
pip show google-meridian 2>/dev/null | head -2
```

If not installed: `pip install -e .`

## Step 2: Context Check

Check if brand context exists:
- Glob for `context/*.md` files

If context exists:
- "Great, you've already set up your brand context. This walkthrough will use the example dataset but your context will still inform the analysis."

If no context:
- "You haven't set up brand context yet. For this walkthrough, I'll create a demo context for a fictional DTC brand so you can see how context-aware analysis works."
- Create minimal demo context files:
  - `context/brand-profile.md`: A fictional DTC supplement brand, beginner experience level
  - `context/channels.md`: Meta, Google, TikTok with example budgets
  - `context/goals-and-kpis.md`: Target CPA of $30, primary KPI is conversions
- Tell the user: "After the walkthrough, run `/init` to replace this demo with your real brand context."

## Step 3: Explore the Example Data

Read and explain the example dataset:

```bash
head -5 data/examples/sample_data.csv
```

Explain:
- "This is weekly marketing data with 3 geos (US, UK, AU) and 3 channels (Meta, Google, TikTok)"
- Point out the column structure: date, geo, conversions, {channel}_spend, {channel}_impressions
- "This is the format your own data needs to be in"

## Step 4: Validate the Data

Run validation:

```bash
python -m mmm.cli.main validate data/examples/sample_data.csv
```

Walk through each validation check:
- "**Minimum time periods**: The model needs at least 26 weeks of data to identify patterns"
- "**Geographic coverage**: Multiple geos help the model separate marketing effects from other factors"
- "**KPI completeness**: We need conversion data for every time period"
- "**Spend values**: No negative spend allowed"
- "**Media channels**: 3-7 channels is the sweet spot"
- And so on...

## Step 5: Run the Model

**This step costs ~$0.30 on Modal and takes ~10-15 minutes.**

Ask the user: "Ready to run the model? This will fit a Bayesian MMM on a Modal GPU. It costs about $0.30 and takes 10-15 minutes."

If they confirm:
```bash
python run_weekly.py data/examples/sample_data.csv
```

While waiting, explain what's happening:
1. "**Validating** -- checking data format one more time before spending GPU money"
2. "**Checking calibration** -- looking for prior information in data/calibration.json"
3. "**Fitting on GPU** -- the Bayesian model is sampling from the posterior distribution using MCMC. This is where the magic happens."
4. "**Generating reports** -- creating the HTML visual report and running the recommendation engine"
5. "**Tracking quality** -- logging model metrics so we can compare across runs"

If they decline:
- "No problem! Let me show you what the output looks like with a previous run's results instead."
- Check for any existing results in `outputs/full_results_*.json` and use those. If none exist, explain what the output would contain.

## Step 6: Interpret the Results

After the model completes (or using existing results), read the results JSON and walk through:

### ROI by Channel
- "These numbers tell you how much revenue (or conversions) each dollar of ad spend generates"
- "ROI > 1.0 means profitable. The 90% confidence interval shows how certain the model is"
- Explain what the specific numbers mean for the example data

### Marginal ROI
- "This is the ROI of your NEXT dollar of spend, not the average"
- "If marginal ROI < average ROI, you're hitting diminishing returns -- the channel is saturated"
- "If marginal ROI > average ROI, there's room to spend more efficiently"

### Model Quality
- "R-squared tells you how much of the variation in conversions the model explains"
- "MAPE is the average prediction error -- lower is better"
- "Convergence means the statistical sampling process worked correctly"

### Contributions
- "This shows how many conversions each channel drives"
- "Compare this to what the ad platforms report -- MMM typically shows lower numbers because it corrects for over-attribution"

## Step 7: Review the Generated Report

Point the user to the HTML report:
```bash
ls -la outputs/full_results_*.html | tail -1
```

"Open this file in your browser to see the visual report with charts. This is what you'd share with stakeholders."

Also point to the analysis markdown:
```bash
ls -la outputs/analysis_*.md | tail -1
```

"This is the AI-generated analysis with recommendations."

## Step 8: Next Steps

Summarize what they learned, then guide them forward:

1. "**Set up your brand context**: Run `/init` to replace the demo context with your real brand information"
2. "**Prepare your data**: Format your marketing data like the example CSV. See the README for the full column spec."
3. "**Run your first real model**: `/sommmelier data/raw/your_data.csv`"
4. "**Run it weekly**: Each time you run, the system learns more and recommendations get more specific"

"The improvement backlog in `context/improvement-backlog.md` will tell you what to do next to make the model more accurate -- things like adding control variables, running incrementality tests, or collecting more data."
