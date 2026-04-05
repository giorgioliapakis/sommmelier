---
description: Set up Sommmelier for your brand
allowed-tools: Bash, Read, Write, Glob, Grep
---

You are running the Sommmelier onboarding interview. Your goal is to learn about the user's brand and marketing setup, then write context files that will make all future analysis tailored to them.

## Step 0: Check Existing Context

First, check what already exists:
- Glob for `context/*.md` files (excluding README.md)
- If context files exist, read them and summarize what's already known
- Ask the user: "I found existing context for [summary]. Would you like to continue from here, or start fresh?"
- If no context files exist, proceed to Step 1

## Step 1: Experience Level

Ask: "How would you describe your experience with marketing analytics?"

Offer three options:
- **Beginner** -- "I run ads but haven't done modeling or incrementality testing"
- **Intermediate** -- "I understand attribution and have some experience with experiments"
- **Advanced** -- "I'm comfortable with Bayesian statistics and MMM concepts"

This determines how you phrase all subsequent questions:
- **Beginner**: Explain why each question matters. No jargon. Use analogies.
- **Intermediate**: Brief context, assume familiarity with marketing concepts.
- **Advanced**: Direct questions, technical options, skip explanations.

Store the answer -- it goes into `context/brand-profile.md`.

## Step 2: Brand Profile

Ask about:
- Company/brand name
- Industry and business model (DTC, B2B SaaS, marketplace, etc.)
- Products or services
- Target customer

For beginners, explain: "Understanding your business model helps the AI interpret your marketing data in the right context -- a DTC brand's funnel looks very different from B2B SaaS."

Write to `context/brand-profile.md` with this structure:
```
Last updated: [TODAY'S DATE]

# Brand Profile

**Experience level:** [beginner/intermediate/advanced]

## Company
- **Name:** ...
- **Industry:** ...
- **Business model:** ...
- **Products/services:** ...
- **Target customer:** ...

## Seasonality
[Added in Step 6]
```

## Step 3: Media Channels

Ask about:
- Which advertising channels they run (Meta/Facebook, Google Ads, TikTok, YouTube, LinkedIn, etc.)
- Approximate monthly budget per channel (ranges are fine: <$5k, $5-25k, $25-100k, $100k+)
- Whether they have impression data for each channel
- Any channels they've recently started or stopped

For beginners, list common channels as options to make it easy.

Write to `context/channels.md`:
```
Last updated: [TODAY'S DATE]

# Media Channels

| Channel | Monthly Budget | Has Impressions | Notes |
|---------|---------------|-----------------|-------|
| ...     | ...           | Yes/No          | ...   |
```

## Step 4: KPIs and Goals

Ask about:
- Primary KPI: what are they optimizing for? (conversions, signups, revenue, purchases)
- How is the KPI measured? (analytics platform, CRM, data warehouse)
- Target CPA or ROAS if they have one
- Total marketing budget and any constraints
- Are there channels with fixed/contractual budgets they can't change?

Write to `context/goals-and-kpis.md`:
```
Last updated: [TODAY'S DATE]

# Goals and KPIs

**Primary KPI:** ...
**Measurement:** ...
**Target CPA/ROAS:** ...
**Total monthly budget:** ...

## Constraints
- [Any budget locks, contractual obligations, or non-negotiables]
```

## Step 5: Data and Attribution

Ask about:
- Where does their marketing spend data come from? (platform exports, data warehouse, spreadsheets)
- What attribution model do they currently use? (last-click, first-click, data-driven, MTA)
- Do they already have data in CSV format? If so, what does it look like?
- Any known data quality issues?

For beginners, explain:
- What attribution is and why it matters ("Attribution is how platforms decide which ad gets credit for a sale. Most platforms over-count because they take credit for sales that would've happened anyway -- that's exactly what MMM helps correct.")
- How to export data from Meta Ads Manager / Google Ads (high-level steps)

Write to `context/data-sources.md`:
```
Last updated: [TODAY'S DATE]

# Data Sources

**Spend data source:** ...
**Attribution model:** ...
**Data format:** ...
**Known issues:** ...

## Data Readiness
[Added in Step 8]
```

## Step 6: Seasonality and Events

Ask about:
- Major seasonal patterns (holiday peaks, summer slowdowns, etc.)
- Upcoming promotions or product launches
- Any events that significantly affect their business
- Recurring patterns (end-of-quarter pushes, back-to-school, etc.)

Add a "## Seasonality" section to `context/brand-profile.md`.

## Step 7: Past Experiments

Ask about:
- Have they ever run incrementality tests? (geo-lift, holdout tests, A/B tests on spend)
- Have they used tools like Haus, Measured, or run tests in-platform?
- If yes: which channels were tested, what were the results, how confident are they?

For beginners, explain: "Incrementality tests are the gold standard for measuring if an ad channel actually drives results. If you've run any, the results make our model much more accurate."

If they have experiment results, capture: channel, test type, lift estimate, confidence, when it was run.

Write to `context/calibration-rationale.md`:
```
Last updated: [TODAY'S DATE]

# Calibration Rationale

## Experiment Results
[Any past tests and their results, or "No experiments run yet"]

## Platform-Reported Performance
[What the ad platforms report as ROAS/CPA, noted as likely over-attributed]

## Prior Beliefs
[Any strong beliefs about channel effectiveness and their basis]

## Calibration Decisions
[Filled in during prior elicitation -- Phase 2]
```

## Step 8: Data Readiness Assessment

If the user has a data file ready:
1. Run `python -m mmm.cli.main validate [path]` to check format
2. Beyond format checks, assess data richness based on what you learned:
   - Do they have impression data? (improves accuracy)
   - Do they have control variables like seasonality, promotions, holidays?
   - How many weeks of data? (52+ is recommended, 26 minimum)
   - How many geos? (more geos = more statistical power)
   - Are channels too aggregated? (e.g., "social" instead of "meta" and "tiktok")
3. Summarize findings and add a "## Data Readiness" section to `context/data-sources.md`

If no data yet:
- Provide clear instructions on what to collect, based on what they told you about their data sources
- Reference `data/examples/sample_data.csv` as a format template
- Explain the minimum requirements: date, geo, conversions, {channel}_spend columns

## Step 9: Generate Calibration Template

Check if `data/calibration.json` already exists. If not:

- If the user provided NO experiment data or platform ROAS in Step 7:
  Run `python -c "from mmm.calibration.calibration_data import create_calibration_template; create_calibration_template('data/calibration.json')"`

- If the user DID provide experiment results, platform ROAS numbers, or beliefs:
  Write a pre-filled `data/calibration.json` directly using the Write tool, following this schema:
  ```json
  {
    "experiments": [{"channel": "...", "experiment_type": "geo_lift", "lift_estimate": 0.0, ...}],
    "platform_conversions": [{"channel": "...", "platform_conversions": 0, "period_weeks": 4, "spend": 0, ...}],
    "prior_beliefs": [{"channel": "...", "expected_roi_low": 0.0, "expected_roi_high": 0.0, "confidence": "medium", ...}],
    "control_variables": {},
    "notes": ""
  }
  ```

Update `context/calibration-rationale.md` with what was set and why.

## Step 10: Summary and Next Steps

Summarize everything you captured:
- Brand profile and experience level
- Channels and budgets
- KPI and targets
- Data readiness status
- Calibration status

Then tell the user their next step:
- If data is ready: "Run `/sommmelier data/raw/your_data.csv` to fit your first model"
- If data needs prep: explain what format to prepare, reference the sample data
- If they want to try it first: "Run `/walkthrough` to see the full pipeline with example data"

Mention: "You can edit any file in `context/` directly. Everything you told me is saved there."
