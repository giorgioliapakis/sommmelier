---
name: sommmelier-onboarding
description: Set up or refresh Sommmelier brand context, marketing channels, KPIs, data sources, and calibration evidence. Use whenever a user wants to onboard a brand, personalize future MMM analysis, prepare their data, or replace demo context.
---

# Sommmelier onboarding

Interview the user in short stages, adapting explanations to their marketing-analytics experience. Preserve existing context unless they explicitly choose to start fresh.

## 1. Inspect existing context

Read `context/*.md`, excluding `README.md`. If files exist, summarize what is known and ask whether to continue or replace it. Check `Last updated:` dates and flag context older than 90 days.

## 2. Capture the operating context

Ask only a few related questions at a time and record:

1. Experience: beginner, intermediate, or advanced.
2. Brand: name, industry, business model, products, target customer, and seasonality.
3. Channels: platforms, approximate monthly budgets, execution metrics, and recent starts/stops.
4. Goals: primary KPI, measurement source, target CPA/ROAS, total budget, and fixed constraints.
5. Data: sources, attribution method, CSV readiness, known quality issues, number of weeks/geos, controls, and channel granularity.
6. Evidence: incrementality tests, platform-reported performance, and prior beliefs with provenance.

Explain jargon for beginners; be direct and statistical with advanced users.

## 3. Write durable context

Use today’s date and maintain these files:

- `context/brand-profile.md`: experience, company, customer, seasonality.
- `context/channels.md`: channel, budget, impressions or reach/frequency availability, notes.
- `context/goals-and-kpis.md`: KPI, measurement, targets, budget, constraints.
- `context/data-sources.md`: sources, attribution, format, known issues, readiness.
- `context/calibration-rationale.md`: experiments, platform figures, beliefs, and calibration decisions.

Do not record secrets or raw client data in context files.

## 4. Assess data readiness

If the user supplies a file, run:

```bash
uv run --frozen sommmelier validate <path>
```

Assess both validity and modeling signal: 26 exact weekly periods minimum (52+ preferred), complete geo/time grid, non-negative values, population, execution data for every paid channel, useful spend variation, controls, and sensible channel granularity.

If no file exists, point to `data/examples/meridian_sample.csv` and explain the required weekly schema. Do not fabricate client observations.

## 5. Handle calibration safely

Create `data/calibration.json` only from evidence the user has supplied or as an explicitly empty template. Every entry must declare one compatible metric:

- Use `monetary_roi` only with revenue or revenue-per-KPI data.
- Otherwise use `incremental_kpi_per_currency`.
- Never mix metric units in one calibration file.

Document each prior’s source and reasoning in `context/calibration-rationale.md`. Remember that `data/calibration.json` is intentionally gitignored.

## 6. Handoff

Summarize what was captured, what is missing, whether the data validates, and the next concrete step. Recommend the `sommmelier-analysis` workflow when data is ready or `sommmelier-walkthrough` for a guided sample.
