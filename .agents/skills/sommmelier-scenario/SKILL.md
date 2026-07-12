---
name: sommmelier-scenario
description: Explore Sommmelier budget reallocations and what-if scenarios from fitted Meridian optimizer output and response curves. Use whenever a user asks to move, cut, add, or optimize channel budget, compare spend scenarios, or estimate outcome changes.
---

# Sommmelier scenario planning

Use the latest decision-ready result and brand constraints to explore budget changes without overstating precision.

## 1. Establish safety and current state

Read the latest `outputs/full_results_*.json` plus relevant `context/*.md`. Require a complete technical run and passed model quality before advising a reallocation. If not decision-ready, explain the failed gate and stop at diagnostic exploration.

Present current spend, ROI or KPI efficiency, marginal efficiency, contribution, and uncertainty by channel. Use monetary language only when `metadata.roi_is_monetary` is true.

## 2. Use fitted scenarios first

Present populated `optimization` scenarios for reduced, current, and increased total budget. Show total budget, per-channel allocation and change, and expected outcome. Respect fixed spends, caps, contracts, and channel minimums from context.

Never derive an “optimal” allocation by ranking average ROI. If optimizer output is absent, state that optimization is unavailable.

## 3. Handle custom scenarios

For a custom change within the modeled range, interpolate the fitted `response_curves` and show:

- Current and proposed spend by affected channel.
- Estimated response loss and gain.
- Net expected change.
- Uncertainty based on affected channel intervals.
- Whether the proposal goes beyond observed support or the curve’s range.

Label interpolation as a projection, not a guarantee. Do not extrapolate beyond available curves without a prominent warning and no false precision.

## 4. Recommend a test

Give a clear proposed reallocation only when supported, explain confidence, and recommend an incremental 2–4 week test with success and rollback criteria. Call out any conflict with brand constraints.
