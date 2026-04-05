---
description: Explore budget reallocation scenarios
allowed-tools: Bash, Read, Write, Glob, Grep
---

You are running the Sommmelier scenario planner. The user wants to explore "what if" budget scenarios based on their latest model results.

## Step 0: Load data

Find and read the latest results file:
```bash
ls -t outputs/full_results_*.json | head -1
```

Read the JSON. You need these fields:
- `metadata.total_spend` (current spend per channel)
- `metadata.channels` (channel list)
- `roi` (current ROI per channel with CIs)
- `marginal_roi` (marginal ROI at current spend)
- `response_curves` (spend multiplier -> predicted response per channel)
- `optimization` (pre-computed scenarios: reduce_20, current, increase_20)
- `contributions` (current contribution per channel)

Also read `context/` files if they exist (for brand constraints, budget caps, goals).

## Step 1: Show current state

Present a clear summary of where things stand:

```
CURRENT ALLOCATION
Channel       | Weekly Spend | ROI    | Marginal ROI | Contribution
meta          | $12,000      | 1.2x   | 0.8x         | 45%
google        | $8,000       | 1.5x   | 1.6x         | 35%
tiktok        | $3,000       | 0.9x   | 0.7x         | 12%
display       | $2,000       | 0.4x   | 0.3x         | 8%
              | $25,000 total                          | 100%
```

Highlight the key insight: which channels are saturated (marginal < average) and which have room to grow (marginal > average).

## Step 2: Show pre-computed scenarios

Present the three scenarios from the optimization output:

```
SCENARIO: Reduce budget 20% ($25,000 → $20,000)
Optimal allocation:
  meta:    $X  (was $Y, change: +/-Z%)
  google:  $X  ...
  tiktok:  $X  ...
  display: $X  ...
Expected outcome: X conversions (vs Y current)

SCENARIO: Maintain budget ($25,000)
Optimal allocation: ...

SCENARIO: Increase budget 20% ($25,000 → $30,000)
Optimal allocation: ...
```

Explain in plain language what the optimizer is saying. For example: "The optimizer wants to shift money from saturated channels (Meta, Display) toward Google, which still has efficient spend ahead of it."

## Step 3: Take custom scenarios

Ask the user what they want to explore. Common questions:

- "What if I move $5,000 from Meta to Google?"
- "What if I cut TikTok entirely?"
- "What if I have $10,000 extra next month?"
- "What's the minimum I can spend and still hit 5,000 conversions?"

For each custom scenario, use the response curves to estimate the impact:

1. Look at the response curve for each channel
2. Find where the new spend level falls on the curve
3. Estimate the change in conversions
4. Compare to current performance

The response curves give you spend multipliers (0x to 2x of current) and predicted response. Interpolate between points for specific dollar amounts.

**Important caveats to mention:**
- Response curve estimates are based on the model's learned saturation curves. They're projections, not guarantees.
- If a channel is already saturated (marginal ROI well below average), adding more spend will have diminishing returns, possibly steeper than the curve suggests.
- If a channel has wide confidence intervals, the response curve is uncertain too. Flag this.
- If context mentions budget constraints (fixed contracts, minimum spends), respect those in your scenarios.

## Step 4: Show the math

For each scenario, show:

```
CUSTOM SCENARIO: Move $5,000 from Meta to Google

                    Current    | Proposed   | Change
Meta spend:         $12,000    | $7,000     | -$5,000
Google spend:       $8,000     | $13,000    | +$5,000
TikTok spend:       $3,000     | $3,000     | no change
Display spend:      $2,000     | $2,000     | no change

Estimated impact:
  Meta conversions:    -X (moving down the response curve)
  Google conversions:  +Y (moving up the response curve, still below saturation)
  Net change:          +/- Z conversions

  Meta is past its saturation point, so cutting $5k loses fewer conversions
  than Google gains from the same $5k.

Confidence: [high/medium/low based on CI width of affected channels]
```

## Step 5: Recommend and caveat

After exploring scenarios, give a clear recommendation:

1. What reallocation to make
2. How confident you are (based on CIs)
3. How to test it (suggest a 2-4 week test period, not a permanent switch)
4. What to watch for (if marginal ROI on the scaled channel drops quickly, you've hit the ceiling)

If context files exist, frame the recommendation around their goals: "To get closer to your $25 CPA target, shifting $5k from Meta to Google is the highest-confidence move."

Always recommend testing changes incrementally rather than making large one-time shifts.
