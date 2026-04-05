# Context Files

This folder stores brand-specific knowledge that makes Sommmelier's analysis tailored to your business. These files are created during onboarding (`/init`) and updated after each model run (`/sommmelier`).

You can edit any of these files directly -- they're plain markdown.

## Files Created by `/init`

| File | What it captures |
|------|-----------------|
| `brand-profile.md` | Company overview, industry, business model, experience level, seasonality |
| `channels.md` | Media channels you run, platforms, approximate budget ranges |
| `goals-and-kpis.md` | Primary KPI, target CPA/ROAS, budget constraints, marketing goals |
| `data-sources.md` | Where your data comes from, format, known issues, data readiness assessment |
| `calibration-rationale.md` | Why calibration priors are set the way they are -- experiment results, platform data, beliefs |

## Files Created by `/sommmelier` (after model runs)

| File | What it captures |
|------|-----------------|
| `model-learnings.md` | Key findings from each model run, ROI trends, what changed and why |
| `improvement-backlog.md` | Prioritized suggestions for improving model accuracy, tracked over time |

## How They're Used

Every time you run `/sommmelier`, Claude reads these files to:
- Reference your specific goals and constraints in recommendations
- Adapt explanation depth to your experience level
- Flag when suggestions conflict with your stated constraints
- Track what's working and what needs attention across runs

## Getting Started

Run `/init` in Claude Code to populate these files through a guided conversation.
