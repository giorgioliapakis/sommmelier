"""
Model Improvement Advisor for Sommmelier.

Analyzes model diagnostics and recommends specific data or configurations
that would improve model quality. This is the "self-improving" component
that guides brands on what to collect next.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ImprovementQuestion:
    """A question or data request to improve the model."""
    category: str  # "calibration", "controls", "data_quality", "configuration", "experiments"
    priority: str  # "high", "medium", "low"
    question: str  # The question to ask
    why_it_helps: str  # Explanation of why this improves the model
    example: Optional[str] = None  # Example of what good data looks like
    impact_estimate: Optional[str] = None  # How much it might tighten estimates


def analyze_confidence_intervals(results: dict) -> list[ImprovementQuestion]:
    """Generate questions for channels with wide confidence intervals."""
    questions = []
    roi_data = results.get("roi", {})

    for channel, data in roi_data.items():
        if not isinstance(data, dict):
            continue

        ci_lo = data.get("ci_lower", 0)
        ci_hi = data.get("ci_upper", 0)
        mean = data.get("mean", 0)

        if mean <= 0:
            continue

        ci_width_ratio = (ci_hi - ci_lo) / mean

        # Very wide CI (more than 200% of mean)
        if ci_width_ratio > 2.0:
            questions.append(ImprovementQuestion(
                category="experiments",
                priority="high",
                question=f"Have you run any incrementality tests or geo-lift experiments for {channel}?",
                why_it_helps=f"The model is very uncertain about {channel}'s true ROI (range: {ci_lo:.1f}x to {ci_hi:.1f}x). "
                            "Experiment data would dramatically narrow this range by providing a 'ground truth' to calibrate against.",
                example=f"Example: 'We ran a 4-week holdout test in California where we paused {channel}. "
                        "Conversions dropped 12% compared to control regions.'",
                impact_estimate="Could reduce uncertainty by 50-80%"
            ))

            questions.append(ImprovementQuestion(
                category="calibration",
                priority="medium",
                question=f"What does {channel}'s platform (e.g., Meta/Google) report as attributed conversions?",
                why_it_helps="Platform-reported conversions serve as a reference point. While they're biased "
                            "(platforms overcount), the model can use them as a soft upper bound on channel impact.",
                example="Example: 'Meta reports 5,000 conversions last month for this campaign.'",
                impact_estimate="Helps set reasonable priors, reducing extreme estimates"
            ))

        # Moderately wide CI (100-200% of mean)
        elif ci_width_ratio > 1.0:
            questions.append(ImprovementQuestion(
                category="data_quality",
                priority="medium",
                question=f"Does {channel} spend vary naturally week-to-week, or is it relatively constant?",
                why_it_helps="The model learns from spend variation. If spend is flat, the model can't "
                            "distinguish this channel's effect from baseline trends.",
                example="Ideal: Spend varies by 30%+ week-to-week. Problematic: Same spend every week.",
                impact_estimate="Natural variation improves signal; if flat, consider running a deliberate test"
            ))

    return questions


def analyze_model_fit(results: dict) -> list[ImprovementQuestion]:
    """Generate questions based on model fit metrics."""
    questions = []
    model_fit = results.get("model_fit", {})

    r_squared = model_fit.get("r_squared", 0)
    mape = model_fit.get("mape", 0)

    # Poor R-squared (model doesn't explain much variance)
    if r_squared < 0.5:
        questions.append(ImprovementQuestion(
            category="controls",
            priority="high",
            question="What major events or factors affected your conversions beyond media spend?",
            why_it_helps=f"The model only explains {r_squared*100:.0f}% of conversion variation. "
                        "This means 'something else' is driving {100-r_squared*100:.0f}% of your conversions. "
                        "Adding control variables for these factors improves accuracy.",
            example="Examples:\n"
                    "- Promotions/sales events (dates and discount %)\n"
                    "- Pricing changes\n"
                    "- Seasonality patterns (holidays, back-to-school)\n"
                    "- PR/viral moments\n"
                    "- Product launches\n"
                    "- Competitor activity",
            impact_estimate="Good controls can improve R² by 10-30 percentage points"
        ))

    elif r_squared < 0.7:
        questions.append(ImprovementQuestion(
            category="controls",
            priority="medium",
            question="Do you have data on promotions, seasonality events, or other non-media factors?",
            why_it_helps=f"Model explains {r_squared*100:.0f}% of variance - decent but improvable. "
                        "Control variables help the model separate media effects from other business drivers.",
            example="A simple CSV with columns: date, had_promotion (0/1), promo_discount_pct, is_holiday (0/1)",
            impact_estimate="Could improve R² to 0.8+"
        ))

    # High MAPE (prediction errors)
    if mape > 0.25:
        questions.append(ImprovementQuestion(
            category="data_quality",
            priority="high",
            question="Are there weeks with unusual spikes or drops in conversions that had special causes?",
            why_it_helps=f"The model's predictions are off by {mape*100:.0f}% on average. "
                        "This often means there are outlier weeks with unexplained spikes/drops. "
                        "Identifying these helps the model fit better.",
            example="Look for weeks where actuals differed from model predictions by >30%. "
                    "Were there outages, PR events, or data issues?",
            impact_estimate="Addressing outliers can reduce MAPE by 5-10 points"
        ))

    return questions


def analyze_channel_structure(results: dict) -> list[ImprovementQuestion]:
    """Generate questions about channel granularity and structure."""
    questions = []
    channels = results.get("metadata", {}).get("channels", [])
    roi_data = results.get("roi", {})

    # Check for aggregated channel names that could be split
    aggregated_names = ["paid_social", "social", "display", "programmatic", "paid_media", "digital"]

    for channel in channels:
        if any(agg in channel.lower() for agg in aggregated_names):
            questions.append(ImprovementQuestion(
                category="configuration",
                priority="medium",
                question=f"Can you break down '{channel}' into individual platforms?",
                why_it_helps=f"'{channel}' likely contains multiple platforms (e.g., Meta + TikTok + Pinterest). "
                            "These have very different effectiveness. Aggregating them hides which actually works.",
                example=f"Instead of '{channel}_spend', provide: meta_spend, tiktok_spend, pinterest_spend",
                impact_estimate="Enables optimization at platform level, often reveals hidden winners/losers"
            ))

    # Check for suspiciously high ROI (might indicate attribution overlap)
    for channel, data in roi_data.items():
        mean_roi = data.get("mean", data) if isinstance(data, dict) else data
        if mean_roi > 50:
            questions.append(ImprovementQuestion(
                category="calibration",
                priority="high",
                question=f"Is the {mean_roi:.0f}x ROI for {channel} plausible based on your business knowledge?",
                why_it_helps=f"{channel} shows unusually high ROI. This could be real (brand search often has 20x+), "
                            "or could indicate: correlation with another factor, data issues, or conversion attribution overlap.",
                example="Questions to consider:\n"
                        f"- Is {channel} capturing brand demand vs. creating it?\n"
                        f"- Could {channel} conversions be double-counted somewhere?\n"
                        f"- Does {channel} scale up when other channels drive awareness?",
                impact_estimate="Understanding this helps set appropriate priors and interpret results"
            ))

    return questions


def analyze_data_completeness(results: dict, has_impressions: bool = False) -> list[ImprovementQuestion]:
    """Generate questions about missing data that would help."""
    questions = []

    if not has_impressions:
        questions.append(ImprovementQuestion(
            category="data_quality",
            priority="low",
            question="Can you provide impression data alongside spend for each channel?",
            why_it_helps="The model currently estimates impressions from spend (assuming $10 CPM). "
                        "Actual impressions allow the model to separate 'reach' effects from 'frequency' effects.",
            example="Add columns like: meta_impressions, google_impressions alongside spend columns",
            impact_estimate="Enables reach/frequency analysis and more accurate saturation curves"
        ))

    # Always suggest this if not already using
    questions.append(ImprovementQuestion(
        category="calibration",
        priority="medium",
        question="Do you have any prior beliefs or benchmarks about expected ROI for each channel?",
        why_it_helps="The model uses generic priors (expecting ~1x ROI). If you know from industry benchmarks "
                    "or past experience that 'Meta usually returns 1.5x for us', this information helps.",
        example="Example: 'Based on our experience, we expect: Meta 1.2-1.8x, Google Search 2-4x, Display 0.5-1x'",
        impact_estimate="Informative priors significantly tighten confidence intervals"
    ))

    return questions


def analyze_time_dynamics(results: dict) -> list[ImprovementQuestion]:
    """Generate questions about time-based effects."""
    questions = []
    metadata = results.get("metadata", {})
    n_periods = metadata.get("n_time_periods", 0)

    if n_periods < 52:
        questions.append(ImprovementQuestion(
            category="data_quality",
            priority="high",
            question=f"Can you provide more historical data? Currently have {n_periods} weeks.",
            why_it_helps="With <52 weeks, the model can't reliably separate seasonal patterns from media effects. "
                        "It also has less 'natural experiments' to learn from.",
            example="Ideal: 104 weeks (2 years) to capture full seasonal cycle",
            impact_estimate="Each additional quarter of data typically improves estimates by 10-20%"
        ))

    # Ask about conversion lag
    questions.append(ImprovementQuestion(
        category="configuration",
        priority="medium",
        question="What's the typical time from ad exposure to conversion for your product?",
        why_it_helps="The model uses a default 8-week decay. If your product has a 2-day purchase cycle (impulse) "
                    "or 12-week cycle (B2B), the model should be configured accordingly.",
        example="Examples:\n"
                "- E-commerce impulse buy: 1-3 days\n"
                "- Subscription SaaS: 2-4 weeks\n"
                "- B2B enterprise: 8-16 weeks",
        impact_estimate="Correct decay settings improve attribution accuracy by 10-30%"
    ))

    return questions


def analyze_spend_concentration(results: dict) -> list[ImprovementQuestion]:
    """Flag channels that dominate the budget, making other channels hard to measure."""
    questions = []
    metadata = results.get("metadata", {})
    total_spend = metadata.get("total_spend", {})

    if not total_spend:
        return questions

    grand_total = sum(total_spend.values())
    if grand_total == 0:
        return questions

    for channel, spend in total_spend.items():
        share = spend / grand_total
        if share > 0.7:
            questions.append(ImprovementQuestion(
                category="data_quality",
                priority="medium",
                question=f"{channel} accounts for {share*100:.0f}% of your total spend. Can you get more variation in the other channels?",
                why_it_helps=f"When one channel dominates the budget, the model has very little signal to measure the smaller channels. "
                            f"The confidence intervals on everything except {channel} will be wide because there's not much spend variation to learn from.",
                example="Options: (1) run a deliberate test where you increase a smaller channel's budget for 4-6 weeks, "
                        "or (2) find historical periods where spend was more balanced.",
                impact_estimate="More balanced spend gives the model better signal for all channels"
            ))

    return questions


def analyze_brand_search_inflation(results: dict) -> list[ImprovementQuestion]:
    """Flag brand/search channels that may have inflated ROI from capturing existing demand."""
    questions = []
    roi_data = results.get("roi", {})
    channels = results.get("metadata", {}).get("channels", [])

    brand_keywords = ["brand", "search", "sem", "google_brand", "branded"]

    for channel in channels:
        if not any(kw in channel.lower() for kw in brand_keywords):
            continue

        data = roi_data.get(channel, {})
        roi = data.get("mean", data) if isinstance(data, dict) else data

        if roi > 2.0:
            questions.append(ImprovementQuestion(
                category="configuration",
                priority="high",
                question=f"{channel} shows {roi:.1f}x ROI. Is this channel capturing demand that other channels created?",
                why_it_helps="Brand search and branded keywords often look extremely profitable in MMM because they capture "
                            "people who were already going to convert. Someone sees a Meta ad, decides to buy, googles your "
                            "brand name, and brand search gets the credit. The ROI is real spend-wise, but scaling brand search "
                            "won't create new demand the way upper-funnel channels do.",
                example="Questions to consider:\n"
                        f"- If you turned off all other channels, would {channel} conversions drop?\n"
                        f"- Does {channel} volume correlate with spend on other channels?\n"
                        "- Can you separate brand and non-brand search in your data?",
                impact_estimate="Separating brand from non-brand search often reveals the true ROI picture"
            ))

    return questions


def analyze_organic_baseline(results: dict) -> list[ImprovementQuestion]:
    """Surface what share of conversions the model attributes to organic/baseline."""
    questions = []
    contributions = results.get("contributions", {})
    metadata = results.get("metadata", {})

    if not contributions:
        return questions

    total_kpi = metadata.get("total_kpi", 0)
    media_total = sum(
        c.get("absolute", 0) if isinstance(c, dict) else c
        for c in contributions.values()
    )

    if total_kpi <= 0:
        return questions

    media_pct = (media_total / total_kpi) * 100
    organic_pct = 100 - media_pct

    if organic_pct > 80:
        questions.append(ImprovementQuestion(
            category="data_quality",
            priority="medium",
            question=f"The model attributes {organic_pct:.0f}% of conversions to organic/baseline (not driven by media). Does that match your intuition?",
            why_it_helps="A very high organic baseline means the model thinks most of your conversions would happen even without ads. "
                        "This could be correct (strong brand, organic traffic), but it could also mean the model is underestimating "
                        "media impact because it lacks the right data to detect it.",
            example="Things that help: more time periods (so the model sees more variation), "
                    "control variables (so organic factors are accounted for), "
                    "and calibration priors (to anchor the model's expectations).",
            impact_estimate="If organic baseline seems too high, calibration data can correct it"
        ))
    elif organic_pct < 20:
        questions.append(ImprovementQuestion(
            category="calibration",
            priority="medium",
            question=f"The model attributes only {organic_pct:.0f}% of conversions to organic/baseline. Are you confident media drives {media_pct:.0f}% of your conversions?",
            why_it_helps="A very low organic baseline means the model thinks almost all conversions are media-driven. "
                        "This is unusual and could indicate the model is overcounting media impact. "
                        "Most businesses have a meaningful organic baseline from brand awareness, word of mouth, and direct traffic.",
            example="If you paused all ads for a week, would conversions really drop by {:.0f}%? If not, the model may need informative priors.".format(media_pct),
            impact_estimate="Calibration priors help anchor the organic baseline at a realistic level"
        ))

    return questions


def analyze_adstock_plausibility(results: dict) -> list[ImprovementQuestion]:
    """Check if adstock decay estimates are plausible for the channel type."""
    questions = []
    adstock = results.get("adstock_decay", {})

    if not adstock:
        return questions

    for channel, data in adstock.items():
        decay_mean = data.get("mean", None) if isinstance(data, dict) else None
        if decay_mean is None:
            continue

        # Very fast decay (< 0.05) means ad effect disappears within a week
        if decay_mean < 0.05:
            questions.append(ImprovementQuestion(
                category="configuration",
                priority="medium",
                question=f"The model estimates {channel}'s ad effect decays almost instantly (decay rate: {decay_mean:.3f}). Is that realistic for your product?",
                why_it_helps="A near-zero decay means the model thinks ads only affect conversions in the same week they run. "
                            "This is plausible for impulse purchases (food delivery, fast fashion) but unrealistic for products "
                            "with longer consideration periods (B2B, real estate, luxury goods).",
                example="If your typical customer takes 2-4 weeks from seeing an ad to converting, "
                        "this decay rate is too fast and may be underestimating the channel's true impact.",
                impact_estimate="Correct adstock settings can shift ROI estimates by 10-30%"
            ))

        # Very slow decay (> 0.5) means ad effects last months
        elif decay_mean > 0.5:
            questions.append(ImprovementQuestion(
                category="configuration",
                priority="low",
                question=f"The model estimates {channel}'s ad effect persists for a long time (decay rate: {decay_mean:.3f}). Does your product have a long purchase cycle?",
                why_it_helps="A high decay rate means the model thinks ads have effects that carry over many weeks. "
                            "This is plausible for high-consideration products (cars, enterprise software) but may indicate "
                            "overfitting if your product is a quick purchase.",
                example="Quick check: if you paused this channel for a month, would you still see effects 2-3 months later? "
                        "If not, the model may be attributing unrelated trends to media carry-over.",
                impact_estimate="Usually a minor issue, but worth verifying for accuracy"
            ))

    return questions


def analyze_geographic_signal(results: dict) -> list[ImprovementQuestion]:
    """Check if the geographic structure provides enough signal."""
    questions = []
    metadata = results.get("metadata", {})
    n_geos = metadata.get("n_geos", 0)
    channels = metadata.get("channels", [])

    if n_geos == 1:
        questions.append(ImprovementQuestion(
            category="data_quality",
            priority="high",
            question="You only have 1 geography. Can you break your data into regions, states, or markets?",
            why_it_helps="MMMs learn by comparing what happens in different places when spend varies. "
                        "With only one geo, the model can only learn from week-to-week variation in spend. "
                        "Multiple geos give the model much more signal because spend naturally varies across regions.",
            example="Common splits: US states, countries, DMAs (designated market areas), or metro areas. "
                    "Even a rough split (East/Central/West, or top 5 states by revenue) helps significantly.",
            impact_estimate="Going from 1 geo to 5+ geos can cut confidence interval width in half"
        ))
    elif n_geos == 2:
        questions.append(ImprovementQuestion(
            category="data_quality",
            priority="medium",
            question="You have 2 geos. Can you split your data into more regions?",
            why_it_helps="Two geos is better than one, but the model still has limited cross-sectional variation. "
                        "With more geos, the model can better separate media effects from local factors.",
            example="If you have state-level or regional data, even 5-10 geos makes a noticeable difference.",
            impact_estimate="More geos improve statistical power for all channel estimates"
        ))

    if n_geos > 1 and n_geos < 5 and len(channels) > 3:
        questions.append(ImprovementQuestion(
            category="data_quality",
            priority="medium",
            question=f"You have {len(channels)} channels but only {n_geos} geos. Can you add more geographic granularity?",
            why_it_helps=f"With {len(channels)} channels and only {n_geos} geos, the model has to estimate many parameters "
                        "from limited cross-sectional variation. The rule of thumb: you want at least as many geos as channels "
                        "for reliable estimates.",
            example="If your data supports it, split into more regions to give the model more signal per channel.",
            impact_estimate="Better geo coverage directly improves per-channel estimate quality"
        ))

    return questions


def analyze_prior_posterior_divergence(results: dict) -> list[ImprovementQuestion]:
    """Flag channels where the posterior moved far from any provided prior."""
    questions = []
    roi_data = results.get("roi", {})

    # This checks for extreme posterior values that suggest the model disagrees
    # with typical platform expectations (a proxy for prior-posterior divergence
    # when we don't have the actual prior values in the results)
    for channel, data in roi_data.items():
        if not isinstance(data, dict):
            continue

        mean = data.get("mean", 0)
        ci_lo = data.get("ci_lower", 0)
        ci_hi = data.get("ci_upper", 0)

        # ROI very close to zero despite likely having some effect
        if 0 < mean < 0.1 and ci_hi > 0.5:
            questions.append(ImprovementQuestion(
                category="calibration",
                priority="medium",
                question=f"The model estimates near-zero ROI for {channel} ({mean:.2f}x), but the upper CI goes to {ci_hi:.1f}x. Should you re-examine priors?",
                why_it_helps="A near-zero point estimate with a wide interval usually means the model doesn't have enough "
                            "signal to distinguish this channel's effect from noise. If you know from platform data or "
                            "experiments that this channel does have some effect, providing calibration priors would help.",
                example=f"Check: what does the ad platform report for {channel}? Even if overcounted, "
                        "it sets a useful ceiling. Add it as platform_conversions in calibration.json.",
                impact_estimate="Calibration data typically moves these 'near-zero' estimates to more realistic levels"
            ))

    return questions


def generate_improvement_questions(results: dict, has_impressions: bool = False) -> list[ImprovementQuestion]:
    """
    Main entry point: analyze model results and generate improvement questions.

    Returns questions sorted by priority, with the most impactful improvements first.
    """
    all_questions = []

    # Gather questions from all analyzers
    all_questions.extend(analyze_confidence_intervals(results))
    all_questions.extend(analyze_model_fit(results))
    all_questions.extend(analyze_channel_structure(results))
    all_questions.extend(analyze_data_completeness(results, has_impressions))
    all_questions.extend(analyze_time_dynamics(results))
    all_questions.extend(analyze_spend_concentration(results))
    all_questions.extend(analyze_brand_search_inflation(results))
    all_questions.extend(analyze_organic_baseline(results))
    all_questions.extend(analyze_adstock_plausibility(results))
    all_questions.extend(analyze_geographic_signal(results))
    all_questions.extend(analyze_prior_posterior_divergence(results))

    # Deduplicate by question text (some analyzers might ask similar things)
    seen_questions = set()
    unique_questions = []
    for q in all_questions:
        if q.question not in seen_questions:
            seen_questions.add(q.question)
            unique_questions.append(q)

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    unique_questions.sort(key=lambda x: priority_order.get(x.priority, 99))

    return unique_questions


def format_questions_for_user(questions: list[ImprovementQuestion], max_questions: int = 5) -> str:
    """Format the improvement questions as user-friendly text."""
    lines = []
    lines.append("=" * 60)
    lines.append("HOW TO IMPROVE YOUR MODEL")
    lines.append("=" * 60)
    lines.append("\nBased on your results, here's what would help most:\n")

    # Group by priority
    high = [q for q in questions if q.priority == "high"][:max_questions]
    medium = [q for q in questions if q.priority == "medium"][:max(0, max_questions - len(high))]

    shown = high + medium

    for i, q in enumerate(shown, 1):
        priority_marker = {"high": "[HIGH IMPACT]", "medium": "[MEDIUM IMPACT]", "low": "[NICE TO HAVE]"}.get(q.priority, "")

        lines.append(f"\n{i}. {priority_marker} {q.question}")
        lines.append(f"   Category: {q.category}")
        lines.append(f"\n   Why this helps:")
        lines.append(f"   {q.why_it_helps}")

        if q.example:
            lines.append(f"\n   {q.example}")

        if q.impact_estimate:
            lines.append(f"\n   Expected impact: {q.impact_estimate}")

        lines.append("")

    remaining = len(questions) - len(shown)
    if remaining > 0:
        lines.append(f"\n({remaining} additional suggestions available)")

    lines.append("\n" + "=" * 60)
    lines.append("Provide any of this data and re-run the model to see improved results.")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_questions_as_checklist(questions: list[ImprovementQuestion]) -> str:
    """Format as a simple checklist for quick scanning."""
    lines = []
    lines.append("## Model Improvement Checklist\n")

    by_category = {}
    for q in questions:
        if q.category not in by_category:
            by_category[q.category] = []
        by_category[q.category].append(q)

    category_names = {
        "experiments": "Run Experiments",
        "calibration": "Provide Calibration Data",
        "controls": "Add Control Variables",
        "data_quality": "Improve Data Quality",
        "configuration": "Adjust Model Configuration"
    }

    for cat, cat_questions in by_category.items():
        lines.append(f"### {category_names.get(cat, cat.title())}")
        for q in cat_questions:
            priority_marker = {"high": "(!) ", "medium": "(*) ", "low": "(-) "}.get(q.priority, "")
            lines.append(f"- [ ] {priority_marker}{q.question}")
        lines.append("")

    return "\n".join(lines)
