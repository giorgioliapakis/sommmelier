"""
Calibration data support for Sommmelier.

Allows users to provide experiment results, platform conversions,
and prior beliefs to improve model accuracy.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentResult:
    """
    Results from an incrementality experiment (geo-lift, holdout, etc.)

    Example:
        ExperimentResult(
            channel="meta",
            experiment_type="geo_lift",
            lift_estimate=0.12,  # 12% lift
            lift_ci_lower=0.08,
            lift_ci_upper=0.16,
            test_period_weeks=4,
            test_spend=50000,
            notes="Ran in CA, OR, WA vs control in TX, FL, GA"
        )
    """
    channel: str
    experiment_type: str  # "geo_lift", "holdout", "synthetic_control", "rct"
    lift_estimate: float  # Incremental lift as decimal (0.12 = 12%)
    lift_ci_lower: Optional[float] = None
    lift_ci_upper: Optional[float] = None
    test_period_weeks: Optional[int] = None
    test_spend: Optional[float] = None
    test_conversions: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class PlatformConversions:
    """
    Platform-reported conversion data for calibration.

    Platform data is biased (over-attributed) but useful as a soft upper bound.

    Example:
        PlatformConversions(
            channel="meta",
            platform_conversions=5000,
            period_weeks=4,
            attribution_window="7d_click_1d_view",
            spend=25000,
            notes="From Meta Ads Manager, last 28 days"
        )
    """
    channel: str
    platform_conversions: float
    period_weeks: int
    spend: float
    attribution_window: Optional[str] = None  # e.g., "7d_click_1d_view"
    notes: Optional[str] = None


@dataclass
class PriorBelief:
    """
    User's prior belief about expected channel performance.

    Example:
        PriorBelief(
            channel="google_search",
            expected_roi_low=1.5,
            expected_roi_high=3.0,
            confidence="high",
            source="Historical data from 2023"
        )
    """
    channel: str
    expected_roi_low: float
    expected_roi_high: float
    confidence: str = "medium"  # "high", "medium", "low"
    source: Optional[str] = None  # Where this belief comes from


@dataclass
class CalibrationData:
    """
    Complete calibration data for a model run.

    Collect all available calibration information in one place.
    """
    experiments: list[ExperimentResult] = field(default_factory=list)
    platform_conversions: list[PlatformConversions] = field(default_factory=list)
    prior_beliefs: list[PriorBelief] = field(default_factory=list)
    control_variables: dict = field(default_factory=dict)  # {column_name: description}
    notes: str = ""


def experiment_to_prior(exp: ExperimentResult) -> dict:
    """
    Convert experiment result to Meridian prior parameters.

    Returns dict with roi_mean (log-space) and roi_sigma for LogNormal distribution.
    LogNormal(roi_mean, roi_sigma) means the ROI median = exp(roi_mean).
    """
    import math

    # Calculate implied ROI = incremental_conversions / spend
    if exp.test_conversions and exp.test_spend and exp.test_spend > 0:
        # test_conversions is TOTAL conversions during test period
        # lift_estimate * total = incremental conversions
        incremental = exp.lift_estimate * exp.test_conversions
        implied_roi = incremental / exp.test_spend
    elif exp.test_spend and exp.test_spend > 0:
        # No conversion count — use lift as rough proxy
        implied_roi = exp.lift_estimate
    else:
        implied_roi = exp.lift_estimate

    # Clamp to avoid log(0)
    implied_roi = max(implied_roi, 1e-8)

    # Convert to log-space for LogNormal parameterization
    log_roi = math.log(implied_roi)

    # Calculate sigma from CI if available
    if (exp.lift_ci_lower is not None and exp.lift_ci_upper is not None
            and exp.test_conversions and exp.test_spend and exp.test_spend > 0):
        roi_lower = max(exp.lift_ci_lower * exp.test_conversions / exp.test_spend, 1e-8)
        roi_upper = max(exp.lift_ci_upper * exp.test_conversions / exp.test_spend, 1e-8)
        # 90% CI spans ~3.29 sigma in log-space
        log_range = math.log(roi_upper) - math.log(roi_lower)
        sigma = max(log_range / 3.29, 0.5)
    else:
        sigma = 1.0  # Default wide uncertainty

    # Floor sigma to prevent priors from dominating the posterior
    sigma = max(sigma, 1.5)

    return {
        "channel": exp.channel,
        "roi_mean": log_roi,
        "roi_sigma": sigma,
        "source": f"experiment:{exp.experiment_type}"
    }


def platform_data_to_upper_bound(platform: PlatformConversions) -> dict:
    """
    Convert platform data to a soft upper bound on ROI.

    Platform-reported conversions are typically 2-5x over-attributed.
    """
    platform_roi = platform.platform_conversions / platform.spend if platform.spend > 0 else 0

    # Platform data is an upper bound - true ROI is likely 30-70% of this
    return {
        "channel": platform.channel,
        "roi_upper_bound": platform_roi,
        "suggested_roi_range": (platform_roi * 0.3, platform_roi * 0.7),
        "source": f"platform:{platform.attribution_window or 'default'}"
    }


def belief_to_prior(belief: PriorBelief) -> dict:
    """
    Convert user belief to Meridian prior parameters (log-space).
    """
    import math

    # Convert ROI range to log-space
    roi_mid = (belief.expected_roi_low + belief.expected_roi_high) / 2
    log_roi = math.log(max(roi_mid, 1e-8))

    # Sigma from range width in log-space
    log_range = math.log(max(belief.expected_roi_high, 1e-8)) - math.log(max(belief.expected_roi_low, 1e-8))

    # Adjust sigma based on confidence
    confidence_multiplier = {"high": 0.5, "medium": 1.0, "low": 2.0}.get(belief.confidence, 1.0)
    sigma = max((log_range / 3.29) * confidence_multiplier, 1.5)

    return {
        "channel": belief.channel,
        "roi_mean": log_roi,
        "roi_sigma": sigma,
        "source": f"belief:{belief.source or 'user_input'}"
    }


def calculate_channel_priors(calibration: CalibrationData) -> dict[str, dict]:
    """
    Combine all calibration data to compute per-channel priors.

    Priority: Experiments > Platform data > Prior beliefs > Default
    """
    channel_priors = {}

    # Start with prior beliefs (lowest priority)
    for belief in calibration.prior_beliefs:
        channel_priors[belief.channel] = belief_to_prior(belief)

    # Layer in platform data (medium priority)
    for platform in calibration.platform_conversions:
        bounds = platform_data_to_upper_bound(platform)
        ch = platform.channel

        if ch in channel_priors:
            # Constrain existing prior by upper bound
            current = channel_priors[ch]
            suggested_mean = (bounds["suggested_roi_range"][0] + bounds["suggested_roi_range"][1]) / 2
            # Average with existing belief
            current["roi_mean"] = (current["roi_mean"] + suggested_mean) / 2
            current["platform_upper_bound"] = bounds["roi_upper_bound"]
        else:
            channel_priors[ch] = {
                "channel": ch,
                "roi_mean": bounds["suggested_roi_range"][1],  # Conservative
                "roi_sigma": 0.7,  # Moderate uncertainty
                "platform_upper_bound": bounds["roi_upper_bound"],
                "source": bounds["source"]
            }

    # Layer in experiments (highest priority) — average multiple per channel
    from collections import defaultdict
    import math

    channel_experiments = defaultdict(list)
    for exp in calibration.experiments:
        channel_experiments[exp.channel].append(experiment_to_prior(exp))

    for channel, priors in channel_experiments.items():
        # Average log-space means across experiments
        avg_mean = sum(p["roi_mean"] for p in priors) / len(priors)

        # Combine intra-experiment sigma with inter-experiment variance
        avg_intra_sigma = sum(p["roi_sigma"] for p in priors) / len(priors)
        if len(priors) > 1:
            means = [p["roi_mean"] for p in priors]
            inter_variance = sum((m - avg_mean) ** 2 for m in means) / (len(means) - 1)
            combined_sigma = math.sqrt(avg_intra_sigma ** 2 + inter_variance)
        else:
            combined_sigma = avg_intra_sigma

        # Floor sigma to keep priors from dominating the posterior
        combined_sigma = max(combined_sigma, 1.5)

        channel_priors[channel] = {
            "channel": channel,
            "roi_mean": avg_mean,
            "roi_sigma": combined_sigma,
            "source": f"experiment:averaged({len(priors)})"
        }

    return channel_priors


def save_calibration(calibration: CalibrationData, path: Path | str) -> None:
    """Save calibration data to JSON."""
    path = Path(path)

    data = {
        "experiments": [
            {
                "channel": e.channel,
                "experiment_type": e.experiment_type,
                "lift_estimate": e.lift_estimate,
                "lift_ci_lower": e.lift_ci_lower,
                "lift_ci_upper": e.lift_ci_upper,
                "test_period_weeks": e.test_period_weeks,
                "test_spend": e.test_spend,
                "test_conversions": e.test_conversions,
                "notes": e.notes
            }
            for e in calibration.experiments
        ],
        "platform_conversions": [
            {
                "channel": p.channel,
                "platform_conversions": p.platform_conversions,
                "period_weeks": p.period_weeks,
                "spend": p.spend,
                "attribution_window": p.attribution_window,
                "notes": p.notes
            }
            for p in calibration.platform_conversions
        ],
        "prior_beliefs": [
            {
                "channel": b.channel,
                "expected_roi_low": b.expected_roi_low,
                "expected_roi_high": b.expected_roi_high,
                "confidence": b.confidence,
                "source": b.source
            }
            for b in calibration.prior_beliefs
        ],
        "control_variables": calibration.control_variables,
        "notes": calibration.notes
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(path: Path | str) -> CalibrationData:
    """Load calibration data from JSON."""
    path = Path(path)

    with open(path) as f:
        data = json.load(f)

    return CalibrationData(
        experiments=[
            ExperimentResult(**e) for e in data.get("experiments", [])
        ],
        platform_conversions=[
            PlatformConversions(**p) for p in data.get("platform_conversions", [])
        ],
        prior_beliefs=[
            PriorBelief(**b) for b in data.get("prior_beliefs", [])
        ],
        control_variables=data.get("control_variables", {}),
        notes=data.get("notes", "")
    )


# Template for users to fill in
CALIBRATION_TEMPLATE = """
{
  "experiments": [
    {
      "channel": "meta",
      "experiment_type": "geo_lift",
      "lift_estimate": 0.12,
      "lift_ci_lower": 0.08,
      "lift_ci_upper": 0.16,
      "test_period_weeks": 4,
      "test_spend": 50000,
      "test_conversions": 10000,
      "notes": "Ran in CA, OR, WA vs control in TX, FL, GA"
    }
  ],
  "platform_conversions": [
    {
      "channel": "meta",
      "platform_conversions": 5000,
      "period_weeks": 4,
      "spend": 25000,
      "attribution_window": "7d_click_1d_view",
      "notes": "From Meta Ads Manager"
    }
  ],
  "prior_beliefs": [
    {
      "channel": "google_search",
      "expected_roi_low": 1.5,
      "expected_roi_high": 3.0,
      "confidence": "high",
      "source": "Historical performance data"
    }
  ],
  "control_variables": {
    "is_promotion": "Binary flag for promotion weeks",
    "seasonality": "Seasonal index (1.0 = normal)"
  },
  "notes": "Q4 2025 calibration data"
}
"""


def create_calibration_template(output_path: Path | str) -> None:
    """Create a calibration template file for users to fill in."""
    path = Path(output_path)
    with open(path, "w") as f:
        f.write(CALIBRATION_TEMPLATE)
