"""Main Sommmelier model wrapper around Meridian."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mmm.data.schema import MMMDataset
from mmm.model.builder import build_meridian_input

if TYPE_CHECKING:
    from meridian.model import model as meridian_model


@dataclass
class ModelConfig:
    """Configuration for the MMM model."""

    # Sampling parameters
    n_chains: int = 4
    n_adapt: int = 1000
    n_burnin: int = 500
    n_keep: int = 500
    seed: int = 0

    # Prior configuration
    roi_prior_mean: float = 0.2
    roi_prior_sigma: float = 0.9

    # Model options - knots for baseline (None = auto, or list like [0, 26, 52, 78, 104])
    knots: list[int] | None = None
    max_lag: int = 8

    # Output paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))


@dataclass
class ModelResults:
    """Container for model results."""

    # Fit metrics
    r_squared: float | None = None
    mape: float | None = None

    # Channel contributions
    channel_contributions: dict[str, float] = field(default_factory=dict)
    channel_roi: dict[str, float] = field(default_factory=dict)

    # Diagnostics
    convergence_passed: bool = False
    r_hat_max: float | None = None

    # Raw model reference
    meridian_model: Any = None

    def summary(self) -> str:
        """Return human-readable summary of results."""
        lines = [
            "Model Results Summary",
            "=" * 40,
            "",
            "Fit Metrics:",
            f"  R-squared: {self.r_squared:.3f}" if self.r_squared else "  R-squared: N/A",
            f"  MAPE: {self.mape:.1%}" if self.mape else "  MAPE: N/A",
            "",
            "Convergence:",
            f"  Passed: {'' if self.convergence_passed else ''}",
            f"  Max R-hat: {self.r_hat_max:.3f}" if self.r_hat_max else "",
            "",
            "Channel ROI:",
        ]

        for channel, roi in sorted(self.channel_roi.items(), key=lambda x: -x[1]):
            lines.append(f"  {channel}: {roi:.2f}x")

        lines.extend([
            "",
            "Channel Contribution to KPI:",
        ])

        total_contrib = sum(self.channel_contributions.values())
        for channel, contrib in sorted(self.channel_contributions.items(), key=lambda x: -x[1]):
            pct = contrib / total_contrib * 100 if total_contrib > 0 else 0
            lines.append(f"  {channel}: {pct:.1f}%")

        return "\n".join(lines)


class AutoMMM:
    """
    Main Sommmelier class wrapping Meridian.

    This provides a simplified interface for:
    - Loading and validating data
    - Configuring and running the model
    - Extracting insights and recommendations
    """

    def __init__(self, dataset: MMMDataset, config: ModelConfig | None = None):
        """
        Initialize Sommmelier with a dataset.

        Args:
            dataset: Validated MMMDataset
            config: Optional ModelConfig (uses defaults if not provided)
        """
        self.dataset = dataset
        self.config = config or ModelConfig()
        self._meridian: "meridian_model.Meridian | None" = None
        self._results: ModelResults | None = None
        self._input_data = None

    def prepare(
        self,
        calibration_priors: dict[str, dict] | None = None,
    ) -> None:
        """Prepare the model for fitting (build InputData, initialize Meridian).

        Args:
            calibration_priors: Per-channel priors from calibration module.
                Keys are channel names, values have roi_mean, roi_sigma, source.
        """
        from meridian.model import model, prior_distribution, spec

        import tensorflow_probability as tfp

        # Build Meridian InputData from our dataset
        self._input_data = build_meridian_input(self.dataset)

        # Configure per-channel priors (single batched LogNormal with batch_shape=[n_channels])
        if calibration_priors:
            roi_means = []
            roi_sigmas = []
            for ch in self.dataset.media_channels:
                if ch in calibration_priors:
                    p = calibration_priors[ch]
                    roi_means.append(p["roi_mean"])
                    roi_sigmas.append(p["roi_sigma"])
                else:
                    roi_means.append(self.config.roi_prior_mean)
                    roi_sigmas.append(self.config.roi_prior_sigma)
            prior = prior_distribution.PriorDistribution(
                roi_m=tfp.distributions.LogNormal(roi_means, roi_sigmas)
            )
        else:
            prior = prior_distribution.PriorDistribution(
                roi_m=tfp.distributions.LogNormal(
                    self.config.roi_prior_mean,
                    self.config.roi_prior_sigma,
                )
            )

        # Use AKS when dataset is large enough, fall back to manual knots
        n_periods = self.dataset.n_time_periods
        model_spec_kwargs = dict(prior=prior, max_lag=self.config.max_lag)

        if n_periods >= 26:
            model_spec_kwargs["enable_aks"] = True
        else:
            if n_periods <= 13:
                knots = [0, n_periods - 1]
            elif n_periods <= 52:
                knots = [0, n_periods // 2, n_periods - 1]
            else:
                knots = list(range(0, n_periods, 13))
                if knots[-1] != n_periods - 1:
                    knots.append(n_periods - 1)
            model_spec_kwargs["knots"] = knots

        model_spec = spec.ModelSpec(**model_spec_kwargs)

        # Initialize Meridian model
        self._meridian = model.Meridian(
            input_data=self._input_data,
            model_spec=model_spec,
        )

    def fit(
        self,
        sample_prior: bool = True,
        calibration_priors: dict[str, dict] | None = None,
    ) -> ModelResults:
        """
        Fit the MMM model.

        Args:
            sample_prior: Whether to sample from prior first (recommended)
            calibration_priors: Per-channel priors from calibration module.

        Returns:
            ModelResults with fit metrics and insights
        """
        if self._meridian is None:
            self.prepare(calibration_priors=calibration_priors)

        # Sample from prior (optional but recommended)
        if sample_prior:
            self._meridian.sample_prior(500)

        # Sample from posterior
        self._meridian.sample_posterior(
            n_chains=self.config.n_chains,
            n_adapt=self.config.n_adapt,
            n_burnin=self.config.n_burnin,
            n_keep=self.config.n_keep,
            seed=self.config.seed,
        )

        # Extract results
        self._results = self._extract_results()

        return self._results

    def _extract_results(self) -> ModelResults:
        """Extract results from fitted Meridian model."""
        from meridian.analysis import summarizer

        results = ModelResults(meridian_model=self._meridian)

        try:
            # Get model summary
            mmm_summarizer = summarizer.Summarizer(self._meridian)

            # Extract ROI estimates
            roi_summary = mmm_summarizer.get_roi_summary()
            if roi_summary is not None:
                for channel in self.dataset.media_channels:
                    if channel in roi_summary.index:
                        results.channel_roi[channel] = float(roi_summary.loc[channel, "mean"])

            # Extract contributions
            contrib_summary = mmm_summarizer.get_contribution_summary()
            if contrib_summary is not None:
                for channel in self.dataset.media_channels:
                    if channel in contrib_summary.index:
                        results.channel_contributions[channel] = float(
                            contrib_summary.loc[channel, "contribution"]
                        )

        except Exception:
            # Results extraction failed, return partial results
            pass

        return results

    def review(self) -> dict[str, Any]:
        """
        Run model diagnostics and review.

        Runs all 7 ModelReviewer checks: convergence, negative baseline,
        Bayesian PPP, goodness of fit, prior-posterior shift, ROI consistency,
        model diagnostics.

        Returns:
            Dictionary of diagnostic results with check names, pass/fail, details.
        """
        if self._meridian is None:
            raise ValueError("Model must be fitted first. Call fit() before review().")

        from meridian.analysis.review import reviewer

        model_reviewer = reviewer.ModelReviewer(self._meridian)
        raw_result = model_reviewer.run()

        # Structure the results for downstream consumption
        structured = {}
        if isinstance(raw_result, dict):
            for check_name, check_result in raw_result.items():
                structured[check_name] = {
                    "passed": bool(check_result.get("passed", True)) if isinstance(check_result, dict) else True,
                    "details": str(check_result),
                }
        else:
            structured["raw"] = str(raw_result)

        return structured

    def optimize_budget(
        self,
        budget: float | None = None,
        constraints: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, float]:
        """
        Run budget optimization.

        Args:
            budget: Total budget to optimize (defaults to current spend)
            constraints: Optional per-channel min/max constraints

        Returns:
            Recommended budget allocation per channel
        """
        if self._meridian is None:
            raise ValueError("Model must be fitted first. Call fit() before optimize_budget().")

        from meridian.analysis import optimizer

        budget_optimizer = optimizer.BudgetOptimizer(self._meridian)

        if budget is None:
            budget = self.dataset.total_spend

        # Run optimization
        result = budget_optimizer.optimize(
            total_budget=budget,
        )

        return result.optimal_allocation

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "meridian": self._meridian,
                "dataset": self.dataset,
                "config": self.config,
                "results": self._results,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "AutoMMM":
        """Load a fitted model from disk."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(data["dataset"], data["config"])
        instance._meridian = data["meridian"]
        instance._results = data["results"]

        return instance

    @property
    def results(self) -> ModelResults | None:
        """Get the model results (None if not fitted)."""
        return self._results

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._meridian is not None and self._results is not None
