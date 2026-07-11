"""Main Sommmelier model wrapper around Meridian."""

from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mmm.data.schema import MMMDataset
from mmm.meridian_compat import (
    extract_channel_contributions,
    extract_optimization_result,
    extract_predictive_accuracy,
    extract_rhat_diagnostics,
    serialize_model_review,
    summarize_channel_tensor,
)
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
            f"  R-squared: {self.r_squared:.3f}"
            if self.r_squared is not None
            else "  R-squared: N/A",
            f"  MAPE: {self.mape:.1%}" if self.mape is not None else "  MAPE: N/A",
            "",
            "Convergence:",
            f"  Passed: {'yes' if self.convergence_passed else 'no'}",
            f"  Max R-hat: {self.r_hat_max:.3f}"
            if self.r_hat_max is not None
            else "  Max R-hat: N/A",
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
        self._meridian: meridian_model.Meridian | None = None
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
        import tensorflow_probability as tfp
        from meridian.model import model, prior_distribution, spec

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
        if self._meridian is None:
            raise RuntimeError("Model preparation did not initialize Meridian")

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
        if self._meridian is None:
            raise ValueError("Model must be prepared before extracting results")

        from meridian.analysis import analyzer

        mmm_analyzer = analyzer.Analyzer(self._meridian)
        results = ModelResults(meridian_model=self._meridian)

        roi = summarize_channel_tensor(
            mmm_analyzer.roi(use_posterior=True), self.dataset.media_channels
        )
        results.channel_roi = {channel: summary["mean"] for channel, summary in roi.items()}

        contributions = extract_channel_contributions(
            mmm_analyzer.incremental_outcome(use_posterior=True),
            self.dataset.media_channels,
        )
        results.channel_contributions = {
            channel: summary["absolute"] for channel, summary in contributions.items()
        }

        accuracy = extract_predictive_accuracy(mmm_analyzer.predictive_accuracy())
        results.r_squared = accuracy.get("r_squared")
        results.mape = accuracy.get("mape")

        diagnostics = extract_rhat_diagnostics(mmm_analyzer.rhat_summary())
        results.convergence_passed = diagnostics["convergence_ok"]
        results.r_hat_max = diagnostics["max_rhat"]

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

        return serialize_model_review(raw_result)

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
        if constraints:
            raise NotImplementedError(
                "Per-channel constraints are not supported by this wrapper yet; "
                "omit constraints rather than assuming they were applied"
            )

        from meridian.analysis import optimizer

        budget_optimizer = optimizer.BudgetOptimizer(self._meridian)

        if budget is None:
            budget = self.dataset.total_spend

        # Run optimization
        result = budget_optimizer.optimize(fixed_budget=True, budget=budget)
        allocation, _ = extract_optimization_result(result)
        if not allocation:
            raise RuntimeError("Meridian returned no optimized channel allocation")
        return allocation

    def save(self, path: str | Path) -> None:
        """Save a non-executable model bundle using protobuf, Parquet, and JSON."""
        import json
        import shutil
        import tempfile

        if self._meridian is None or self._results is None:
            raise ValueError("Model must be fitted before it can be saved")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Model bundle already exists: {path}")

        from meridian.schema.serde import meridian_serde

        temporary_path = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        try:
            meridian_serde.save_meridian(
                self._meridian,
                str(temporary_path / "model.binpb"),
            )
            self.dataset.df.to_parquet(temporary_path / "dataset.parquet", index=False)

            model_config = {
                item.name: (
                    str(getattr(self.config, item.name))
                    if item.name == "output_dir"
                    else getattr(self.config, item.name)
                )
                for item in fields(ModelConfig)
            }
            result_data = {
                item.name: getattr(self._results, item.name)
                for item in fields(ModelResults)
                if item.name != "meridian_model"
            }
            metadata = {
                "schema_version": 1,
                "dataset": {
                    "config": self.dataset.config.model_dump(mode="json"),
                    "date_range": [value.isoformat() for value in self.dataset.date_range],
                    "geos": self.dataset.geos,
                    "n_time_periods": self.dataset.n_time_periods,
                    "n_geos": self.dataset.n_geos,
                    "media_channels": self.dataset.media_channels,
                    "total_spend": self.dataset.total_spend,
                    "total_kpi": self.dataset.total_kpi,
                },
                "model_config": model_config,
                "results": result_data,
            }
            (temporary_path / "metadata.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False)
            )
            temporary_path.rename(path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> "AutoMMM":
        """Load a protobuf/Parquet/JSON model bundle created by :meth:`save`."""
        import json

        import pandas as pd
        from meridian.schema.serde import meridian_serde

        from mmm.data.schema import DataConfig

        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"Model bundle directory not found: {path}")

        required_files = ("model.binpb", "dataset.parquet", "metadata.json")
        missing_files = [name for name in required_files if not (path / name).is_file()]
        if missing_files:
            raise ValueError("Invalid model bundle; missing: " + ", ".join(missing_files))

        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported model bundle schema: {metadata.get('schema_version')}"
            )

        dataset_data = metadata["dataset"]
        date_values = dataset_data["date_range"]
        if not isinstance(date_values, list) or len(date_values) != 2:
            raise ValueError("Invalid model bundle date_range")
        date_range = (
            date.fromisoformat(date_values[0]),
            date.fromisoformat(date_values[1]),
        )
        dataset = MMMDataset(
            df=pd.read_parquet(path / "dataset.parquet"),
            config=DataConfig.model_validate(dataset_data["config"]),
            date_range=date_range,
            geos=dataset_data["geos"],
            n_time_periods=dataset_data["n_time_periods"],
            n_geos=dataset_data["n_geos"],
            media_channels=dataset_data["media_channels"],
            total_spend=dataset_data["total_spend"],
            total_kpi=dataset_data["total_kpi"],
        )
        model_config_data = metadata["model_config"]
        model_config_data["output_dir"] = Path(model_config_data["output_dir"])
        instance = cls(dataset, ModelConfig(**model_config_data))
        instance._meridian = meridian_serde.load_meridian(str(path / "model.binpb"))
        instance._results = ModelResults(
            **metadata["results"], meridian_model=instance._meridian
        )

        return instance

    @property
    def results(self) -> ModelResults | None:
        """Get the model results (None if not fitted)."""
        return self._results

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._meridian is not None and self._results is not None
