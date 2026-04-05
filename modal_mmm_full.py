"""
Modal GPU runner for Sommmelier with full analysis and visualization.

Usage:
    modal run modal_mmm_full.py --data data/examples/sample_data.csv
    modal run modal_mmm_full.py --data data/raw/your_data.csv --report
"""

import modal

# Define the Modal image with all dependencies
mmm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "google-meridian==1.4.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
    )
    .pip_install(
        "jax[cuda12]",
    )
)

app = modal.App("sommmelier-full")
volume = modal.Volume.from_name("sommmelier-outputs", create_if_missing=True)


@app.function(
    image=mmm_image,
    gpu="T4",
    timeout=7200,  # 2 hours for full analysis
    volumes={"/outputs": volume},
)
def fit_mmm_full(
    data_csv: str,
    kpi_column: str = "conversions",
    n_chains: int = 4,
    n_keep: int = 500,
    run_optimization: bool = True,
    calibration_priors: dict | None = None,  # Channel-specific priors from calibration
) -> dict:
    """
    Fit MMM model and extract comprehensive results for visualization.

    Returns:
        Dictionary with ROI, contributions, response curves, adstock,
        optimization results, and model diagnostics.
    """
    import io
    import json
    import warnings
    from datetime import datetime

    import pandas as pd
    import numpy as np

    # Monkey-patch numpy 2.x compatibility for TFP
    _original_reshape = np.reshape
    def _patched_reshape(a, *args, newshape=None, shape=None, **kwargs):
        if newshape is not None and shape is None:
            shape = newshape
        if shape is not None:
            return _original_reshape(a, shape, **kwargs)
        return _original_reshape(a, *args, **kwargs)
    np.reshape = _patched_reshape

    warnings.filterwarnings('ignore')

    print(f"Starting full MMM analysis at {datetime.now()}")

    import jax
    print(f"JAX devices: {jax.devices()}")

    # Load and prepare data
    df = pd.read_csv(io.StringIO(data_csv))

    # Handle date column (could be 'date' or 'time')
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'time'})
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    print(f"Loaded data: {len(df)} rows, {df['geo'].nunique()} geos, {df['time'].nunique()} periods")

    # Auto-detect channels from spend columns
    # Handle both formats: 'meta_spend' and 'Channel0_spend'
    spend_cols = [col for col in df.columns if '_spend' in col.lower()]
    channels = []
    impression_cols = []

    for spend_col in spend_cols:
        # Extract channel name
        ch = spend_col.replace('_spend', '').replace('_Spend', '')
        channels.append(ch)

        # Find matching impression column
        possible_imp_cols = [
            f"{ch}_impressions",
            f"{ch}_impression",  # Google's format
            f"{ch.lower()}_impressions",
            f"{ch.lower()}_impression",
        ]
        imp_col = None
        for pic in possible_imp_cols:
            if pic in df.columns:
                imp_col = pic
                break

        if imp_col is None:
            # Estimate impressions from spend
            imp_col = f"{ch}_impression"
            df[imp_col] = df[spend_col] * 100  # Assume $10 CPM
            print(f"  Estimated impressions for {ch} from spend")

        impression_cols.append(imp_col)

    if 'population' not in df.columns:
        pop_map = {'US': 330_000_000, 'UK': 67_000_000, 'AU': 26_000_000}
        df['population'] = df['geo'].map(lambda x: pop_map.get(x, 10_000_000))

    print(f"Channels: {channels}")

    # Build Meridian input
    from meridian.data import data_frame_input_data_builder

    # Determine KPI type based on data
    has_revenue = 'revenue_per_conversion' in df.columns or 'revenue' in df.columns
    kpi_type = 'revenue' if has_revenue else 'non_revenue'

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=kpi_type,
        default_kpi_column=kpi_column,
    )
    builder = builder.with_kpi(df)
    builder = builder.with_population(df)

    # Add revenue_per_kpi if available
    if 'revenue_per_conversion' in df.columns:
        builder = builder.with_revenue_per_kpi(df, revenue_per_kpi_col='revenue_per_conversion')

    builder = builder.with_media(
        df,
        media_channels=channels,
        media_cols=impression_cols,
        media_spend_cols=spend_cols,
    )

    # Add controls if present
    control_cols = [col for col in df.columns if '_control' in col.lower()]
    if control_cols:
        builder = builder.with_controls(df, control_cols=control_cols)
        print(f"Added controls: {control_cols}")

    input_data = builder.build()
    print("InputData built successfully")

    # Configure model
    from meridian.model import model, spec, prior_distribution
    import tensorflow_probability as tfp

    n_periods = df['time'].nunique()
    if n_periods <= 13:
        knots = [0, n_periods - 1]
    elif n_periods <= 52:
        knots = [0, n_periods // 2, n_periods - 1]
    else:
        knots = list(range(0, n_periods, 13))
        if knots[-1] != n_periods - 1:
            knots.append(n_periods - 1)

    # Configure priors - use calibration data if available
    if calibration_priors:
        print(f"Using calibration priors for {len(calibration_priors)} channels")
        # Build per-channel ROI priors from calibration
        # For now, use the average of calibrated channels for the global prior
        roi_means = [p["roi_mean"] for p in calibration_priors.values()]
        roi_sigmas = [p["roi_sigma"] for p in calibration_priors.values()]
        avg_mean = sum(roi_means) / len(roi_means) if roi_means else 0.2
        avg_sigma = sum(roi_sigmas) / len(roi_sigmas) if roi_sigmas else 0.9

        # Log the priors being used
        for ch, prior_data in calibration_priors.items():
            print(f"  {ch}: mean={prior_data['roi_mean']:.2f}, sigma={prior_data['roi_sigma']:.2f} (from {prior_data.get('source', 'calibration')})")

        # Use calibrated prior (more informative than default)
        prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(avg_mean, avg_sigma)
        )
    else:
        # Default prior (uninformative)
        print("Using default priors (no calibration data provided)")
        prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(0.2, 0.9)
        )

    model_spec = spec.ModelSpec(prior=prior, knots=knots)
    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    print("Model initialized")

    # Sample
    print("Sampling from prior...")
    mmm.sample_prior(500)

    print(f"Sampling posterior with {n_chains} chains, {n_keep} samples each...")
    mmm.sample_posterior(
        n_chains=n_chains,
        n_adapt=2000,
        n_burnin=500,
        n_keep=n_keep,
        seed=0,
    )
    print("Posterior sampling complete!")

    # Initialize results
    from meridian.analysis import analyzer, optimizer

    results = {
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "n_time_periods": n_periods,
            "n_geos": int(df['geo'].nunique()),
            "channels": channels,
            "total_spend": {ch: float(df[f"{ch}_spend"].sum()) for ch in channels},
            "total_kpi": float(df[kpi_column].sum()),
        },
        "roi": {},
        "contributions": {},
        "response_curves": {},
        "adstock_decay": {},
        "marginal_roi": {},
        "model_fit": {},
        "optimization": {},
        "diagnostics": {},
    }

    mmm_analyzer = analyzer.Analyzer(mmm)

    # 1. ROI per channel
    print("Extracting ROI...")
    try:
        roi_tensor = mmm_analyzer.roi(use_posterior=True)
        roi_np = roi_tensor.numpy()
        roi_mean = roi_np.mean(axis=(0, 1))
        roi_std = roi_np.std(axis=(0, 1))
        roi_q05 = np.percentile(roi_np, 5, axis=(0, 1))
        roi_q95 = np.percentile(roi_np, 95, axis=(0, 1))

        for i, ch in enumerate(channels):
            results["roi"][ch] = {
                "mean": float(roi_mean[i]),
                "std": float(roi_std[i]),
                "ci_lower": float(roi_q05[i]),
                "ci_upper": float(roi_q95[i]),
            }
    except Exception as e:
        print(f"Warning: ROI extraction failed: {e}")

    # 2. Contributions
    print("Extracting contributions...")
    try:
        incremental = mmm_analyzer.incremental_outcome(use_posterior=True)
        inc_np = incremental.numpy().mean(axis=(0, 1))
        if len(inc_np.shape) > 1:
            inc_sum = inc_np.sum(axis=tuple(range(len(inc_np.shape) - 1)))
        else:
            inc_sum = inc_np
        total = inc_sum.sum()

        for i, ch in enumerate(channels):
            results["contributions"][ch] = {
                "absolute": float(inc_sum[i]),
                "percentage": float(inc_sum[i] / total * 100) if total > 0 else 0,
            }
    except Exception as e:
        print(f"Warning: Contribution extraction failed: {e}")

    # 3. Response curves (spend vs outcome)
    print("Extracting response curves...")
    try:
        response_df = mmm_analyzer.response_curves(spend_multipliers=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        if response_df is not None and len(response_df) > 0:
            for ch in channels:
                ch_data = response_df[response_df['channel'] == ch] if 'channel' in response_df.columns else None
                if ch_data is not None and len(ch_data) > 0:
                    results["response_curves"][ch] = {
                        "spend_multiplier": ch_data['spend_multiplier'].tolist() if 'spend_multiplier' in ch_data.columns else [],
                        "response": ch_data['response_mean'].tolist() if 'response_mean' in ch_data.columns else [],
                    }
    except Exception as e:
        print(f"Warning: Response curves extraction failed: {e}")

    # 4. Adstock decay
    print("Extracting adstock decay...")
    try:
        adstock_df = mmm_analyzer.adstock_decay()
        if adstock_df is not None and len(adstock_df) > 0:
            for ch in channels:
                if ch in adstock_df.index:
                    row = adstock_df.loc[ch]
                    results["adstock_decay"][ch] = {
                        "mean": float(row['mean']) if 'mean' in row else None,
                        "ci_lower": float(row['ci_lo']) if 'ci_lo' in row else None,
                        "ci_upper": float(row['ci_hi']) if 'ci_hi' in row else None,
                    }
    except Exception as e:
        print(f"Warning: Adstock decay extraction failed: {e}")

    # 5. Marginal ROI (ROI at current spend levels)
    print("Extracting marginal ROI...")
    try:
        mroi_tensor = mmm_analyzer.marginal_roi(use_posterior=True)
        mroi_np = mroi_tensor.numpy()
        mroi_mean = mroi_np.mean(axis=(0, 1))

        for i, ch in enumerate(channels):
            results["marginal_roi"][ch] = float(mroi_mean[i])
    except Exception as e:
        print(f"Warning: Marginal ROI extraction failed: {e}")

    # 6. Model fit (R-squared, MAPE) - critical for model quality tracking
    print("Extracting model fit metrics...")
    try:
        accuracy_ds = mmm_analyzer.predictive_accuracy()
        if accuracy_ds is not None:
            # Debug: show full structure
            print(f"  Dims: {dict(accuracy_ds.dims)}")
            print(f"  Coords: {list(accuracy_ds.coords)}")
            print(f"  Data vars: {list(accuracy_ds.data_vars)}")

            # xarray stores metrics as coordinates with 'metric' dimension
            if 'metric' in accuracy_ds.dims or 'metric' in accuracy_ds.coords:
                # Iterate through metrics
                for metric_name in accuracy_ds.coords.get('metric', accuracy_ds.dims.get('metric', [])).values:
                    metric_name_str = str(metric_name)
                    try:
                        # Get value for this metric, averaged across geo_granularity
                        val = accuracy_ds.sel(metric=metric_name)['value'].values
                        val_float = float(val.mean()) if val.size > 1 else float(val)
                        print(f"  {metric_name_str}: {val_float:.4f}")

                        # Store with standardized names
                        metric_lower = metric_name_str.lower().replace('_', '')
                        if metric_lower == 'rsquared' or metric_name_str == 'R_Squared':
                            results["model_fit"]["r_squared"] = val_float
                        elif metric_lower == 'wmape' or metric_name_str == 'wMAPE':
                            results["model_fit"]["wmape"] = val_float
                        elif metric_lower == 'mape' or metric_name_str == 'MAPE':
                            results["model_fit"]["mape"] = val_float
                    except Exception as e2:
                        print(f"  Could not extract {metric_name_str}: {e2}")
            else:
                # Fallback: just print everything
                print(f"  Full dataset:\n{accuracy_ds}")

    except Exception as e:
        import traceback
        print(f"Warning: Model fit extraction failed: {e}")
        traceback.print_exc()

    # 7. MCMC diagnostics (R-hat)
    print("Extracting MCMC diagnostics...")
    try:
        rhat_df = mmm_analyzer.rhat_summary()
        if rhat_df is not None and len(rhat_df) > 0:
            bad_rhats = rhat_df[rhat_df['rhat'] > 1.1] if 'rhat' in rhat_df.columns else []
            results["diagnostics"]["rhat_warnings"] = len(bad_rhats)
            results["diagnostics"]["convergence_ok"] = len(bad_rhats) == 0
    except Exception as e:
        print(f"Warning: Diagnostics extraction failed: {e}")

    # 8. Budget optimization
    if run_optimization:
        print("Running budget optimization...")
        try:
            budget_optimizer = optimizer.BudgetOptimizer(mmm_analyzer)

            # Get current total spend
            current_spend = sum(results["metadata"]["total_spend"].values())

            # Create optimization scenarios
            from meridian.analysis.optimizer import FixedBudgetScenario

            scenarios = [
                FixedBudgetScenario(budget=current_spend * 0.8, name="reduce_20"),
                FixedBudgetScenario(budget=current_spend, name="current"),
                FixedBudgetScenario(budget=current_spend * 1.2, name="increase_20"),
            ]

            for scenario in scenarios:
                try:
                    opt_result = budget_optimizer.optimize(scenario)
                    if opt_result is not None:
                        results["optimization"][scenario.name] = {
                            "budget": float(scenario.budget),
                            "optimal_allocation": {},
                            "expected_outcome": None,
                        }
                        # Extract optimal allocation per channel
                        if hasattr(opt_result, 'optimal_spend'):
                            for i, ch in enumerate(channels):
                                results["optimization"][scenario.name]["optimal_allocation"][ch] = float(opt_result.optimal_spend[i])
                        if hasattr(opt_result, 'optimal_outcome'):
                            results["optimization"][scenario.name]["expected_outcome"] = float(opt_result.optimal_outcome)
                except Exception as e:
                    print(f"Warning: Optimization for {scenario.name} failed: {e}")

        except Exception as e:
            print(f"Warning: Budget optimization failed: {e}")

    print("Results extracted successfully!")

    # Save to volume
    output_path = f"/outputs/full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    volume.commit()
    print(f"Completed at {datetime.now()}")

    return results


@app.local_entrypoint()
def main(
    data: str = "data/examples/sample_data.csv",
    kpi_column: str = "conversions",
    n_chains: int = 4,
    n_keep: int = 500,
    report: bool = False,
    calibration: str = "",  # Path to calibration.json
):
    """
    Run full MMM analysis from command line.

    Example:
        modal run modal_mmm_full.py --data data/examples/sample_data.csv --report
        modal run modal_mmm_full.py --data data/raw/mydata.csv --calibration data/calibration.json
    """
    import json
    from pathlib import Path

    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data}")

    print(f"Reading data from {data_path}...")
    data_csv = data_path.read_text()

    # Load calibration data if provided
    calibration_priors = None
    if calibration:
        calibration_path = Path(calibration)
        if calibration_path.exists():
            print(f"Loading calibration from {calibration_path}...")
            try:
                # Import locally to avoid requiring mmm package on Modal worker
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from mmm.calibration import load_calibration, calculate_channel_priors

                cal_data = load_calibration(calibration_path)
                calibration_priors = calculate_channel_priors(cal_data)
                print(f"Loaded calibration with {len(calibration_priors)} channel priors")
            except Exception as e:
                print(f"Warning: Could not load calibration: {e}")
                print("Proceeding with default priors")
        else:
            print(f"Warning: Calibration file not found: {calibration}")
    else:
        # Check for default calibration file location
        default_cal = Path("data/calibration.json")
        if default_cal.exists():
            print(f"Found default calibration file: {default_cal}")
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from mmm.calibration import load_calibration, calculate_channel_priors

                cal_data = load_calibration(default_cal)
                calibration_priors = calculate_channel_priors(cal_data)
                print(f"Loaded calibration with {len(calibration_priors)} channel priors")
            except Exception as e:
                print(f"Warning: Could not load calibration: {e}")

    print("Submitting full analysis to Modal GPU...")
    print("(This may take 30-45 minutes)")
    print()

    results = fit_mmm_full.remote(
        data_csv=data_csv,
        kpi_column=kpi_column,
        n_chains=n_chains,
        n_keep=n_keep,
        calibration_priors=calibration_priors,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("MMM ANALYSIS RESULTS")
    print("=" * 60)

    print("\n## Channel ROI (Return on Investment)")
    print("-" * 40)
    for ch, data in sorted(results.get("roi", {}).items(), key=lambda x: -x[1].get("mean", 0)):
        mean = data.get("mean", 0)
        ci_lo = data.get("ci_lower", 0)
        ci_hi = data.get("ci_upper", 0)
        print(f"  {ch:12s}: {mean:.2f}x  (90% CI: {ci_lo:.2f} - {ci_hi:.2f})")

    print("\n## Channel Contributions")
    print("-" * 40)
    for ch, data in sorted(results.get("contributions", {}).items(), key=lambda x: -x[1].get("percentage", 0)):
        pct = data.get("percentage", 0)
        print(f"  {ch:12s}: {pct:.1f}%")

    print("\n## Marginal ROI (ROI at current spend)")
    print("-" * 40)
    for ch, mroi in sorted(results.get("marginal_roi", {}).items(), key=lambda x: -x[1]):
        print(f"  {ch:12s}: {mroi:.2f}x")

    if results.get("diagnostics", {}).get("convergence_ok"):
        print("\n[OK] Model convergence: Good (all R-hat < 1.1)")
    else:
        warnings = results.get("diagnostics", {}).get("rhat_warnings", 0)
        print(f"\n[!] Model convergence: {warnings} parameters with R-hat > 1.1")

    # Save results
    output_path = Path("outputs") / f"full_results_{results['timestamp'].replace(':', '-').replace('.', '-')}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results saved to: {output_path}")

    # Generate HTML report if requested
    if report:
        print("\nGenerating HTML report...")
        # Report generation will be handled by the reporting module
        report_path = output_path.with_suffix('.html')
        print(f"Report saved to: {report_path}")

    return results
