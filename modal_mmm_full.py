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
        "matplotlib>=3.8.0",
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
    holdout_weeks: int = 0,  # Number of trailing weeks to hold out (0 = no holdout)
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

    # ─── Auto-detect channels and variable types from column names ───
    spend_cols_all = [col for col in df.columns if '_spend' in col.lower()]

    # Separate channels into spend+impressions vs reach+frequency
    si_channels = []       # spend+impressions channel names
    si_impression_cols = []
    si_spend_cols = []
    rf_channels = []       # reach+frequency channel names
    rf_reach_cols = []
    rf_frequency_cols = []
    rf_spend_cols = []

    for spend_col in spend_cols_all:
        ch = spend_col.replace('_spend', '').replace('_Spend', '')

        # Check for reach+frequency columns
        reach_col = next((c for c in df.columns if c.lower() == f"{ch.lower()}_reach"), None)
        freq_col = next((c for c in df.columns if c.lower() == f"{ch.lower()}_frequency"), None)

        if reach_col and freq_col:
            # R&F channel
            rf_channels.append(ch)
            rf_reach_cols.append(reach_col)
            rf_frequency_cols.append(freq_col)
            rf_spend_cols.append(spend_col)
            print(f"  {ch}: reach+frequency channel")
        elif reach_col and not freq_col:
            print(f"  Warning: {ch} has _reach but no _frequency — treating as spend+impressions")
            # Fall through to spend+impressions
        elif freq_col and not reach_col:
            print(f"  Warning: {ch} has _frequency but no _reach — treating as spend+impressions")
            # Fall through to spend+impressions

        if not (reach_col and freq_col):
            # Spend+impressions channel
            si_channels.append(ch)
            si_spend_cols.append(spend_col)

            # Find matching impression column
            imp_col = None
            for suffix in ["_impressions", "_impression"]:
                for prefix in [ch, ch.lower()]:
                    if f"{prefix}{suffix}" in df.columns:
                        imp_col = f"{prefix}{suffix}"
                        break
                if imp_col:
                    break

            if imp_col is None:
                imp_col = f"{ch}_impression"
                df[imp_col] = df[spend_col] * 100  # Assume $10 CPM
                print(f"  Estimated impressions for {ch} from spend")

            si_impression_cols.append(imp_col)

    # All paid media channels (spend+impressions + R&F) — used for priors, ROI, etc.
    channels = si_channels + rf_channels

    # Detect organic media columns (suffix: _organic)
    organic_cols = [col for col in df.columns if col.lower().endswith('_organic')]
    organic_channels = [col.rsplit('_organic', 1)[0] for col in organic_cols]
    if organic_channels:
        print(f"Organic media: {organic_channels}")

    # Detect non-media treatment columns (suffix: _treatment)
    treatment_cols = [col for col in df.columns if col.lower().endswith('_treatment')]
    treatment_names = [col.rsplit('_treatment', 1)[0] for col in treatment_cols]
    if treatment_names:
        print(f"Non-media treatments: {treatment_names}")

    # Detect control columns (suffix: _control, or common names like is_holiday)
    control_cols = [col for col in df.columns if '_control' in col.lower()]
    # Treatment columns take precedence over control columns
    control_cols = [c for c in control_cols if c not in treatment_cols]

    if 'population' not in df.columns:
        pop_map = {'US': 330_000_000, 'UK': 67_000_000, 'AU': 26_000_000}
        df['population'] = df['geo'].map(lambda x: pop_map.get(x, 10_000_000))

    print(f"Paid media channels: {channels} ({len(si_channels)} spend+imp, {len(rf_channels)} R&F)")

    # ─── Build Meridian InputData ───
    from meridian.data import data_frame_input_data_builder

    has_revenue = 'revenue_per_conversion' in df.columns or 'revenue' in df.columns
    kpi_type = 'revenue' if has_revenue else 'non_revenue'

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=kpi_type,
        default_kpi_column=kpi_column,
    )
    builder = builder.with_kpi(df)
    builder = builder.with_population(df)

    if 'revenue_per_conversion' in df.columns:
        builder = builder.with_revenue_per_kpi(df, revenue_per_kpi_col='revenue_per_conversion')

    # Add spend+impressions media channels
    if si_channels:
        builder = builder.with_media(
            df,
            media_channels=si_channels,
            media_cols=si_impression_cols,
            media_spend_cols=si_spend_cols,
        )

    # Add reach+frequency media channels
    if rf_channels:
        builder = builder.with_media_rf(
            df,
            media_channels=rf_channels,
            reach_cols=rf_reach_cols,
            frequency_cols=rf_frequency_cols,
            spend_cols=rf_spend_cols,
        )
        print(f"Added R&F channels: {rf_channels}")

    # Add organic media channels
    if organic_cols:
        builder = builder.with_organic_media(
            df,
            organic_channels=organic_channels,
            organic_cols=organic_cols,
        )
        print(f"Added organic channels: {organic_channels}")

    # Add non-media treatment variables
    if treatment_cols:
        builder = builder.with_non_media_treatments(df, treatment_cols=treatment_cols)
        print(f"Added treatments: {treatment_names}")

    # Add controls
    if control_cols:
        builder = builder.with_controls(df, control_cols=control_cols)
        print(f"Added controls: {control_cols}")

    input_data = builder.build()
    print("InputData built successfully")

    # Configure model
    from meridian.model import model, spec, prior_distribution
    import tensorflow_probability as tfp

    n_periods = df['time'].nunique()

    # Configure priors - use calibration data if available
    # Build per-channel ROI priors: each channel gets its own LogNormal distribution
    default_roi_mean = 0.2
    default_roi_sigma = 0.9

    if calibration_priors:
        print(f"Using per-channel calibration priors for {len(calibration_priors)} channels")
        # Build parallel arrays of means and sigmas for a single batched LogNormal
        # Meridian expects roi_m to be a single distribution with batch_shape=[n_channels]
        roi_means = []
        roi_sigmas = []
        for ch in channels:
            if ch in calibration_priors:
                p = calibration_priors[ch]
                roi_means.append(p["roi_mean"])
                roi_sigmas.append(p["roi_sigma"])
                print(f"  {ch}: mean={p['roi_mean']:.2f}, sigma={p['roi_sigma']:.2f} (from {p.get('source', 'calibration')})")
            else:
                roi_means.append(default_roi_mean)
                roi_sigmas.append(default_roi_sigma)
                print(f"  {ch}: mean={default_roi_mean}, sigma={default_roi_sigma} (default, no calibration)")

        # Single LogNormal with batch_shape=[n_channels]
        prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(roi_means, roi_sigmas)
        )
    else:
        # Default prior (uninformative) - single scalar applies to all channels
        print("Using default priors (no calibration data provided)")
        prior = prior_distribution.PriorDistribution(
            roi_m=tfp.distributions.LogNormal(default_roi_mean, default_roi_sigma)
        )

    # Infer adstock type per channel: upper-funnel channels get binomial,
    # direct response channels get geometric (the default).
    UPPER_FUNNEL_KEYWORDS = {"youtube", "tv", "video", "brand_awareness", "awareness"}
    adstock_decay_spec = {}
    for ch in channels:
        ch_lower = ch.lower()
        if any(kw in ch_lower for kw in UPPER_FUNNEL_KEYWORDS):
            adstock_decay_spec[ch] = "binomial"
        else:
            adstock_decay_spec[ch] = "geometric"
    print(f"Adstock types: {adstock_decay_spec}")

    # Build holdout mask if requested (out-of-time validation)
    holdout_id = None
    if holdout_weeks and holdout_weeks > 0:
        n_geos = int(df['geo'].nunique())
        if holdout_weeks > n_periods // 2:
            print(f"Warning: holdout_weeks ({holdout_weeks}) > half the data ({n_periods // 2}). Skipping holdout.")
        else:
            holdout_id = np.zeros((n_geos, n_periods), dtype=bool)
            holdout_id[:, -holdout_weeks:] = True
            print(f"Holdout validation: last {holdout_weeks} weeks held out ({holdout_id.sum()} observations)")

    # Use AKS when the dataset is large enough, fall back to manual knots otherwise.
    # AKS requires enough time periods for backward elimination to work.
    USE_AKS_MIN_PERIODS = 26  # AKS needs meaningful time range

    # Only include adstock_decay_spec if any channels are non-default (binomial)
    has_binomial = any(v == "binomial" for v in adstock_decay_spec.values())
    model_spec_kwargs = dict(prior=prior)
    if has_binomial:
        model_spec_kwargs["adstock_decay_spec"] = adstock_decay_spec

    if n_periods >= USE_AKS_MIN_PERIODS:
        model_spec_kwargs["enable_aks"] = True
        print(f"Using Automatic Knot Selection (AKS) — {n_periods} periods")
    else:
        # Manual knot placement for small datasets
        if n_periods <= 13:
            knots = [0, n_periods - 1]
        elif n_periods <= 52:
            knots = [0, n_periods // 2, n_periods - 1]
        else:
            knots = list(range(0, n_periods, 13))
            if knots[-1] != n_periods - 1:
                knots.append(n_periods - 1)
        model_spec_kwargs["knots"] = knots
        print(f"Using manual knots (dataset too small for AKS): {knots}")

    if holdout_id is not None:
        model_spec_kwargs["holdout_id"] = holdout_id

    # Try full ModelSpec; strip unsupported kwargs if needed
    model_spec = spec.ModelSpec(**model_spec_kwargs)
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
        "cpik": {},
        "contributions": {},
        "response_curves": {},
        "adstock_decay": {},
        "marginal_roi": {},
        "model_fit": {},
        "optimal_frequency": {},
        "organic_contributions": {},
        "treatment_effects": {},
        "optimization": {},
        "diagnostics": {},
        "model_review": {},
        "charts": {},
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

    # 1b. CPIK (cost per incremental KPI) - inverse of ROI, more intuitive for marketers
    print("Extracting CPIK...")
    try:
        cpik_tensor = mmm_analyzer.cpik()
        cpik_np = cpik_tensor.numpy()
        cpik_mean = cpik_np.mean(axis=(0, 1))

        for i, ch in enumerate(channels):
            results["cpik"][ch] = float(cpik_mean[i])
    except Exception as e:
        print(f"Warning: CPIK extraction failed: {e}")

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
        spend_multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        response_ds = mmm_analyzer.response_curves(spend_multipliers=spend_multipliers)
        if response_ds is not None:
            # Meridian 1.4.x returns xarray Dataset with dims: spend_multiplier, channel, metric
            # and data vars: spend, incremental_outcome
            if 'incremental_outcome' in response_ds.data_vars and 'channel' in response_ds.dims:
                for ch in channels:
                    try:
                        # Get mean incremental outcome across the metric dimension
                        ch_data = response_ds['incremental_outcome'].sel(channel=ch)
                        # metric dim has [mean, ci_lo, ci_hi] — take mean (index 0)
                        if 'metric' in ch_data.dims:
                            mean_response = ch_data.sel(metric=ch_data.coords['metric'].values[0]).values.tolist()
                        else:
                            mean_response = ch_data.values.tolist()
                        results["response_curves"][ch] = {
                            "spend_multiplier": spend_multipliers,
                            "response": mean_response,
                        }
                    except (KeyError, IndexError):
                        results["response_curves"][ch] = {
                            "spend_multiplier": spend_multipliers,
                            "response": [],
                        }
            else:
                print(f"  Response curves xarray: vars={list(response_ds.data_vars)}, dims={dict(response_ds.dims)}")
    except Exception as e:
        print(f"Warning: Response curves extraction failed: {e}")

    # 4. Adstock decay
    print("Extracting adstock decay...")
    try:
        adstock_data = mmm_analyzer.adstock_decay()
        if adstock_data is not None:
            if hasattr(adstock_data, 'numpy'):
                # Tensor output — shape (chains, draws, channels) or similar
                ad_np = adstock_data.numpy()
                ad_mean = ad_np.mean(axis=tuple(range(ad_np.ndim - 1)))
                ad_q05 = np.percentile(ad_np, 5, axis=tuple(range(ad_np.ndim - 1)))
                ad_q95 = np.percentile(ad_np, 95, axis=tuple(range(ad_np.ndim - 1)))
                for i, ch in enumerate(channels):
                    if i < len(ad_mean):
                        results["adstock_decay"][ch] = {
                            "mean": float(ad_mean[i]),
                            "ci_lower": float(ad_q05[i]),
                            "ci_upper": float(ad_q95[i]),
                        }
            elif hasattr(adstock_data, 'index'):
                # DataFrame output
                for ch in channels:
                    if ch in adstock_data.index:
                        row = adstock_data.loc[ch]
                        results["adstock_decay"][ch] = {
                            "mean": float(row.get('mean', row.iloc[0])),
                            "ci_lower": float(row.get('ci_lo', row.iloc[0] * 0.7)),
                            "ci_upper": float(row.get('ci_hi', row.iloc[0] * 1.3)),
                        }
            else:
                print(f"  Adstock decay type: {type(adstock_data)}")
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

    # 5b. Optimal frequency for R&F channels
    if rf_channels:
        print("Extracting optimal frequency for R&F channels...")
        try:
            opt_freq = mmm_analyzer.optimal_freq()
            if opt_freq is not None:
                opt_freq_np = opt_freq.numpy()
                opt_freq_mean = opt_freq_np.mean(axis=(0, 1))
                for i, ch in enumerate(rf_channels):
                    results["optimal_frequency"][ch] = float(opt_freq_mean[i])
        except Exception as e:
            print(f"Warning: Optimal frequency extraction failed: {e}")

    # 5c. Organic media contributions
    if organic_channels:
        print("Extracting organic media contributions...")
        try:
            organic_inc = mmm_analyzer.incremental_outcome(use_posterior=True)
            organic_np = organic_inc.numpy().mean(axis=(0, 1))
            # Organic channels come after paid media channels in the model output
            n_paid = len(channels)
            for i, ch in enumerate(organic_channels):
                idx = n_paid + i
                if idx < len(organic_np) if organic_np.ndim == 1 else idx < organic_np.shape[-1]:
                    results["organic_contributions"][ch] = {
                        "absolute": float(organic_np[idx] if organic_np.ndim == 1 else organic_np[..., idx].mean()),
                    }
        except Exception as e:
            print(f"Warning: Organic contribution extraction failed: {e}")

    # 5d. Non-media treatment effects
    if treatment_cols:
        print("Extracting treatment effects...")
        try:
            # Treatment effects are part of the model's non-media contribution
            treatment_inc = mmm_analyzer.incremental_outcome(use_posterior=True)
            treatment_np = treatment_inc.numpy().mean(axis=(0, 1))
            for i, tname in enumerate(treatment_names):
                results["treatment_effects"][tname] = {
                    "name": tname,
                    "column": treatment_cols[i],
                }
        except Exception as e:
            print(f"Warning: Treatment effects extraction failed: {e}")

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

    # 6b. Holdout validation (if holdout was requested)
    if holdout_id is not None:
        print("Extracting holdout validation metrics...")
        try:
            # predictive_accuracy with holdout gives in-sample and out-of-sample metrics
            holdout_accuracy = mmm_analyzer.predictive_accuracy()
            if holdout_accuracy is not None:
                results["holdout_validation"] = {
                    "holdout_weeks": holdout_weeks,
                }
                # Extract in-sample and out-of-sample R-squared if available
                if 'metric' in holdout_accuracy.dims or 'metric' in holdout_accuracy.coords:
                    for metric_name in holdout_accuracy.coords.get('metric', holdout_accuracy.dims.get('metric', [])).values:
                        metric_str = str(metric_name).lower().replace('_', '')
                        if 'rsquared' in metric_str:
                            val = holdout_accuracy.sel(metric=metric_name)['value'].values
                            val_float = float(val.mean()) if val.size > 1 else float(val)
                            results["holdout_validation"]["r_squared"] = val_float
                print(f"  Holdout validation: {results.get('holdout_validation', {})}")
        except Exception as e:
            print(f"Warning: Holdout validation extraction failed: {e}")

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

    # 8. ModelReviewer (diagnostic checks)
    print("Running ModelReviewer...")
    try:
        from meridian.analysis.review import reviewer

        model_reviewer = reviewer.ModelReviewer(mmm)
        review_result = model_reviewer.run()

        # Store structured results — handle various return types
        results["model_review"] = {}
        if isinstance(review_result, dict):
            for check_name, check_result in review_result.items():
                results["model_review"][check_name] = {
                    "passed": bool(check_result.get("passed", True)) if isinstance(check_result, dict) else True,
                    "details": str(check_result),
                }
        elif isinstance(review_result, list):
            for item in review_result:
                name = item.get("name", "unknown") if isinstance(item, dict) else str(item)
                results["model_review"][name] = {
                    "passed": item.get("passed", True) if isinstance(item, dict) else True,
                    "details": str(item),
                }
        else:
            results["model_review"]["raw"] = str(review_result)

        print(f"  ModelReviewer completed: {len(results['model_review'])} checks")

    except Exception as e:
        print(f"Warning: ModelReviewer failed: {e}")
        results["model_review"] = {}

    # 8b. Native Meridian visualizations (generate PNGs on GPU)
    # Meridian 1.4.x visualizer classes: MediaSummary, MediaEffects, ModelFit, ModelDiagnostics
    # They take an Analyzer instance and have specific plot_* method names.
    print("Generating native Meridian charts...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from meridian.analysis import visualizer

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chart_dir = f"/outputs/charts_{timestamp}"
        import os
        os.makedirs(chart_dir, exist_ok=True)

        results["charts"] = {}

        # Meridian 1.4.0 visualizer classes take the Meridian model, not the Analyzer
        media_summary = visualizer.MediaSummary(mmm)
        media_effects = visualizer.MediaEffects(mmm)
        model_fit_viz = visualizer.ModelFit(mmm)
        model_diag = visualizer.ModelDiagnostics(mmm)

        chart_configs = [
            ("roi_bar_chart.png", media_summary.plot_roi_bar_chart),
            ("contribution_pie.png", media_summary.plot_contribution_pie_chart),
            ("cpik_chart.png", media_summary.plot_cpik),
            ("roi_vs_mroi.png", media_summary.plot_roi_vs_mroi),
            ("response_curves.png", media_effects.plot_response_curves),
            ("adstock_decay.png", media_effects.plot_adstock_decay),
            ("hill_curves.png", media_effects.plot_hill_curves),
            ("model_fit.png", model_fit_viz.plot_model_fit),
            ("prior_posterior.png", model_diag.plot_prior_and_posterior_distribution),
            ("rhat_boxplot.png", model_diag.plot_rhat_boxplot),
        ]

        for chart_name, plot_fn in chart_configs:
            try:
                fig = plot_fn()
                chart_path = f"{chart_dir}/{chart_name}"
                if fig is not None and hasattr(fig, 'savefig'):
                    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                    plt.close('all')
                results["charts"][chart_name.replace('.png', '')] = chart_path
                print(f"  Saved {chart_name}")
            except Exception as e_chart:
                print(f"  Warning: {chart_name} failed: {e_chart}")

        print(f"  Generated {len(results['charts'])} charts in {chart_dir}")

    except Exception as e:
        print(f"Warning: Native chart generation failed: {e}")
        results["charts"] = {}

    # 9. Budget optimization
    if run_optimization:
        print("Running budget optimization...")
        try:
            # BudgetOptimizer takes the Meridian model, not the Analyzer
            budget_optimizer = optimizer.BudgetOptimizer(mmm)

            current_spend = sum(results["metadata"]["total_spend"].values())

            spend_multipliers = {"reduce_20": 0.8, "current": 1.0, "increase_20": 1.2}
            for name, mult in spend_multipliers.items():
                try:
                    budget = current_spend * mult
                    opt_result = budget_optimizer.optimize(fixed_budget=budget)
                    if opt_result is not None:
                        results["optimization"][name] = {
                            "budget": float(budget),
                            "optimal_allocation": {},
                            "expected_outcome": None,
                        }
                        if hasattr(opt_result, 'optimal_spend'):
                            for i, ch in enumerate(channels):
                                if i < len(opt_result.optimal_spend):
                                    results["optimization"][name]["optimal_allocation"][ch] = float(opt_result.optimal_spend[i])
                        if hasattr(opt_result, 'optimal_outcome'):
                            results["optimization"][name]["expected_outcome"] = float(opt_result.optimal_outcome)
                except Exception as e:
                    print(f"  Warning: Optimization for {name} failed: {e}")

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
    holdout_weeks: int = 0,  # Hold out last N weeks for validation (~$0.30 extra GPU)
):
    """
    Run full MMM analysis from command line.

    Example:
        modal run modal_mmm_full.py --data data/examples/sample_data.csv --report
        modal run modal_mmm_full.py --data data/raw/mydata.csv --calibration data/calibration.json
        modal run modal_mmm_full.py --data data/raw/mydata.csv --holdout-weeks 8
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
        holdout_weeks=holdout_weeks,
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

    if results.get("cpik"):
        print("\n## CPIK (Cost per Incremental KPI)")
        print("-" * 40)
        for ch, cpik in sorted(results.get("cpik", {}).items(), key=lambda x: x[1]):
            print(f"  {ch:12s}: ${cpik:.2f}")

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
