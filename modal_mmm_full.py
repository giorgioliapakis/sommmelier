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
        "google-meridian[and-cuda]==1.6.2",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        # ArviZ 0.19 (required by Meridian 1.6) uses a Matplotlib API removed in 3.10+.
        "matplotlib>=3.8.0,<3.10",
        "vl-convert-python>=1.7.0",
    )
    .add_local_python_source("mmm", copy=True)
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
    n_adapt: int = 2000,
    n_burnin: int = 500,
    max_lag: int = 8,
    run_optimization: bool = True,
    calibration_priors: dict | None = None,  # Channel-specific priors from calibration
    holdout_weeks: int = 0,  # Number of trailing weeks to hold out (0 = no holdout)
    adstock_overrides: dict | None = None,  # {"channel": "geometric"|"binomial"}
    force_aks: bool | None = None,  # None=auto, True=force AKS, False=force manual knots
    allow_population_estimates: bool = False,
    allow_impression_estimates: bool = False,
    run_id: str | None = None,
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

    import numpy as np
    import pandas as pd

    from mmm.detection import detect_columns
    from mmm.meridian_compat import (
        extract_channel_contributions,
        extract_optimization_result,
        extract_predictive_accuracy,
        extract_rhat_diagnostics,
        save_chart,
        serialize_model_review,
        summarize_channel_tensor,
    )
    from mmm.observability import configure_run_logger, log_event
    from mmm.result_manifest import (
        create_run_manifest,
        finalize_run_manifest,
        record_section_error,
    )

    # Monkey-patch numpy 2.x compatibility for TFP
    _original_reshape = np.reshape

    def _patched_reshape(a, *args, newshape=None, shape=None, **kwargs):
        if newshape is not None and shape is None:
            shape = newshape
        if shape is not None:
            return _original_reshape(a, shape, **kwargs)
        return _original_reshape(a, *args, **kwargs)

    np.reshape = _patched_reshape

    warnings.filterwarnings("ignore")

    run_manifest = create_run_manifest(run_id=run_id)
    run_logger = configure_run_logger(run_manifest["run_id"], "modal")
    log_event(
        run_logger,
        "run_started",
        n_chains=n_chains,
        n_keep=n_keep,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
    )

    print(f"Starting full MMM analysis at {datetime.now()}")

    import tensorflow as tf

    print(f"TensorFlow GPUs: {tf.config.list_physical_devices('GPU')}")

    # Load and prepare data
    df = pd.read_csv(io.StringIO(data_csv))

    # Handle date column (could be 'date' or 'time')
    if "date" in df.columns:
        df = df.rename(columns={"date": "time"})
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    print(
        f"Loaded data: {len(df)} rows, {df['geo'].nunique()} geos, {df['time'].nunique()} periods"
    )

    # ─── Auto-detect channels and variable types from column names ───
    detected = detect_columns(df.columns)

    # Separate channels into spend+impressions vs reach+frequency
    si_channels = []  # spend+impressions channel names
    si_impression_cols = []
    si_spend_cols = []
    rf_channels = []  # reach+frequency channel names
    rf_reach_cols = []
    rf_frequency_cols = []
    rf_spend_cols = []
    estimated_impression_channels = []

    spend_cols_by_channel = {}
    for channel in detected.media_channels:
        ch = channel["name"]
        spend_col = channel["spend_column"]
        reach_col = channel["reach_column"]
        freq_col = channel["frequency_column"]
        spend_cols_by_channel[ch] = spend_col

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

            imp_col = channel["impressions_column"]

            if imp_col is None:
                if not allow_impression_estimates:
                    raise ValueError(
                        f"Channel '{ch}' needs impressions or reach/frequency data. "
                        "Pass --allow-impression-estimates only when the coarse "
                        "$10 CPM fallback is acceptable."
                    )
                imp_col = f"{ch}_impression"
                df[imp_col] = df[spend_col] * 100  # Assume $10 CPM
                estimated_impression_channels.append(ch)
                print(f"  Estimated impressions for {ch} from spend")

            si_impression_cols.append(imp_col)

    # All paid media channels (spend+impressions + R&F) — used for priors, ROI, etc.
    channels = si_channels + rf_channels
    if not channels:
        raise ValueError("No paid media columns ending in '_spend' were found")

    duplicate_rows = int(df.duplicated(["geo", "time"]).sum())
    if duplicate_rows:
        raise ValueError(f"Found {duplicate_rows} duplicate geo/time rows")
    expected_rows = int(df["geo"].nunique() * df["time"].nunique())
    if len(df) != expected_rows:
        raise ValueError(
            f"Geo/time panel is incomplete: expected {expected_rows} rows, found {len(df)}"
        )
    unique_dates = pd.Series(df["time"].drop_duplicates().sort_values())
    cadence_days = unique_dates.diff().dropna().dt.days
    irregular_cadence = sorted(set(cadence_days[cadence_days != 7].astype(int).tolist()))
    if irregular_cadence:
        raise ValueError(
            "Model data must use exact weekly cadence; found gaps of "
            + ", ".join(map(str, irregular_cadence))
            + " days"
        )
    numeric_columns = [kpi_column, *spend_cols_by_channel.values()]
    if df[numeric_columns].isna().any().any():
        raise ValueError("KPI and spend columns cannot contain missing values")
    if not np.isfinite(df[numeric_columns].to_numpy()).all():
        raise ValueError("KPI and spend columns must contain only finite numeric values")
    if (df[numeric_columns] < 0).any().any():
        raise ValueError("KPI and spend columns cannot contain negative values")
    zero_spend_channels = [
        ch for ch, column in spend_cols_by_channel.items() if df[column].sum() <= 0
    ]
    if zero_spend_channels:
        raise ValueError(f"Channels with zero total spend: {', '.join(zero_spend_channels)}")

    # Detect organic media columns (suffix: _organic)
    organic_cols = list(detected.organic_columns)
    organic_channels = [col.rsplit("_organic", 1)[0] for col in organic_cols]
    if organic_channels:
        print(f"Organic media: {organic_channels}")

    # Detect non-media treatment columns (suffix: _treatment)
    treatment_cols = list(detected.treatment_columns)
    treatment_names = [col.rsplit("_treatment", 1)[0] for col in treatment_cols]
    if treatment_names:
        print(f"Non-media treatments: {treatment_names}")

    # Detect control columns (suffix: _control, or common names like is_holiday)
    control_cols = list(detected.control_columns)

    if "population" not in df.columns:
        if not allow_population_estimates:
            raise ValueError(
                "A population column is required. Pass --allow-population-estimates "
                "only when the coarse built-in fallback is acceptable."
            )
        pop_map = {"US": 330_000_000, "UK": 67_000_000, "AU": 26_000_000}
        df["population"] = df["geo"].map(lambda x: pop_map.get(x, 10_000_000))

    auxiliary_columns = list(
        dict.fromkeys(
            [
                *si_impression_cols,
                *rf_reach_cols,
                *rf_frequency_cols,
                *organic_cols,
                *treatment_cols,
                *control_cols,
                "population",
                *[
                    column
                    for column in ("revenue", "revenue_per_kpi", "revenue_per_conversion")
                    if column in df.columns
                ],
            ]
        )
    )
    if df[auxiliary_columns].isna().any().any():
        raise ValueError("Additional model inputs cannot contain missing values")
    try:
        auxiliary_values = df[auxiliary_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError("Additional model inputs must be numeric") from e
    if not np.isfinite(auxiliary_values).all():
        raise ValueError("Additional model inputs must contain only finite values")
    nonnegative_columns = [
        *si_impression_cols,
        *rf_reach_cols,
        *rf_frequency_cols,
        *organic_cols,
        "population",
        *[
            column
            for column in ("revenue", "revenue_per_kpi", "revenue_per_conversion")
            if column in df.columns
        ],
    ]
    if (df[list(dict.fromkeys(nonnegative_columns))] < 0).any().any():
        raise ValueError("Media, population, organic, and revenue inputs cannot be negative")

    print(f"Paid media channels: {channels} ({len(si_channels)} spend+imp, {len(rf_channels)} R&F)")

    # ─── Build Meridian InputData ───
    from meridian.data import data_frame_input_data_builder

    kpi_type = "revenue" if kpi_column.lower() == "revenue" else "non_revenue"
    revenue_per_kpi_col = next(
        (
            column
            for column in ("revenue_per_kpi", "revenue_per_conversion")
            if column in df.columns
        ),
        None,
    )
    if kpi_type == "non_revenue" and revenue_per_kpi_col is None and "revenue" in df.columns:
        invalid_revenue_rows = (df[kpi_column] == 0) & (df["revenue"] != 0)
        if invalid_revenue_rows.any():
            raise ValueError("Revenue cannot be non-zero when KPI is zero")
        revenue_per_kpi_col = "_revenue_per_kpi"
        df[revenue_per_kpi_col] = (
            df["revenue"].div(df[kpi_column].where(df[kpi_column] != 0)).fillna(0)
        )

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type=kpi_type,
        default_kpi_column=kpi_column,
    )
    builder = builder.with_kpi(df)
    builder = builder.with_population(df)

    if revenue_per_kpi_col:
        builder = builder.with_revenue_per_kpi(df, revenue_per_kpi_col=revenue_per_kpi_col)

    # Add spend+impressions media channels
    if si_channels:
        builder = builder.with_media(
            df,
            media_channels=si_channels,
            media_cols=si_impression_cols,
            media_spend_cols=si_spend_cols,
        )

    # Add reach+frequency media channels (Meridian 1.4.x: with_reach, not with_media_rf)
    if rf_channels:
        builder = builder.with_reach(
            df,
            reach_cols=rf_reach_cols,
            frequency_cols=rf_frequency_cols,
            rf_spend_cols=rf_spend_cols,
            rf_channels=rf_channels,
        )
        print(f"Added R&F channels: {rf_channels}")

    # Add organic media channels
    if organic_cols:
        builder = builder.with_organic_media(
            df,
            organic_media_cols=organic_cols,
            organic_media_channels=organic_channels,
        )
        print(f"Added organic channels: {organic_channels}")

    # Add non-media treatment variables
    if treatment_cols:
        builder = builder.with_non_media_treatments(df, non_media_treatment_cols=treatment_cols)
        print(f"Added treatments: {treatment_names}")

    # Add controls
    if control_cols:
        builder = builder.with_controls(df, control_cols=control_cols)
        print(f"Added controls: {control_cols}")

    input_data = builder.build()
    print("InputData built successfully")

    # Configure model
    import tensorflow_probability as tfp
    from meridian.model import model, prior_distribution, spec

    n_periods = df["time"].nunique()

    # Configure priors - use calibration data if available
    # Build per-channel ROI priors: each channel gets its own LogNormal distribution
    default_roi_mean = 0.2
    default_roi_sigma = 0.9

    # Meridian's roi_m prior applies to with_media() channels only (spend+impressions).
    # R&F channels added via with_reach() have separate priors (rf_prior_type on ModelSpec).
    prior_channels = si_channels  # Only spend+impressions channels get roi_m priors

    if calibration_priors:
        print(f"Using per-channel calibration priors for media channels: {prior_channels}")
        # Build parallel arrays of means and sigmas for a single batched LogNormal
        # Meridian expects roi_m to be a single distribution with batch_shape=[n_media_channels]
        roi_means = []
        roi_sigmas = []
        for ch in prior_channels:
            if ch in calibration_priors:
                p = calibration_priors[ch]
                roi_means.append(p["roi_mean"])
                roi_sigmas.append(p["roi_sigma"])
                print(
                    f"  {ch}: mean={p['roi_mean']:.2f}, sigma={p['roi_sigma']:.2f} (from {p.get('source', 'calibration')})"
                )
            else:
                roi_means.append(default_roi_mean)
                roi_sigmas.append(default_roi_sigma)
                print(
                    f"  {ch}: mean={default_roi_mean}, sigma={default_roi_sigma} (default, no calibration)"
                )

        # Single LogNormal with batch_shape=[n_media_channels]
        # If only 1 channel, use scalar to avoid batch_shape issues
        if len(roi_means) == 1:
            prior = prior_distribution.PriorDistribution(
                roi_m=tfp.distributions.LogNormal(roi_means[0], roi_sigmas[0])
            )
        else:
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
    # User overrides take precedence over auto-detection.
    upper_funnel_keywords = {"youtube", "tv", "video", "brand_awareness", "awareness"}
    adstock_decay_spec = {}
    for ch in channels:
        if adstock_overrides and ch in adstock_overrides:
            adstock_decay_spec[ch] = adstock_overrides[ch]
        else:
            ch_lower = ch.lower()
            if any(kw in ch_lower for kw in upper_funnel_keywords):
                adstock_decay_spec[ch] = "binomial"
            else:
                adstock_decay_spec[ch] = "geometric"
    print(f"Adstock types: {adstock_decay_spec}")

    # Build holdout mask if requested (out-of-time validation)
    holdout_id = None
    if holdout_weeks and holdout_weeks > 0:
        n_geos = int(df["geo"].nunique())
        if holdout_weeks > n_periods // 2:
            print(
                f"Warning: holdout_weeks ({holdout_weeks}) > half the data ({n_periods // 2}). Skipping holdout."
            )
        else:
            holdout_id = np.zeros((n_geos, n_periods), dtype=bool)
            holdout_id[:, -holdout_weeks:] = True
            print(
                f"Holdout validation: last {holdout_weeks} weeks held out ({holdout_id.sum()} observations)"
            )

    # Determine knot strategy: AKS vs manual
    use_aks_min_periods = 26
    use_aks = force_aks if force_aks is not None else (n_periods >= use_aks_min_periods)

    # Only include adstock_decay_spec if any channels are non-default (binomial)
    has_binomial = any(v == "binomial" for v in adstock_decay_spec.values())
    model_spec_kwargs = dict(prior=prior, max_lag=max_lag)
    if has_binomial:
        model_spec_kwargs["adstock_decay_spec"] = adstock_decay_spec

    if use_aks:
        model_spec_kwargs["enable_aks"] = True
        print(f"Using Automatic Knot Selection (AKS) — {n_periods} periods")
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
        print(f"Using manual knots: {knots}")

    if holdout_id is not None:
        model_spec_kwargs["holdout_id"] = holdout_id

    # Try full ModelSpec; strip unsupported kwargs if needed
    model_spec = spec.ModelSpec(**model_spec_kwargs)
    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    print("Model initialized")

    # Sample
    print("Sampling from prior...")
    mmm.sample_prior(500)

    print(
        f"Sampling posterior with {n_chains} chains, {n_keep} samples each (adapt={n_adapt}, burnin={n_burnin})..."
    )
    mmm.sample_posterior(
        n_chains=n_chains,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
        n_keep=n_keep,
        seed=0,
    )
    print("Posterior sampling complete!")
    log_event(run_logger, "posterior_sampling_completed")

    # Initialize results
    from importlib.metadata import version

    from meridian.analysis import analyzer, optimizer

    results = {
        "timestamp": datetime.now().isoformat(),
        "run_manifest": run_manifest,
        "metadata": {
            "meridian_version": version("google-meridian"),
            "n_time_periods": n_periods,
            "n_geos": int(df["geo"].nunique()),
            "channels": channels,
            "estimated_impression_channels": estimated_impression_channels,
            "total_spend": {ch: float(df[spend_cols_by_channel[ch]].sum()) for ch in channels},
            "total_kpi": float(df[kpi_column].sum()),
            "kpi_type": kpi_type,
            "roi_is_monetary": kpi_type == "revenue" or revenue_per_kpi_col is not None,
            "config": {
                "n_chains": n_chains,
                "n_keep": n_keep,
                "n_adapt": n_adapt,
                "n_burnin": n_burnin,
                "max_lag": max_lag,
                "use_aks": use_aks,
                "adstock_decay_spec": adstock_decay_spec,
                "holdout_weeks": holdout_weeks,
                "allow_population_estimates": allow_population_estimates,
                "allow_impression_estimates": allow_impression_estimates,
            },
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
        results["roi"] = summarize_channel_tensor(mmm_analyzer.roi(use_posterior=True), channels)
    except Exception as e:
        print(f"Warning: ROI extraction failed: {e}")
        record_section_error(results, "roi", e, required=True)

    # 1b. CPIK (cost per incremental KPI) - inverse of ROI, more intuitive for marketers
    print("Extracting CPIK...")
    try:
        cpik_summary = summarize_channel_tensor(mmm_analyzer.cpik(), channels)
        results["cpik"] = {channel: summary["mean"] for channel, summary in cpik_summary.items()}
    except Exception as e:
        print(f"Warning: CPIK extraction failed: {e}")
        record_section_error(results, "cpik", e)

    # 2. Contributions
    print("Extracting contributions...")
    try:
        results["contributions"] = extract_channel_contributions(
            mmm_analyzer.incremental_outcome(use_posterior=True), channels
        )
    except Exception as e:
        print(f"Warning: Contribution extraction failed: {e}")
        record_section_error(results, "contributions", e, required=True)

    # 3. Response curves (spend vs outcome)
    print("Extracting response curves...")
    try:
        spend_multipliers = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        response_ds = mmm_analyzer.response_curves(spend_multipliers=spend_multipliers)
        if response_ds is not None:
            # Meridian 1.4.x returns xarray Dataset with dims: spend_multiplier, channel, metric
            # and data vars: spend, incremental_outcome
            if "incremental_outcome" in response_ds.data_vars and "channel" in response_ds.dims:
                for ch in channels:
                    try:
                        # Get mean incremental outcome across the metric dimension
                        ch_data = response_ds["incremental_outcome"].sel(channel=ch)
                        # metric dim has [mean, ci_lo, ci_hi] — take mean (index 0)
                        if "metric" in ch_data.dims:
                            mean_response = ch_data.sel(
                                metric=ch_data.coords["metric"].values[0]
                            ).values.tolist()
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
                print(
                    f"  Response curves xarray: vars={list(response_ds.data_vars)}, dims={dict(response_ds.dims)}"
                )
    except Exception as e:
        print(f"Warning: Response curves extraction failed: {e}")
        record_section_error(results, "response_curves", e)

    # 4. Adstock decay
    print("Extracting adstock decay...")
    try:
        adstock_data = mmm_analyzer.adstock_decay()
        if adstock_data is not None and hasattr(adstock_data, "columns"):
            # DataFrame with columns: metric, channel, time_units, ..., mean, ...
            # Extract the decay at time_unit=1 (one-period decay rate) per channel
            for ch in channels:
                ch_data = (
                    adstock_data[adstock_data["channel"] == ch]
                    if "channel" in adstock_data.columns
                    else None
                )
                if ch_data is not None and len(ch_data) > 0 and "mean" in ch_data.columns:
                    # Get decay at integer time points for a summary
                    int_data = ch_data[
                        ch_data.get("is_int_time_unit", pd.Series([True] * len(ch_data)))
                    ]
                    if len(int_data) > 1:
                        # Decay at t=1 gives the retention rate
                        t1 = (
                            int_data[int_data["time_units"] == 1.0]
                            if "time_units" in int_data.columns
                            else None
                        )
                        if t1 is not None and len(t1) > 0:
                            results["adstock_decay"][ch] = {
                                "retention_at_1_period": float(t1["mean"].iloc[0]),
                            }
                        else:
                            # Just use the overall mean decay
                            results["adstock_decay"][ch] = {
                                "mean_decay": float(ch_data["mean"].mean()),
                            }
    except Exception as e:
        print(f"Warning: Adstock decay extraction failed: {e}")
        record_section_error(results, "adstock_decay", e)

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
        record_section_error(results, "marginal_roi", e)

    # 5b. Optimal frequency for R&F channels
    if rf_channels:
        print("Extracting optimal frequency for R&F channels...")
        try:
            opt_freq = mmm_analyzer.optimal_freq()
            if opt_freq is not None:
                if hasattr(opt_freq, "numpy"):
                    opt_freq_np = opt_freq.numpy()
                    opt_freq_mean = opt_freq_np.mean(axis=(0, 1))
                    for i, ch in enumerate(rf_channels):
                        results["optimal_frequency"][ch] = float(opt_freq_mean[i])
                elif hasattr(opt_freq, "data_vars"):
                    # xarray Dataset with dims: frequency, rf_channel, metric
                    # and var: optimal_frequency
                    for ch in rf_channels:
                        try:
                            # Select by rf_channel first, then get optimal_frequency scalar
                            ch_data = (
                                opt_freq.sel(rf_channel=ch)
                                if "rf_channel" in opt_freq.dims
                                else opt_freq
                            )
                            val = ch_data["optimal_frequency"]
                            # val may have metric dim or be scalar
                            if val.dims:
                                val = val.isel({d: 0 for d in val.dims})  # take first element
                            results["optimal_frequency"][ch] = float(val.values)
                        except Exception as e:
                            print(f"  Warning: Could not extract optimal freq for {ch}: {e}")
                            record_section_error(results, "optimal_frequency", e)
        except Exception as e:
            print(f"Warning: Optimal frequency extraction failed: {e}")
            record_section_error(results, "optimal_frequency", e)

    # 5c. Organic media contributions
    # Organic channels are not in summary_metrics (which only covers paid media).
    # Instead, extract from the model's expected_outcome or incremental_outcome
    # by looking at the organic media channel dimension.
    if organic_channels:
        print("Extracting organic media contributions...")
        try:
            # Check if the model has organic media data
            organic_media_ch = getattr(input_data, "organic_media_channel", None)
            if organic_media_ch is not None:
                print(
                    f"  Model organic channels: {organic_media_ch.values if hasattr(organic_media_ch, 'values') else organic_media_ch}"
                )
            # Use expected_vs_actual_data which includes all components
            ev = mmm_analyzer.expected_vs_actual_data()
            if ev is not None:
                print(f"  expected_vs_actual: vars={list(ev.data_vars)}, dims={dict(ev.dims)}")
                # Organic contributions show up in the decomposition
                for ch in organic_channels:
                    results["organic_contributions"][ch] = {
                        "included_in_model": True,
                    }
            if not results["organic_contributions"]:
                # Mark as included even without quantified contribution
                for ch in organic_channels:
                    results["organic_contributions"][ch] = {"included_in_model": True}
        except Exception as e:
            print(f"Warning: Organic contribution extraction failed: {e}")
            record_section_error(results, "organic_contributions", e)
            for ch in organic_channels:
                results["organic_contributions"][ch] = {"included_in_model": True}

    # 5d. Non-media treatment effects (from baseline_summary_metrics)
    if treatment_cols:
        print("Extracting treatment effects...")
        try:
            # Treatment effects show up in baseline_summary_metrics
            bs = mmm_analyzer.baseline_summary_metrics()
            if bs is not None:
                for i, tname in enumerate(treatment_names):
                    results["treatment_effects"][tname] = {
                        "name": tname,
                        "column": treatment_cols[i],
                        "included_in_model": True,
                    }
                    # Try to get the treatment effect from the baseline
                    if "baseline_outcome" in bs.data_vars:
                        mean_val = bs["baseline_outcome"].sel(
                            metric="mean", distribution="posterior"
                        )
                        results["treatment_effects"][tname]["baseline_impact"] = float(
                            mean_val.values
                        )
        except Exception as e:
            print(f"Warning: Treatment effects extraction failed: {e}")
            record_section_error(results, "treatment_effects", e)

    # 6. Model fit (R-squared, MAPE) - critical for model quality tracking
    print("Extracting model fit metrics...")
    try:
        accuracy_ds = mmm_analyzer.predictive_accuracy()
        results["model_fit"] = extract_predictive_accuracy(accuracy_ds)
        for metric_name, value in results["model_fit"].items():
            print(f"  {metric_name}: {value:.4f}")

    except Exception as e:
        print(f"Warning: Model fit extraction failed: {e}")
        record_section_error(results, "model_fit", e, required=True)

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
                if "metric" in holdout_accuracy.dims or "metric" in holdout_accuracy.coords:
                    for metric_name in holdout_accuracy.coords.get(
                        "metric", holdout_accuracy.dims.get("metric", [])
                    ).values:
                        metric_str = str(metric_name).lower().replace("_", "")
                        if "rsquared" in metric_str:
                            val = holdout_accuracy.sel(metric=metric_name)["value"].values
                            val_float = float(val.mean()) if val.size > 1 else float(val)
                            results["holdout_validation"]["r_squared"] = val_float
                print(f"  Holdout validation: {results.get('holdout_validation', {})}")
        except Exception as e:
            print(f"Warning: Holdout validation extraction failed: {e}")
            record_section_error(results, "holdout_validation", e)

    # 7. MCMC diagnostics (R-hat)
    print("Extracting MCMC diagnostics...")
    try:
        rhat_df = mmm_analyzer.rhat_summary()
        results["diagnostics"].update(extract_rhat_diagnostics(rhat_df))
    except Exception as e:
        print(f"Warning: Diagnostics extraction failed: {e}")
        record_section_error(results, "diagnostics", e, required=True)
        results["diagnostics"].update(
            {
                "convergence_ok": False,
                "diagnostics_available": False,
                "error": str(e),
            }
        )

    # 8. ModelReviewer (diagnostic checks)
    print("Running ModelReviewer...")
    try:
        from meridian.analysis.review import reviewer

        model_reviewer = reviewer.ModelReviewer(mmm)
        review_result = model_reviewer.run()

        results["model_review"] = serialize_model_review(review_result)
        review_checks = results["model_review"].get("checks", [])
        convergence_check = next(
            (check for check in review_checks if check["name"] == "Convergence"), None
        )
        if convergence_check:
            results["diagnostics"]["model_reviewer_convergence"] = convergence_check["status"]
            if convergence_check["status"] != "PASS":
                results["diagnostics"]["convergence_ok"] = False

        print(f"  ModelReviewer completed: {len(review_checks)} checks")

    except Exception as e:
        print(f"Warning: ModelReviewer failed: {e}")
        results["model_review"] = {}
        record_section_error(results, "model_review", e)

    # 8b. Native Meridian visualizations (generate PNGs on GPU)
    # Meridian 1.4.x visualizer classes: MediaSummary, MediaEffects, ModelFit, ModelDiagnostics
    # They take an Analyzer instance and have specific plot_* method names.
    print("Generating native Meridian charts...")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from meridian.analysis import visualizer

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                chart = plot_fn()
                chart_path = f"{chart_dir}/{chart_name}"
                save_chart(chart, chart_path)
                plt.close("all")
                results["charts"][chart_name.replace(".png", "")] = chart_path
                print(f"  Saved {chart_name}")
            except Exception as e_chart:
                print(f"  Warning: {chart_name} failed: {e_chart}")
                record_section_error(results, "charts", e_chart)

        print(f"  Generated {len(results['charts'])} charts in {chart_dir}")

    except Exception as e:
        print(f"Warning: Native chart generation failed: {e}")
        results["charts"] = {}
        record_section_error(results, "charts", e)

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
                    opt_result = budget_optimizer.optimize(fixed_budget=True, budget=budget)
                    if opt_result is not None:
                        allocation, expected_outcome = extract_optimization_result(opt_result)
                        results["optimization"][name] = {
                            "budget": float(budget),
                            "optimal_allocation": allocation,
                            "expected_outcome": expected_outcome,
                        }
                except Exception as e:
                    print(f"  Warning: Optimization for {name} failed: {e}")
                    record_section_error(results, "optimization", e)

        except Exception as e:
            print(f"Warning: Budget optimization failed: {e}")
            record_section_error(results, "optimization", e)

    print("Results extracted successfully!")
    manifest = finalize_run_manifest(results)
    log_event(
        run_logger,
        "run_completed",
        status=manifest["status"],
        quality_status=manifest["quality_status"],
        errors=len(manifest["errors"]),
    )
    print(f"Run status: {manifest['status']}; model quality: {manifest['quality_status']}")

    # Save to volume
    output_path = f"/outputs/full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
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
    n_adapt: int = 2000,
    n_burnin: int = 500,
    max_lag: int = 8,
    report: bool = False,
    calibration: str = "",  # Path to calibration.json
    holdout_weeks: int = 0,  # Hold out last N weeks for validation (~$0.30 extra GPU)
    allow_population_estimates: bool = False,
    allow_impression_estimates: bool = False,
):
    """
    Run full MMM analysis from command line.

    Example:
        modal run modal_mmm_full.py --data data/examples/sample_data.csv --report
        modal run modal_mmm_full.py --data data/raw/mydata.csv --calibration data/calibration.json
        modal run modal_mmm_full.py --data data/raw/mydata.csv --holdout-weeks 8
        modal run modal_mmm_full.py --data data/raw/mydata.csv --max-lag 12 --n-keep 1000
    """
    import csv
    import io
    import json
    from pathlib import Path

    from mmm.calibration import (
        calculate_channel_priors,
        infer_calibration_metric,
        load_calibration,
    )
    from mmm.observability import configure_run_logger, log_event, new_run_id
    from mmm.result_manifest import finalize_run_manifest, record_section_error

    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data}")

    run_id = new_run_id()
    run_logger = configure_run_logger(run_id, "local")
    log_event(run_logger, "input_read_started", data_path=str(data_path))
    print(f"Reading data from {data_path}...")
    data_csv = data_path.read_text()
    columns = next(csv.reader(io.StringIO(data_csv)))
    calibration_metric = infer_calibration_metric(columns, kpi_column)

    # Load calibration data if provided
    calibration_priors = None
    if calibration:
        calibration_path = Path(calibration)
        if calibration_path.exists():
            print(f"Loading calibration from {calibration_path}...")
            try:
                cal_data = load_calibration(calibration_path)
                calibration_priors = calculate_channel_priors(
                    cal_data, expected_metric=calibration_metric
                )
                print(f"Loaded calibration with {len(calibration_priors)} channel priors")
            except Exception as e:
                raise ValueError(f"Could not load explicit calibration file: {e}") from e
        else:
            print(f"Warning: Calibration file not found: {calibration}")
    else:
        # Check for default calibration file location
        default_cal = Path("data/calibration.json")
        if default_cal.exists():
            print(f"Found default calibration file: {default_cal}")
            try:
                cal_data = load_calibration(default_cal)
                calibration_priors = calculate_channel_priors(
                    cal_data, expected_metric=calibration_metric
                )
                print(f"Loaded calibration with {len(calibration_priors)} channel priors")
            except Exception as e:
                raise ValueError(f"Could not load default calibration file: {e}") from e

    print("Submitting full analysis to Modal GPU...")
    print("(This may take 30-45 minutes)")
    print()
    log_event(run_logger, "remote_submission_started")

    results = fit_mmm_full.remote(
        data_csv=data_csv,
        kpi_column=kpi_column,
        n_chains=n_chains,
        n_keep=n_keep,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
        max_lag=max_lag,
        calibration_priors=calibration_priors,
        holdout_weeks=holdout_weeks,
        allow_population_estimates=allow_population_estimates,
        allow_impression_estimates=allow_impression_estimates,
        run_id=run_id,
    )
    log_event(
        run_logger,
        "remote_submission_completed",
        status=results.get("run_manifest", {}).get("status", "unknown"),
        quality_status=results.get("run_manifest", {}).get("quality_status", "unknown"),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("MMM ANALYSIS RESULTS")
    print("=" * 60)

    roi_is_monetary = results.get("metadata", {}).get("roi_is_monetary", False)
    roi_label = (
        "Channel ROI (Return on Investment)" if roi_is_monetary else "Channel KPI Efficiency"
    )
    roi_suffix = "x" if roi_is_monetary else " KPI/currency"
    print(f"\n## {roi_label}")
    print("-" * 40)
    for ch, data in sorted(results.get("roi", {}).items(), key=lambda x: -x[1].get("mean", 0)):
        mean = data.get("mean", 0)
        ci_lo = data.get("ci_lower", 0)
        ci_hi = data.get("ci_upper", 0)
        print(f"  {ch:12s}: {mean:.2f}{roi_suffix}  (90% CI: {ci_lo:.2f} - {ci_hi:.2f})")

    print("\n## Channel Contributions")
    print("-" * 40)
    for ch, data in sorted(
        results.get("contributions", {}).items(), key=lambda x: -x[1].get("percentage", 0)
    ):
        pct = data.get("percentage", 0)
        print(f"  {ch:12s}: {pct:.1f}%")

    if results.get("cpik"):
        print("\n## CPIK (Cost per Incremental KPI)")
        print("-" * 40)
        for ch, cpik in sorted(results.get("cpik", {}).items(), key=lambda x: x[1]):
            print(f"  {ch:12s}: ${cpik:.2f}")

    marginal_label = (
        "Marginal ROI (ROI at current spend)"
        if roi_is_monetary
        else "Marginal KPI Efficiency (at current spend)"
    )
    print(f"\n## {marginal_label}")
    print("-" * 40)
    for ch, mroi in sorted(results.get("marginal_roi", {}).items(), key=lambda x: -x[1]):
        print(f"  {ch:12s}: {mroi:.2f}{roi_suffix}")

    if results.get("diagnostics", {}).get("convergence_ok"):
        print("\n[OK] Model convergence: Good (all R-hat < 1.1)")
    else:
        warnings = results.get("diagnostics", {}).get("rhat_warnings", 0)
        print(f"\n[!] Model convergence: {warnings} parameters with R-hat > 1.1")

    # Save results
    output_path = (
        Path("outputs")
        / f"full_results_{results['timestamp'].replace(':', '-').replace('.', '-')}.json"
    )
    output_path.parent.mkdir(exist_ok=True)

    # Native chart paths point into the remote Modal Volume. Download them so
    # the local HTML report can actually embed the charts it was promised.
    local_chart_dir = (
        output_path.parent / f"charts_{output_path.stem.removeprefix('full_results_')}"
    )
    downloaded_charts = {}
    for chart_name, remote_path in results.get("charts", {}).items():
        try:
            volume_path = str(remote_path).removeprefix("/outputs/")
            chart_bytes = b"".join(volume.read_file(volume_path))
            local_chart_dir.mkdir(parents=True, exist_ok=True)
            local_path = local_chart_dir / Path(remote_path).name
            local_path.write_bytes(chart_bytes)
            downloaded_charts[chart_name] = str(local_path)
        except Exception as e:
            print(f"Warning: Could not download chart {chart_name}: {e}")
            record_section_error(results, "charts", e)
            log_event(run_logger, "chart_download_failed", chart=chart_name, error=str(e))
    results["charts"] = downloaded_charts
    manifest = finalize_run_manifest(results)

    output_path.write_text(json.dumps(results, indent=2, default=str))
    log_event(
        run_logger,
        "artifacts_written",
        results_path=str(output_path),
        charts=len(downloaded_charts),
    )
    print(f"\nFull results saved to: {output_path}")

    if manifest["status"] == "failed":
        failed_sections = [
            section
            for section, status in manifest["sections"].items()
            if status != "complete" and section in manifest["required_sections"]
        ]
        raise RuntimeError(
            "MMM run did not produce required sections: " + ", ".join(failed_sections)
        )

    # Generate HTML report if requested
    if report:
        print("\nGenerating HTML report...")
        from mmm.analysis.visualize import generate_html_report

        report_path = output_path.with_suffix(".html")
        generate_html_report(results, report_path)
        print(f"Report saved to: {report_path}")

    return results
