"""
Modal GPU runner for Sommmelier.

Usage:
    modal run modal_mmm.py --data data/examples/sample_data.csv
    modal run modal_mmm.py --data data/raw/your_data.csv --output outputs/
"""

import modal

# Define the Modal image with all dependencies
# Let pip resolve version conflicts - Meridian will pull in what it needs
mmm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Install Meridian first - it will pull compatible TFP/JAX
        "google-meridian==1.4.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
    )
    .pip_install(
        # Then add CUDA support for JAX
        "jax[cuda12]",
    )
)

app = modal.App("sommmelier")

# Create a volume to persist model outputs
volume = modal.Volume.from_name("sommmelier-outputs", create_if_missing=True)


@app.function(
    image=mmm_image,
    gpu="T4",  # Cheapest GPU, good for MMM
    timeout=3600,  # 1 hour max
    volumes={"/outputs": volume},
)
def fit_mmm(
    data_csv: str,
    kpi_column: str = "conversions",
    n_chains: int = 4,
    n_keep: int = 500,
) -> dict:
    """
    Fit MMM model on GPU and return results.

    Args:
        data_csv: CSV data as string
        kpi_column: Name of KPI column
        n_chains: Number of MCMC chains
        n_keep: Samples to keep per chain

    Returns:
        Dictionary with ROI, contributions, and diagnostics
    """
    import io
    import json
    import warnings
    from datetime import datetime

    import pandas as pd
    import numpy as np

    # Monkey-patch numpy 2.x compatibility for TFP
    # TFP uses np.reshape(x, newshape=...) but numpy 2.x changed it to shape=
    _original_reshape = np.reshape
    def _patched_reshape(a, *args, newshape=None, shape=None, **kwargs):
        if newshape is not None and shape is None:
            shape = newshape
        if shape is not None:
            return _original_reshape(a, shape, **kwargs)
        return _original_reshape(a, *args, **kwargs)
    np.reshape = _patched_reshape

    warnings.filterwarnings('ignore')

    print(f"Starting MMM fit at {datetime.now()}")
    print(f"GPU available: checking JAX devices...")

    import jax
    print(f"JAX devices: {jax.devices()}")

    # Load data
    df = pd.read_csv(io.StringIO(data_csv), parse_dates=['date'])
    df = df.rename(columns={'date': 'time'})
    print(f"Loaded data: {len(df)} rows, {df['geo'].nunique()} geos, {df['time'].nunique()} periods")

    # Auto-detect channels
    spend_cols = [col for col in df.columns if '_spend' in col.lower()]
    channels = [col.replace('_spend', '').replace('_Spend', '') for col in spend_cols]
    impression_cols = [f"{ch}_impressions" for ch in channels]

    # Check for impressions columns, estimate if missing
    for i, (ch, imp_col) in enumerate(zip(channels, impression_cols)):
        if imp_col not in df.columns:
            df[imp_col] = df[spend_cols[i]] * 100  # Estimate at $10 CPM

    # Add population if missing
    if 'population' not in df.columns:
        pop_map = {'US': 330_000_000, 'UK': 67_000_000, 'AU': 26_000_000}
        df['population'] = df['geo'].map(lambda x: pop_map.get(x, 10_000_000))

    print(f"Channels: {channels}")

    # Build Meridian input
    from meridian.data import data_frame_input_data_builder

    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(
        kpi_type='non_revenue',
        default_kpi_column=kpi_column,
    )

    builder = builder.with_kpi(df)
    builder = builder.with_population(df)
    builder = builder.with_media(
        df,
        media_channels=channels,
        media_cols=impression_cols,
        media_spend_cols=spend_cols,
    )

    input_data = builder.build()
    print("InputData built successfully")

    # Configure model
    from meridian.model import model, spec, prior_distribution
    import tensorflow_probability as tfp

    # Auto-calculate knots
    n_periods = df['time'].nunique()
    if n_periods <= 13:
        knots = [0, n_periods - 1]
    elif n_periods <= 52:
        knots = [0, n_periods // 2, n_periods - 1]
    else:
        knots = list(range(0, n_periods, 13))
        if knots[-1] != n_periods - 1:
            knots.append(n_periods - 1)

    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(0.2, 0.9)
    )

    model_spec = spec.ModelSpec(prior=prior, knots=knots)
    mmm = model.Meridian(input_data=input_data, model_spec=model_spec)
    print("Model initialized")

    # Sample prior
    print("Sampling from prior...")
    mmm.sample_prior(500)

    # Sample posterior (this is the slow part)
    print(f"Sampling posterior with {n_chains} chains, {n_keep} samples each...")
    mmm.sample_posterior(
        n_chains=n_chains,
        n_adapt=2000,
        n_burnin=500,
        n_keep=n_keep,
        seed=0,
    )
    print("Posterior sampling complete!")

    # Extract results
    from meridian.analysis import analyzer

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_time_periods": n_periods,
        "n_geos": df['geo'].nunique(),
        "channels": channels,
        "channel_roi": {},
        "channel_contributions": {},
    }

    try:
        mmm_analyzer = analyzer.Analyzer(mmm)

        # Get ROI - returns tensor of shape (n_samples, n_chains, n_channels)
        roi_tensor = mmm_analyzer.roi(use_posterior=True)
        roi_mean = roi_tensor.numpy().mean(axis=(0, 1))  # Average across samples and chains
        for i, ch in enumerate(channels):
            results["channel_roi"][ch] = float(roi_mean[i])

        # Get incremental outcome (contribution)
        incremental = mmm_analyzer.incremental_outcome(use_posterior=True)
        incremental_mean = incremental.numpy().mean(axis=(0, 1))  # Average
        # Sum across geos and times
        if len(incremental_mean.shape) > 1:
            incremental_sum = incremental_mean.sum(axis=tuple(range(len(incremental_mean.shape) - 1)))
        else:
            incremental_sum = incremental_mean
        total = incremental_sum.sum()
        for i, ch in enumerate(channels):
            results["channel_contributions"][ch] = float(incremental_sum[i] / total) if total > 0 else 0.0

        print("Results extracted successfully")

    except Exception as e:
        import traceback
        print(f"Warning: Could not extract all results: {e}")
        traceback.print_exc()

    # Save to volume
    output_path = f"/outputs/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Commit volume changes
    volume.commit()

    print(f"Completed at {datetime.now()}")
    return results


@app.local_entrypoint()
def main(
    data: str = "data/examples/sample_data.csv",
    kpi_column: str = "conversions",
    n_chains: int = 4,
    n_keep: int = 500,
):
    """
    Run MMM from command line.

    Example:
        modal run modal_mmm.py --data data/examples/sample_data.csv
    """
    import json
    from pathlib import Path

    # Read the data file
    data_path = Path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data}")

    print(f"Reading data from {data_path}...")
    data_csv = data_path.read_text()

    print("Submitting to Modal GPU...")
    print("(This may take 20-30 minutes for a full run)")
    print()

    # Call the remote function
    results = fit_mmm.remote(
        data_csv=data_csv,
        kpi_column=kpi_column,
        n_chains=n_chains,
        n_keep=n_keep,
    )

    # Print results
    print("\n" + "=" * 50)
    print("MMM RESULTS")
    print("=" * 50)

    print("\nChannel ROI:")
    for ch, roi in sorted(results.get("channel_roi", {}).items(), key=lambda x: -x[1]):
        print(f"  {ch}: {roi:.2f}x")

    print("\nChannel Contributions:")
    total = sum(results.get("channel_contributions", {}).values())
    for ch, contrib in sorted(results.get("channel_contributions", {}).items(), key=lambda x: -x[1]):
        pct = contrib / total * 100 if total > 0 else 0
        print(f"  {ch}: {pct:.1f}%")

    # Save locally too
    output_path = Path("outputs") / f"results_{results['timestamp'].replace(':', '-').replace('.', '-')}.json"
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {output_path}")

    return results
