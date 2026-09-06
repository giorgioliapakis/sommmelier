"""Exercise production preflight against reproducible synthetic scenarios."""

import numpy as np
import pandas as pd
import pytest

from evals.synthetic import INVALID_SCENARIOS, SCENARIOS, generate_scenario
from mmm.preflight import preflight_dataframe


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synthetic_scenario_preflight_contract(scenario):
    frame, truth = generate_scenario(scenario)
    if scenario in INVALID_SCENARIOS:
        with pytest.raises(ValueError, match="preflight failed"):
            preflight_dataframe(frame)
    else:
        dataset = preflight_dataframe(frame)
        assert dataset.n_geos == truth["n_geos"]
        assert dataset.n_time_periods == truth["n_weeks"]
        assert set(dataset.media_channels) == set(truth["true_roi"])


def test_synthetic_seed_and_ground_truth():
    frame, truth = generate_scenario("zero_effect", seed=7)
    repeated, repeated_truth = generate_scenario("zero_effect", seed=7)
    pd.testing.assert_frame_equal(frame, repeated)
    assert truth == repeated_truth
    assert truth["true_roi"]["video"] == 0
    assert truth["true_incremental_kpi"]["video"] == 0
    other, _ = generate_scenario("zero_effect", seed=8)
    assert not frame.equals(other)
    for channel, value in truth["true_roi"].items():
        assert value == pytest.approx(
            truth["true_incremental_kpi"][channel] * 20 / frame[f"{channel}_spend"].sum()
        )


def test_correlated_scenario_is_deliberately_unidentifiable():
    frame, _ = generate_scenario("correlated")
    np.testing.assert_array_equal(frame.meta_spend, frame.search_spend)
