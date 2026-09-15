import numpy as np
import pandas as pd
import pytest

from pipeline_def import SeasonPercentileTransformer


def make_training_frame():
    return pd.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021, 2021],
            "scoring_avg": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        }
    )


def test_fit_returns_self():
    transformer = SeasonPercentileTransformer(feature_columns=["scoring_avg"])
    result = transformer.fit(make_training_frame())
    assert result is transformer


def test_transform_before_fit_has_no_learned_state():
    transformer = SeasonPercentileTransformer(feature_columns=["scoring_avg"])
    assert not hasattr(transformer, "season_distributions_")


def test_percentile_within_known_season():
    transformer = SeasonPercentileTransformer(feature_columns=["scoring_avg"]).fit(
        make_training_frame()
    )
    row = pd.DataFrame({"season": [2020], "scoring_avg": [20.0]})
    result = transformer.transform(row)
    assert result["scoring_avg"].iloc[0] == pytest.approx(2 / 3)


def test_missing_value_imputed_with_season_median():
    transformer = SeasonPercentileTransformer(feature_columns=["scoring_avg"]).fit(
        make_training_frame()
    )
    row_with_nan = pd.DataFrame({"season": [2020], "scoring_avg": [np.nan]})
    row_with_median = pd.DataFrame({"season": [2020], "scoring_avg": [20.0]})
    result_nan = transformer.transform(row_with_nan)
    result_median = transformer.transform(row_with_median)
    assert result_nan["scoring_avg"].iloc[0] == pytest.approx(
        result_median["scoring_avg"].iloc[0]
    )


def test_unseen_season_falls_back_to_overall_distribution():
    transformer = SeasonPercentileTransformer(feature_columns=["scoring_avg"]).fit(
        make_training_frame()
    )
    row = pd.DataFrame({"season": [2099], "scoring_avg": [20.0]})
    result = transformer.transform(row)
    overall_sorted = np.sort([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
    expected = np.searchsorted(overall_sorted, 20.0, side="right") / len(overall_sorted)
    assert result["scoring_avg"].iloc[0] == pytest.approx(expected)


def test_percentiles_are_bounded_between_0_and_1():
    transformer = SeasonPercentileTransformer(feature_columns=["scoring_avg"]).fit(
        make_training_frame()
    )
    rows = pd.DataFrame({"season": [2020, 2020, 2021], "scoring_avg": [5.0, 30.0, 500.0]})
    result = transformer.transform(rows)
    assert (result["scoring_avg"] >= 0).all()
    assert (result["scoring_avg"] <= 1).all()


def test_init_only_assigns_arguments():
    transformer = SeasonPercentileTransformer(feature_columns=["a", "b"], season_column="yr")
    assert transformer.feature_columns == ["a", "b"]
    assert transformer.season_column == "yr"
    assert not hasattr(transformer, "season_distributions_")
