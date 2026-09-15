import joblib
import pandas as pd


def test_artifact_has_expected_structure():
    bundle = joblib.load("pipeline.joblib")
    assert set(bundle.keys()) == {
        "pipeline",
        "feature_columns",
        "season_column",
        "feature_bounds",
        "metadata",
    }
    assert bundle["metadata"]["steps"] == ["percentile", "clf"]
    assert len(bundle["feature_columns"]) == 16
    assert bundle["metadata"]["n_training_rows"] > 1000
    assert "sklearn_version" in bundle["metadata"]
    assert "built_at" in bundle["metadata"]


def test_artifact_can_predict():
    bundle = joblib.load("pipeline.joblib")
    row = {col: bounds["min"] for col, bounds in bundle["feature_bounds"].items()}
    row[bundle["season_column"]] = 2023
    df = pd.DataFrame([row])
    proba = bundle["pipeline"].predict_proba(df)[0]
    assert len(proba) == 2
    assert 0.0 <= proba[1] <= 1.0
