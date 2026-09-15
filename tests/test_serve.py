from fastapi.testclient import TestClient

import serve

client = TestClient(serve.app)


def test_info_returns_metadata():
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert "metadata" in body
    assert "feature_bounds" in body
    assert body["metadata"]["steps"] == ["percentile", "clf"]


def test_predict_valid_input_returns_200():
    bundle = serve._bundle
    payload = {col: bounds["min"] for col, bounds in bundle["feature_bounds"].items()}
    payload["season"] = 2023
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["predicted_win_season"], bool)
    assert 0.0 <= body["win_probability"] <= 1.0


def test_predict_out_of_bounds_returns_422():
    bundle = serve._bundle
    payload = {col: bounds["min"] for col, bounds in bundle["feature_bounds"].items()}
    payload["season"] = 2023
    first_feature = bundle["feature_columns"][0]
    payload[first_feature] = bundle["feature_bounds"][first_feature]["max"] + 1000
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_field_returns_422():
    bundle = serve._bundle
    payload = {col: bounds["min"] for col, bounds in bundle["feature_bounds"].items()}
    payload["season"] = 2023
    del payload[bundle["feature_columns"][0]]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_artifact_returns_503_on_info(monkeypatch):
    monkeypatch.setattr(serve, "_bundle", None)
    monkeypatch.setattr(serve, "_load_error", "simulated failure")
    response = client.get("/info")
    assert response.status_code == 503


def test_missing_artifact_returns_503_on_predict(monkeypatch):
    monkeypatch.setattr(serve, "_bundle", None)
    monkeypatch.setattr(serve, "_load_error", "simulated failure")
    bundle = joblib_bundle_for_valid_payload()
    payload = {col: bounds["min"] for col, bounds in bundle["feature_bounds"].items()}
    payload["season"] = 2023
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


def joblib_bundle_for_valid_payload():
    import joblib

    return joblib.load("pipeline.joblib")
