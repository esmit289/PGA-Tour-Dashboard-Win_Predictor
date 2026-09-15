"""Generates the Postman collection for the deployed win-predictor API,
using real feature bounds from pipeline.joblib so request bodies are
realistic rather than placeholder numbers.

Usage:
    python3 generate_postman_collection.py https://your-modal-url.modal.run
"""
import json
import sys

import joblib


LOWER_IS_BETTER = {"scoring_avg", "putting_avg", "putts_per_round", "bogey_avoidance_pct"}


def near_best_value(col, b):
    span = b["max"] - b["min"]
    if col in LOWER_IS_BETTER:
        return round(b["min"] + 0.10 * span, 3)
    return round(b["min"] + 0.90 * span, 3)


def build_collection(base_url: str) -> dict:
    bundle = joblib.load("pipeline.joblib")
    bounds = bundle["feature_bounds"]

    valid_body = {col: near_best_value(col, b) for col, b in bounds.items()}
    valid_body["season"] = 2024

    invalid_body = dict(valid_body)
    first_feature = bundle["feature_columns"][0]
    invalid_body[first_feature] = bounds[first_feature]["max"] + 1000

    return {
        "info": {
            "name": "PGA Tour Win Predictor",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "variable": [{"key": "base_url", "value": base_url}],
        "item": [
            {
                "name": "GET /info (health + artifact description)",
                "request": {"method": "GET", "url": "{{base_url}}/info"},
                "event": [
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.test('status is 200', function () {",
                                "    pm.response.to.have.status(200);",
                                "});",
                                "pm.test('has metadata and feature_bounds', function () {",
                                "    const json = pm.response.json();",
                                "    pm.expect(json).to.have.property('metadata');",
                                "    pm.expect(json).to.have.property('feature_bounds');",
                                "    pm.expect(json.metadata).to.have.property('sklearn_version');",
                                "    pm.expect(json.metadata).to.have.property('built_at');",
                                "    pm.expect(json.metadata).to.have.property('steps');",
                                "});",
                            ],
                        },
                    }
                ],
            },
            {
                "name": "POST /predict (valid input -> 200)",
                "request": {
                    "method": "POST",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": {"mode": "raw", "raw": json.dumps(valid_body, indent=2)},
                    "url": "{{base_url}}/predict",
                },
                "event": [
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.test('status is 200', function () {",
                                "    pm.response.to.have.status(200);",
                                "});",
                                "pm.test('response has prediction fields', function () {",
                                "    const json = pm.response.json();",
                                "    pm.expect(json).to.have.property('predicted_win_season');",
                                "    pm.expect(json).to.have.property('win_probability');",
                                "    pm.expect(json.win_probability).to.be.within(0, 1);",
                                "});",
                            ],
                        },
                    }
                ],
            },
            {
                "name": "POST /predict (out-of-bounds input -> 422)",
                "request": {
                    "method": "POST",
                    "header": [{"key": "Content-Type", "value": "application/json"}],
                    "body": {"mode": "raw", "raw": json.dumps(invalid_body, indent=2)},
                    "url": "{{base_url}}/predict",
                },
                "event": [
                    {
                        "listen": "test",
                        "script": {
                            "type": "text/javascript",
                            "exec": [
                                "pm.test('status is 422', function () {",
                                "    pm.response.to.have.status(422);",
                                "});",
                            ],
                        },
                    }
                ],
            },
        ],
    }


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 generate_postman_collection.py <base_url>")
    base_url = sys.argv[1].rstrip("/")
    collection = build_collection(base_url)
    with open("postman/pga-win-predictor.postman_collection.json", "w") as f:
        json.dump(collection, f, indent=2)
    print("Wrote postman/pga-win-predictor.postman_collection.json")


if __name__ == "__main__":
    main()
