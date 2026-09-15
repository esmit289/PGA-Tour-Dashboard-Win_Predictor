# PGA Tour Win Predictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fitted scikit-learn Pipeline (with a custom, stateful transformer) that predicts whether a PGA Tour player's season stat line looks like a winning season, serve it via FastAPI, deploy it to Modal, call the live Modal API from a new page in the existing PGA Tour dashboard on Vercel, and produce a Postman collection that tests the deployed URL.

**Architecture:** A two-step scikit-learn Pipeline (`SeasonPercentileTransformer` → `LogisticRegression`) is fit once against Supabase's `player_season_stats` table and dumped as a joblib bundle. A FastAPI service loads that bundle once at import and exposes `GET /info` + `POST /predict`. Modal ships the service (plus the pipeline's source file and artifact) as a container and exposes it as a public ASGI app. The existing Next.js dashboard gets a new `/win-predictor` page that calls that live Modal URL directly from the browser.

**Tech Stack:** Python 3.12, scikit-learn, pandas, FastAPI, Pydantic v2, joblib, Modal, pytest, Next.js/React (existing dashboard), Postman collection format v2.1.

**Spec:** `docs/superpowers/specs/2026-09-14-pga-win-predictor-design.md`

## Global Constraints

- Pipeline must have real learned state — wrong/absent if the process were rebuilt fresh without fitting (satisfied by `SeasonPercentileTransformer`'s per-season distributions).
- The joblib artifact is a dict bundle (`pipeline`, `feature_columns`, `season_column`, `feature_bounds`, `metadata`), never a bare pipeline.
- `metadata` must include `steps`, `built_at`, and `sklearn_version`.
- The custom transformer's `__init__` only assigns its arguments — no computation.
- The custom transformer's `fit` must return `self`.
- FastAPI loads the artifact once at import, never per-request.
- Every `POST /predict` field has a Pydantic bound; out-of-bounds/wrong-type input must produce 422.
- A missing/unloadable artifact must produce 503, never 500 or a crash at import.
- The Modal image ships exactly three project files: `serve.py`, `pipeline_def.py`, `pipeline.joblib`.
- scikit-learn is pinned in the Modal image to the exact version recorded in the artifact's metadata (both come from the same `requirements.txt`).
- The FastAPI `app` object is imported *inside* the Modal function, not at module scope.
- The Vercel frontend calls the live Modal URL — never localhost, never mock/fake data.
- The Postman collection targets the live deployed URL — never localhost.
- Out of scope for this plan (the user does these themselves): Canvas submission, Postman screenshots, the 3–5 sentence writeup, pushing/zipping the repo for submission.

---

## File Structure

```
pga-win-predictor/
  pipeline_def.py                          # SeasonPercentileTransformer (Task 2)
  build_pipeline.py                        # fetch data, fit pipeline, dump pipeline.joblib (Task 3)
  pipeline.joblib                          # committed artifact, produced by Task 3
  serve.py                                 # FastAPI app (Task 4)
  modal_serve.py                           # Modal deployment wrapper (Task 6)
  generate_postman_collection.py           # builds the Postman collection from the real artifact + live URL (Task 7)
  requirements.txt                         # pinned deps, single source of truth for local + Modal (Task 1)
  pytest.ini                               # so `import pipeline_def` etc. work from tests/ (Task 1)
  .gitignore                               # (Task 1)
  README.md                                # (Task 1)
  tests/
    test_pipeline_def.py                   # Task 2
    test_build_pipeline.py                 # Task 3
    test_serve.py                          # Task 4
  postman/
    pga-win-predictor.postman_collection.json   # Task 7

pga-tour-dashboard/                        # existing separate repo, only these two files touch it
  src/app/win-predictor/page.tsx           # Task 8
  src/components/nav.tsx                   # Task 8 (add one nav link)
```

---

### Task 1: Project scaffolding

**Files:**
- Create: `pga-win-predictor/.gitignore`
- Create: `pga-win-predictor/pytest.ini`
- Create: `pga-win-predictor/README.md`
- Create: `pga-win-predictor/requirements.txt`

**Interfaces:**
- Produces: an activated `.venv` at `pga-win-predictor/.venv` with `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `pytest`, `httpx` installed; `requirements.txt` pinning their exact resolved versions. Every later task's Python code runs inside this venv.

- [ ] **Step 1: Create the directory layout**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
mkdir -p tests postman
```

- [ ] **Step 2: Create and activate a virtualenv, install dependencies**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pydantic scikit-learn pandas numpy joblib pytest httpx
```

- [ ] **Step 3: Freeze exact versions into requirements.txt**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
pip freeze > requirements.txt
cat requirements.txt
```

Expected: a file with pinned `==` versions for `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `pandas`, `numpy`, `joblib`, `pytest`, `httpx`, and their transitive dependencies. This file is the single source of truth both `pip install -r requirements.txt` locally and Modal's `pip_install_from_requirements("requirements.txt")` (Task 6) will use — that's what keeps the Modal image's scikit-learn version identical to whatever `build_pipeline.py` (Task 3) actually fit with.

- [ ] **Step 4: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
```

- [ ] **Step 5: Write `pytest.ini`**

```ini
[pytest]
pythonpath = .
```

This lets `tests/*.py` do `import pipeline_def`, `import serve`, etc. regardless of which directory `pytest` is invoked from.

- [ ] **Step 6: Write a README stub**

```markdown
# PGA Tour Win Predictor

A fitted scikit-learn Pipeline (with a custom `SeasonPercentileTransformer`)
that predicts whether a PGA Tour player's season stat line looks like a
winning season, served via FastAPI and deployed on Modal. Companion project
to the [PGA Tour dashboard](https://pga-tour-dashboard.vercel.app) — trains
on that project's Supabase `player_season_stats` table.

## Local development

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python3 build_pipeline.py        # fetches data, fits the pipeline, writes pipeline.joblib
    pytest                           # run the test suite
    uvicorn serve:app --reload       # http://localhost:8000/docs

## Deploying

    modal deploy modal_serve.py
    python3 generate_postman_collection.py <the URL modal deploy prints>

## Live URLs

- Modal API: _fill in after `modal deploy`_
- API docs: _Modal URL_ + `/docs`
- Vercel page using this API: https://pga-tour-dashboard.vercel.app/win-predictor
```

- [ ] **Step 7: Verify and commit**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
pip list | grep -Ei "fastapi|scikit-learn|pandas|pydantic|joblib|pytest|httpx"
git add .gitignore pytest.ini README.md requirements.txt
git commit -m "Scaffold PGA Tour win predictor project"
```

Expected: `pip list` shows all the installed packages; `git commit` succeeds.

---

### Task 2: Custom transformer — `SeasonPercentileTransformer`

**Files:**
- Create: `pipeline_def.py`
- Test: `tests/test_pipeline_def.py`

**Interfaces:**
- Produces: `SeasonPercentileTransformer(feature_columns: list[str], season_column: str = "season")`, a scikit-learn-compatible transformer with `.fit(X, y=None) -> self` and `.transform(X) -> pd.DataFrame`. `X` must be (or be convertible via `pd.DataFrame(X)` to) a DataFrame containing `season_column` plus every name in `feature_columns`. Later tasks (Task 3, Task 4) import this as `from pipeline_def import SeasonPercentileTransformer`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline_def.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
pytest tests/test_pipeline_def.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipeline_def'` (the file doesn't exist yet).

- [ ] **Step 3: Write `pipeline_def.py`**

```python
"""Custom scikit-learn transformer for the PGA Tour win predictor pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SeasonPercentileTransformer(BaseEstimator, TransformerMixin):
    """Converts raw stat columns into percentile ranks within each season.

    A raw stat value isn't comparable across seasons (driving distance has
    crept up tour-wide over time, for example), so this transformer learns,
    per season and per feature column, the empirical distribution of values
    seen during fit(). At transform time, each value is mapped to its
    percentile rank against its own season's distribution (falling back to
    the pooled distribution across all seasons for a season never seen
    during fit). Missing values are imputed with that season's median first
    (falling back to the overall median).
    """

    def __init__(self, feature_columns, season_column="season"):
        self.feature_columns = feature_columns
        self.season_column = season_column

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.season_distributions_ = {}
        self.overall_distributions_ = {}
        self.season_medians_ = {}
        self.overall_medians_ = {}

        for col in self.feature_columns:
            self.season_distributions_[col] = {}
            self.season_medians_[col] = {}

            overall_values = X[col].dropna().to_numpy(dtype=float)
            self.overall_distributions_[col] = np.sort(overall_values)
            self.overall_medians_[col] = (
                float(np.median(overall_values)) if len(overall_values) else 0.0
            )

            for season, group in X.groupby(self.season_column):
                values = group[col].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                self.season_distributions_[col][season] = np.sort(values)
                self.season_medians_[col][season] = float(np.median(values))

        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        output = {}

        for col in self.feature_columns:
            season_dists = self.season_distributions_[col]
            season_meds = self.season_medians_[col]
            overall_dist = self.overall_distributions_[col]
            overall_med = self.overall_medians_[col]

            seasons = X[self.season_column].to_numpy()
            values = X[col].to_numpy(dtype=float)
            percentiles = np.empty(len(X), dtype=float)

            for i in range(len(X)):
                season = seasons[i]
                value = values[i]

                if np.isnan(value):
                    value = season_meds.get(season, overall_med)

                dist = season_dists.get(season)
                if dist is None or len(dist) == 0:
                    dist = overall_dist
                percentiles[i] = np.searchsorted(dist, value, side="right") / len(dist)

            output[col] = percentiles

        return pd.DataFrame(output, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_columns)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
pytest tests/test_pipeline_def.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
git add pipeline_def.py tests/test_pipeline_def.py
git commit -m "Add SeasonPercentileTransformer with tests"
```

---

### Task 3: Fetch data and fit the pipeline — `build_pipeline.py`

**Files:**
- Create: `build_pipeline.py`
- Test: `tests/test_build_pipeline.py`

**Interfaces:**
- Consumes: `SeasonPercentileTransformer` from Task 2 (`from pipeline_def import SeasonPercentileTransformer`).
- Produces: `pipeline.joblib` at the repo root, a dict with keys `pipeline`, `feature_columns`, `season_column`, `feature_bounds`, `metadata` (with sub-keys `steps`, `built_at`, `sklearn_version`, `n_training_rows`, `features`, `target`, `coefficients`). Task 4 (`serve.py`), Task 6 (Modal), and Task 7 (Postman generation) all load this exact structure via `joblib.load("pipeline.joblib")`.

- [ ] **Step 1: Write `build_pipeline.py`**

```python
"""Fetches PGA Tour season stats from Supabase, fits the win-predictor
pipeline, and dumps the artifact bundle to pipeline.joblib.

Usage:
    python3 build_pipeline.py
"""
import json
import urllib.request
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline_def import SeasonPercentileTransformer

# Same public, read-only anon key already embedded client-side in the
# pga-tour-dashboard Next.js app -- safe to reuse here, no new credentials.
SUPABASE_URL = "https://clbdjrnvtqvzgcnetqfx.supabase.co"
SUPABASE_ANON_KEY = "sb_publishable_DVuVnzIpFOD7gCTqWT1JKw_mLcQjSp5"

# Deliberately excludes wins/top_10/official_money/fedexcup_*/world_rank*/
# all_around_*/finish_*/*_rank -- those are outcomes (or close proxies) of
# winning, not skill inputs to it.
FEATURE_COLUMNS = [
    "scoring_avg",
    "driving_distance",
    "driving_accuracy_pct",
    "gir_pct",
    "putting_avg",
    "putts_per_round",
    "scrambling_pct",
    "sand_save_pct",
    "sg_total",
    "sg_off_the_tee",
    "sg_approach",
    "sg_around_green",
    "sg_putting",
    "birdie_avg",
    "birdie_or_better_pct",
    "bogey_avoidance_pct",
]
SEASON_COLUMN = "season"
TARGET_COLUMN = "has_win"
MAX_MISSING_FEATURES = 8  # drop a row if more than half its features are null


def fetch_player_season_stats() -> pd.DataFrame:
    columns = ",".join(FEATURE_COLUMNS + [SEASON_COLUMN, "wins"])
    rows = []
    offset = 0
    page_size = 1000
    while True:
        url = (
            f"{SUPABASE_URL}/rest/v1/player_season_stats"
            f"?select={columns}&offset={offset}&limit={page_size}"
        )
        req = urllib.request.Request(
            url,
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
            },
        )
        page = json.loads(urllib.request.urlopen(req, timeout=30).read())
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return pd.DataFrame(rows)


def prepare_training_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df[TARGET_COLUMN] = (df["wins"].fillna(0) > 0).astype(int)
    missing_count = df[FEATURE_COLUMNS].isna().sum(axis=1)
    df = df[missing_count <= MAX_MISSING_FEATURES].reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "percentile",
                SeasonPercentileTransformer(
                    feature_columns=FEATURE_COLUMNS, season_column=SEASON_COLUMN
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def compute_feature_bounds(df: pd.DataFrame) -> dict:
    bounds = {}
    for col in FEATURE_COLUMNS:
        series = df[col].dropna()
        bounds[col] = {"min": float(series.min()), "max": float(series.max())}
    return bounds


def main():
    print("Fetching player_season_stats from Supabase...")
    raw = fetch_player_season_stats()
    print(f"  {len(raw)} rows fetched")

    df = prepare_training_data(raw)
    print(f"  {len(df)} rows after dropping too-sparse rows")
    print(f"  {int(df[TARGET_COLUMN].sum())} win-seasons out of {len(df)}")

    pipeline = build_pipeline()
    X = df[FEATURE_COLUMNS + [SEASON_COLUMN]]
    y = df[TARGET_COLUMN]
    pipeline.fit(X, y)

    clf = pipeline.named_steps["clf"]
    coefficients = dict(zip(FEATURE_COLUMNS, clf.coef_[0].tolist()))

    bundle = {
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "season_column": SEASON_COLUMN,
        "feature_bounds": compute_feature_bounds(df),
        "metadata": {
            "steps": [name for name, _ in pipeline.steps],
            "built_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "n_training_rows": int(len(df)),
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "coefficients": coefficients,
        },
    }

    joblib.dump(bundle, "pipeline.joblib")
    print("Wrote pipeline.joblib")
    print(json.dumps(bundle["metadata"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
python3 build_pipeline.py
```

Expected: prints fetched/kept row counts and a win-season count greater than zero, ends by printing the `metadata` dict, and leaves a `pipeline.joblib` file in the current directory.

- [ ] **Step 3: Write the failing test, then confirm it passes against the real artifact**

Create `tests/test_build_pipeline.py`:

```python
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
```

Run it (this is a "run and confirm" step rather than TDD-red-first, since it verifies output already produced by Step 2):

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
pytest tests/test_build_pipeline.py -v
```

Expected: both tests PASS. If `test_artifact_has_expected_structure` fails on the row-count assertion, re-check the Supabase fetch paginated correctly (Step 2's printed row count should be in the thousands, matching the dashboard's known ~7,500 `player_season_stats` rows).

- [ ] **Step 4: Commit**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
git add build_pipeline.py tests/test_build_pipeline.py pipeline.joblib
git commit -m "Fetch training data and fit the win predictor pipeline"
```

---

### Task 4: FastAPI service — `serve.py`

**Files:**
- Create: `serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `pipeline.joblib` (Task 3) via `joblib.load`; `SeasonPercentileTransformer` (Task 2) must be importable as `pipeline_def.SeasonPercentileTransformer` for that `joblib.load` to succeed (the class is referenced by the pickled pipeline).
- Produces: a FastAPI instance named `app` in `serve.py`, with routes `GET /info` and `POST /predict`. Task 6 (Modal) does `from serve import app`. Module-level globals `_bundle` and `_load_error` are used directly by tests (Task 4) via `monkeypatch.setattr(serve, ...)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_serve.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
pytest tests/test_serve.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'serve'`.

- [ ] **Step 3: Write `serve.py`**

```python
"""FastAPI service for the PGA Tour win predictor pipeline."""
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, create_model

# Required for joblib.load to unpickle the fitted pipeline's custom step.
from pipeline_def import SeasonPercentileTransformer  # noqa: F401

ARTIFACT_PATH = os.environ.get(
    "PIPELINE_ARTIFACT_PATH", str(Path(__file__).parent / "pipeline.joblib")
)


def load_bundle(path: str):
    try:
        return joblib.load(path), None
    except Exception as exc:  # noqa: BLE001 -- any load failure should 503, not crash the process
        return None, str(exc)


_bundle, _load_error = load_bundle(ARTIFACT_PATH)


def _build_predict_model():
    if _bundle is None:
        # No real bounds available; keep the app importable so it still
        # serves a schema and every endpoint can 503 cleanly.
        return create_model("PredictRequest", season=(int, Field(..., ge=2000, le=2100)))
    fields = {
        col: (float, Field(..., ge=bounds["min"], le=bounds["max"]))
        for col, bounds in _bundle["feature_bounds"].items()
    }
    fields["season"] = (int, Field(..., ge=2000, le=2100))
    return create_model("PredictRequest", **fields)


PredictRequest = _build_predict_model()


class PredictResponse(BaseModel):
    predicted_win_season: bool
    win_probability: float


app = FastAPI(title="PGA Tour Win Predictor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_bundle():
    if _bundle is None:
        raise HTTPException(status_code=503, detail=f"Model artifact not loaded: {_load_error}")
    return _bundle


@app.get("/info")
def info():
    bundle = _require_bundle()
    return {
        "status": "ok",
        "metadata": bundle["metadata"],
        "feature_bounds": bundle["feature_bounds"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    bundle = _require_bundle()
    row = request.model_dump()
    df = pd.DataFrame([row])
    proba = float(bundle["pipeline"].predict_proba(df)[0][1])
    return PredictResponse(predicted_win_season=proba >= 0.5, win_probability=proba)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
pytest tests/test_serve.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Run the full test suite together**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
pytest -v
```

Expected: all tests across `test_pipeline_def.py`, `test_build_pipeline.py`, and `test_serve.py` PASS.

- [ ] **Step 6: Commit**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
git add serve.py tests/test_serve.py
git commit -m "Add FastAPI service with /info and /predict"
```

---

### Task 5: Local manual verification checkpoint

**Files:** none (verification only).

**Interfaces:** none — this task only exercises Task 4's `serve.py` over HTTP to catch anything the test suite's in-process `TestClient` might not (e.g. `uvicorn` startup issues, `/docs` rendering).

- [ ] **Step 1: Start the local server**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
uvicorn serve:app --reload
```

Expected: starts without error, logs `Uvicorn running on http://127.0.0.1:8000`.

- [ ] **Step 2: Smoke-test with curl (in a second terminal, or via a background run in this session)**

```bash
curl -s http://localhost:8000/info | python3 -m json.tool
```

Expected: JSON with `metadata` (including `steps`, `built_at`, `sklearn_version`) and `feature_bounds`.

- [ ] **Step 3: Confirm interactive docs render**

Open `http://localhost:8000/docs` in a browser. Expected: Swagger UI listing `GET /info` and `POST /predict`, with the `/predict` request schema showing bounded fields for all 16 features plus `season`.

- [ ] **Step 4: Stop the server**

```bash
# Ctrl+C in the terminal running uvicorn, or:
pkill -f "uvicorn serve:app"
```

No commit for this task — it's a manual checkpoint, not a code change.

---

### Task 6: Deploy to Modal — `modal_serve.py`

**Files:**
- Create: `modal_serve.py`

**Interfaces:**
- Consumes: `requirements.txt` (Task 1) for `pip_install_from_requirements`; `serve.py`, `pipeline_def.py`, `pipeline.joblib` (Tasks 2–4) as the three files shipped into the image.
- Produces: a public Modal URL (printed by `modal deploy`). Task 7 (Postman) and Task 8 (frontend) both consume this exact URL string.

- [ ] **Step 1: Write `modal_serve.py`**

```python
"""Modal deployment wrapper for the PGA Tour win predictor FastAPI service.

Usage:
    modal deploy modal_serve.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("pipeline_def", "serve")
    .add_local_file("pipeline.joblib", "/root/pipeline.joblib")
)

app = modal.App("pga-win-predictor", image=image)


@app.function()
@modal.asgi_app()
def fastapi_app():
    # Imported inside the function, not at module scope, per the
    # assignment's requirement -- this also ensures pipeline_def.py and
    # pipeline.joblib (mounted into the container at /root) are already in
    # place before serve.py tries to import/load them.
    from serve import app as web_app

    return web_app
```

- [ ] **Step 2: Deploy**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
modal deploy modal_serve.py
```

Expected: build logs showing the image installing packages from `requirements.txt`, then a line like:

```
✓ Created objects.
├── 🔨 Created fastapi_app => https://<workspace>--pga-win-predictor-fastapi-app.modal.run
```

Copy that URL — it's needed for every remaining task.

- [ ] **Step 3: Verify the live deployment matches local behavior**

```bash
MODAL_URL="<paste the URL from Step 2>"
curl -s "$MODAL_URL/info" | python3 -m json.tool
curl -s -o /dev/null -w "%{http_code}\n" -X POST "$MODAL_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"season": 2099999}'
```

Expected: the first command returns the same `metadata`/`feature_bounds` shape as the local server did in Task 5. The second command (a deliberately invalid payload — out-of-bounds season and every required feature missing) returns `422`.

- [ ] **Step 4: Commit**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
git add modal_serve.py
git commit -m "Add Modal deployment for the win predictor API"
```

---

### Task 7: Postman collection — `generate_postman_collection.py`

**Files:**
- Create: `generate_postman_collection.py`
- Produces: `postman/pga-win-predictor.postman_collection.json`

**Interfaces:**
- Consumes: `pipeline.joblib` (Task 3, for real `feature_bounds`/`feature_columns`) and the live Modal URL (Task 6's output).
- Produces: a Postman Collection v2.1 JSON file with a `base_url` collection variable set to the real deployed URL, and three requests with `pm.test` assertions. This is the file the user imports into Postman themselves to take screenshots (out of scope for this plan).

- [ ] **Step 1: Write `generate_postman_collection.py`**

```python
"""Generates the Postman collection for the deployed win-predictor API,
using real feature bounds from pipeline.joblib so request bodies are
realistic rather than placeholder numbers.

Usage:
    python3 generate_postman_collection.py https://your-modal-url.modal.run
"""
import json
import sys

import joblib


def build_collection(base_url: str) -> dict:
    bundle = joblib.load("pipeline.joblib")
    bounds = bundle["feature_bounds"]

    valid_body = {col: round((b["min"] + b["max"]) / 2, 3) for col, b in bounds.items()}
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
```

- [ ] **Step 2: Generate the collection using the real deployed URL from Task 6**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
source .venv/bin/activate
python3 generate_postman_collection.py "<the URL from Task 6, Step 2>"
cat postman/pga-win-predictor.postman_collection.json
```

Expected: a well-formed JSON file with three `item` entries and a `base_url` variable set to the real Modal URL (not localhost).

- [ ] **Step 3: Verify the collection actually passes against the live API**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
npx --yes newman run postman/pga-win-predictor.postman_collection.json
```

Expected: Newman's summary shows all assertions passing (0 failures) across the 3 requests / ~7 assertions. This is a self-check before handing the collection to the user for manual screenshotting — it does not replace that manual step.

- [ ] **Step 4: Commit**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
git add generate_postman_collection.py postman/pga-win-predictor.postman_collection.json
git commit -m "Add Postman collection generator and generated collection"
```

---

### Task 8: Frontend page in the existing dashboard

**Files:**
- Create: `pga-tour-dashboard/src/app/win-predictor/page.tsx`
- Modify: `pga-tour-dashboard/src/components/nav.tsx` (add one entry to the `LINKS` array)

Note: `pga-tour-dashboard` here refers to the **existing, separate** repo at `/Users/evansmith/Desktop/Studio 3/Homework 3/pga-tour-dashboard` — not this plan's repo.

**Interfaces:**
- Consumes: the live Modal URL from Task 6 (hardcoded as a constant — see Step 1); `GET /info` and `POST /predict` response shapes from Task 4's `serve.py` (`metadata.features`, `metadata.coefficients`, `metadata.steps`, `metadata.sklearn_version`, `metadata.n_training_rows`, `feature_bounds[col].min/max`, `predicted_win_season`, `win_probability`); `STAT_DESCRIPTIONS` from `@/lib/glossary` and the existing `StatLabel`, `Card`, `Input`, `Button` components already used elsewhere in the dashboard.

- [ ] **Step 1: Create the page**

Create `pga-tour-dashboard/src/app/win-predictor/page.tsx`. Replace `REPLACE_WITH_MODAL_URL` with the exact URL printed by `modal deploy` in Task 6, Step 2 (e.g. `https://esmit289--pga-win-predictor-fastapi-app.modal.run`):

```tsx
"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { StatLabel } from "@/components/stat-label";
import { STAT_DESCRIPTIONS } from "@/lib/glossary";

// Deployed Modal API for the win-predictor pipeline (separate class
// project: github.com/esmit289/PGA-Tour-Dashboard-Win_Predictor).
// Intentionally hardcoded to the real live URL -- never localhost, never
// mock data.
const API_URL = "REPLACE_WITH_MODAL_URL";

interface FeatureBounds {
  min: number;
  max: number;
}

interface InfoResponse {
  metadata: {
    steps: string[];
    built_at: string;
    sklearn_version: string;
    n_training_rows: number;
    features: string[];
    target: string;
    coefficients: Record<string, number>;
  };
  feature_bounds: Record<string, FeatureBounds>;
}

interface PredictResponse {
  predicted_win_season: boolean;
  win_probability: number;
}

const FEATURE_LABELS: Record<string, string> = {
  scoring_avg: "Scoring Average",
  driving_distance: "Driving Distance",
  driving_accuracy_pct: "Driving Accuracy %",
  gir_pct: "Greens in Regulation %",
  putting_avg: "Putting Average",
  putts_per_round: "Putts Per Round",
  scrambling_pct: "Scrambling %",
  sand_save_pct: "Sand Save %",
  sg_total: "SG: Total",
  sg_off_the_tee: "SG: Off-the-Tee",
  sg_approach: "SG: Approach",
  sg_around_green: "SG: Around-the-Green",
  sg_putting: "SG: Putting",
  birdie_avg: "Birdie Average",
  birdie_or_better_pct: "Birdie or Better %",
  bogey_avoidance_pct: "Bogey Avoidance %",
};

export default function WinPredictorPage() {
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [values, setValues] = useState<Record<string, number>>({});
  const [season, setSeason] = useState(2024);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/info`)
      .then((res) => {
        if (!res.ok) throw new Error(`API returned ${res.status}`);
        return res.json();
      })
      .then((data: InfoResponse) => {
        setInfo(data);
        const midpoints: Record<string, number> = {};
        for (const [col, bounds] of Object.entries(data.feature_bounds)) {
          midpoints[col] = Math.round(((bounds.min + bounds.max) / 2) * 1000) / 1000;
        }
        setValues(midpoints);
      })
      .catch((err) => setInfoError(String(err)));
  }, []);

  async function handlePredict() {
    setPredicting(true);
    setPredictError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...values, season }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(
          body?.detail ? JSON.stringify(body.detail) : `API returned ${res.status}`
        );
      }
      setResult(await res.json());
    } catch (err) {
      setPredictError(String(err));
    } finally {
      setPredicting(false);
    }
  }

  const sortedCoefficients = info
    ? Object.entries(info.metadata.coefficients).sort(
        (a, b) => Math.abs(b[1]) - Math.abs(a[1])
      )
    : [];
  const maxAbsCoefficient = sortedCoefficients.length
    ? Math.max(...sortedCoefficients.map(([, v]) => Math.abs(v)))
    : 1;

  return (
    <div className="mx-auto max-w-6xl space-y-8 px-4 py-10 sm:px-6">
      <div>
        <h1 className="text-2xl font-heading font-bold sm:text-3xl">Win Predictor</h1>
        <p className="text-muted-foreground">
          A logistic regression pipeline trained on 2016-2026 PGA Tour season stats predicts
          whether a stat line looks like a winning season. Live model served from Modal.
        </p>
      </div>

      {infoError && (
        <Card className="border-destructive/60">
          <CardContent className="py-4 text-sm text-destructive">
            Couldn&apos;t reach the prediction API: {infoError}
          </CardContent>
        </Card>
      )}

      {info && (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card className="border-border/60">
            <CardHeader>
              <CardTitle>Enter a season stat line</CardTitle>
              <CardDescription>
                Defaults are the midpoint of each stat&apos;s real range in the training data.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <StatLabel
                  label="Season"
                  description="Season year, used to compare this stat line against its own season."
                />
                <Input
                  type="number"
                  value={season}
                  min={2000}
                  max={2100}
                  onChange={(e) => setSeason(Number(e.target.value))}
                />
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {info.metadata.features.map((col) => (
                  <div key={col}>
                    <StatLabel
                      label={FEATURE_LABELS[col] ?? col}
                      description={STAT_DESCRIPTIONS[col as keyof typeof STAT_DESCRIPTIONS]}
                    />
                    <Input
                      type="number"
                      step="any"
                      value={values[col] ?? ""}
                      min={info.feature_bounds[col].min}
                      max={info.feature_bounds[col].max}
                      onChange={(e) =>
                        setValues((prev) => ({ ...prev, [col]: Number(e.target.value) }))
                      }
                    />
                  </div>
                ))}
              </div>
              <Button onClick={handlePredict} disabled={predicting} className="w-full">
                {predicting ? "Predicting..." : "Predict"}
              </Button>
              {predictError && <p className="text-sm text-destructive">{predictError}</p>}
              {result && (
                <div className="rounded-lg border border-border/60 bg-secondary/40 p-4 text-center">
                  <p className="font-heading text-3xl font-black text-accent">
                    {Math.round(result.win_probability * 100)}%
                  </p>
                  <p className="text-sm text-muted-foreground">
                    predicted chance of a win-season
                    {result.predicted_win_season ? " — looks like a winner" : " — not quite there"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="border-border/60">
            <CardHeader>
              <CardTitle>What leads to tournament wins</CardTitle>
              <CardDescription>
                Logistic regression coefficients: which season-relative stat percentiles push
                win-probability up (accent) or down (muted), and by how much.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {sortedCoefficients.map(([col, coef]) => (
                <div key={col} className="flex items-center gap-2 text-sm">
                  <span className="w-40 shrink-0 truncate">{FEATURE_LABELS[col] ?? col}</span>
                  <div className="relative h-4 flex-1 overflow-hidden rounded-sm bg-secondary/60">
                    <div
                      className={
                        coef >= 0
                          ? "absolute inset-y-0 left-1/2 bg-accent"
                          : "absolute inset-y-0 right-1/2 bg-muted-foreground/50"
                      }
                      style={{ width: `${(Math.abs(coef) / maxAbsCoefficient) * 50}%` }}
                    />
                    <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
                  </div>
                  <span className="w-16 shrink-0 text-right text-xs text-muted-foreground">
                    {coef >= 0 ? "+" : ""}
                    {coef.toFixed(2)}
                  </span>
                </div>
              ))}
              <p className="pt-2 text-xs text-muted-foreground">
                Model: {info.metadata.steps.join(" → ")} · scikit-learn {info.metadata.sklearn_version} ·
                trained on {info.metadata.n_training_rows} player-seasons
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Add the nav link**

In `pga-tour-dashboard/src/components/nav.tsx`, the `LINKS` array currently reads:

```tsx
const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/leaderboard", label: "Leaderboard" },
  { href: "/players", label: "Players" },
  { href: "/compare", label: "Compare" },
  { href: "/glossary", label: "Glossary" },
];
```

Change it to:

```tsx
const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/leaderboard", label: "Leaderboard" },
  { href: "/players", label: "Players" },
  { href: "/compare", label: "Compare" },
  { href: "/win-predictor", label: "Win Predictor" },
  { href: "/glossary", label: "Glossary" },
];
```

- [ ] **Step 3: Type-check and lint**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 3/pga-tour-dashboard"
npx tsc --noEmit
npx eslint src/app/win-predictor/page.tsx src/components/nav.tsx
```

Expected: no errors from either command.

- [ ] **Step 4: Manual local verification**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 3/pga-tour-dashboard"
lsof -ti:3000,3001 2>/dev/null | xargs -r kill -9
npm run dev &
sleep 5
curl -s http://localhost:3000/win-predictor | grep -o "Win Predictor" | head -1
```

Expected: prints `Win Predictor` (confirms the page renders without a server error). Since the page's data fetch happens client-side against the live Modal URL, the curl check only confirms the page shell renders — open `http://localhost:3000/win-predictor` in an actual browser to confirm the form loads real bounds and a prediction round-trips successfully end to end. Stop the dev server afterward:

```bash
pkill -f "next dev"
```

- [ ] **Step 5: Commit (in the dashboard repo)**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 3/pga-tour-dashboard"
git add src/app/win-predictor/page.tsx src/components/nav.tsx
git commit -m "Add Win Predictor page calling the deployed Modal API"
```

---

### Task 9: Deploy the frontend and verify end to end

**Files:** none (deploy + verification only).

**Interfaces:** none — final integration check across every earlier task's output.

- [ ] **Step 1: Push the dashboard repo**

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 3/pga-tour-dashboard"
git push
```

- [ ] **Step 2: Wait for the Vercel deploy and confirm it's live**

```bash
for i in $(seq 1 40); do
  if curl -s "https://pga-tour-dashboard.vercel.app/win-predictor" | grep -q "Win Predictor"; then
    echo "LIVE"
    break
  fi
  sleep 10
done
```

Expected: prints `LIVE` within a few minutes.

- [ ] **Step 3: End-to-end confirmation**

Open `https://pga-tour-dashboard.vercel.app/win-predictor` in a browser. Expected: the form loads with real midpoint defaults (proving it successfully called the live Modal `/info` endpoint from the browser, not localhost), and clicking Predict shows a win-probability percentage (proving the live `/predict` round trip and CORS both work from a real browser origin, not just curl).

- [ ] **Step 4: Update the README with the live URLs**

In `pga-win-predictor/README.md` (Task 1), replace the `## Live URLs` section's placeholders with the real values now known:

```markdown
## Live URLs

- Modal API: <the URL from Task 6, Step 2>
- API docs: <the URL from Task 6, Step 2>/docs
- Vercel page using this API: https://pga-tour-dashboard.vercel.app/win-predictor
```

```bash
cd "/Users/evansmith/Desktop/Studio 3/Homework 4/pga-win-predictor"
git add README.md
git commit -m "Record live deployment URLs"
```

No push is done automatically for this repo — the user said they'll handle pushing/zipping it for submission themselves.

---

## What the user still needs to do (explicitly not part of this plan)

1. Push `pga-win-predictor` to `https://github.com/esmit289/PGA-Tour-Dashboard-Win_Predictor.git` (or zip it) for Canvas submission.
2. Import `postman/pga-win-predictor.postman_collection.json` into the Postman app and screenshot the valid (200) and invalid (422) calls.
3. Write the 3–5 sentence explanation of model choice, custom transformer, and scikit-learn version for Canvas.
4. Submit on Canvas: Modal API URL, `/docs` URL, the Vercel URL, Postman screenshots, the writeup, and the repo/zip.
