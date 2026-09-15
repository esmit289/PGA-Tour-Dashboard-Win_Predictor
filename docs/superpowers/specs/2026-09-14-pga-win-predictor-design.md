# PGA Tour Win Predictor — Design Spec

Date: 2026-09-14
Repo: https://github.com/esmit289/PGA-Tour-Dashboard-Win_Predictor.git

## Purpose

Class assignment: build a fitted scikit-learn `Pipeline` with a custom
transformer, serve it via FastAPI, deploy to Modal, call it from a Vercel
frontend, and test it with Postman. Continues the golf/PGA Tour theme from
the existing dashboard project (`Studio 3/Homework 3/pga-tour-dashboard`),
framed as "what leads to tournament wins?"

## Data & target

- Source: the existing Supabase `player_season_stats` table (public anon
  key — the same one already used client-side in the dashboard; no new
  credentials needed). The build script queries it once via the public
  REST endpoint, paginating past the 1000-row default cap.
- Rows: all 2016–2026 player-season rows (~7,500 rows).
- Target: `has_win` (bool) = `wins > 0` for that player-season.
- Features (16 continuous "skill" stats): `scoring_avg`, `driving_distance`,
  `driving_accuracy_pct`, `gir_pct`, `putting_avg`, `putts_per_round`,
  `scrambling_pct`, `sand_save_pct`, `sg_total`, `sg_off_the_tee`,
  `sg_approach`, `sg_around_green`, `sg_putting`, `birdie_avg`,
  `birdie_or_better_pct`, `bogey_avoidance_pct`.
- Deliberately excluded: `wins` (the target itself), `top_10`,
  `official_money`, `fedexcup_*`, `world_rank*`, `all_around_*`,
  `finish_1st/2nd/3rd`, and every `*_rank` column — these are outcomes or
  close proxies of winning, not inputs to it, and including them would make
  the classifier trivially (and uninterestingly) accurate.
- Rows missing more than a handful of the 16 features are dropped; the rest
  keep their nulls, which the custom transformer imputes (see below).

## Custom transformer — `SeasonPercentileTransformer`

File: `pipeline_def.py`. Inherits `BaseEstimator`, `TransformerMixin`.

Rationale: raw stat values aren't comparable across seasons (a 69.5 scoring
average meant something different in 2016 than 2024, and driving distance
has crept up tour-wide over time). The transformer converts each raw stat
into its **percentile rank within that player-season's own season**,
learned from the training data's per-season empirical distributions.

- `__init__(self, feature_columns, season_column="season")` — only assigns
  its arguments, per the assignment's constraint.
- `fit(self, X, y=None)`: `X` is a DataFrame including `season_column` plus
  the feature columns. For each feature column, groups training rows by
  season and stores the sorted array of non-null values
  (`self.season_distributions_[col][season]`), plus a pooled sorted array
  across all seasons as a fallback for a season never seen in training
  (`self.overall_distributions_[col]`). Also stores per-season (and
  overall-fallback) medians for imputing missing values
  (`self.season_medians_`, `self.overall_medians_`). Returns `self`.
- `transform(self, X)`: for each row and each feature column, imputes
  nulls with that row's season median (falling back to the overall
  median), then maps the value to a percentile via `np.searchsorted`
  against the stored sorted array for that row's season (falling back to
  the pooled array for an unseen season). Returns a DataFrame of
  percentiles (0–1), with the season column dropped from the output.

This is genuine learned state: a freshly-constructed
`SeasonPercentileTransformer` has no distributions or medians and cannot
transform correctly — it must be fit first, which is exactly the "wrong if
rebuilt from scratch at boot" property the assignment requires.

## Pipeline

Exactly two steps, no separate scaler needed since percentiles are already
bounded to [0, 1]:

```python
Pipeline([
    ("percentile", SeasonPercentileTransformer(feature_columns=FEATURE_COLUMNS)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
```

`class_weight="balanced"` because win-seasons are a small minority of all
player-seasons.

## Artifact bundle

`build_pipeline.py` fits the pipeline and dumps a dict (not a bare
pipeline) via `joblib.dump`:

```python
{
    "pipeline": fitted_pipeline,
    "feature_columns": FEATURE_COLUMNS,       # order matters for input construction
    "season_column": "season",
    "feature_bounds": {col: {"min": ..., "max": ...} for col in FEATURE_COLUMNS},
    "metadata": {
        "steps": [name for name, _ in fitted_pipeline.steps],
        "built_at": <ISO 8601 timestamp>,
        "sklearn_version": sklearn.__version__,
        "n_training_rows": N,
        "features": FEATURE_COLUMNS,
        "target": "has_win",
        "coefficients": {col: float for col, coef in zip(FEATURE_COLUMNS, clf.coef_[0])},
    },
}
```

`feature_bounds` and `coefficients` let the frontend build sane input
constraints and show "what matters most" without hardcoding either on the
frontend side.

## FastAPI (`serve.py`)

- Loads `pipeline.joblib` once at import into a module-level global. Load
  failure is caught and recorded (not raised) so the process still starts.
- `GET /info` — returns the stored `metadata` plus `feature_bounds`. Returns
  503 if the artifact failed to load.
- `POST /predict` — Pydantic request model with one bounded `float` field
  per feature (bounds pulled from the real training data's min/max, padded
  slightly) plus a bounded `season` int field. Out-of-range or wrong-type
  input is rejected automatically as 422 by FastAPI/Pydantic. Returns
  `{"predicted_win_season": bool, "win_probability": float}`. Returns 503
  (not 500) if the artifact isn't loaded, checked at the top of the
  handler.
- No live Supabase/DB dependency at serve time — the joblib file is fully
  self-contained.

## Modal (`modal_serve.py`)

- Builds a Modal image containing exactly `serve.py`, `pipeline_def.py`,
  and `pipeline.joblib`, with `scikit-learn` pinned to the exact version
  recorded in the artifact's metadata (plus `fastapi`, `pydantic`,
  `pandas`, `numpy`, `joblib`).
- Exposes the FastAPI app via `@modal.asgi_app()`; the FastAPI `app` object
  is imported *inside* the Modal function (not at module scope) per the
  assignment's requirement.
- `modal deploy modal_serve.py` produces the public URL that both the
  frontend and Postman collection target.

## Frontend

New `/win-predictor` page inside the existing dashboard
(`Studio 3/Homework 3/pga-tour-dashboard`, already deployed on Vercel),
linked from the top nav:

- Numeric inputs/sliders for the 16 features, defaulted to tour-average
  values and constrained by the bounds from `GET /info`.
- "Predict" button POSTs to the live Modal URL (stored as
  `NEXT_PUBLIC_WIN_PREDICTOR_API_URL`, an env var — never hardcoded
  localhost, never fake data).
- Shows the returned win probability (visually, e.g. a gauge/bar) and the
  boolean prediction.
- A "what leads to wins" panel built from `GET /info`'s `coefficients`,
  showing which stats push win-probability up or down and by how much —
  ties the tool back to the assignment's framing instead of being a bare
  prediction toy.

## Postman

A committed collection (`postman/pga-win-predictor.postman_collection.json`)
targeting the deployed Modal URL (never localhost), with requests +
assertions for:

1. `GET /info` → 200, response has `metadata` and `feature_bounds`.
2. `POST /predict` with valid, in-bounds input → 200, response has
   `predicted_win_season` (bool) and `win_probability` (0–1 float).
3. `POST /predict` with an out-of-bounds value → 422.

Screenshots of these three calls, plus writing the 3–5 sentence
explanation and the actual Canvas submission, are the user's own work —
not part of this build.

## Repo layout

```
pga-win-predictor/
  pipeline_def.py                 # SeasonPercentileTransformer
  build_pipeline.py               # fetches data, fits pipeline, dumps pipeline.joblib
  pipeline.joblib                 # committed artifact
  serve.py                        # FastAPI app
  modal_serve.py                  # Modal deployment wrapper
  requirements.txt
  postman/
    pga-win-predictor.postman_collection.json
  README.md
```

`pipeline.joblib` is committed directly (small logistic-regression
artifact) alongside `build_pipeline.py` as the rebuild script, satisfying
"artifact or rebuild script" with both.

## Out of scope (explicitly the user's own work)

- Canvas submission itself.
- Postman screenshots (collection is provided; running it and capturing
  screenshots is manual).
- The 3–5 sentence writeup explaining model/transformer choices.
- Creating the GitHub repo (already done:
  https://github.com/esmit289/PGA-Tour-Dashboard-Win_Predictor.git) and
  pushing/zipping it for submission.
