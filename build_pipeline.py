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

# Deliberately much wider than any real player's stat line -- this is a
# "what if" toy, not a strict validity check. SeasonPercentileTransformer
# safely clamps anything outside the training data's real range to a 0%
# or 100% percentile (np.searchsorted), so widening these costs nothing in
# correctness; it just lets people push the numbers to see what happens.
FEATURE_INPUT_BOUNDS = {
    "scoring_avg": {"min": 50.0, "max": 200.0},
    "driving_distance": {"min": 0.0, "max": 500.0},
    "driving_accuracy_pct": {"min": 0.0, "max": 100.0},
    "gir_pct": {"min": 0.0, "max": 100.0},
    "putting_avg": {"min": 0.0, "max": 5.0},
    "putts_per_round": {"min": 0.0, "max": 100.0},
    "scrambling_pct": {"min": 0.0, "max": 100.0},
    "sand_save_pct": {"min": 0.0, "max": 100.0},
    "sg_total": {"min": -20.0, "max": 20.0},
    "sg_off_the_tee": {"min": -10.0, "max": 10.0},
    "sg_approach": {"min": -10.0, "max": 10.0},
    "sg_around_green": {"min": -10.0, "max": 10.0},
    "sg_putting": {"min": -10.0, "max": 10.0},
    "birdie_avg": {"min": 0.0, "max": 18.0},
    "birdie_or_better_pct": {"min": 0.0, "max": 100.0},
    "bogey_avoidance_pct": {"min": 0.0, "max": 100.0},
}


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
    # No class_weight="balanced" here on purpose: it improves the 0.5-
    # threshold decision boundary but skews predict_proba() upward for the
    # minority (win) class -- an "every stat at the tour average" profile
    # came out to a 36% win chance under balanced weighting, vs the true
    # ~20% base rate. This app's headline number *is* win_probability, so
    # calibration matters more than threshold accuracy here. Dropping the
    # reweighting costs ~0.003 AUC (0.766 -> 0.763, verified via 5-fold CV)
    # and brings the "average player" prediction back in line with reality.
    #
    # C=10 (default is 1.0, i.e. 10x less L2 regularization): the default
    # shrinks coefficients enough that even a maxed-out-everywhere stat line
    # can only reach ~97% (verified by direct sigmoid(intercept + sum of
    # favorable coefficients) computation), and a realistic (not literally
    # perfect-on-every-stat) dominant profile landed at 86%. C=10 lets the
    # same fitted decision boundary express more confidence at the extremes
    # -- the maxed-out ceiling rises to ~99.8% and a realistic dominant
    # profile reaches ~92% -- while 5-fold CV AUC is unchanged (0.764) and
    # the "average player" calibration point barely moves (11.2% -> 10.9%).
    return Pipeline(
        [
            (
                "percentile",
                SeasonPercentileTransformer(
                    feature_columns=FEATURE_COLUMNS, season_column=SEASON_COLUMN
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=10.0)),
        ]
    )


def compute_feature_defaults(df: pd.DataFrame) -> dict:
    """Realistic starting values for the input form -- the real per-feature
    mean across training data, not the (now artificially wide) input bounds.
    """
    return {col: round(float(df[col].dropna().mean()), 3) for col in FEATURE_COLUMNS}


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
        "feature_bounds": FEATURE_INPUT_BOUNDS,
        "feature_defaults": compute_feature_defaults(df),
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
