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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
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

# Stats where a LOWER raw value is the better golf outcome. Used both to
# impose monotonic constraints on the classifier (see build_pipeline())
# and to sign the permutation-importance chart the frontend displays.
LOWER_IS_BETTER = {"scoring_avg", "putting_avg", "putts_per_round", "bogey_avoidance_pct"}

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
    # History: this started as a LogisticRegression (see git history for the
    # class_weight/C tuning that fixed probability calibration and pushed
    # its ceiling from 86% to 92% for a realistic dominant stat line). That
    # ceiling turned out to be a genuine property of *linear* models on this
    # data: several of the 16 stats are naturally correlated in the real
    # training data (e.g. GIR% and Scrambling% partially trade off once
    # driving/SG stats are already in the model), which both caps how
    # confident an additive model can get about a realistic profile, and
    # occasionally makes an individual coefficient's sign counterintuitive
    # (e.g. sg_total, which is literally the sum of the other 4 SG stats
    # already in the model, could come out negative).
    #
    # Switched to HistGradientBoostingClassifier with per-feature monotonic
    # constraints (LOWER_IS_BETTER above) to fix both: a nonlinear model can
    # recognize "this whole profile looks elite" as a pattern rather than a
    # strict sum of parts, and the monotonic constraints guarantee every
    # feature's effect points the intuitively-correct direction (no more
    # sign flips), which also lets genuinely great profiles compound toward
    # high confidence instead of fighting a wrong-signed feature.
    #
    # This is a deliberate accuracy/confidence trade-off, not a free win:
    # 5-fold CV AUC drops from 0.763 (tuned LogisticRegression) to 0.742 --
    # tested 6+ configurations (varying depth/learning-rate/regularization,
    # with and without monotonic constraints) and AUC consistently landed
    # ~0.74-0.75 regardless, so this is a real property of gradient
    # boosting on a dataset this size (1,919 rows), not a tuning miss. In
    # exchange, a realistic dominant profile reaches ~99.5% (up from 92%)
    # and a bad-amateur profile is ~0.0%. Accepted deliberately for this
    # project: it's a class assignment meant to be fun to play with, not a
    # production system where the extra AUC would matter more than the
    # more satisfying (and now guaranteed-intuitive-direction) predictions.
    monotonic_cst = [(-1 if c in LOWER_IS_BETTER else 1) for c in FEATURE_COLUMNS]
    return Pipeline(
        [
            (
                "percentile",
                SeasonPercentileTransformer(
                    feature_columns=FEATURE_COLUMNS, season_column=SEASON_COLUMN
                ),
            ),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    max_depth=3,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    learning_rate=0.05,
                    monotonic_cst=monotonic_cst,
                    random_state=42,
                ),
            ),
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

    # HistGradientBoostingClassifier has no .coef_ (it's a tree ensemble,
    # not linear), so "what leads to wins" is expressed as permutation
    # importance -- how much shuffling one column hurts the pipeline's AUC
    # -- signed by the monotonic direction we already imposed on that
    # feature (LOWER_IS_BETTER), so the frontend's existing signed-bar
    # chart keeps working unchanged and every sign is now guaranteed to
    # point the intuitively-correct way (no more sign flips from
    # collinearity, unlike the earlier LogisticRegression version).
    print("Computing permutation importance...")
    importance = permutation_importance(
        pipeline, X, y, n_repeats=20, random_state=42, scoring="roc_auc"
    )
    # Permutation importance can come out slightly negative for a
    # near-zero-importance feature purely from shuffling noise, even though
    # monotonic_cst guarantees the model's true effect for that feature is
    # non-negative in its assigned direction -- clip before signing so the
    # chart never shows a spuriously wrong-signed sliver for a redundant
    # feature (e.g. sg_total, whose info is already captured by its own
    # four components).
    importance_by_col = dict(zip(X.columns, importance.importances_mean))
    coefficients = {
        col: max(0.0, float(importance_by_col[col])) * (-1.0 if col in LOWER_IS_BETTER else 1.0)
        for col in FEATURE_COLUMNS
    }

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
