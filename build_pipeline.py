"""Fetches PGA Tour season stats from Supabase, fits the win-predictor
pipeline, and dumps the artifact bundle to pipeline.joblib.

Usage:
    python3 build_pipeline.py
"""
import json
import urllib.parse
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
# winning, not skill inputs to it. Also excludes Strokes Gained (sg_total
# and its 4 components) and birdie_avg (redundant with birdie_or_better_pct)
# in favor of 6 stats a recreational golfer can self-track from their own
# scorecard -- see EXTENDED_STAT_CONFIG below for where those come from.
FEATURE_COLUMNS = [
    "scoring_avg",
    "driving_distance",
    "driving_accuracy_pct",
    "gir_pct",
    "putting_avg",
    "putts_per_round",
    "scrambling_pct",
    "sand_save_pct",
    "par3_scoring_avg",
    "par4_scoring_avg",
    "par5_scoring_avg",
    "three_putt_avoidance",
    "bounce_back",
    "birdie_to_bogey_ratio",
    "birdie_or_better_pct",
    "bogey_avoidance_pct",
]
SEASON_COLUMN = "season"
TARGET_COLUMN = "has_win"
MAX_MISSING_FEATURES = 8  # drop a row if more than half its features are null

# The amateur-friendly replacements for Strokes Gained live in the
# long-format player_extended_stats table (season, player_id, stat_key,
# stat_name, numeric_value). Each stat_key bundles several stat_name
# sub-metrics -- e.g. par4_scoring_avg also has "Total Strokes" and "Total
# Holes" rows alongside the actual "Avg" we want -- so this pins down which
# stat_name is the real per-hole/per-round metric for each key.
EXTENDED_STAT_CONFIG = {
    "par3_scoring_avg": "Avg",
    "par4_scoring_avg": "Avg",
    "par5_scoring_avg": "Avg",
    "three_putt_avoidance": "%",
    "bounce_back": "%",
    "birdie_to_bogey_ratio": "Birdie to Bogey Ratio",
}

# Stats where a LOWER raw value is the better golf outcome. Used both to
# impose monotonic constraints on the classifier (see build_pipeline())
# and to sign the permutation-importance chart the frontend displays.
LOWER_IS_BETTER = {
    "scoring_avg",
    "putting_avg",
    "putts_per_round",
    "bogey_avoidance_pct",
    "par3_scoring_avg",
    "par4_scoring_avg",
    "par5_scoring_avg",
    "three_putt_avoidance",
}

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
    "par3_scoring_avg": {"min": 1.0, "max": 10.0},
    "par4_scoring_avg": {"min": 2.0, "max": 15.0},
    "par5_scoring_avg": {"min": 3.0, "max": 18.0},
    "three_putt_avoidance": {"min": 0.0, "max": 100.0},
    "bounce_back": {"min": 0.0, "max": 100.0},
    "birdie_to_bogey_ratio": {"min": 0.0, "max": 10.0},
    "birdie_or_better_pct": {"min": 0.0, "max": 100.0},
    "bogey_avoidance_pct": {"min": 0.0, "max": 100.0},
}


def fetch_player_season_stats() -> pd.DataFrame:
    core_columns = [c for c in FEATURE_COLUMNS if c not in EXTENDED_STAT_CONFIG]
    columns = ",".join(core_columns + [SEASON_COLUMN, "player_id", "wins"])
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


def fetch_extended_stats() -> pd.DataFrame:
    """Fetches and pivots the 6 amateur-friendly replacement stats out of
    the long-format player_extended_stats table into one row per
    (season, player_id), one column per EXTENDED_STAT_CONFIG key.
    """
    frames = []
    for stat_key, stat_name in EXTENDED_STAT_CONFIG.items():
        rows = []
        offset = 0
        page_size = 1000
        while True:
            url = (
                f"{SUPABASE_URL}/rest/v1/player_extended_stats"
                f"?select=season,player_id,numeric_value"
                f"&stat_key=eq.{stat_key}"
                f"&stat_name=eq.{urllib.parse.quote(stat_name)}"
                f"&offset={offset}&limit={page_size}"
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
        frame = pd.DataFrame(rows).rename(columns={"numeric_value": stat_key})
        frames.append(frame[[SEASON_COLUMN, "player_id", stat_key]])

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=[SEASON_COLUMN, "player_id"], how="outer")
    return merged


def prepare_training_data(raw: pd.DataFrame, extended: pd.DataFrame) -> pd.DataFrame:
    df = raw.merge(extended, on=[SEASON_COLUMN, "player_id"], how="left")
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
    # driving stats are already in the model), which both caps how
    # confident an additive model can get about a realistic profile, and
    # occasionally makes an individual coefficient's sign counterintuitive
    # (at the time, sg_total -- literally the sum of the other 4 Strokes
    # Gained stats then in the model -- could come out negative; the SG
    # stats and birdie_avg have since been replaced with amateur-trackable
    # equivalents -- see EXTENDED_STAT_CONFIG -- since Strokes Gained needs
    # tournament-grade ShotLink tracking a recreational golfer can't produce
    # about their own game).
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

    print("Fetching extended stats (par-type scoring, 3-putt avoidance, bounce back, birdie:bogey ratio)...")
    extended = fetch_extended_stats()
    print(f"  {len(extended)} player-seasons with extended stats")

    df = prepare_training_data(raw, extended)
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
