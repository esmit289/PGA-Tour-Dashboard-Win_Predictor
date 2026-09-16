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

    pip install modal              # deliberately not in requirements.txt, since that
                                    # file also serves as the Modal image's own dependency manifest
    modal deploy modal_serve.py
    python3 generate_postman_collection.py <the URL modal deploy prints>

## Live URLs

- Modal API: https://esmit289--pga-win-predictor-fastapi-app.modal.run
- API docs: https://esmit289--pga-win-predictor-fastapi-app.modal.run/docs
- Vercel page using this API: https://pga-tour-dashboard.vercel.app/win-predictor

## Model performance

The classifier is a `HistGradientBoostingClassifier` with per-feature
monotonic constraints (not the more common choice of a plain
`LogisticRegression` -- see "Why gradient boosting" below for the full
story of that switch).

5-fold stratified cross-validation on the 1,973 player-seasons used for
training (19.2% of which included at least one win):

- ROC AUC: 0.747
- Balanced accuracy: 0.629

### Why gradient boosting, not logistic regression

This project started with a tuned `LogisticRegression` (`class_weight=None`,
`C=50` -- see git history for that tuning story: fixing an inflated-
probability calibration bug, then finding a real, non-adversarial dominant
stat line's predicted win chance plateaued at ~92% no matter how much
further the regularization was loosened). That 92% ceiling turned out to be
a genuine property of *linear* models on this data: several of the 16 stats
are naturally correlated in the real training data (e.g. GIR% and
Scrambling% partially trade off once driving/SG stats are already in the
model), which caps how confident an additive model can get about a
realistic profile, and occasionally made an individual coefficient's sign
counterintuitive (`sg_total`, the arithmetic sum of the other 4 SG stats
already in the model, could come out negative).

Switching to gradient-boosted trees with monotonic constraints
(`LOWER_IS_BETTER` in `build_pipeline.py`, one `+1`/`-1` per feature based
on golf domain knowledge) fixes both: a nonlinear model can recognize "this
whole profile looks elite" as a pattern instead of a strict sum of parts,
and the monotonic constraints guarantee every feature's effect points the
intuitively-correct direction, so genuinely great profiles compound toward
high confidence instead of fighting a wrong-signed feature.

This is a deliberate trade-off, not a free win: 5-fold CV AUC drops from
0.763 (tuned logistic regression) to 0.742. Six-plus gradient-boosting
configurations were tested (varying tree depth, learning rate, L2
regularization, with and without monotonic constraints) and AUC
consistently landed around 0.74-0.75 regardless -- a real property of
gradient boosting on a dataset this size (1,919 rows), not a tuning miss.
In exchange, a realistic dominant profile's predicted win chance rises from
~92% to ~99.5%, and a bad-amateur profile is ~0.0%. Accepted deliberately
for this project: it's a class assignment meant to be fun to play with, not
a production system where the extra AUC would matter more than the more
satisfying (and now guaranteed-intuitive-direction) predictions.

### Why not Strokes Gained

The original 16 features included Strokes Gained (`sg_total` and its 4
components: off-the-tee, approach, around-the-green, putting). SG compares
every shot to the field average that round, which requires tournament-grade
shot tracking (ShotLink) -- data only the PGA Tour itself can produce. A
recreational golfer can never generate an SG number about their own game,
no matter how good they are, which undercuts the "plug in your own stats
and see" premise of this tool.

The 5 SG stats (plus `birdie_avg`, redundant with `birdie_or_better_pct`)
were replaced with 6 stats any golfer can compute from their own scorecard,
sourced from the same PGA Tour dataset's extended stats table:

| Replaced | With | Why |
|---|---|---|
| `sg_total` | `birdie_to_bogey_ratio` | One number summarizing a round's quality -- count birdies, count bogeys, divide. |
| `sg_off_the_tee` | `par5_scoring_avg` | Reaching a par 5 in two strokes rewards long, well-placed drives. |
| `sg_approach` | `par3_scoring_avg` | On a par 3, the tee shot *is* the approach shot -- a clean proxy for iron precision. |
| `sg_around_green` | `bounce_back` | % of holes where a bogey is immediately followed by a birdie-or-better -- short-game recovery and mental composure. |
| `sg_putting` | `three_putt_avoidance` | % of holes *without* a 3-putt -- a distinct putting signal from `putting_avg`/`putts_per_round`, and the number amateurs actually track ("did I 3-putt today"). |
| `birdie_avg` | `par4_scoring_avg` | Rounds out the par-3/4/5 scoring breakdown; `birdie_avg` was redundant with `birdie_or_better_pct`. |

Retraining on this feature set barely moved model quality (ROC AUC 0.742 ->
0.747, balanced accuracy 0.639 -> 0.629 -- noise-level differences), and a
dominant profile still reaches ~99% predicted win chance, confirming the
amateur-trackable stats carry comparable signal to Strokes Gained for this
model.

### Reading the "what leads to tournament wins" chart

`HistGradientBoostingClassifier` has no `.coef_` (it's a tree ensemble, not
linear), so the per-feature chart is built from permutation importance --
how much shuffling one column hurts the pipeline's cross-validated AUC --
signed by the same monotonic direction imposed during training. Read it as
"how much the model leans on this stat", not a literal per-unit effect size
the way a linear coefficient would be.

## Testing

Run `pytest` for the unit tests (15 tests across pipeline_def, build output,
and the API). The Postman collection testing the live deployment lives in
`postman/pga-win-predictor.postman_collection.json`.
