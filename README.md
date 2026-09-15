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

5-fold stratified cross-validation on the 1,919 player-seasons used for
training (19.6% of which included at least one win):

- ROC AUC: 0.766
- Balanced accuracy: 0.701

Note on interpreting the coefficients: several features are structurally
correlated (`sg_total` is the arithmetic sum of `sg_off_the_tee` +
`sg_approach` + `sg_around_green` + `sg_putting`; `birdie_avg` and
`birdie_or_better_pct` move together; `putting_avg` and `putts_per_round`
move together). With correlated inputs, a logistic regression's individual
coefficient signs can look counterintuitive even when the model's overall
predictions are sound -- credit gets split arbitrarily among correlated
features rather than reflecting each one's true independent effect. The
AUC/accuracy numbers above are the right way to judge this model; the
per-feature coefficient chart is best read as "what the model weighted",
not "what causes wins" in a strict sense.

## Testing

Run `pytest` for the unit tests (15 tests across pipeline_def, build output,
and the API). The Postman collection testing the live deployment lives in
`postman/pga-win-predictor.postman_collection.json`.
