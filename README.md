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

- Modal API: https://esmit289--pga-win-predictor-fastapi-app.modal.run
- API docs: https://esmit289--pga-win-predictor-fastapi-app.modal.run/docs
- Vercel page using this API: https://pga-tour-dashboard.vercel.app/win-predictor
