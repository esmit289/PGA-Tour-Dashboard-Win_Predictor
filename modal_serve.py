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
