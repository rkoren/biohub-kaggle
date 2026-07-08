"""Predictor for biohub-cell-tracking — plug your trained model in here.

This module is loaded by ``kitchen serve local`` (and the Lambda handler) via
``kitchen.serve.loader``.  It must expose::

    def predict(payload: dict) -> dict: ...

Optionally export ``RequestModel`` and ``ResponseModel`` (Pydantic
``BaseModel`` subclasses) to enable typed OpenAPI docs on ``/predict``.
If either is absent the endpoint accepts and returns raw dicts.

Reserved environment variables — set by ``kitchen serve local`` / the loader; do
not reuse them for your own settings (use a project-specific name instead):
``KITCHEN_PREDICTOR_DIR`` (directory of this file), ``KITCHEN_MODEL_NAME``,
``KITCHEN_MODEL_VERSION``.

Optionally export ``MODEL_NAME`` / ``MODEL_VERSION`` (strings) to surface the
model identity on ``GET /metadata``.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Uncomment once your champion model is promoted to the registry.
# lazy_model defers the (slow) load to the first prediction instead of import
# time, so Lambda cold starts are faster; it loads once and caches thereafter.
# load_champion translates an unreachable-artifact failure (e.g. after migrating
# the tracking store from local SQLite to a remote server) into a clear error.
# ---------------------------------------------------------------------------
# from kitchen.serve import lazy_model, load_champion
# model = lazy_model(lambda: load_champion("models:/biohub-cell-tracking-model@champion"))
# # model.predict(...) works transparently and triggers the load on first use.

# ---------------------------------------------------------------------------
# Optional: typed OpenAPI schema (requires pydantic, already a FastAPI dep)
# ---------------------------------------------------------------------------
# from pydantic import BaseModel
#
# class RequestModel(BaseModel):
#     feature_a: float
#     feature_b: str
#
# class ResponseModel(BaseModel):
#     label: int
#     score: float

# ---------------------------------------------------------------------------
# Optional: feature list + model identity — surfaced on GET /metadata so callers
# know which input keys the model expects and which model is serving.
# ---------------------------------------------------------------------------
# FEATURES: list[str] = ["feature_a", "feature_b"]
# MODEL_NAME = "biohub-cell-tracking-model"
# MODEL_VERSION = "champion"


def predict(payload: dict) -> dict:
    """Return a prediction for *payload*.

    Args:
        payload: Arbitrary JSON dict from the caller.  When ``RequestModel``
                 is configured this will be ``RequestModel.model_dump()``.

    Returns:
        Prediction result (must be JSON-serialisable).  When ``ResponseModel``
        is configured FastAPI validates the return value against the schema.
    """
    # TODO: replace with real model inference, e.g.:
    #   features = [payload["feature_a"], payload["feature_b"]]
    #   return {"label": int(model.predict([features])[0]), "score": 0.0}
    raise NotImplementedError(
        "Implement predict() in src/serve/predictor.py — "
        "see the commented examples above."
    )
