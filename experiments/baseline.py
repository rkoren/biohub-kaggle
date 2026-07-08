"""Baseline experiment for biohub-cell-tracking.

First approach — simpler features, default hyperparams.
Tag: model_variant=baseline.

Usage:
    python experiments/baseline.py
"""
from __future__ import annotations

import os

import yaml
from dotenv import load_dotenv

load_dotenv()

import mlflow

from kitchen.store import DataStore
from kitchen.tracking import Tracker, configure_from_env, init_experiment

EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "biohub-cell-tracking")
VARIANT = "baseline"


def run_variant(params: dict, variant: str) -> None:
    from src.features.run import build
    from src.train.run import train

    configure_from_env()
    init_experiment(EXPERIMENT)

    store = DataStore()
    tracker = Tracker(EXPERIMENT)

    with tracker.run(run_name=variant, params=params) as _run:
        mlflow.set_tag("model_variant", variant)
        build(params, store)
        train(params, store, tracker)
        print(f"{variant} run complete — see MLflow for val metrics")


if __name__ == "__main__":
    with open("menu.yaml") as f:
        params = yaml.safe_load(f)
    run_variant(params, VARIANT)
