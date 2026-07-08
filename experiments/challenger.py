"""Challenger experiment for biohub-cell-tracking.

Extend the baseline: add features, tune hyperparams, or swap the model.
Tag: model_variant=challenger.

Usage:
    python experiments/challenger.py
"""
from __future__ import annotations

import yaml
from dotenv import load_dotenv

load_dotenv()

from experiments.baseline import run_variant

VARIANT = "challenger"


def challenger(params_file: str = "menu.yaml") -> None:
    with open(params_file) as f:
        params = yaml.safe_load(f)

    # TODO: Override params for the challenger approach, e.g.:
    # params["model"]["max_depth"] = 8
    # params["model"]["learning_rate"] = 0.01

    run_variant(params, VARIANT)


if __name__ == "__main__":
    challenger()
