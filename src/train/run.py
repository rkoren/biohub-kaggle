"""Model training for biohub-cell-tracking."""
from __future__ import annotations

import pandas as pd
from kitchen.steps import Trainer
from kitchen.store import DataStore
from kitchen.tracking import Tracker


class BiohubCellTrackingTrainer(Trainer):
    model_flavour = "sklearn"  # change to "xgboost" or "pyfunc" as needed

    def fit(self, df: pd.DataFrame, params: dict) -> object:
        """Train and return a model. Log metrics to the active MLflow run."""
        raise NotImplementedError


def train(params: dict, store: DataStore, tracker: Tracker) -> object:
    return BiohubCellTrackingTrainer().run(store, tracker, params)
