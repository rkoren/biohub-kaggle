"""Evaluation for biohub-cell-tracking."""
from __future__ import annotations

import pandas as pd
from kitchen.steps import Evaluator
from kitchen.store import DataStore


class BiohubCellTrackingEvaluator(Evaluator):
    def evaluate(self, model: object, df: pd.DataFrame) -> dict[str, float]:
        """Return metric_name -> value. Logged to MLflow and written to metrics.json."""
        raise NotImplementedError


def evaluate(model: object, params: dict, store: DataStore) -> dict[str, float]:
    return BiohubCellTrackingEvaluator().run(model, store, params)
