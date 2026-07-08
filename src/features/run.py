"""Feature engineering for biohub-cell-tracking.

TODO:
  1. Implement BiohubCellTrackingFeatures.build() to transform raw CSV into model-ready features.
  2. Update FEATURES to list every column passed to the model (exclude the target).
  3. Keep the target column in the returned DataFrame — train.py separates it.
  4. If your project has multiple raw input files, override sources() to declare them:
       def sources(self, params: dict) -> list[str]:
           return ["train.csv", "other.csv"]
     build() will then receive a dict[filename, DataFrame] instead of a plain DataFrame.
"""
from __future__ import annotations

import pandas as pd
from kitchen.steps import FeatureBuilder
from kitchen.store import DataStore

# Columns passed to the model (exclude the target column).
FEATURES: list[str] = []  # TODO: fill in after feature engineering


class BiohubCellTrackingFeatures(FeatureBuilder):
    def build(self, raw: pd.DataFrame | dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
        """Transform raw data into model-ready features + target column.

        ``raw`` is a plain DataFrame for single-source projects (the default).
        Override sources() and ``raw`` becomes a dict[filename, DataFrame].
        """
        raise NotImplementedError


def build(params: dict, store: DataStore) -> None:
    BiohubCellTrackingFeatures().run(store, params)
