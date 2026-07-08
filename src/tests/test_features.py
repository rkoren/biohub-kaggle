"""Tests for biohub-cell-tracking feature engineering."""
import pandas as pd
import pytest

from src.features.run import BiohubCellTrackingFeatures, FEATURES


@pytest.fixture
def raw_row() -> pd.DataFrame:
    # TODO: replace with a representative row from your raw training data
    return pd.DataFrame([{}])


def test_feature_builder_raises_not_implemented(raw_row):
    # build() raises NotImplementedError until you implement it.
    # Remove this test and add real assertions once build() is done.
    with pytest.raises(NotImplementedError):
        BiohubCellTrackingFeatures().build(raw_row, params={})


def test_features_list_is_defined():
    # Populate FEATURES once build() is implemented.
    assert isinstance(FEATURES, list)
