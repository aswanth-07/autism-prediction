from __future__ import annotations

import pandas as pd
import pytest

from asd_screening.config import AQ_FEATURES, FEATURES
from asd_screening.data import DataValidationError, load_training_data, normalize_features


def test_data_audit_matches_reviewed_release() -> None:
    features, target, audit = load_training_data()

    assert list(features.columns) == list(FEATURES)
    assert len(features) == len(target) == 577
    assert audit.source_rows == 800
    assert audit.excluded_under_18 == 223
    assert audit.excluded_over_100 == 0
    assert audit.positive_rows == 124
    assert audit.negative_rows == 453


def test_normalization_rejects_nonbinary_screening_score() -> None:
    features, _, _ = load_training_data()
    invalid = features.head(1).copy()
    invalid.loc[0, AQ_FEATURES[0]] = 2

    with pytest.raises(DataValidationError, match="binary scores"):
        normalize_features(invalid)


def test_normalization_rejects_out_of_scope_age() -> None:
    features, _, _ = load_training_data()
    invalid = features.head(1).copy()
    invalid.loc[0, "age"] = 17

    with pytest.raises(DataValidationError, match="between 18 and 100"):
        normalize_features(invalid)


def test_missing_markers_become_actual_missing_values() -> None:
    features, _, _ = load_training_data()
    record = features.head(1).copy()
    record.loc[0, "ethnicity"] = "?"

    normalized = normalize_features(record)

    assert pd.isna(normalized.loc[0, "ethnicity"])
