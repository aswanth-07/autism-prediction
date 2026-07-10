"""Data loading, validation, and label-independent normalization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asd_screening.config import (
    AQ_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    MAX_AGE,
    MIN_AGE,
    RAW_DATA_PATH,
    RAW_TO_CANONICAL,
    TARGET,
)


class DataValidationError(ValueError):
    """Raised when input data violates the documented model contract."""


@dataclass(frozen=True)
class DataAudit:
    source_rows: int
    usable_rows: int
    excluded_under_18: int
    excluded_over_100: int
    positive_rows: int
    negative_rows: int
    duplicate_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}")


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical feature values without learning statistics from the data.

    This function is safe inside a scikit-learn pipeline: it only standardizes known
    spellings and validates values. Imputation and encoding are fitted later and only
    on the training folds.
    """

    _require_columns(frame, FEATURES)
    clean = frame.loc[:, FEATURES].copy()

    for column in AQ_FEATURES:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        invalid = clean[column].dropna().loc[lambda values: ~values.isin([0, 1])]
        if not invalid.empty:
            raise DataValidationError(f"{column} must contain only binary scores 0 or 1")

    clean["age"] = pd.to_numeric(clean["age"], errors="coerce")
    non_missing_age = clean["age"].dropna()
    if ((non_missing_age < MIN_AGE) | (non_missing_age > MAX_AGE)).any():
        raise DataValidationError(f"age must be between {MIN_AGE} and {MAX_AGE}")

    for column in CATEGORICAL_FEATURES:
        values = clean[column].astype("string").str.strip()
        values = values.replace({"?": pd.NA, "": pd.NA})
        clean[column] = values.astype(object).where(values.notna(), np.nan)

    clean["ethnicity"] = clean["ethnicity"].mask(clean["ethnicity"].eq("others"), "Others")
    for column in ("gender", "jaundice", "family_asd_history", "used_app_before"):
        clean[column] = clean[column].str.lower()

    return clean


def load_training_data(
    path: str | Path = RAW_DATA_PATH,
) -> tuple[pd.DataFrame, pd.Series, DataAudit]:
    """Load the course dataset and enforce the adult-only modeling scope."""

    raw = pd.read_csv(path).rename(columns=RAW_TO_CANONICAL)
    required = (*FEATURES, TARGET)
    _require_columns(raw, required)

    raw[TARGET] = pd.to_numeric(raw[TARGET], errors="coerce")
    if raw[TARGET].isna().any() or not set(raw[TARGET].unique()).issubset({0, 1}):
        raise DataValidationError(f"{TARGET} must be complete and binary")

    ages = pd.to_numeric(raw["age"], errors="coerce")
    under_18 = int((ages < MIN_AGE).sum())
    over_100 = int((ages > MAX_AGE).sum())
    adult_mask = ages.between(MIN_AGE, MAX_AGE, inclusive="both")
    scoped = raw.loc[adult_mask].copy()
    features = normalize_features(scoped.loc[:, FEATURES])
    target = scoped[TARGET].astype("int8").reset_index(drop=True)
    features = features.reset_index(drop=True)

    audit = DataAudit(
        source_rows=len(raw),
        usable_rows=len(scoped),
        excluded_under_18=under_18,
        excluded_over_100=over_100,
        positive_rows=int(target.sum()),
        negative_rows=int((target == 0).sum()),
        duplicate_rows=int(raw.duplicated().sum()),
    )
    return features, target, audit


def make_inference_frame(values: dict[str, Any]) -> pd.DataFrame:
    """Validate a single user record and return it in canonical feature order."""

    missing = sorted(set(FEATURES).difference(values))
    if missing:
        raise DataValidationError(f"Missing inference values: {', '.join(missing)}")
    frame = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
    return normalize_features(frame)


def category_options(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Extract stable UI options from the reviewed training scope."""

    clean = normalize_features(frame)
    return {
        column: sorted(str(value) for value in clean[column].dropna().unique())
        for column in CATEGORICAL_FEATURES
    }


def target_rate(target: pd.Series) -> float:
    """Return the positive-class prevalence."""

    return float(np.mean(target.to_numpy(dtype=float)))
