from __future__ import annotations

from pathlib import Path

import pytest

from asd_screening.config import MODEL_PATH
from asd_screening.data import load_training_data
from asd_screening.inference import ArtifactError, load_artifact, predict_record


def test_missing_artifact_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        load_artifact(tmp_path / "missing.joblib")


def test_release_artifact_predicts_canonical_record() -> None:
    artifact = load_artifact(MODEL_PATH)
    features, _, _ = load_training_data()
    outcome = predict_record(artifact, features.iloc[0].to_dict())

    assert outcome["class"] in {0, 1}
    assert 0.0 <= outcome["score"] <= 1.0
    assert 0.0 < outcome["threshold"] < 1.0
