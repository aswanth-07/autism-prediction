"""Versioned artifact loading and validated inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from asd_screening.config import MODEL_PATH, PROJECT_VERSION
from asd_screening.data import make_inference_frame
from asd_screening.evaluation import positive_probabilities


class ArtifactError(RuntimeError):
    """Raised when a serialized artifact is missing or incompatible."""


def load_artifact(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise ArtifactError(f"Model artifact was not found: {artifact_path}")
    artifact = joblib.load(artifact_path)
    required = {"pipeline", "threshold", "metadata"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ArtifactError("Model artifact does not match the production schema")
    version = artifact["metadata"].get("project_version")
    if version != PROJECT_VERSION:
        raise ArtifactError(f"Artifact version {version!r} does not match app {PROJECT_VERSION!r}")
    return artifact


def predict_record(artifact: dict[str, Any], values: dict[str, Any]) -> dict[str, float | int]:
    frame = make_inference_frame(values)
    score = float(positive_probabilities(artifact["pipeline"], frame)[0])
    threshold = float(artifact["threshold"])
    return {
        "class": int(score >= threshold),
        "score": float(np.clip(score, 0.0, 1.0)),
        "threshold": threshold,
    }
