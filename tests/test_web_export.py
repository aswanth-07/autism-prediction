from __future__ import annotations

from asd_screening.config import MODEL_PATH
from asd_screening.data import load_training_data
from asd_screening.inference import load_artifact
from asd_screening.web_export import export_payload, verify_export


def test_browser_export_matches_sklearn_probabilities() -> None:
    artifact = load_artifact(MODEL_PATH)
    features, _, _ = load_training_data()
    payload = export_payload(artifact)

    maximum_error = verify_export(artifact, payload, features.iloc[::29])

    assert payload["format"] == "asd-extra-trees-web-v1"
    assert payload["featureCount"] > 18
    assert len(payload["trees"]) == 300
    assert maximum_error < 1e-12
