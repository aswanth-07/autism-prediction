"""Export the fitted Extra Trees pipeline to a deterministic browser format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier

from asd_screening.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PROJECT_ROOT
from asd_screening.data import make_inference_frame
from asd_screening.inference import load_artifact

DEFAULT_EXPORT_PATH = PROJECT_ROOT / "exports" / "asd_extra_trees_web.json"


class WebExportError(RuntimeError):
    """Raised when the release artifact cannot be represented by the web runtime."""


def _categorical_vectors(encoder: Any, options: dict[str, list[str]]) -> dict[str, Any]:
    vectors: dict[str, Any] = {}
    width = len(encoder.get_feature_names_out(list(CATEGORICAL_FEATURES)))
    for column_index, column in enumerate(CATEGORICAL_FEATURES):
        mappings: dict[str, list[list[float | int]]] = {}
        for category in options[column]:
            row = np.full((1, len(CATEGORICAL_FEATURES)), "__UNKNOWN__", dtype=object)
            row[0, column_index] = category
            encoded = encoder.transform(row)
            dense = encoded.toarray()[0] if hasattr(encoded, "toarray") else np.asarray(encoded)[0]
            mappings[category] = [
                [int(index), float(dense[index])] for index in np.flatnonzero(dense)
            ]
        vectors[column] = {"width": width, "values": mappings}
    return vectors


def export_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    pipeline = artifact["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, ExtraTreesClassifier):
        raise WebExportError(
            "The browser exporter currently requires an ExtraTreesClassifier release"
        )

    preprocess = pipeline.named_steps["preprocess"]
    numeric_pipeline = preprocess.named_transformers_["numeric"]
    categorical_pipeline = preprocess.named_transformers_["categorical"]
    imputer = numeric_pipeline.named_steps["imputer"]
    categorical_imputer = categorical_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]
    encoder = categorical_pipeline.named_steps["one_hot"]

    trees = []
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        leaf_positive_probability = []
        for node_values in tree.value:
            class_values = np.asarray(node_values[0], dtype=float)
            total = float(class_values.sum())
            leaf_positive_probability.append(float(class_values[1] / total) if total else 0.0)
        trees.append(
            {
                "left": tree.children_left.astype(int).tolist(),
                "right": tree.children_right.astype(int).tolist(),
                "feature": tree.feature.astype(int).tolist(),
                "threshold": tree.threshold.astype(float).tolist(),
                "positiveProbability": leaf_positive_probability,
            }
        )

    metadata = artifact["metadata"]
    categorical_width = len(encoder.get_feature_names_out(list(CATEGORICAL_FEATURES)))
    return {
        "format": "asd-extra-trees-web-v1",
        "projectVersion": metadata["project_version"],
        "modelLabel": metadata["model_label"],
        "threshold": float(artifact["threshold"]),
        "scope": metadata["scope"],
        "intendedUse": metadata["intended_use"],
        "datasetSha256": metadata["dataset_sha256"],
        "numeric": {
            "features": list(NUMERIC_FEATURES),
            "median": np.asarray(imputer.statistics_, dtype=float).tolist(),
            "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        },
        "categorical": {
            "features": list(CATEGORICAL_FEATURES),
            "width": categorical_width,
            "vectors": _categorical_vectors(encoder, metadata["category_options"]),
            "options": metadata["category_options"],
            "imputed": {
                feature: str(categorical_imputer.statistics_[index])
                for index, feature in enumerate(CATEGORICAL_FEATURES)
            },
        },
        "featureCount": int(classifier.n_features_in_),
        "trees": trees,
    }


def write_export(payload: dict[str, Any], path: str | Path = DEFAULT_EXPORT_PATH) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    return destination


def web_score(payload: dict[str, Any], values: dict[str, Any]) -> float:
    """Reference implementation matching the TypeScript browser evaluator."""

    frame = make_inference_frame(values)
    numeric = payload["numeric"]
    transformed: list[float] = []
    for index, feature in enumerate(numeric["features"]):
        value = frame.iloc[0][feature]
        raw_value = numeric["median"][index] if pd.isna(value) else float(value)
        transformed.append((raw_value - numeric["mean"][index]) / numeric["scale"][index])

    categorical = payload["categorical"]
    encoded = np.zeros(int(categorical["width"]), dtype=float)
    for feature in categorical["features"]:
        value = frame.iloc[0][feature]
        if pd.isna(value):
            value = categorical["imputed"][feature]
        pairs = categorical["vectors"][feature]["values"].get(str(value), [])
        for position, amount in pairs:
            encoded[int(position)] = float(amount)
    feature_vector = np.concatenate([np.asarray(transformed), encoded])

    probabilities = []
    for tree in payload["trees"]:
        node = 0
        while tree["left"][node] != -1:
            feature_index = tree["feature"][node]
            node = (
                tree["left"][node]
                if feature_vector[feature_index] <= tree["threshold"][node]
                else tree["right"][node]
            )
        probabilities.append(tree["positiveProbability"][node])
    return float(np.mean(probabilities))


def verify_export(
    artifact: dict[str, Any],
    payload: dict[str, Any],
    features: pd.DataFrame,
) -> float:
    python_scores = artifact["pipeline"].predict_proba(features)[:, 1]
    exported_scores: NDArray[np.float64] = np.asarray(
        [web_score(payload, row.to_dict()) for _, row in features.iterrows()],
        dtype=float,
    )
    return float(np.max(np.abs(python_scores - exported_scores)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORT_PATH)
    arguments = parser.parse_args()
    artifact = load_artifact()
    destination = write_export(export_payload(artifact), arguments.output)
    print(f"Exported browser model to {destination}")


if __name__ == "__main__":
    main()
