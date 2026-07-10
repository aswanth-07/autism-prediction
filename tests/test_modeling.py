from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from asd_screening.data import load_training_data
from asd_screening.evaluation import (
    classification_metrics,
    positive_probabilities,
    select_f1_threshold,
)
from asd_screening.modeling import build_pipeline


def test_pipeline_handles_missing_and_unseen_categories() -> None:
    features, target, _ = load_training_data()
    pipeline = build_pipeline(LogisticRegression(max_iter=2_000, class_weight="balanced"))
    pipeline.fit(features, target)

    unseen = features.head(2).copy()
    unseen.loc[0, "country_of_residence"] = "Unseen country"
    unseen.loc[1, "ethnicity"] = None
    scores = positive_probabilities(pipeline, unseen)

    assert scores.shape == (2,)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_threshold_selection_and_metrics_are_consistent() -> None:
    _, target, _ = load_training_data()
    scores = target.to_numpy(dtype=float) * 0.8 + 0.1

    threshold, best_f1 = select_f1_threshold(target, scores)
    metrics = classification_metrics(target, scores, threshold)

    assert best_f1 == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[453, 0], [0, 124]]
