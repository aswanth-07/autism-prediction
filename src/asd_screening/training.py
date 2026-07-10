"""Train, select, evaluate, and serialize the production classical ML pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from asd_screening.config import (
    FEATURES,
    MODEL_PATH,
    PROJECT_VERSION,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TEST_SIZE,
)
from asd_screening.data import category_options, load_training_data, target_rate
from asd_screening.evaluation import (
    bootstrap_interval,
    classification_metrics,
    positive_probabilities,
    select_f1_threshold,
)
from asd_screening.modeling import build_pipeline, model_candidates

plt.switch_backend("Agg")

SCORING = {
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    quick: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outer_folds = 3 if quick else 5
    inner_folds = 3 if quick else 4
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE + 1)
    records: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for key, candidate in model_candidates().items():
        pipeline = build_pipeline(candidate.estimator)
        nested_search = GridSearchCV(
            estimator=pipeline,
            param_grid=candidate.parameter_grid,
            scoring="f1",
            cv=inner_cv,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        nested_scores = cross_validate(
            nested_search,
            features,
            target,
            scoring=SCORING,
            cv=outer_cv,
            n_jobs=-1,
            error_score="raise",
        )

        final_search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=candidate.parameter_grid,
            scoring=SCORING,
            cv=inner_cv,
            refit="f1",
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )
        final_search.fit(features, target)
        fitted[key] = final_search.best_estimator_

        record: dict[str, Any] = {
            "key": key,
            "model": candidate.label,
            "best_parameters": final_search.best_params_,
            "inner_cv_best_f1": float(final_search.best_score_),
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
        }
        for metric in SCORING:
            values = np.asarray(nested_scores[f"test_{metric}"], dtype=float)
            record[f"nested_{metric}_mean"] = float(values.mean())
            record[f"nested_{metric}_std"] = float(values.std(ddof=1))
        records.append(record)
        print(
            f"{candidate.label}: nested F1 "
            f"{record['nested_f1_mean']:.3f} ± {record['nested_f1_std']:.3f}"
        )

    records.sort(
        key=lambda row: (row["nested_f1_mean"], row["nested_roc_auc_mean"]),
        reverse=True,
    )
    selected_key = str(records[0]["key"])
    return records, {"key": selected_key, "pipeline": fitted[selected_key]}


def _plot_reports(
    comparison: pd.DataFrame,
    target: pd.Series,
    scores: NDArray[np.float64],
    threshold: float,
    importance: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    ordered = comparison.sort_values("nested_f1_mean")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(
        ordered["model"],
        ordered["nested_f1_mean"],
        xerr=ordered["nested_f1_std"],
        color="#167D9A",
        alpha=0.92,
        capsize=4,
    )
    axis.set(xlabel="Nested cross-validation F1", xlim=(0.0, 1.0), title="Model comparison")
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=180)
    plt.close(fig)

    metrics = classification_metrics(target, scores, threshold)
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(5.2, 4.7))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Predicted class",
        ylabel="True class",
        title=f"Untouched holdout confusion matrix (threshold {threshold:.2f})",
    )
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures / "holdout_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    RocCurveDisplay.from_predictions(target, scores, ax=axes[0], color="#167D9A")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#666666")
    axes[0].set_title("Holdout ROC curve")
    PrecisionRecallDisplay.from_predictions(target, scores, ax=axes[1], color="#8B5CF6")
    axes[1].axhline(float(target.mean()), linestyle="--", color="#666666")
    axes[1].set_title("Holdout precision–recall curve")
    fig.tight_layout()
    fig.savefig(figures / "holdout_roc_pr_curves.png", dpi=180)
    plt.close(fig)

    top = importance.sort_values("importance_mean").tail(12)
    fig, axis = plt.subplots(figsize=(8, 5.5))
    axis.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#167D9A")
    axis.set(xlabel="Decrease in holdout ROC-AUC after permutation", title="Permutation importance")
    fig.tight_layout()
    fig.savefig(figures / "permutation_importance.png", dpi=180)
    plt.close(fig)


def train(*, quick: bool = False) -> dict[str, Any]:
    """Execute the reproducible training and evaluation protocol."""

    features, target, audit = load_training_data(RAW_DATA_PATH)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    train_x = train_x.reset_index(drop=True)
    test_x = test_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    print(f"Adult dataset: {len(features)} rows; train={len(train_x)}, holdout={len(test_x)}")
    comparison, selected = _nested_evaluate(train_x, train_y, quick=quick)
    pipeline = selected["pipeline"]

    threshold_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 2)
    out_of_fold = cross_val_predict(
        clone(pipeline),
        train_x,
        train_y,
        cv=threshold_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    threshold, training_oof_f1 = select_f1_threshold(train_y, out_of_fold)

    pipeline.fit(train_x, train_y)
    test_scores = positive_probabilities(pipeline, test_x)
    holdout = classification_metrics(test_y, test_scores, threshold)
    predictions = (test_scores >= threshold).astype(int)
    holdout["f1_95_ci"] = bootstrap_interval(
        test_y,
        predictions,
        lambda truth, predicted: f1_score(truth, predicted, zero_division=0),
    )
    holdout["roc_auc_95_ci"] = bootstrap_interval(test_y, test_scores, roc_auc_score)

    permutation = permutation_importance(
        pipeline,
        test_x,
        test_y,
        scoring="roc_auc",
        n_repeats=30 if not quick else 5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    generated_at = datetime.now(timezone.utc).isoformat()
    selected_row = next(row for row in comparison if row["key"] == selected["key"])
    metadata = {
        "project_version": PROJECT_VERSION,
        "generated_at_utc": generated_at,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "dataset_sha256": _sha256(RAW_DATA_PATH),
        "model_key": selected["key"],
        "model_label": selected_row["model"],
        "features": list(FEATURES),
        "scope": "Adults aged 18–100 in the supplied course dataset",
        "intended_use": "Educational research demonstration only; not diagnosis or clinical triage",
        "category_options": category_options(features),
    }
    artifact = {"pipeline": pipeline, "threshold": threshold, "metadata": metadata}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)

    report = {
        "protocol": {
            "selection": "Nested stratified cross-validation on the training partition",
            "primary_metric": "F1 for the positive class",
            "holdout": "20% stratified split, evaluated once after model and threshold selection",
            "threshold_selection": "Maximum F1 on five-fold out-of-fold training probabilities",
            "random_state": RANDOM_STATE,
            "quick_mode": quick,
        },
        "data_audit": audit.to_dict(),
        "split": {
            "training_rows": len(train_x),
            "holdout_rows": len(test_x),
            "training_positive_rate": target_rate(train_y),
            "holdout_positive_rate": target_rate(test_y),
        },
        "model_comparison": comparison,
        "selected_model": {
            "key": selected["key"],
            "label": selected_row["model"],
            "threshold": threshold,
            "training_oof_f1_at_threshold": training_oof_f1,
        },
        "holdout_metrics": holdout,
        "limitations": [
            "Small course dataset with incomplete provenance and demographic imbalance.",
            (
                "The source marks all rows as adults but contains 223 ages below 18; "
                "those rows are excluded."
            ),
            (
                "The source result field is excluded because its semantics cannot be "
                "reproduced by the app."
            ),
            (
                "Unknown values are common in ethnicity and relation, and subgroup "
                "performance is not established."
            ),
            "Performance estimates have wide uncertainty and do not establish clinical validity.",
        ],
        "artifact_metadata": metadata,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_frame = pd.DataFrame(comparison)
    comparison_frame.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    importance.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    with (REPORTS_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)
        handle.write("\n")
    _plot_reports(comparison_frame, test_y, test_scores, threshold, importance, REPORTS_DIR)

    print(
        f"Selected {selected_row['model']} | threshold={threshold:.3f} | "
        f"holdout F1={holdout['f1']:.3f} | ROC-AUC={holdout['roc_auc']:.3f}"
    )
    print(f"Saved model: {MODEL_PATH}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer outer folds/repeats for a local smoke run; not for published metrics.",
    )
    arguments = parser.parse_args()
    train(quick=arguments.quick)


if __name__ == "__main__":
    main()
