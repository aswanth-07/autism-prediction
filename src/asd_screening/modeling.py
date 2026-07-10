"""Model definitions and leakage-safe preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from asd_screening.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE
from asd_screening.data import normalize_features


@dataclass(frozen=True)
class Candidate:
    label: str
    estimator: BaseEstimator
    parameter_grid: dict[str, list[Any]]


def build_preprocessor() -> ColumnTransformer:
    """Build transformations whose learned state remains inside each CV fold."""

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Create an end-to-end pipeline from canonical records to predictions."""

    return Pipeline(
        steps=[
            (
                "normalize",
                FunctionTransformer(
                    normalize_features,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("preprocess", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def build_hybrid_mlp_stack() -> StackingClassifier:
    """Build the repository's hybrid idea with genuine out-of-fold meta-features.

    The MLP receives three base-model probabilities plus the preprocessed source
    features through ``passthrough=True``. ``StackingClassifier`` constructs the
    training probabilities with stratified cross-validation, then refits each base
    learner on the complete fold-local training partition for inference.
    """

    base_estimators = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        (
            "linear_svc",
            SVC(
                kernel="linear",
                C=0.5,
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "adaboost",
            AdaBoostClassifier(
                n_estimators=150,
                learning_rate=0.05,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    meta_learner = Pipeline(
        steps=[
            ("select", SelectKBest(score_func=f_classif, k=12)),
            ("scale", StandardScaler(with_mean=False)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(8,),
                    activation="logistic",
                    solver="lbfgs",
                    alpha=0.01,
                    max_iter=2_000,
                    tol=0.001,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        passthrough=True,
        n_jobs=1,
    )


def model_candidates() -> dict[str, Candidate]:
    """Return a focused set of interpretable and nonlinear classical baselines."""

    return {
        "hybrid_mlp_stack": Candidate(
            label="Hybrid OOF Stack + MLP",
            estimator=build_hybrid_mlp_stack(),
            parameter_grid={
                "classifier__final_estimator__select__k": [8, 12],
                "classifier__final_estimator__mlp__alpha": [0.01, 0.1],
            },
        ),
        "logistic_regression": Candidate(
            label="Logistic Regression",
            estimator=LogisticRegression(
                max_iter=4_000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),
            parameter_grid={
                "classifier__C": [0.1, 0.5, 1.0, 2.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "rbf_svc": Candidate(
            label="RBF Support Vector Classifier",
            estimator=SVC(probability=True, random_state=RANDOM_STATE),
            parameter_grid={
                "classifier__C": [0.5, 1.0, 2.0, 4.0],
                "classifier__gamma": ["scale", 0.01, 0.03],
                "classifier__class_weight": ["balanced"],
            },
        ),
        "random_forest": Candidate(
            label="Random Forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            parameter_grid={
                "classifier__max_depth": [None, 6, 10],
                "classifier__min_samples_leaf": [2, 4, 8],
                "classifier__max_features": ["sqrt", 0.7],
                "classifier__class_weight": ["balanced_subsample"],
            },
        ),
        "extra_trees": Candidate(
            label="Extremely Randomized Trees",
            estimator=ExtraTreesClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            parameter_grid={
                "classifier__max_depth": [None, 6, 10],
                "classifier__min_samples_leaf": [2, 4, 8],
                "classifier__max_features": ["sqrt", 0.7],
                "classifier__class_weight": ["balanced"],
            },
        ),
    }
