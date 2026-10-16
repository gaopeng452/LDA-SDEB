from __future__ import annotations

import math

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


METRIC_NAMES = ("precision", "recall", "accuracy", "f1", "auc", "aupr")


def train_logit_blending_ensemble(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    seed: int,
    estimators: int,
    threads: int,
) -> np.ndarray:
    common_subsample = 0.8
    xgb_model = XGBClassifier(
        n_estimators=estimators,
        max_depth=8,
        learning_rate=0.05,
        subsample=common_subsample,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=threads,
        verbosity=0,
    )
    lgb_model = LGBMClassifier(
        n_estimators=estimators,
        learning_rate=0.05,
        num_leaves=63,
        subsample=common_subsample,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed + 1,
        n_jobs=threads,
        verbosity=-1,
    )
    cat_model = CatBoostClassifier(
        iterations=estimators,
        depth=8,
        learning_rate=0.05,
        subsample=common_subsample,
        rsm=0.8,
        loss_function="Logloss",
        random_seed=seed + 2,
        verbose=False,
        allow_writing_files=False,
        thread_count=threads,
    )
    models = (xgb_model, lgb_model, cat_model)
    logits: list[np.ndarray] = []
    for model in models:
        model.fit(train_features, train_labels)
        probability = np.clip(model.predict_proba(test_features)[:, 1], 1e-6, 1.0 - 1e-6)
        logits.append(np.log(probability / (1.0 - probability)))
    mean_logit = np.mean(logits, axis=0)
    return 1.0 / (1.0 + np.exp(-np.clip(mean_logit, -40.0, 40.0)))


def compute_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute the six evaluation metrics used by LDA-SDEB."""
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = (probability >= threshold).astype(np.int64)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probability)
    return {
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "auc": float(roc_auc_score(labels, probability)),
        "aupr": float(auc(recall_curve, precision_curve)),
    }


def summarize_metrics(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "sem": float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
        }
    return summary
