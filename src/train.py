from __future__ import annotations

import argparse
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import MODEL_DIR, RANDOM_STATE, REPORT_DIR, TEST_SIZE, VALIDATION_SIZE
from .data import load_adult_dataset, split_features_target
from .evaluate import (
    classification_metrics,
    find_best_f1_threshold,
    save_evaluation_plots,
    save_json,
)
from .features import make_linear_preprocessor, make_tree_preprocessor


def build_models(X):
    return {
        "logistic_regression": Pipeline([
            ("preprocessor", make_linear_preprocessor(X)),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", make_tree_preprocessor(X)),
            ("model", RandomForestClassifier(
                n_estimators=350,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=2,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocessor", make_tree_preprocessor(X)),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=250,
                max_leaf_nodes=31,
                l2_regularization=0.5,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def main():
    parser = argparse.ArgumentParser(description="Train upgraded Adult income-status classifiers.")
    parser.add_argument("--quick", action="store_true", help="Use fewer tuning iterations for a faster run.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_adult_dataset()
    X, y = split_features_target(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    validation_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=validation_fraction,
        stratify=y_train_full, random_state=RANDOM_STATE
    )

    models = build_models(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    fitted = {}

    for name, pipe in models.items():
        scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            pipe.fit(X_tr, y_tr)
            prob = pipe.predict_proba(X_va)[:, 1]
            from sklearn.metrics import roc_auc_score
            scores.append(roc_auc_score(y_va, prob))
        pipe.fit(X_train, y_train)
        valid_prob = pipe.predict_proba(X_valid)[:, 1]
        threshold, _ = find_best_f1_threshold(y_valid, valid_prob)
        valid_metrics = classification_metrics(y_valid, valid_prob, threshold)
        rows.append({
            "model": name,
            "cv_roc_auc_mean": float(np.mean(scores)),
            "cv_roc_auc_std": float(np.std(scores)),
            **{f"validation_{k}": v for k, v in valid_metrics.items()},
        })
        fitted[name] = (pipe, threshold)

    comparison = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)

    # Tune the top two candidates using only the training portion.
    candidates = comparison["model"].head(2).tolist()
    tuned = {}
    for name in candidates:
        pipe = models[name]
        if name == "logistic_regression":
            params = {
                "model__C": np.logspace(-2, 1, 8),
            }
        elif name == "random_forest":
            params = {
                "model__n_estimators": [250, 400, 550],
                "model__max_depth": [None, 12, 20, 30],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.5],
            }
        else:
            params = {
                "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.5, 1.0, 2.0],
            }
        search = RandomizedSearchCV(
            pipe,
            params,
            n_iter=3 if args.quick else 8,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train_full, y_train_full)
        tuned[name] = search

    best_name = max(tuned, key=lambda n: tuned[n].best_score_)
    best_model = tuned[best_name].best_estimator_

    # Threshold is selected on a holdout validation split from train_full.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=RANDOM_STATE
    )
    best_model.fit(X_fit, y_fit)
    val_prob = best_model.predict_proba(X_val)[:, 1]
    threshold, _ = find_best_f1_threshold(y_val, val_prob)
    best_model.fit(X_train_full, y_train_full)
    test_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, test_prob, 0.50)
    tuned_threshold_metrics = classification_metrics(y_test, test_prob, threshold)

    artifact = {
        "model": best_model,
        "threshold": threshold,
        "model_name": best_name,
        "feature_columns": X.columns.tolist(),
    }
    joblib.dump(artifact, MODEL_DIR / "best_model.joblib")

    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    save_json({
        "selected_model": best_name,
        "tuned_cv_roc_auc": float(tuned[best_name].best_score_),
        "best_params": tuned[best_name].best_params_,
        "test_metrics_threshold_0_50": metrics,
        "test_metrics_optimized_f1_threshold": tuned_threshold_metrics,
        "optimized_threshold": threshold,
    }, REPORT_DIR / "final_metrics.json")
    save_evaluation_plots(y_test, test_prob, REPORT_DIR, threshold=threshold)

    print(f"Selected model: {best_name}")
    print(f"Optimized threshold: {threshold:.3f}")
    print(json.dumps(tuned_threshold_metrics, indent=2))


if __name__ == "__main__":
    main()
