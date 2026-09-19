from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import RANDOM_STATE


def classification_metrics(y_true, y_prob, threshold=0.50):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def find_best_f1_threshold(y_true, y_prob):
    thresholds = np.linspace(0.10, 0.90, 161)
    scores = [f1_score(y_true, np.asarray(y_prob) >= t, zero_division=0) for t in thresholds]
    idx = int(np.argmax(scores))
    return float(thresholds[idx]), float(scores[idx])


def save_evaluation_plots(y_true, y_prob, output_dir: Path, threshold=0.50):
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)


def save_permutation_importance(model, X_test, y_test, output_dir: Path, top_n=20):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    names = getattr(model, "feature_names_in_", X_test.columns)
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp.head(top_n).to_csv(output_dir / "permutation_importance.csv", index=False)
    return imp


def save_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
