"""
14_final_evaluation.py  —  Final Test Set Evaluation

Loads the best model selected by 13_model_selection.py and evaluates it
on the held-out test set.  The test set is used ONCE here and never touched
during model development or hyperparameter tuning.

Metrics
-------
Threshold-free (model ranking ability):
    AUC-ROC, AUC-PR, KS Statistic

Threshold-based (business decision metrics):
    Threshold determined by Youden's J = argmax(TPR - FPR), the point of
    maximum separation on the ROC curve — equivalent to the KS statistic.
    At this threshold: Accuracy, Precision, Recall, F1, Confusion Matrix.

MLflow
------
    Logs test metrics to the existing 'lending_club_default_prediction'
    experiment as a new run named 'FinalTestEvaluation'.

Input  : data/processed/test_features.csv
         data/processed/best_model.pkl
Output : console log + MLflow run
"""

import logging
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET = "charged_off"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_proba(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Threshold-free metrics: AUC-ROC, AUC-PR, KS."""
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr  = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float((tpr - fpr).max())
    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "ks": ks}


def _evaluate_at_youden(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """
    Threshold-based metrics at Youden's J optimal threshold.

    Youden's J = argmax(TPR - FPR): the threshold that maximises the
    separation between true positive rate and false positive rate,
    corresponding to the KS statistic point on the ROC curve.
    Widely used in credit risk to set a decision cutoff without requiring
    an explicit business rule.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_idx    = int((tpr - fpr).argmax())
    best_thr = float(thresholds[j_idx])
    y_pred   = (y_score >= best_thr).astype(int)

    return {
        "threshold": best_thr,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                    model_name: str, threshold: float) -> plt.Figure:
    """ROC curve with Youden's J threshold point marked."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    # find the point closest to the Youden threshold
    j_idx = int((tpr - fpr).argmax())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", linewidth=2,
            label=f"ROC curve (AUC = {auc:.4f})")
    ax.scatter(fpr[j_idx], tpr[j_idx], color="crimson", zorder=5, s=80,
               label=f"Youden's J threshold = {threshold:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — Test Set  ({model_name})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_confusion_matrix(cm: np.ndarray, model_name: str) -> plt.Figure:
    """Confusion matrix heatmap with counts and row-normalised percentages."""
    labels = ["Fully Paid (0)", "Charged Off (1)"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    annot = np.array([
        [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — Test Set  ({model_name})")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(data_dir: str, artifact_dir: str) -> None:
    # ── Load model ─────────────────────────────────────────────────────────────
    model_path = os.path.join(artifact_dir, "best_model.pkl")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    model_name = type(model).__name__
    logger.info("Loaded best model: %s from %s", model_name, model_path)

    # ── Load test data ─────────────────────────────────────────────────────────
    test_path = os.path.join(data_dir, "test_features.csv")
    test_df   = pd.read_csv(test_path)
    logger.info("Loaded test set: %s", test_df.shape)

    X_test = test_df.drop(columns=[TARGET]).values
    y_test = test_df[TARGET].values.astype(int)
    logger.info(
        "Class balance (test) — 0: %.1f%%  1: %.1f%%",
        (y_test == 0).mean() * 100, (y_test == 1).mean() * 100,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_score = model.predict_proba(X_test)[:, 1]
    proba   = _evaluate_proba(y_test, y_score)
    thresh  = _evaluate_at_youden(y_test, y_score)

    sep = "=" * 60
    logger.info("\n%s", sep)
    logger.info("FINAL TEST SET EVALUATION — %s", model_name)
    logger.info("%s", sep)
    logger.info("Threshold-free metrics:")
    logger.info("  AUC-ROC    : %.4f", proba["auc_roc"])
    logger.info("  AUC-PR     : %.4f", proba["auc_pr"])
    logger.info("  KS         : %.4f", proba["ks"])
    logger.info("Threshold-based metrics (Youden's J threshold = %.4f):", thresh["threshold"])
    logger.info("  Accuracy   : %.4f", thresh["accuracy"])
    logger.info("  Precision  : %.4f", thresh["precision"])
    logger.info("  Recall     : %.4f", thresh["recall"])
    logger.info("  F1         : %.4f", thresh["f1"])
    logger.info("Confusion Matrix:")
    cm = thresh["confusion_matrix"]
    logger.info("  TN=%-8d FP=%d", cm[0, 0], cm[0, 1])
    logger.info("  FN=%-8d TP=%d", cm[1, 0], cm[1, 1])
    logger.info("%s", sep)

    # ── MLflow logging ─────────────────────────────────────────────────────────
    mlflow_uri = f"file:{os.path.join(artifact_dir, 'mlruns')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("lending_club_default_prediction")

    roc_fig = _plot_roc_curve(y_test, y_score, model_name, thresh["threshold"])
    cm_fig  = _plot_confusion_matrix(thresh["confusion_matrix"], model_name)

    with mlflow.start_run(run_name="FinalTestEvaluation"):
        mlflow.log_param("model", model_name)
        mlflow.log_metrics({
            "test_auc_roc":   proba["auc_roc"],
            "test_auc_pr":    proba["auc_pr"],
            "test_ks":        proba["ks"],
            "test_threshold": thresh["threshold"],
            "test_accuracy":  thresh["accuracy"],
            "test_precision": thresh["precision"],
            "test_recall":    thresh["recall"],
            "test_f1":        thresh["f1"],
        })
        mlflow.log_figure(roc_fig, "plots/roc_curve_test.png")
        mlflow.log_figure(cm_fig,  "plots/confusion_matrix_test.png")

    plt.close(roc_fig)
    plt.close(cm_fig)
    logger.info("Test metrics and plots logged to MLflow (%s)", mlflow_uri)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    _src_dir  = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.join(_src_dir, "..")

    DATA_DIR     = os.getenv("DATA_DIR",     os.path.join(_root_dir, "data", "processed"))
    ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", DATA_DIR)

    evaluate(DATA_DIR, ARTIFACT_DIR)
