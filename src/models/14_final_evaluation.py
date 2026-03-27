"""
14_final_evaluation.py - Final Test Set Evaluation

Loads the best model from 13_model_selection.py and runs it once on the
held-out test set. 

"""

import logging
import os
import pickle
import warnings

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
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


def eval_proba(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "auc_roc": roc_auc_score(y_true, y_score),
        "auc_pr": average_precision_score(y_true, y_score),
        "ks": float((tpr - fpr).max()),
    }


def eval_at_youden(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_idx = int((tpr - fpr).argmax())
    best_thr = float(thresholds[j_idx])
    y_pred = (y_score >= best_thr).astype(int)
    return {
        "threshold": best_thr,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def plot_roc_curve(y_true, y_score, model_name, threshold):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    j_idx = int((tpr - fpr).argmax())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC curve (AUC={auc:.4f})")
    ax.scatter(fpr[j_idx], tpr[j_idx], color="crimson", zorder=5, s=80,
               label=f"Youden's J threshold={threshold:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - Test Set ({model_name})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, model_name):
    labels = ["Fully Paid (0)", "Charged Off (1)"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([
        [f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix - Test Set ({model_name})")
    fig.tight_layout()
    return fig


def evaluate(data_dir, artifact_dir):
    model_path = os.path.join(artifact_dir, "best_model.pkl")
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    model_name = type(model).__name__
    logger.info("Loaded best model: %s", model_name)

    test_df = pd.read_csv(os.path.join(data_dir, "test_features.csv"))
    logger.info("Test set: %s", test_df.shape)

    X_test = test_df.drop(columns=[TARGET]).values
    y_test = test_df[TARGET].values.astype(int)
    logger.info("Class balance (test) — 0: %.1f%%  1: %.1f%%",
                (y_test == 0).mean() * 100, (y_test == 1).mean() * 100)

    y_score = model.predict_proba(X_test)[:, 1]
    proba = eval_proba(y_test, y_score)
    thresh = eval_at_youden(y_test, y_score)

    logger.info("FINAL TEST SET EVALUATION - %s", model_name)
    logger.info("AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f",
                proba["auc_roc"], proba["auc_pr"], proba["ks"])
    logger.info("Youden threshold=%.4f | Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
                thresh["threshold"], thresh["accuracy"],
                thresh["precision"], thresh["recall"], thresh["f1"])
    cm = thresh["confusion_matrix"]
    logger.info("Confusion matrix: TN=%d  FP=%d  FN=%d  TP=%d",
                cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])

    mlflow_uri = f"file:{os.path.join(artifact_dir, 'mlruns')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("lending_club_default_prediction")

    roc_fig = plot_roc_curve(y_test, y_score, model_name, thresh["threshold"])
    cm_fig = plot_confusion_matrix(thresh["confusion_matrix"], model_name)

    with mlflow.start_run(run_name="FinalTestEvaluation"):
        mlflow.log_param("model", model_name)
        mlflow.log_metrics({
            "test_auc_roc": proba["auc_roc"],
            "test_auc_pr": proba["auc_pr"],
            "test_ks": proba["ks"],
            "test_threshold": thresh["threshold"],
            "test_accuracy": thresh["accuracy"],
            "test_precision": thresh["precision"],
            "test_recall": thresh["recall"],
            "test_f1": thresh["f1"],
        })
        mlflow.log_figure(roc_fig, "plots/roc_curve_test.png")
        mlflow.log_figure(cm_fig, "plots/confusion_matrix_test.png")

    plt.close(roc_fig)
    plt.close(cm_fig)
    logger.info("Logged to MLflow (%s)", mlflow_uri)


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(root, "data", "processed"))
    ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", DATA_DIR)

    evaluate(DATA_DIR, ARTIFACT_DIR)
