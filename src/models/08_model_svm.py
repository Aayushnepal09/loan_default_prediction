"""
Phase 2: Support Vector Machine (LinearSVC)
This script trains a fast, linear SVM on the full dataset. We wrap it in a CalibratedClassifierCV
so we can extract predicted probabilities required for calculating the ROC-AUC score.
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


TARGET = "charged_off"
RANDOM_STATE = 42
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "reports", "figures", "svm",
)


def evaluate(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "auc_roc": roc_auc_score(y_true, y_score),
        "auc_pr": average_precision_score(y_true, y_score),
        "ks": float((tpr - fpr).max()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
    }


def save_metrics_table(metrics, best_C, train_n, val_n, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    rows = [
        ["AUC-ROC",   f"{metrics['auc_roc']:.4f}"],
        ["AUC-PR",    f"{metrics['auc_pr']:.4f}"],
        ["KS",        f"{metrics['ks']:.4f}"],
        ["Accuracy",  f"{metrics['accuracy']:.4f}"],
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall",    f"{metrics['recall']:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.4, 1.6)

    for col in range(2):
        tbl[0, col].set_facecolor("#2c5f8a")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    for row in range(1, len(rows) + 1):
        color = "#eaf2fb" if row % 2 == 0 else "white"
        for col in range(2):
            tbl[row, col].set_facecolor(color)

    ax.set_title(
        f"LinearSVC Performance Metrics (C={best_C})\ntrain={train_n:,}  val={val_n:,}  threshold=0.5",
        fontsize=10, pad=12,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "svm_metrics_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % (path,))


def save_plots(y_val, y_score, metrics, best_C, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(metrics["fpr"], metrics["tpr"], color="steelblue", lw=2,
            label=f"LinearSVC (C={best_C})  AUC={metrics['auc_roc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("LinearSVC - ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, "svm_roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved %s" % (path,))

    # Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_score)
    baseline = y_val.mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall_vals, precision_vals, color="darkorange", lw=2,
            label=f"LinearSVC (C={best_C})  AP={metrics['auc_pr']:.4f}")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("LinearSVC - Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = os.path.join(out_dir, "svm_pr_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved %s" % (path,))


def run(X_train, y_train, X_val, y_val, save_plots_flag=False, figures_dir=FIGURES_DIR):
    best_auc, best_C, best_model, best_fit_time = -1.0, None, None, 0.0

    for C in [0.01, 0.1, 1.0]:
        print("Training LinearSVC C=%.2f on %d rows" % (C, len(X_train)))
        t0 = time.time()
        base = LinearSVC(C=C, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
        model = CalibratedClassifierCV(base, cv=3, n_jobs=-1)
        model.fit(X_train, y_train)
        fit_time = time.time() - t0

        y_score = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_score)
        print("  C=%.2f | fit=%.1fs | AUC-ROC=%.4f" % (C, fit_time, auc))

        if auc > best_auc:
            best_auc, best_C, best_model, best_fit_time = auc, C, model, fit_time

    y_score_best = best_model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, y_score_best)

    print("Best C=%.2f | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f" % (
                best_C, metrics["auc_roc"], metrics["auc_pr"], metrics["ks"]))
    print("Accuracy=%.4f  Precision=%.4f  Recall=%.4f" % (
                metrics["accuracy"], metrics["precision"], metrics["recall"]))

    if save_plots_flag:
        save_metrics_table(metrics, best_C, len(X_train), len(X_val), figures_dir)
        save_plots(y_val, y_score_best, metrics, best_C, figures_dir)

    return {
        "model": "LinearSVC",
        "type": "dead_end",
        "params": {"kernel": "linear", "C": best_C, "class_weight": "balanced"},
        "val_auc_roc": metrics["auc_roc"],
        "val_auc_pr": metrics["auc_pr"],
        "val_ks": metrics["ks"],
        "val_accuracy": metrics["accuracy"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "fit_time_s": best_fit_time,
    }


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(root, "data", "processed"))

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_features.csv"))

    feature_cols = [c for c in train_df.columns if c != TARGET]
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET].values.astype(int)
    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET].values.astype(int)

    run(X_train, y_train, X_val, y_val, save_plots_flag=True)
