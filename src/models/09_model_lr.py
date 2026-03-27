"""
09_model_lr.py - Logistic Regression (Baseline)

No hyperparameter tuning. This sets the minimum performance bar
that all other models need to beat.
"""

import logging
import os
import time
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET = "charged_off"
RANDOM_STATE = 42


def evaluate(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "auc_roc": roc_auc_score(y_true, y_score),
        "auc_pr": average_precision_score(y_true, y_score),
        "ks": float((tpr - fpr).max()),
    }


def run(X_train, y_train, X_val, y_val):
    logger.info("Training Logistic Regression on %d rows", len(X_train))

    t0 = time.time()
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, y_score)

    logger.info("Logistic Regression | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f  fit=%.1fs",
                metrics["auc_roc"], metrics["auc_pr"], metrics["ks"], fit_time)

    return {
        "model": "LogisticRegression",
        "type": "baseline",
        "params": {"class_weight": "balanced", "max_iter": 1000},
        "val_auc_roc": metrics["auc_roc"],
        "val_auc_pr": metrics["auc_pr"],
        "val_ks": metrics["ks"],
        "fit_time_s": fit_time,
        "model_obj": model,
    }


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(root, "data", "processed"))

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_features.csv"))

    X_train = train_df.drop(columns=[TARGET]).values
    y_train = train_df[TARGET].values.astype(int)
    X_val = val_df.drop(columns=[TARGET]).values
    y_val = val_df[TARGET].values.astype(int)

    run(X_train, y_train, X_val, y_val)
