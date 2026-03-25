"""
09_model_lr.py  —  Baseline: Logistic Regression

Default parameters, no hyperparameter tuning.
Serves as the minimum performance bar that all tuned models must exceed.

class_weight='balanced' adjusts loss weights to account for the 80/20
class imbalance without resampling.

Input  : data/processed/train_features.csv, val_features.csv
Output : console log (no artifact saved)
"""

import logging
import os
import time
import warnings

import numpy as np
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

TARGET       = "charged_off"
RANDOM_STATE = 42


def _evaluate_proba(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr  = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float((tpr - fpr).max())
    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "ks": ks}


def run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
) -> dict:
    logger.info("Training Logistic Regression on %d rows", len(X_train))

    t0    = time.time()
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = _evaluate_proba(y_val, y_score)

    sep = "=" * 60
    logger.info(sep)
    logger.info("RESULT  Logistic Regression  (baseline, no tuning)")
    logger.info("  AUC-ROC  : %.4f", metrics["auc_roc"])
    logger.info("  AUC-PR   : %.4f", metrics["auc_pr"])
    logger.info("  KS       : %.4f", metrics["ks"])
    logger.info("  Fit time : %.1fs", fit_time)
    logger.info(sep)

    return {
        "model":       "LogisticRegression",
        "type":        "baseline",
        "params":      {"class_weight": "balanced", "max_iter": 1000},
        "val_auc_roc": metrics["auc_roc"],
        "val_auc_pr":  metrics["auc_pr"],
        "val_ks":      metrics["ks"],
        "fit_time_s":  fit_time,
        "model_obj":   model,
    }


if __name__ == "__main__":
    _src_dir  = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.join(_src_dir, "..")
    DATA_DIR  = os.getenv("DATA_DIR", os.path.join(_root_dir, "data", "processed"))

    logger.info("Loading data from %s", DATA_DIR)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val_features.csv"))

    X_train = train_df.drop(columns=[TARGET]).values
    y_train = train_df[TARGET].values.astype(int)
    X_val   = val_df.drop(columns=[TARGET]).values
    y_val   = val_df[TARGET].values.astype(int)

    run(X_train, y_train, X_val, y_val)
