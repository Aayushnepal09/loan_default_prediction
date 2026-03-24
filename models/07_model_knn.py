"""
07_model_knn.py  —  K-Nearest Neighbors

Trains KNN on a 20K random subset of the training data (the full 833K set
is infeasible due to O(n × d) prediction cost) and evaluates on the full
val set.  Tries n_neighbors ∈ {5, 10, 20}; reports the best result.

Input  : data/processed/train_features.csv, val_features.csv
Output : console log (no artifact saved)
"""

import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET       = "charged_off"
RANDOM_STATE = 42
KNN_SUBSET   = 20_000


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
    subset_size: int = KNN_SUBSET,
) -> dict:
    # Random subset 
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=min(subset_size, len(X_train)), replace=False)
    X_sub, y_sub = X_train[idx], y_train[idx]
    logger.info("KNN training subset : %d rows  (full train: %d)", len(X_sub), len(X_train))

    # Try k ∈ {5, 10, 20} — keep best val AUC-ROC
    best_auc, best_k, best_model, best_fit_time = -1.0, None, None, 0.0

    for k in [5, 10, 20]:
        t0  = time.time()
        mdl = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        mdl.fit(X_sub, y_sub)
        fit_time = time.time() - t0

        t1        = time.time()
        y_score   = mdl.predict_proba(X_val)[:, 1]
        pred_time = time.time() - t1

        auc = roc_auc_score(y_val, y_score)
        logger.info(
            "  k=%2d | fit %.1fs | predict %.1fs | val AUC-ROC=%.4f",
            k, fit_time, pred_time, auc,
        )
        if auc > best_auc:
            best_auc, best_k, best_model, best_fit_time = auc, k, mdl, fit_time

    # Final evaluation
    y_score_best = best_model.predict_proba(X_val)[:, 1]
    metrics = _evaluate_proba(y_val, y_score_best)

   
    logger.info(sep)
    logger.info("RESULT  KNN  (best k=%d, subset=%d)", best_k, subset_size)
    logger.info("  AUC-ROC  : %.4f", metrics["auc_roc"])
    logger.info("  AUC-PR   : %.4f", metrics["auc_pr"])
    logger.info("  KS       : %.4f", metrics["ks"])
    logger.info("  Fit time : %.1fs  (subset fit only)", best_fit_time)
    logger.info(sep)

    return {
        "model":       "KNN",
        "type":        "dead_end",
        "params":      {"n_neighbors": best_k, "subset_size": subset_size},
        "val_auc_roc": metrics["auc_roc"],
        "val_auc_pr":  metrics["auc_pr"],
        "val_ks":      metrics["ks"],
        "fit_time_s":  best_fit_time,
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
