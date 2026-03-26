"""
13_model_selection.py  —  Model Selection with Optuna Hyperparameter Tuning

Trains and compares 4 models on the val set, selects the best by AUC-ROC.

Models:

  Logistic Regression      : baseline, no tuning
  Decision Tree            : Optuna TPE, 50 trials
  XGBoost                  : Optuna TPE, 50 trials
  HistGradientBoosting     : Optuna TPE, 50 trials

Optuna objective          : maximize val AUC-ROC
Class imbalance handling  : class_weight='balanced' (LR, DT, HGB)
                            scale_pos_weight (XGBoost)

"""

import logging
import os
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET        = "charged_off"
RANDOM_STATE  = 42
OPTUNA_TRIALS = 50


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_proba(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    auc_roc = roc_auc_score(y_true, y_score)
    auc_pr  = average_precision_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float((tpr - fpr).max())
    return {"auc_roc": auc_roc, "auc_pr": auc_pr, "ks": ks}


def _plot_roc_comparison(results: list, y_val: np.ndarray) -> plt.Figure:
    """
    Plot ROC curves for all models on the val set in a single figure.
    Each curve is labeled with model name and AUC-ROC value.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for r in results:
        fpr, tpr, _ = roc_curve(y_val, r["y_score_val"])
        ax.plot(fpr, tpr, label=f"{r['model']}  (AUC={r['val_auc_roc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — Val Set")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Baseline — Logistic Regression (no Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic(X_train, y_train, X_val, y_val) -> dict:
    logger.info("── Logistic Regression (baseline) ──────────────────────────")
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
    logger.info(
        "  LR | fit %.1fs | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f",
        fit_time, metrics["auc_roc"], metrics["auc_pr"], metrics["ks"],
    )
    return {
        "model":        "LogisticRegression",
        "type":         "baseline",
        "params":       {"class_weight": "balanced", "max_iter": 1000},
        "val_auc_roc":  metrics["auc_roc"],
        "val_auc_pr":   metrics["auc_pr"],
        "val_ks":       metrics["ks"],
        "fit_time_s":   fit_time,
        "model_obj":    model,
        "y_score_val":  y_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tuned — Decision Tree
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna_dt(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS) -> dict:
    logger.info("── Decision Tree (Optuna, %d trials) ──────────────────────", n_trials)

    def objective(trial):
        params = {
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 50),
            "criterion":         trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight":      "balanced",
            "random_state":      RANDOM_STATE,
        }
        mdl = DecisionTreeClassifier(**params)
        mdl.fit(X_train, y_train)
        return roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    fit_time = time.time() - t0

    best_params = {
        **study.best_params,
        "class_weight": "balanced",
        "random_state":  RANDOM_STATE,
    }
    model = DecisionTreeClassifier(**best_params)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = _evaluate_proba(y_val, y_score)
    logger.info(
        "  DT best | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f",
        metrics["auc_roc"], metrics["auc_pr"], metrics["ks"],
    )
    logger.info("  Best params: %s", study.best_params)

    return {
        "model":        "DecisionTree",
        "type":         "tuned",
        "params":       best_params,
        "val_auc_roc":  metrics["auc_roc"],
        "val_auc_pr":   metrics["auc_pr"],
        "val_ks":       metrics["ks"],
        "fit_time_s":   fit_time,
        "model_obj":    model,
        "y_score_val":  y_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tuned — XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna_xgb(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS) -> dict:
    logger.info("── XGBoost (Optuna, %d trials) ─────────────────────────────", n_trials)

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    logger.info("  scale_pos_weight = %.4f", scale_pos_weight)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": scale_pos_weight,
            "tree_method":      "hist",
            "eval_metric":      "auc",
            "random_state":     RANDOM_STATE,
            "verbosity":        0,
        }
        mdl = XGBClassifier(**params)
        mdl.fit(X_train, y_train)
        return roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    fit_time = time.time() - t0

    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "tree_method":      "hist",
        "eval_metric":      "auc",
        "random_state":     RANDOM_STATE,
        "verbosity":        0,
    }
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = _evaluate_proba(y_val, y_score)
    logger.info(
        "  XGB best | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f",
        metrics["auc_roc"], metrics["auc_pr"], metrics["ks"],
    )
    logger.info("  Best params: %s", study.best_params)

    return {
        "model":        "XGBoost",
        "type":         "tuned",
        "params":       best_params,
        "val_auc_roc":  metrics["auc_roc"],
        "val_auc_pr":   metrics["auc_pr"],
        "val_ks":       metrics["ks"],
        "fit_time_s":   fit_time,
        "model_obj":    model,
        "y_score_val":  y_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tuned — HistGradientBoosting
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna_hgb(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS):
    logger.info("── HistGradientBoosting (Optuna, %d trials) ────────────────", n_trials)

    def objective(trial):
        params = {
            "max_iter":          trial.suggest_int("max_iter", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 15),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 10, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
            "max_leaf_nodes":    trial.suggest_int("max_leaf_nodes", 15, 127),
            "class_weight":      "balanced",
            "random_state":      RANDOM_STATE,
        }
        mdl = HistGradientBoostingClassifier(**params)
        mdl.fit(X_train, y_train)
        return roc_auc_score(y_val, mdl.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    fit_time = time.time() - t0

    best_params = {
        **study.best_params,
        "class_weight": "balanced",
        "random_state":  RANDOM_STATE,
    }
    model = HistGradientBoostingClassifier(**best_params)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = _evaluate_proba(y_val, y_score)
    logger.info(
        "  HGB best | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f",
        metrics["auc_roc"], metrics["auc_pr"], metrics["ks"],
    )
    logger.info("  Best params: %s", study.best_params)

    return {
        "model":        "HistGradientBoosting",
        "type":         "tuned",
        "params":       best_params,
        "val_auc_roc":  metrics["auc_roc"],
        "val_auc_pr":   metrics["auc_pr"],
        "val_ks":       metrics["ks"],
        "fit_time_s":   fit_time,
        "model_obj":    model,
        "y_score_val":  y_score,
    }


# Main

def train(data_dir: str, artifact_dir: str) -> None:
    os.makedirs(artifact_dir, exist_ok=True)

    # Load data
    logger.info("Loading feature data from %s", data_dir)
    train_df = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "val_features.csv"))
    logger.info("Loaded — train %s  val %s", train_df.shape, val_df.shape)

    X_train = train_df.drop(columns=[TARGET]).values
    y_train = train_df[TARGET].values.astype(int)
    X_val   = val_df.drop(columns=[TARGET]).values
    y_val   = val_df[TARGET].values.astype(int)
    logger.info(
        "Class balance (train) — 0: %.1f%%  1: %.1f%%",
        (y_train == 0).mean() * 100, (y_train == 1).mean() * 100,
    )

    # Run all 4 models
    results = [
        run_logistic(X_train, y_train, X_val, y_val),
        run_optuna_dt(X_train, y_train, X_val, y_val),
        run_optuna_xgb(X_train, y_train, X_val, y_val),
        run_optuna_hgb(X_train, y_train, X_val, y_val),
    ]

    # Val comparison table
    sep = "=" * 72
    logger.info("\n%s", sep)
    logger.info("VAL SET COMPARISON")
    logger.info("%s", sep)
    logger.info(
        "%-26s %-10s %-10s %-10s %-8s %-10s",
        "Model", "Type", "AUC-ROC", "AUC-PR", "KS", "Time(s)",
    )
    logger.info("%s", "-" * 72)
    for r in results:
        logger.info(
            "%-26s %-10s %-10.4f %-10.4f %-8.4f %-10.1f",
            r["model"], r["type"],
            r["val_auc_roc"], r["val_auc_pr"], r["val_ks"], r["fit_time_s"],
        )

    # Select best model (by val AUC-ROC among tuned models)
    tuned  = [r for r in results if r["type"] == "tuned"]
    best   = max(tuned, key=lambda r: r["val_auc_roc"])
    logger.info(
        "\nBest model: %s  (val AUC-ROC = %.4f)",
        best["model"], best["val_auc_roc"],
    )

    # Save artifacts
    model_path = os.path.join(artifact_dir, "best_model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(best["model_obj"], fh)
    logger.info("Best model saved → %s", model_path)

    results_df = pd.DataFrame([
        {
            "model":       r["model"],
            "type":        r["type"],
            "val_auc_roc": r["val_auc_roc"],
            "val_auc_pr":  r["val_auc_pr"],
            "val_ks":      r["val_ks"],
            "fit_time_s":  r["fit_time_s"],
        }
        for r in results
    ])
    results_csv = os.path.join(artifact_dir, "model_results.csv")
    results_df.to_csv(results_csv, index=False)
    logger.info("Model results saved → %s", results_csv)

    # MLflow logging
    mlflow_uri = f"file:{os.path.join(artifact_dir, 'mlruns')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("lending_club_default_prediction")
    logger.info("Logging to MLflow (%s)", mlflow_uri)

    # ROC comparison figure — generated once, logged to the best model's run
    roc_fig = _plot_roc_comparison(results, y_val)

    for r in results:
        is_best  = r["model"] == best["model"]
        run_name = f"{r['model']}_BEST" if is_best else r["model"]

        with mlflow.start_run(run_name=run_name):
            flat_params = {
                k: v for k, v in r["params"].items()
                if isinstance(v, (int, float, str, bool))
            }
            mlflow.log_params(flat_params)
            mlflow.log_metrics({
                "val_auc_roc": r["val_auc_roc"],
                "val_auc_pr":  r["val_auc_pr"],
                "val_ks":      r["val_ks"],
                "fit_time_s":  r["fit_time_s"],
            })
            if is_best:
                mlflow.sklearn.log_model(r["model_obj"], "best_model")
                mlflow.log_figure(roc_fig, "plots/roc_comparison_val.png")

    plt.close(roc_fig)

    logger.info("MLflow logging complete")
    logger.info("Done. Next: 14_final_evaluation.py")


if __name__ == "__main__":
    _src_dir  = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.join(_src_dir, "..")

    DATA_DIR     = os.getenv("DATA_DIR",     os.path.join(_root_dir, "data", "processed"))
    ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", DATA_DIR)

    train(DATA_DIR, ARTIFACT_DIR)
