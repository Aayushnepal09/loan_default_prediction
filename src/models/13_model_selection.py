"""
Phase 2: Model Selection
This script compares the validation metrics of our different models and saves the single
best-performing algorithm as a serialized pickle file for final evaluation and deployment.
"""

import os
import pickle
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


TARGET = "charged_off"
RANDOM_STATE = 42
OPTUNA_TRIALS = 50
FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"


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
    }


def save_model_plots(result, y_val, figures_dir):
    name = result["model"].lower().replace("histgradientboosting", "hgb")
    out_dir = figures_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]
    y_score = result["y_score_val"]
    label = result["model"]

    # Metrics table
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
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"], cellLoc="center", loc="center")
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
    model_type = result["type"].capitalize()
    ax.set_title(f"{label} ({model_type})\nval={len(y_val):,}  threshold=0.5", fontsize=10, pad=12)
    fig.tight_layout()
    path = out_dir / f"{name}_metrics_table.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved %s" % (path,))

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(metrics["fpr"], metrics["tpr"], color="steelblue", lw=2,
            label=f"{label}  AUC={metrics['auc_roc']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{label} - ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / f"{name}_roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved %s" % (path,))

    # Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_score)
    baseline = y_val.mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall_vals, precision_vals, color="darkorange", lw=2,
            label=f"{label}  AP={metrics['auc_pr']:.4f}")
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{label} - Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / f"{name}_pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved %s" % (path,))


def save_comparison_chart(results, figures_dir):
    out_dir = figures_dir / "model_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = [r["model"] for r in results]
    auc_rocs = [r["val_auc_roc"] for r in results]
    auc_prs = [r["val_auc_pr"] for r in results]
    kss = [r["val_ks"] for r in results]
    bar_colors = ["#d9534f" if r["type"] == "baseline" else "steelblue" for r in results]

    x = range(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width for i in x], auc_rocs, width, label="AUC-ROC",
                   color=bar_colors, alpha=0.9)
    bars2 = ax.bar(list(x), auc_prs, width, label="AUC-PR",
                   color=["#e8837f" if c == "#d9534f" else "lightsteelblue" for c in bar_colors], alpha=0.9)
    bars3 = ax.bar([i + width for i in x], kss, width, label="KS",
                   color=["#f2b3b1" if c == "#d9534f" else "#aed4e8" for c in bar_colors], alpha=0.9)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Validation Set\n(red = LR baseline, blue = tuned models)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, max(auc_rocs) + 0.08)
    fig.tight_layout()
    path = out_dir / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved %s" % (path,))


def run_logistic(X_train, y_train, X_val, y_val):
    print("Logistic Regression (baseline)")
    t0 = time.time()
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, y_score)
    print("LR | fit=%.1fs | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f" % (
                fit_time, metrics["auc_roc"], metrics["auc_pr"], metrics["ks"]))

    return {
        "model": "LogisticRegression",
        "type": "baseline",
        "params": {"class_weight": "balanced", "max_iter": 1000},
        "val_auc_roc": metrics["auc_roc"],
        "val_auc_pr": metrics["auc_pr"],
        "val_ks": metrics["ks"],
        "fit_time_s": fit_time,
        "model_obj": model,
        "y_score_val": y_score,
        "metrics": metrics,
    }


def run_optuna_dt(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS):
    print("Decision Tree (Optuna, %d trials)" % (n_trials,))

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
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

    best_params = {**study.best_params, "class_weight": "balanced", "random_state": RANDOM_STATE}
    model = DecisionTreeClassifier(**best_params)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, y_score)
    print("DT best | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f" % (
                metrics["auc_roc"], metrics["auc_pr"], metrics["ks"]))
    print("DT best params: %s" % (study.best_params,))

    return {
        "model": "DecisionTree",
        "type": "tuned",
        "params": best_params,
        "val_auc_roc": metrics["auc_roc"],
        "val_auc_pr": metrics["auc_pr"],
        "val_ks": metrics["ks"],
        "fit_time_s": fit_time,
        "model_obj": model,
        "y_score_val": y_score,
        "metrics": metrics,
    }


def run_optuna_xgb(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS):
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print("XGBoost (Optuna, %d trials)  scale_pos_weight=%.2f" % (n_trials, scale_pos_weight))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "eval_metric": "auc",
            "random_state": RANDOM_STATE,
            "verbosity": 0,
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
        "tree_method": "hist",
        "eval_metric": "auc",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, y_score)
    print("XGB best | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f" % (
                metrics["auc_roc"], metrics["auc_pr"], metrics["ks"]))
    print("XGB best params: %s" % (study.best_params,))

    return {
        "model": "XGBoost",
        "type": "tuned",
        "params": best_params,
        "val_auc_roc": metrics["auc_roc"],
        "val_auc_pr": metrics["auc_pr"],
        "val_ks": metrics["ks"],
        "fit_time_s": fit_time,
        "model_obj": model,
        "y_score_val": y_score,
        "metrics": metrics,
    }


def run_optuna_hgb(X_train, y_train, X_val, y_val, n_trials=OPTUNA_TRIALS):
    print("HistGradientBoosting (Optuna, %d trials)" % (n_trials,))

    def objective(trial):
        params = {
            "max_iter": trial.suggest_int("max_iter", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
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

    best_params = {**study.best_params, "class_weight": "balanced", "random_state": RANDOM_STATE}
    model = HistGradientBoostingClassifier(**best_params)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_val)[:, 1]
    metrics = evaluate(y_val, y_score)
    print("HGB best | AUC-ROC=%.4f  AUC-PR=%.4f  KS=%.4f" % (
                metrics["auc_roc"], metrics["auc_pr"], metrics["ks"]))
    print("HGB best params: %s" % (study.best_params,))

    return {
        "model": "HistGradientBoosting",
        "type": "tuned",
        "params": best_params,
        "val_auc_roc": metrics["auc_roc"],
        "val_auc_pr": metrics["auc_pr"],
        "val_ks": metrics["ks"],
        "fit_time_s": fit_time,
        "model_obj": model,
        "y_score_val": y_score,
        "metrics": metrics,
    }


def train(data_dir, artifact_dir):
    os.makedirs(artifact_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val_features.csv"))
    print("Loaded train=%s  val=%s" % (train_df.shape, val_df.shape))

    X_train = train_df.drop(columns=[TARGET]).values
    y_train = train_df[TARGET].values.astype(int)
    X_val = val_df.drop(columns=[TARGET]).values
    y_val = val_df[TARGET].values.astype(int)
    print("Class balance (train) — 0: %.1f%%  1: %.1f%%" % (
                (y_train == 0).mean() * 100, (y_train == 1).mean() * 100))

    results = [
        run_logistic(X_train, y_train, X_val, y_val),
        run_optuna_dt(X_train, y_train, X_val, y_val),
        run_optuna_xgb(X_train, y_train, X_val, y_val),
        run_optuna_hgb(X_train, y_train, X_val, y_val),
    ]

    print("%-26s %-10s %-10s %-10s %-8s %-10s" % (
                "Model", "Type", "AUC-ROC", "AUC-PR", "KS", "Time(s)"))
    for r in results:
        print("%-26s %-10s %-10.4f %-10.4f %-8.4f %-10.1f" % (
                    r["model"], r["type"],
                    r["val_auc_roc"], r["val_auc_pr"], r["val_ks"], r["fit_time_s"]))

    tuned = [r for r in results if r["type"] == "tuned"]
    best = max(tuned, key=lambda r: r["val_auc_roc"])
    print("Best model: %s  AUC-ROC=%.4f" % (best["model"], best["val_auc_roc"]))

    # Save per-model figures and comparison chart
    for r in results:
        save_model_plots(r, y_val, FIGURES_DIR)
    save_comparison_chart(results, FIGURES_DIR)

    # Save best model
    model_path = os.path.join(artifact_dir, "best_model.pkl")
    with open(model_path, "wb") as fh:
        pickle.dump(best["model_obj"], fh)
    print("Best model saved -> %s" % (model_path,))

    # Save results CSV
    results_df = pd.DataFrame([
        {
            "model": r["model"],
            "type": r["type"],
            "val_auc_roc": r["val_auc_roc"],
            "val_auc_pr": r["val_auc_pr"],
            "val_ks": r["val_ks"],
            "fit_time_s": r["fit_time_s"],
        }
        for r in results
    ])
    results_csv = os.path.join(artifact_dir, "model_results.csv")
    results_df.to_csv(results_csv, index=False)
    print("Results saved -> %s" % (results_csv,))

    # MLflow logging
    mlflow_uri = f"file:{os.path.join(artifact_dir, 'mlruns')}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("lending_club_default_prediction")

    for r in results:
        is_best = r["model"] == best["model"]
        run_name = f"{r['model']}_BEST" if is_best else r["model"]
        with mlflow.start_run(run_name=run_name):
            flat_params = {k: v for k, v in r["params"].items()
                           if isinstance(v, (int, float, str, bool))}
            mlflow.log_params(flat_params)
            mlflow.log_metrics({
                "val_auc_roc": r["val_auc_roc"],
                "val_auc_pr": r["val_auc_pr"],
                "val_ks": r["val_ks"],
                "fit_time_s": r["fit_time_s"],
            })
            if is_best:
                mlflow.sklearn.log_model(r["model_obj"], "best_model")

    print("MLflow logging complete")
    print("Done. Next: 14_final_evaluation.py")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = str(root / "data" / "processed")
    ARTIFACT_DIR = str(root / "models")
    train(DATA_DIR, ARTIFACT_DIR)
