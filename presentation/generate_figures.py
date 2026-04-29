"""
Generate static figures for the Phase 4 presentation deck.

Reads the held-out 2017 test set (data/processed/test_features.csv) and the
trained model (models/best_model.pkl), then produces:
  - figures/roc_curve.png
  - figures/pr_curve.png
  - figures/confusion_matrix.png
  - figures/model_comparison.png   (val metrics from models/model_results.csv)
  - figures/feature_importance.png (top 15 XGBoost gain importances)
  - figures/pipeline_diagram.png   (8-stage preprocessing pipeline schematic)
  - figures/architecture_diagram.png (P1 -> P2 -> P3 -> P4 evolution)

Also prints a summary metrics block to stdout that gets pasted into the slides.
"""

import __main__
import importlib.util
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "presentation" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_SRC = ROOT / "src" / "06_data_processing_pipeline.py"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
TEST_FEATURES = ROOT / "data" / "processed" / "test_features.csv"
MODEL_RESULTS = ROOT / "models" / "model_results.csv"

# pickle workaround - same as MCP server / streamlit app
spec = importlib.util.spec_from_file_location("_pipeline_module", PIPELINE_SRC)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for name, obj in vars(module).items():
    if not name.startswith("__"):
        setattr(__main__, name, obj)

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

print("Loading test features...")
test_df = pd.read_csv(TEST_FEATURES)
y_test = test_df["charged_off"].values.astype(int)
X_test = test_df.drop(columns=["charged_off"]).values
feature_names = test_df.drop(columns=["charged_off"]).columns.tolist()

print("Scoring test set...")
y_score = MODEL.predict_proba(X_test)[:, 1]


def youden_threshold(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = (tpr - fpr).argmax()
    return float(thr[j]), float(fpr[j]), float(tpr[j])


thr, fpr_at, tpr_at = youden_threshold(y_test, y_score)
y_pred = (y_score >= thr).astype(int)

# headline metrics
fpr_all, tpr_all, _ = roc_curve(y_test, y_score)
auc_roc = auc(fpr_all, tpr_all)
auc_pr = average_precision_score(y_test, y_score)
ks = float((tpr_all - fpr_all).max())
acc = float((y_pred == y_test).mean())
prec = float(((y_pred == 1) & (y_test == 1)).sum() / max((y_pred == 1).sum(), 1))
rec = float(((y_pred == 1) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1))
f1 = 2 * prec * rec / max(prec + rec, 1e-9)

print()
print("=" * 60)
print("PHASE 4 TEST-SET METRICS  (paste into slides)")
print("=" * 60)
print(f"  N (test):          {len(y_test):,}")
print(f"  Default rate:      {y_test.mean():.1%}")
print(f"  AUC-ROC:           {auc_roc:.4f}")
print(f"  AUC-PR:            {auc_pr:.4f}")
print(f"  KS:                {ks:.4f}")
print(f"  Threshold (Youden):{thr:.4f}")
print(f"  Accuracy:          {acc:.4f}")
print(f"  Precision:         {prec:.4f}")
print(f"  Recall:            {rec:.4f}")
print(f"  F1:                {f1:.4f}")
print("=" * 60)
print()


# ROC curve
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
ax.plot(fpr_all, tpr_all, color="#3b6db5", lw=2.4, label=f"XGBoost  (AUC = {auc_roc:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Random")
ax.scatter([fpr_at], [tpr_at], color="#d73027", s=70, zorder=5,
           label=f"Operating point  (thr={thr:.2f}, KS={ks:.3f})")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve  -  Test set (2017 issuances)")
ax.legend(loc="lower right", frameon=True)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "roc_curve.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"  wrote {FIG_DIR/'roc_curve.png'}")


# PR curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_score)
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
ax.plot(rec_curve, prec_curve, color="#3b6db5", lw=2.4, label=f"XGBoost  (AP = {auc_pr:.3f})")
ax.axhline(y_test.mean(), ls="--", color="grey", lw=1,
           label=f"Class prior ({y_test.mean():.1%})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall curve  -  Test set")
ax.legend(loc="upper right", frameon=True)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "pr_curve.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"  wrote {FIG_DIR/'pr_curve.png'}")


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum() * 100
fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
sns.heatmap(
    cm, annot=[[f"{cm[i,j]:,}\n({cm_pct[i,j]:.1f}%)" for j in (0, 1)] for i in (0, 1)],
    fmt="", cmap="Blues", cbar=False, ax=ax,
    xticklabels=["Predicted Paid", "Predicted Default"],
    yticklabels=["Actual Paid", "Actual Default"],
)
ax.set_title(f"Confusion matrix  -  threshold = {thr:.2f}")
plt.tight_layout()
plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"  wrote {FIG_DIR/'confusion_matrix.png'}")


# Model comparison (validation metrics from CSV)
res = pd.read_csv(MODEL_RESULTS)
res = res.sort_values("val_auc_roc", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
y_pos = np.arange(len(res))
bars = ax.barh(y_pos, res["val_auc_roc"], color=["#9ec5e8" if m != "XGBoost" else "#3b6db5"
                                                  for m in res["model"]])
ax.set_yticks(y_pos)
ax.set_yticklabels(res["model"])
ax.set_xlabel("Validation AUC-ROC")
ax.set_xlim(0.65, 0.75)
for i, (auc_v, ks_v) in enumerate(zip(res["val_auc_roc"], res["val_ks"])):
    ax.text(auc_v + 0.001, i, f"AUC {auc_v:.4f}  /  KS {ks_v:.3f}",
            va="center", fontsize=9)
ax.set_title("Model bake-off  -  validation set")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "model_comparison.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"  wrote {FIG_DIR/'model_comparison.png'}")


# XGBoost gain feature importance (top 15)
booster = MODEL.get_booster()
gain = booster.get_score(importance_type="gain")
imp_full = []
for fname, fval in gain.items():
    if fname.startswith("f") and fname[1:].isdigit():
        idx = int(fname[1:])
        imp_full.append((feature_names[idx] if idx < len(feature_names) else fname, fval))
    else:
        imp_full.append((fname, fval))
imp = pd.Series({n: v for n, v in imp_full}).sort_values(ascending=False).head(15)[::-1]

fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
ax.barh(np.arange(len(imp)), imp.values, color="#3b6db5")
ax.set_yticks(np.arange(len(imp)))
ax.set_yticklabels([n.replace("_", " ") for n in imp.index], fontsize=9)
ax.set_xlabel("XGBoost gain")
ax.set_title("Top 15 features by gain  -  global importance")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
plt.close()
print(f"  wrote {FIG_DIR/'feature_importance.png'}")


# Pipeline diagram - 8-stage preprocessing schematic
def draw_pipeline_diagram():
    stages = [
        ("DropCorrelated",        "13 highly-correlated columns"),
        ("DateExtractor",         "issue_d, earliest_cr_line\n-> year, month,\n  credit_history_months"),
        ("FeatureConstructor",    "loan_to_income\nmonthly_payment_to_income\ndelinq_rate"),
        ("OutlierCapper",         "dti capped at p99\n(max 999 was data error)"),
        ("MissingIndicatorAdder", "mths_since_recent_inq\nmissingness flag"),
        ("MacroJoiner",           "FRED UNRATE merged\non issue_year, issue_month"),
        ("RareCategoryMerger",    "purpose, home_ownership\ncategories <50 -> 'other'"),
        ("ColumnPreprocessor",    "skewed -> log1p+scale\nbinarize -> >0\nordinal -> sub_grade\nOHE -> categoricals"),
    ]
    fig, ax = plt.subplots(figsize=(15, 5.5), dpi=150)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")
    box_w, box_h = 1.85, 3.2
    gap = 0.05
    for i, (name, body) in enumerate(stages):
        x = i * (box_w + gap)
        ax.add_patch(FancyBboxPatch(
            (x, 1.5), box_w, box_h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.6, edgecolor="#3b6db5", facecolor="#eaf2fb",
        ))
        ax.text(x + box_w / 2, 1.5 + box_h - 0.45, name,
                ha="center", va="top", fontsize=10, fontweight="bold", color="#0e3d70")
        ax.text(x + box_w / 2, 1.5 + box_h / 2 - 0.2, body,
                ha="center", va="center", fontsize=8.2, color="#333")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + box_w + gap + 0.04, 1.5 + box_h / 2),
                        xytext=(x + box_w - 0.04, 1.5 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#3b6db5", lw=1.5))
    ax.text(0, 6.5, "Preprocessing pipeline  -  fit on train only, applied to val and test",
            fontsize=12, fontweight="bold", color="#0e3d70")
    ax.text(0, 0.6, "Input: cleaned dataframe (49 columns)",
            fontsize=9, fontstyle="italic", color="#666")
    ax.text(15.5, 0.6, "Output: 110 features",
            fontsize=9, fontstyle="italic", color="#666", ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pipeline_diagram.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {FIG_DIR/'pipeline_diagram.png'}")


draw_pipeline_diagram()


# Architecture diagram - P1 -> P2 -> P3 -> P4 evolution
def draw_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    phases = [
        ("Phase 1", "Local pandas",
         "archive.zip\nfilter 2014-2017\ndrop leakage\ntime-based split",
         "#fde0c5", "#cf6f1c"),
        ("Phase 2", "scikit-learn pipeline\n+ XGBoost",
         "8-stage preprocessing\nOptuna tuning\nMLflow tracking\nbest_model.pkl",
         "#cfe2cb", "#3a7a35"),
        ("Phase 3", "Spark + Delta + MCP",
         "Bronze / Silver / Gold\nPySpark ML\nFRED macro overlay\nMCP server (FastMCP)",
         "#cee0f1", "#1f5b9c"),
        ("Phase 4", "Streamlit app\n+ MCP demo",
         "interactive predictor\nSHAP-style \"why\" panel\nclient-facing UI",
         "#e6d9ec", "#6a3c8a"),
    ]
    box_w, box_h = 3.0, 3.8
    gap = 0.4
    for i, (phase, title, body, fill, edge) in enumerate(phases):
        x = i * (box_w + gap) + 0.2
        ax.add_patch(FancyBboxPatch(
            (x, 1.7), box_w, box_h,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            linewidth=2, edgecolor=edge, facecolor=fill,
        ))
        ax.text(x + box_w / 2, 1.7 + box_h - 0.4, phase,
                ha="center", va="top", fontsize=14, fontweight="bold", color=edge)
        ax.text(x + box_w / 2, 1.7 + box_h - 1.2, title,
                ha="center", va="top", fontsize=11, color="#222")
        ax.text(x + box_w / 2, 1.7 + box_h / 2 - 0.5, body,
                ha="center", va="center", fontsize=9.5, color="#333")
        if i < len(phases) - 1:
            ax.annotate("", xy=(x + box_w + gap, 1.7 + box_h / 2),
                        xytext=(x + box_w - 0.1, 1.7 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#666", lw=2))
    ax.text(0.2, 6.7, "Same use case, four implementations",
            fontsize=15, fontweight="bold", color="#222")
    ax.text(0.2, 6.2,
            "Lender's question:  given a borrower, will this loan default?",
            fontsize=10.5, fontstyle="italic", color="#555")
    ax.text(0.2, 0.9, "Local prototype  -->  Tuned ML model  -->  Scalable + macro-aware  -->  Conversational + visual",
            fontsize=9.5, color="#444")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "architecture_diagram.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  wrote {FIG_DIR/'architecture_diagram.png'}")


draw_architecture_diagram()

print()
print("All figures written to", FIG_DIR)
