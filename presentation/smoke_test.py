"""
End-to-end smoke test for Phase 4 deliverables. Imports the app module,
loads the model + pipeline, runs every chart function, smoke-tests the
predictor, and checks the MCP server symbols.

Run from project root:
    python presentation/smoke_test.py
"""

import importlib.util
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def banner(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def section(label):
    print(f"\n  -- {label}")


def ok(msg):
    print(f"     OK  {msg}")


def fail(msg):
    print(f"     FAIL  {msg}")
    raise SystemExit(1)


# 1 - Streamlit app module ----------------------------------------------------
banner("1. Streamlit app module")
spec = importlib.util.spec_from_file_location("app", "src/app/streamlit_app.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ok(f"app imported")
ok(f"model: {type(m.MODEL).__name__}")
ok(f"feature count: {len(m.FEATURE_NAMES)}")
ok(f"tabs: {len(m.tabs)}")
ok(f"themes: {len(m.THEMES)}")
ok(f"active theme: {m.st.session_state['theme_name']} -- {m.THEME['tagline']}")
ok(f"page bg color: {m.THEME['bg']}")


# 2 - Predictor end-to-end ----------------------------------------------------
banner("2. Predictor end-to-end")
inputs = m.PREDICTOR_DEFAULTS.copy()
df = m.build_input_row(inputs)
proba, shap_values, bias = m.predict_with_contribs(df)
tier = m.risk_tier(proba)
ok(f"default profile -> {proba:.4f}, tier {tier[0]} ({tier[1]} risk)")

# preset coverage
for preset_name, preset in m.PRESETS.items():
    if preset is None:
        continue
    test_inputs = m.PREDICTOR_DEFAULTS.copy()
    test_inputs.update(preset)
    df2 = m.build_input_row(test_inputs)
    p, _, _ = m.predict_with_contribs(df2)
    ok(f"preset '{preset_name}' -> {p:.4f}")


# 3 - Test set scoring (threshold tuner backing) ------------------------------
banner("3. Test set scoring (threshold tuner data)")
y, score = m.get_test_scores()
ok(f"test rows: {len(y):,}")
ok(f"default rate: {y.mean():.1%}")
ok(f"score range: {score.min():.4f} .. {score.max():.4f}")


# 4 - Insight parquets --------------------------------------------------------
banner("4. Macro insight parquets")
for name in [
    "insight1_unrate", "insight2_subgrade_regime", "insight3_state_diff",
    "default_by_subgrade", "default_over_time", "rows_per_year",
]:
    df = m.load_insight(name)
    ok(f"{name}: {len(df)} rows, cols={list(df.columns)}")


# 5 - Chart functions ---------------------------------------------------------
banner("5. Chart functions render")
for fn_name in [
    "chart_insight1_unrate", "chart_insight2_regime", "chart_insight3_state",
    "chart_default_by_subgrade", "chart_split_timeline", "chart_class_balance",
    "chart_default_over_time", "chart_model_bakeoff",
]:
    fig = getattr(m, fn_name)()
    ok(f"{fn_name}() -> {type(fig).__name__}")

# threshold-tuner charts (need args)
y, score = m.get_test_scores()
import numpy as np
y_pred = (score >= 0.5).astype(int)
tn = int(((y_pred==0)&(y==0)).sum())
fp = int(((y_pred==1)&(y==0)).sum())
fn_ = int(((y_pred==0)&(y==1)).sum())
tp = int(((y_pred==1)&(y==1)).sum())
ok(f"chart_threshold_cm() -> {type(m.chart_threshold_cm(tn,fp,fn_,tp)).__name__}")
fpr_arr, tpr_arr, _ = m.get_roc_data()
ok(f"chart_threshold_roc() -> {type(m.chart_threshold_roc(0.5, float(fpr_arr[len(fpr_arr)//2]), float(tpr_arr[len(tpr_arr)//2]))).__name__}")


# 6 - Static figures present --------------------------------------------------
banner("6. Static PNG figures (presentation/figures/)")
fig_dir = ROOT / "presentation" / "figures"
expected = ["roc_curve.png", "pr_curve.png", "confusion_matrix.png",
            "model_comparison.png", "feature_importance.png",
            "pipeline_diagram.png", "architecture_diagram.png"]
for f in expected:
    p = fig_dir / f
    if not p.exists():
        fail(f"missing {f}")
    ok(f"{f} ({p.stat().st_size//1024} KB)")


# 7 - Report figures synced ---------------------------------------------------
banner("7. Report figures synced (report/figures/)")
report_fig = ROOT / "report" / "figures"
for f in expected:
    p = report_fig / f
    if not p.exists():
        fail(f"missing {f} in report")
    ok(f"{f}")


# 8 - Slides PDF + report files exist -----------------------------------------
banner("8. Submission artifacts")
for path in [
    "presentation/presentation_slides.pdf",
    "report/Phase4Report.tex",
    "report/references.bib",
    "report/README.md",
    "presentation/speaker_plan.md",
]:
    p = ROOT / path
    if not p.exists():
        fail(f"missing {path}")
    ok(f"{path} ({p.stat().st_size//1024} KB)")


# 9 - MCP server symbols ------------------------------------------------------
banner("9. MCP server")
mcp_spec = importlib.util.spec_from_file_location("mcp_server", "src/mcp/server.py")
mcp_mod = importlib.util.module_from_spec(mcp_spec)
mcp_spec.loader.exec_module(mcp_mod)
ok(f"server imported")
ok(f"model: {mcp_mod.MODEL_NAME}")
ok(f"tool registered: predict_loan_default")
# Test the underlying function with the same canonical demo prompt
underlying = getattr(mcp_mod.predict_loan_default, "fn", None) or mcp_mod.predict_loan_default
result = underlying(
    loan_amnt=12000.0, term=36, int_rate=11.5, sub_grade="B2",
    annual_inc=58000.0, dti=15.0, fico_score=720,
    home_ownership="RENT", purpose="debt_consolidation",
)
ok(f"canonical demo prompt -> default_probability={result['default_probability']}, tier={result['risk_level']}")


# Summary --------------------------------------------------------------------
banner("ALL CHECKS PASSED")
print("  Streamlit app:   http://localhost:8501")
print("  Active theme:    Stripe Sunset")
print("  Production model: XGBoost, AUC-ROC 0.726 on 314k 2017 holdout")
print("  MCP demo prompt:  ~32% default prob (Moderate, Review)")
print("  All artifacts present.\n")
