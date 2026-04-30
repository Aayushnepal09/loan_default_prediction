"""
Phase 4: Streamlit Presentation App
Multi-tab Streamlit application that doubles as the team's live demo and the
Visual-of-Results deliverable for Phase 4.

Each tab maps to one rubric pillar:
    Welcome             -> use case (Results & Impact)
    Phase 1             -> data + leakage discipline (Technical Content)
    Phase 2: Pipeline   -> 8-stage preprocessing (Technical Content)
    Phase 2: Models     -> bake-off + holdout metrics (Technical Content)
    Phase 3: Spark+Macro -> medallion + FRED (Technical Content + secondary data)
    Phase 3: MCP        -> conversational deployment (Technical Content)
    Predict a loan      -> live demo (Visual of Results)
    Insights & next     -> findings, limitations, future work (Results & Impact)

Run from the project root with:
    streamlit run src/app/streamlit_app.py
"""

import __main__
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb


# -----------------------------------------------------------------------------
# Paths and one-time loading
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
PIPELINE_PATH = ROOT / "data" / "processed" / "preprocessing_pipeline.pkl"
PIPELINE_SRC = ROOT / "src" / "06_data_processing_pipeline.py"
FIG_DIR = ROOT / "presentation" / "figures"


@st.cache_resource(show_spinner="Loading model and preprocessing pipeline...")
def load_artifacts():
    # Inject 06_data_processing_pipeline's custom transformer classes into
    # __main__ so the pipeline pickle (saved while that script ran as __main__)
    # can deserialize. Same trick as src/mcp/server.py.
    spec = importlib.util.spec_from_file_location("_pipeline_module", PIPELINE_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, obj in vars(module).items():
        if not name.startswith("__"):
            setattr(__main__, name, obj)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(PIPELINE_PATH, "rb") as f:
        pipeline = pickle.load(f)

    feature_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    return model, pipeline, feature_names


@st.cache_data(show_spinner=False)
def load_model_results():
    return pd.read_csv(ROOT / "models" / "model_results.csv")


MODEL, PIPELINE, FEATURE_NAMES = load_artifacts()
MODEL_NAME = type(MODEL).__name__


# -----------------------------------------------------------------------------
# Constants for the predictor (mirrors src/mcp/server.py)
# -----------------------------------------------------------------------------
SECONDARY_DEFAULTS = {
    "acc_open_past_24mths": 4,
    "mo_sin_old_rev_tl_op": 130,
    "mo_sin_rcnt_rev_tl_op": 12,
    "mo_sin_rcnt_tl": 8,
    "mort_acc": 1,
    "mths_since_recent_bc": 18,
    "mths_since_recent_inq": np.nan,
    "num_actv_rev_tl": 5,
    "num_il_tl": 8,
    "num_rev_accts": 12,
    "num_tl_op_past_12m": 2,
    "pct_tl_nvr_dlq": 94.0,
    "total_il_high_credit_limit": 25000.0,
    "bc_open_to_buy": 5000.0,
    "revol_bal": 10000.0,
    "tot_cur_bal": 60000.0,
    "total_rev_hi_lim": 25000.0,
    "tot_coll_amt": 0,
    "num_tl_90g_dpd_24m": 0,
    "pub_rec_bankruptcies": 0,
    "num_accts_ever_120_pd": 0,
    "addr_state": "CA",
    "initial_list_status": "w",
    "emp_length": 5,
}

SUB_GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
HOME_OWNERSHIPS = ["RENT", "OWN", "MORTGAGE", "OTHER"]
VERIFICATIONS = ["Not Verified", "Verified", "Source Verified"]
PURPOSES = [
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "medical", "small_business", "car", "vacation",
    "moving", "house", "wedding", "renewable_energy", "educational",
]

PREDICTOR_DEFAULTS = {
    "loan_amnt": 15000.0,
    "term": 36,
    "int_rate": 12.5,
    "sub_grade": "B3",
    "purpose": "debt_consolidation",
    "annual_inc": 65000.0,
    "dti": 18.0,
    "fico_score": 700,
    "home_ownership": "RENT",
    "verification_status": "Not Verified",
    "open_acc": 10,
    "revol_util": 30.0,
    "total_acc": 20,
    "delinq_2yrs": 0,
    "pub_rec": 0,
    "inq_last_6mths": 0,
    "earliest_cr_line": "2010-01",
}

PRESETS = {
    "Custom": None,
    "Safe (A2, FICO 760)": {
        "loan_amnt": 8000.0, "term": 36, "int_rate": 6.5, "sub_grade": "A2",
        "purpose": "credit_card", "annual_inc": 95000.0, "dti": 8.0,
        "fico_score": 760, "home_ownership": "MORTGAGE",
        "verification_status": "Source Verified",
        "open_acc": 12, "revol_util": 12.0, "total_acc": 28,
        "delinq_2yrs": 0, "pub_rec": 0, "inq_last_6mths": 0,
        "earliest_cr_line": "2002-05",
    },
    "Borderline (C3, FICO 690)": {
        "loan_amnt": 18000.0, "term": 60, "int_rate": 14.5, "sub_grade": "C3",
        "purpose": "debt_consolidation", "annual_inc": 55000.0, "dti": 22.0,
        "fico_score": 690, "home_ownership": "RENT",
        "verification_status": "Verified",
        "open_acc": 8, "revol_util": 55.0, "total_acc": 18,
        "delinq_2yrs": 0, "pub_rec": 0, "inq_last_6mths": 1,
        "earliest_cr_line": "2009-08",
    },
    "Risky (E4, FICO 620)": {
        "loan_amnt": 32000.0, "term": 60, "int_rate": 24.5, "sub_grade": "E4",
        "purpose": "small_business", "annual_inc": 38000.0, "dti": 31.0,
        "fico_score": 620, "home_ownership": "RENT",
        "verification_status": "Not Verified",
        "open_acc": 6, "revol_util": 88.0, "total_acc": 12,
        "delinq_2yrs": 2, "pub_rec": 1, "inq_last_6mths": 4,
        "earliest_cr_line": "2012-11",
    },
}


# -----------------------------------------------------------------------------
# Predictor logic
# -----------------------------------------------------------------------------
def risk_tier(proba):
    if proba < 0.15:
        return 1, "Very Low", "Approve", "#1a9850"
    if proba < 0.25:
        return 2, "Low", "Approve", "#66bd63"
    if proba < 0.35:
        return 3, "Moderate", "Review", "#fdae61"
    if proba < 0.50:
        return 4, "High", "Decline", "#f46d43"
    return 5, "Very High", "Decline", "#d73027"


def build_input_row(inputs):
    row = {
        "loan_amnt": float(inputs["loan_amnt"]),
        "term": int(inputs["term"]),
        "int_rate": float(inputs["int_rate"]),
        "sub_grade": inputs["sub_grade"],
        "annual_inc": float(inputs["annual_inc"]),
        "dti": float(inputs["dti"]),
        "fico_range_low": int(inputs["fico_score"]),
        "home_ownership": inputs["home_ownership"],
        "purpose": inputs["purpose"],
        "verification_status": inputs["verification_status"],
        "open_acc": int(inputs["open_acc"]),
        "revol_util": float(inputs["revol_util"]),
        "total_acc": int(inputs["total_acc"]),
        "delinq_2yrs": int(inputs["delinq_2yrs"]),
        "pub_rec": int(inputs["pub_rec"]),
        "inq_last_6mths": int(inputs["inq_last_6mths"]),
        "issue_d": pd.Timestamp.now().strftime("%Y-%m"),
        "earliest_cr_line": inputs["earliest_cr_line"],
        **SECONDARY_DEFAULTS,
    }
    df = pd.DataFrame([row])
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].astype(float)
    return df


def predict_with_contribs(df):
    X = PIPELINE.transform(df)
    proba = float(MODEL.predict_proba(X)[0, 1])
    booster = MODEL.get_booster()
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    contribs = booster.predict(dmat, pred_contribs=True)[0]
    return proba, contribs[:-1], float(contribs[-1])


def gauge_chart(proba, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"size": 44}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0, 15],   "color": "#d9f0d3"},
                {"range": [15, 25],  "color": "#f7fcb9"},
                {"range": [25, 35],  "color": "#fee08b"},
                {"range": [35, 50],  "color": "#fdae61"},
                {"range": [50, 100], "color": "#f4a582"},
            ],
            "threshold": {
                "line": {"color": "#222", "width": 3},
                "thickness": 0.85,
                "value": proba * 100,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10))
    return fig


def contribs_chart(shap_values, top_n=8):
    series = pd.Series(shap_values, index=FEATURE_NAMES)
    top = series.reindex(series.abs().sort_values(ascending=False).index).head(top_n)
    top = top[::-1]
    colors = ["#d73027" if v > 0 else "#1a9850" for v in top.values]
    fig = go.Figure(go.Bar(
        x=top.values,
        y=[n.replace("_", " ") for n in top.index],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in top.values],
        textposition="outside",
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title="Contribution to log-odds (positive = higher default risk)",
        yaxis=dict(automargin=True),
        showlegend=False,
    )
    return fig


# -----------------------------------------------------------------------------
# Insight charts (precomputed parquets in presentation/data/)
# -----------------------------------------------------------------------------
INSIGHT_DIR = ROOT / "presentation" / "data"

PLOTLY_LAYOUT = dict(
    margin=dict(l=20, r=20, t=10, b=10),
    height=320,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Calibri, sans-serif", color="#1A2440"),
    legend=dict(orientation="h", yanchor="bottom", y=1.0,
                xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
)


@st.cache_data(show_spinner=False)
def load_insight(name: str) -> pd.DataFrame:
    return pd.read_parquet(INSIGHT_DIR / f"{name}.parquet")


def chart_insight1_unrate():
    df = load_insight("insight1_unrate")
    fig = go.Figure()
    fig.add_bar(x=df["unrate_bin"], y=df["n_loans"], name="# loans",
                marker_color="rgba(62,109,181,0.45)", yaxis="y1")
    fig.add_scatter(x=df["unrate_bin"], y=df["default_rate"],
                    mode="lines+markers", name="Default rate",
                    line=dict(color="#D73027", width=3),
                    marker=dict(size=8), yaxis="y2")
    fig.update_layout(
        xaxis=dict(title="US unemployment rate at issuance (%)",
                   showgrid=False),
        yaxis=dict(title="# loans", side="left", showgrid=False),
        yaxis2=dict(title="Default rate", side="right", overlaying="y",
                    tickformat=".0%", showgrid=True, gridcolor="#E2E8F0"),
        **PLOTLY_LAYOUT,
    )
    return fig


def chart_insight2_regime():
    df = load_insight("insight2_subgrade_regime")
    pivot = df.pivot(index="sub_grade", columns="rate_regime",
                     values="default_rate").reindex(
        [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
    ).dropna(how="all")
    fig = go.Figure()
    fig.add_bar(x=pivot.index, y=pivot["high rate (>0.5%)"],
                name="High Fed Funds (>0.5%)", marker_color="#D73027")
    fig.add_bar(x=pivot.index, y=pivot["low rate (<=0.5%)"],
                name="Low Fed Funds (<=0.5%)", marker_color="#3B6DB5")
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Loan sub-grade", showgrid=False),
        yaxis=dict(title="Default rate", tickformat=".0%",
                   showgrid=True, gridcolor="#E2E8F0"),
        **PLOTLY_LAYOUT,
    )
    return fig


def chart_insight3_state():
    df = load_insight("insight3_state_diff")
    state_colors = {"CA": "#3B6DB5", "TX": "#D73027", "NY": "#1A9850",
                    "FL": "#E89E2C", "IL": "#6A3C8A"}
    fig = go.Figure()
    for st_code in ["CA", "TX", "NY", "FL", "IL"]:
        s = df[df["addr_state"] == st_code].sort_values("excess_unemp")
        if not len(s):
            continue
        fig.add_scatter(x=s["excess_unemp"], y=s["default_rate"],
                        mode="lines+markers", name=st_code,
                        line=dict(color=state_colors[st_code], width=2),
                        marker=dict(size=7))
    fig.add_vline(x=0, line=dict(dash="dash", color="#888", width=1))
    fig.update_layout(
        xaxis=dict(title="State unemployment - National (percentage points)",
                   showgrid=True, gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(title="Default rate", tickformat=".0%",
                   showgrid=True, gridcolor="#E2E8F0"),
        **PLOTLY_LAYOUT,
    )
    return fig


def chart_default_by_subgrade():
    df = load_insight("default_by_subgrade").sort_values("sub_grade")
    grade_colors = {"A": "#1A9850", "B": "#66BD63", "C": "#FDAE61",
                    "D": "#F46D43", "E": "#D73027", "F": "#A50026", "G": "#67001F"}
    colors = [grade_colors[g[0]] for g in df["sub_grade"]]
    fig = go.Figure(go.Bar(
        x=df["sub_grade"], y=df["default_rate"],
        marker_color=colors,
        text=[f"{r:.0%}" for r in df["default_rate"]],
        textposition="outside",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Default rate: %{y:.1%}<br>n: %{customdata:,}<extra></extra>",
        customdata=df["n"],
    ))
    fig.update_layout(
        xaxis=dict(title="Sub-grade (A1 = lowest risk, G5 = highest)",
                   showgrid=False),
        yaxis=dict(title="Default rate", tickformat=".0%",
                   showgrid=True, gridcolor="#E2E8F0"),
        showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"},
    )
    return fig


def chart_split_timeline():
    df = load_insight("rows_per_year").sort_values("year").set_index("year")["n"].to_dict()
    rows = {y: df.get(y, 0) for y in (2014, 2015, 2016, 2017)}
    # Within 2014-16, the actual time-based split puts ~80% in train and 20% in val
    # by issue_d ordering. As a visual approximation we put the bulk of 2016 in val
    # since it's the most recent year before the test cutoff.
    train_total = rows[2014] + rows[2015] + rows[2016]
    val_total   = int(train_total * 0.20)
    train_total = train_total - val_total
    test_total  = rows[2017]

    fig = go.Figure()
    fig.add_bar(x=["Train (~80% of 2014-2016)", "Val (~20% of 2014-2016)", "Test (full 2017)"],
                y=[train_total, val_total, test_total],
                marker_color=["#3B6DB5", "#E89E2C", "#D73027"],
                text=[f"{train_total:,}", f"{val_total:,}", f"{test_total:,}"],
                textposition="outside",
                textfont=dict(size=12),
                hovertemplate="<b>%{x}</b><br>%{y:,} loans<extra></extra>")
    fig.update_layout(
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="# loans", showgrid=True, gridcolor="#E2E8F0"),
        showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"},
    )
    return fig


def chart_class_balance():
    fig = go.Figure(go.Pie(
        labels=["Fully Paid", "Charged Off"],
        values=[79.0, 21.0],
        marker=dict(colors=["#1A9850", "#D73027"]),
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(size=14, color="white"),
        sort=False,
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=10, b=10),
        height=260,
        paper_bgcolor="white",
        annotations=[dict(text="~4:1<br>paid : default",
                          x=0.5, y=0.5, font_size=14,
                          font_color="#1A2440", showarrow=False)],
    )
    return fig


# --- Tab 4 helpers: test-set scoring + threshold tuner + bake-off chart -----

@st.cache_data(show_spinner="Scoring the 314k-row 2017 test set once...")
def get_test_scores():
    df = pd.read_csv(ROOT / "data" / "processed" / "test_features.csv")
    y = df["charged_off"].values.astype(int)
    X = df.drop(columns=["charged_off"]).values
    proba = MODEL.predict_proba(X)[:, 1]
    return y, proba


@st.cache_data(show_spinner=False)
def get_roc_data():
    from sklearn.metrics import roc_curve
    y, proba = get_test_scores()
    fpr, tpr, thr = roc_curve(y, proba)
    return fpr, tpr, thr


def chart_model_bakeoff():
    res = load_model_results().sort_values("val_auc_roc", ascending=True)
    colors = ["#0E3D70" if m == "XGBoost" else "#9EC5E8" for m in res["model"]]
    text = [f"AUC {a:.4f}  /  KS {k:.3f}  ({t:.0f}s)"
            for a, k, t in zip(res["val_auc_roc"], res["val_ks"], res["fit_time_s"])]
    fig = go.Figure(go.Bar(
        x=res["val_auc_roc"], y=res["model"], orientation="h",
        marker_color=colors, text=text, textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>AUC-ROC: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(title="Validation AUC-ROC", range=[0.66, 0.78],
                   showgrid=True, gridcolor="#E2E8F0", zeroline=False),
        yaxis=dict(title=None, automargin=True),
        showlegend=False,
        height=280,
        margin=dict(l=20, r=20, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Calibri, sans-serif", color="#1A2440"),
    )
    return fig


def chart_threshold_cm(tn, fp, fn, tp):
    fig = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted Paid", "Predicted Default"],
        y=["Actual Paid", "Actual Default"],
        text=[[f"{tn:,}<br><span style='font-size:11px'>true neg</span>",
               f"{fp:,}<br><span style='font-size:11px'>false pos</span>"],
              [f"{fn:,}<br><span style='font-size:11px'>false neg</span>",
               f"{tp:,}<br><span style='font-size:11px'>true pos</span>"]],
        texttemplate="%{text}",
        colorscale=[[0, "#EAF2FB"], [1, "#0E3D70"]],
        showscale=False,
        hovertemplate="<b>%{y} / %{x}</b><br>%{z:,}<extra></extra>",
    ))
    fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=10, b=10),
        font=dict(family="Calibri, sans-serif", color="#1A2440"),
        paper_bgcolor="white", plot_bgcolor="white",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def chart_threshold_roc(threshold, op_fpr, op_tpr):
    fpr, tpr, _ = get_roc_data()
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines",
                    line=dict(color="#3B6DB5", width=2.5),
                    name="ROC", showlegend=False,
                    hoverinfo="skip")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                    line=dict(color="#999", width=1, dash="dash"),
                    name="Random", showlegend=False,
                    hoverinfo="skip")
    fig.add_scatter(x=[op_fpr], y=[op_tpr], mode="markers",
                    marker=dict(color="#D73027", size=14,
                                line=dict(color="white", width=2)),
                    name=f"thr = {threshold:.2f}",
                    showlegend=True,
                    hovertemplate=f"<b>Threshold {threshold:.2f}</b><br>"
                                  f"FPR: {op_fpr:.3f}<br>TPR: {op_tpr:.3f}<extra></extra>")
    fig.update_layout(
        xaxis=dict(title="False positive rate", range=[0, 1],
                   showgrid=True, gridcolor="#E2E8F0"),
        yaxis=dict(title="True positive rate", range=[0, 1],
                   showgrid=True, gridcolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=0.0,
                    xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)"),
        height=320, margin=dict(l=20, r=20, t=10, b=10),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Calibri, sans-serif", color="#1A2440"),
    )
    return fig


def chart_default_over_time():
    df = load_insight("default_over_time").sort_values("year_month")
    fig = go.Figure(go.Scatter(
        x=df["year_month"], y=df["default_rate"],
        mode="lines+markers",
        line=dict(color="#0E3D70", width=2),
        marker=dict(size=5, color="#0E3D70"),
        hovertemplate="<b>%{x}</b><br>Default rate: %{y:.1%}<br>n: %{customdata:,}<extra></extra>",
        customdata=df["n"],
    ))
    fig.update_layout(
        xaxis=dict(title="Issue month", showgrid=False),
        yaxis=dict(title="Default rate", tickformat=".0%",
                   showgrid=True, gridcolor="#E2E8F0"),
        showlegend=False,
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"},
    )
    return fig


# -----------------------------------------------------------------------------
# Page setup + theme system
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Lending Club Default Risk - EAS 587 Phase 4",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# 5 hand-tuned modern-startup / Gen-Z themes, each modeled after a real
# product whose visual language already lands with that audience.
THEMES = {
    "Edge Sunset": {
        "tagline":         "Vercel minimalism + Stripe gradient hero",
        "primary":         "#5B5BFF",
        "accent":          "#FF7A66",
        "bg":              "#FFFFFF",
        "surface":         "#FAFAFA",
        "surface_2":       "#F0F0F5",
        "text":            "#0A0A14",
        "muted":           "#5C627A",
        "border":          "#EAEAEA",
        "hero_a":          "#635BFF",
        "hero_b":          "#FF7A66",
        "hero_text":       "#FFFFFF",
        "hero_subtitle":   "#FFD4CC",
        "is_dark":         False,
    },
    "Linear Violet": {
        "tagline":         "Linear / Loom premium dark - violet glow",
        "primary":         "#8B7CFF",
        "accent":          "#FF6BA8",
        "bg":              "#0A0817",
        "surface":         "#16142A",
        "surface_2":       "#221E3D",
        "text":            "#F4F2FF",
        "muted":           "#A9A3CC",
        "border":          "#2A2547",
        "hero_a":          "#5E6AD2",
        "hero_b":          "#C660E0",
        "hero_text":       "#FFFFFF",
        "hero_subtitle":   "#E0D8FF",
        "is_dark":         True,
    },
    "Stripe Sunset": {
        "tagline":         "Stripe-style purple to coral gradient, friendly",
        "primary":         "#635BFF",
        "accent":          "#FF7A66",
        "bg":              "#FAFAFC",
        "surface":         "#F0EEFF",
        "surface_2":       "#E5E1FF",
        "text":            "#0A2540",
        "muted":           "#5C6B8C",
        "border":          "#E0DEF0",
        "hero_a":          "#635BFF",
        "hero_b":          "#FF7A66",
        "hero_text":       "#FFFFFF",
        "hero_subtitle":   "#FFD4CC",
        "is_dark":         False,
    },
    "Vercel Edge": {
        "tagline":         "Vercel-style mono with one electric blue accent",
        "primary":         "#000000",
        "accent":          "#0070F3",
        "bg":              "#FFFFFF",
        "surface":         "#FAFAFA",
        "surface_2":       "#F0F0F0",
        "text":            "#0A0A0A",
        "muted":           "#666666",
        "border":          "#EAEAEA",
        "hero_a":          "#000000",
        "hero_b":          "#1A1A1A",
        "hero_text":       "#FFFFFF",
        "hero_subtitle":   "#B3B3B3",
        "is_dark":         False,
    },
    "Lo-Fi Pastel": {
        "tagline":         "Pastel lavender + hot pink, Y2K Gen-Z energy",
        "primary":         "#7C3AED",
        "accent":          "#EC4899",
        "bg":              "#FBF7FF",
        "surface":         "#F3E8FF",
        "surface_2":       "#E9D5FF",
        "text":            "#2D1065",
        "muted":           "#7E5BAA",
        "border":          "#DDD0F8",
        "hero_a":          "#A855F7",
        "hero_b":          "#EC4899",
        "hero_text":       "#FFFFFF",
        "hero_subtitle":   "#FBE5FF",
        "is_dark":         False,
    },
    "Cyber Glow": {
        "tagline":         "Cyberpunk dark + neon cyan/magenta",
        "primary":         "#22D3EE",
        "accent":          "#F472B6",
        "bg":              "#08081C",
        "surface":         "#14152A",
        "surface_2":       "#1F2042",
        "text":            "#E0F7FF",
        "muted":           "#8AAAC8",
        "border":          "#1F2042",
        "hero_a":          "#06B6D4",
        "hero_b":          "#EC4899",
        "hero_text":       "#FFFFFF",
        "hero_subtitle":   "#C0F0FF",
        "is_dark":         True,
    },
}
DEFAULT_THEME = "Stripe Sunset"
THEME_FILE = ROOT / ".streamlit" / "active_theme.txt"


def load_theme_choice():
    try:
        name = THEME_FILE.read_text(encoding="utf-8").strip()
        return name if name in THEMES else DEFAULT_THEME
    except FileNotFoundError:
        return DEFAULT_THEME


def save_theme_choice(name):
    if name in THEMES:
        THEME_FILE.parent.mkdir(parents=True, exist_ok=True)
        THEME_FILE.write_text(name, encoding="utf-8")


if "theme_name" not in st.session_state:
    st.session_state["theme_name"] = load_theme_choice()
# Guard against stale session_state pointing to a theme we no longer ship
if st.session_state["theme_name"] not in THEMES:
    st.session_state["theme_name"] = DEFAULT_THEME
    save_theme_choice(DEFAULT_THEME)

# Color keys the user can override via the Customize expander
THEME_COLOR_KEYS = ["primary", "accent", "bg", "surface", "surface_2",
                    "text", "muted", "border", "hero_a", "hero_b",
                    "hero_text", "hero_subtitle"]

# Effective theme = base preset + per-session user overrides
_base = THEMES[st.session_state["theme_name"]]
THEME = dict(_base)
for _k, _v in st.session_state.get("custom_overrides", {}).items():
    if _k in THEME_COLOR_KEYS:
        THEME[_k] = _v

# Re-skin Plotly defaults to match the active theme (charts pick this up at call time)
PLOTLY_LAYOUT["plot_bgcolor"] = THEME["bg"]
PLOTLY_LAYOUT["paper_bgcolor"] = THEME["bg"]
PLOTLY_LAYOUT["font"] = dict(family="Calibri, sans-serif", color=THEME["text"])

# Themed CSS - everything else is driven by these CSS variables
st.markdown(
    f"""
    <style>
      :root {{
        --t-primary:    {THEME['primary']};
        --t-accent:     {THEME['accent']};
        --t-bg:         {THEME['bg']};
        --t-surface:    {THEME['surface']};
        --t-surface-2:  {THEME['surface_2']};
        --t-text:       {THEME['text']};
        --t-muted:      {THEME['muted']};
        --t-border:     {THEME['border']};
        --t-hero-a:     {THEME['hero_a']};
        --t-hero-b:     {THEME['hero_b']};
        --t-hero-text:  {THEME['hero_text']};
        --t-hero-sub:   {THEME['hero_subtitle']};
        --risk-red:     #D73027;
        --safe-green:   #1A9850;
      }}

      /* Slim the Streamlit header - don't kill it entirely or the sidebar
         collapse-expand arrow disappears with it. Keep the bar transparent
         and short, hide only the Deploy button + hamburger menu inside. */
      header[data-testid="stHeader"], .stAppHeader {{
          background: transparent !important;
          height: 2.4rem !important;
          min-height: 2.4rem !important;
      }}
      [data-testid="stDecoration"] {{ display: none !important; }}
      [data-testid="stToolbar"],
      [data-testid="stStatusWidget"],
      .stDeployButton,
      [data-testid="stDeployButton"],
      [data-testid="stMainMenu"] {{
          display: none !important;
          visibility: hidden !important;
      }}
      /* Make sure the sidebar collapse expand button stays clickable */
      [data-testid="stSidebarCollapsedControl"],
      [data-testid="stSidebarCollapseButton"],
      button[kind="headerNoPadding"] {{
          display: flex !important;
          visibility: visible !important;
          z-index: 999 !important;
      }}
      /* Background: target every level Streamlit might use */
      html, body,
      .stApp, [data-testid="stApp"],
      [data-testid="stAppViewContainer"],
      section[data-testid="stMain"],
      section.main,
      .main, .main > .block-container,
      [class*="appview-container"] {{
          background-color: var(--t-bg) !important;
      }}

      /* Body text inherits theme - critical for dark themes */
      html, body, .stApp, [data-testid="stApp"],
      .stMarkdown, .stMarkdown p, .stMarkdown li,
      .stMarkdown strong, .stMarkdown em, .stMarkdown a,
      .block-container, .block-container p, .block-container li,
      .block-container span, .block-container strong {{
          color: var(--t-text);
      }}
      .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
      .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{ color: var(--t-text); }}
      .stMarkdown blockquote {{ color: var(--t-muted); border-left-color: var(--t-primary); }}
      .stCaption, [data-testid="stCaptionContainer"] {{ color: var(--t-muted) !important; }}

      .block-container {{ padding-top: 1.5rem; padding-bottom: 1.5rem; max-width: 1400px; }}

      /* Sidebar */
      [data-testid="stSidebar"] {{ background: var(--t-surface) !important; }}
      [data-testid="stSidebar"] * {{ color: var(--t-text); }}
      [data-testid="stSidebar"] a {{ color: var(--t-primary); }}

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid var(--t-border); }}
      .stTabs [data-baseweb="tab"] {{
          padding: 0.6rem 1.0rem; font-weight: 600; color: var(--t-muted);
      }}
      .stTabs [aria-selected="true"] {{
          background: var(--t-surface-2) !important;
          color: var(--t-primary) !important;
          border-bottom: 3px solid var(--t-primary) !important;
      }}

      /* Cards + callouts */
      .ph-card {{
          background: var(--t-surface); border-radius: 10px; padding: 1rem 1.1rem;
          border-left: 4px solid var(--t-primary); color: var(--t-text);
      }}
      .ph-card-amber {{ border-left-color: var(--t-accent); }}
      .ph-card-red   {{ border-left-color: var(--risk-red); }}
      .ph-card-green {{ border-left-color: var(--safe-green); }}
      .ph-callout-num {{
          font-size: 2.4rem; font-weight: 700; color: var(--t-primary); line-height: 1.0;
      }}
      .ph-callout-num-amber {{ color: var(--t-accent); }}
      .ph-callout-num-red {{ color: var(--risk-red); }}
      .ph-callout-label {{ font-size: 0.85rem; color: var(--t-muted); margin-top: 0.3rem; }}

      .ph-tab-header {{
          font-size: 1.7rem; font-weight: 700; color: var(--t-primary);
          margin-bottom: 0.1rem;
      }}
      .ph-tab-sub {{ color: var(--t-muted); font-style: italic; margin-bottom: 1.1rem; }}

      .ph-pill {{
          display: inline-block; background: var(--t-accent); color: white;
          padding: 0.15rem 0.6rem; border-radius: 999px;
          font-size: 0.75rem; font-weight: 700; letter-spacing: 0.04em;
          margin-bottom: 0.4rem;
      }}
      .ph-pill-navy {{ background: var(--t-primary); }}

      /* Form widget surfaces - keep readable on dark themes */
      [data-baseweb="select"] > div, [data-baseweb="input"] > div,
      [data-testid="stNumberInput"] > div > div {{
          background: var(--t-surface) !important;
          color: var(--t-text) !important;
      }}
      .stSlider [data-baseweb="slider"] {{ color: var(--t-primary); }}

      /* Native dataframes / metric */
      [data-testid="stMetricValue"] {{ color: var(--t-text); }}
      [data-testid="stMetricLabel"] {{ color: var(--t-muted); }}
    </style>
    """,
    unsafe_allow_html=True,
)


def tab_header(title: str, subtitle: str, pill: str = None, pill_class: str = "ph-pill"):
    if pill:
        st.markdown(f'<div class="{pill_class}">{pill}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ph-tab-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ph-tab-sub">{subtitle}</div>', unsafe_allow_html=True)


def callout(big: str, label: str, kind: str = "navy"):
    color_class = {
        "navy": "",
        "amber": "ph-callout-num-amber",
        "red": "ph-callout-num-red",
    }[kind]
    st.markdown(
        f"""
        <div class="ph-card">
          <div class="ph-callout-num {color_class}">{big}</div>
          <div class="ph-callout-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body_md: str, kind: str = "navy"):
    extra = {"navy": "", "amber": "ph-card-amber",
             "red": "ph-card-red", "green": "ph-card-green"}[kind]
    st.markdown(
        f"""
        <div class="ph-card {extra}">
          <div style="font-weight:700; color:var(--t-primary); margin-bottom:0.35rem;">{title}</div>
          <div style="color:var(--t-text); font-size:0.95rem;">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Sidebar - constant info
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Lending Club Default Risk")
    st.caption("EAS 587 Spring 2026 - Phase 4")

    st.markdown("---")
    st.caption(f"Active theme: _{st.session_state['theme_name']}_")

    st.markdown("---")
    st.markdown("**Team: 404 Team Not Found**")
    st.markdown(
        "- Aayush Nepal\n"
        "- Junwei Zhang\n"
        "- Lusi Zhang"
    )
    st.markdown("---")
    st.markdown("**About this app**")
    st.caption(
        "This single Streamlit app is both the team's live demo and the visual "
        "deliverable for Phase 4. Walk through the tabs left-to-right to see "
        "the full pipeline; the **Predict a loan** tab is the live model."
    )
    st.markdown("---")
    st.markdown(
        "[Project repository](https://github.com/Aayushnepal09/Eas587_project)"
    )
    st.caption("Phases 1+2 received perfect scores; Phase 3 not yet graded.")


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "Welcome",
    "Phase 1: Data",
    "Phase 2: Pipeline",
    "Phase 2: Models",
    "Phase 3: Spark + Macro",
    "Phase 3: MCP",
    "Predict a loan",
    "Insights + Next",
    "Q&A",
])


# === TAB 1: Welcome (title screen) ==========================================
with tabs[0]:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, var(--t-hero-a) 0%, var(--t-hero-b) 100%);
            color: var(--t-hero-text);
            padding: 2.6rem 2.2rem 2.2rem 2.2rem;
            border-radius: 14px;
            margin: 0.2rem 0 1.6rem 0;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
            border-left: 8px solid var(--t-accent);
        ">
          <div style="font-size:0.82rem; color:var(--t-accent); font-weight:700;
                      letter-spacing:0.16em; margin-bottom:0.5rem;">
            EAS 587  &middot;  SPRING 2026  &middot;  FINAL PRESENTATION
          </div>
          <div style="font-size:2.7rem; font-weight:700; line-height:1.08;
                      margin:0 0 0.4rem 0; color:var(--t-hero-text);">
            Lending Club Loan Default Prediction
          </div>
          <div style="font-size:1.05rem; color:var(--t-hero-sub); font-style:italic;
                      margin-bottom:1.4rem;">
            Same use case, four implementations:
            pandas &rarr; scikit-learn &rarr; Spark + MCP &rarr; conversational UI
          </div>
          <div style="font-size:0.78rem; color:var(--t-accent); font-weight:700;
                      letter-spacing:0.18em; margin-bottom:0.35rem;">
            TEAM &middot; 404 TEAM NOT FOUND
          </div>
          <div style="font-size:1.05rem; color:var(--t-hero-text); letter-spacing:0.02em;">
            <strong>Aayush Nepal</strong>
            &nbsp;&middot;&nbsp;
            <strong>Junwei Zhang</strong>
            &nbsp;&middot;&nbsp;
            <strong>Lusi Zhang</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([1.2, 1])

    with c_left:
        st.markdown(
            """
            ### The lender's question
            > Given a loan application, what is the probability this borrower
            > will charge off rather than fully repay the loan?

            We answer it end-to-end: filter and clean the public Lending Club
            dataset (2014-2017), train a tuned XGBoost model with strict
            leakage discipline, scale the data layer to Spark + Delta Lake,
            and deploy two ways - **as a conversational MCP tool inside Claude
            Desktop** and **as this interactive app** (Predict a loan tab).
            """
        )

        st.markdown(" ")
        st.markdown(
            """
            #### How to navigate this presentation

            1. **Walk the tabs left to right** to follow the project's evolution.
            2. **Predict a loan** runs the actual production model live - try
               the preset selector for quick scenarios, then toggle compare
               mode to score two loans side-by-side.
            3. The **Insights + Next** tab is the team's findings + future work.
            4. **Q&A** is the closing surface for questions.
            """
        )

    with c_right:
        st.markdown("##### At a glance")
        cc1, cc2 = st.columns(2)
        with cc1:
            callout("314,212", "loans in the held-out 2017 test set")
            st.markdown(" ")
            callout("0.726", "test AUC-ROC (XGBoost, full 2017)", kind="amber")
        with cc2:
            callout("21.0%", "actual charge-off rate", kind="red")
            st.markdown(" ")
            callout("110", "preprocessed features fed to the model")


# === TAB 2: Phase 1 - Data ===================================================
with tabs[1]:
    tab_header(
        "Data foundations",
        "Filter, define the target, split with leakage discipline.",
        pill="PHASE 1",
    )

    # Top: split visual on the left, class balance donut on the right
    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("##### Time-based split (no random shuffling)")
        st.plotly_chart(chart_split_timeline(), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            "Train + val drawn from 2014-2016 by `issue_d` order; full 2017 held out as test. "
            "Random splits would leak the macro environment."
        )
    with right:
        st.markdown("##### Class balance")
        st.plotly_chart(chart_class_balance(), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            "We deliberately do NOT resample. Tree models handle the 4:1 "
            "imbalance natively via `scale_pos_weight`."
        )

    st.markdown("---")
    st.markdown("##### Leakage discipline - 21 post-loan columns dropped")

    lc1, lc2 = st.columns([1, 2])
    with lc1:
        callout("21", "post-loan columns dropped", kind="amber")
        st.caption(
            "These only get values **after** the outcome is known. Most public "
            "Lending Club notebooks accidentally train on them and report inflated metrics."
        )
    with lc2:
        st.markdown(
            "| Category | Columns dropped |\n"
            "|---|---|\n"
            "| Payment history | `total_pymnt*`, `total_rec_prncp/int/late_fee`, `last_pymnt_*` |\n"
            "| Outstanding balance | `out_prncp`, `out_prncp_inv` |\n"
            "| Recovery + collection | `recoveries`, `collection_recovery_fee`, `last_credit_pull_d` |\n"
            "| Settlement / hardship | `hardship_flag`, `debt_settlement_flag*`, `settlement_status/date/amount/percentage/term` |\n"
            "| Updated credit signal | `last_fico_range_low`, `last_fico_range_high` |\n"
        )


# === TAB 3: Phase 2 - Pipeline ==============================================
with tabs[2]:
    tab_header(
        "8-stage preprocessing pipeline",
        "Custom scikit-learn transformers, fit on train only, applied identically to val and test.",
        pill="PHASE 2",
    )

    pipeline_img = FIG_DIR / "pipeline_diagram.png"
    if pipeline_img.exists():
        st.image(str(pipeline_img), use_container_width=True)

    st.markdown(" ")
    st.markdown("##### Why each stage exists")

    g1, g2 = st.columns(2)
    with g1:
        card("DropCorrelated",
             "13 highly-correlated columns removed (e.g. fico_range_high which is r=1.0 with fico_range_low).")
        card("DateExtractor",
             "Splits issue_d and earliest_cr_line into year, month, and credit_history_months.")
        card("FeatureConstructor",
             "Engineered features: loan_to_income, monthly_payment_to_income (PMT-formula based), delinq_rate.")
        card("OutlierCapper",
             "dti capped at p99 - the raw max of 999 is a known data-entry error.")

    with g2:
        card("MissingIndicatorAdder",
             "Some missingness is informative; we add a binary _missing flag for mths_since_recent_inq before imputing.",
             kind="amber")
        card("MacroJoiner",
             "Pulls FRED UNRATE at fit time and joins on (issue_year, issue_month). Falls back to a 5.0% constant offline so SimpleImputer doesn't drop the column.",
             kind="amber")
        card("RareCategoryMerger",
             "purpose and home_ownership categories with <50 occurrences fold into 'other' before OneHot.")
        card("ColumnPreprocessor",
             "Final routing: skewed -> log1p+scale, binarize -> >0 flag, ordinal -> sub_grade encoding, OHE -> categoricals, normal -> impute+scale.")


# === TAB 4: Phase 2 - Models =================================================
with tabs[3]:
    tab_header(
        "Model bake-off + holdout test results",
        "Four algorithms, 50 Optuna trials each, MLflow tracked. XGBoost won; tested on the full untouched 2017 set.",
        pill="PHASE 2",
    )

    st.markdown("##### Validation set bake-off")
    st.plotly_chart(chart_model_bakeoff(), use_container_width=True,
                    config={"displayModeBar": False})
    st.caption(
        "All four share the same preprocessing pipeline above. Tuning: Optuna TPE, "
        "50 trials per model, optimizing val AUC-ROC. All 200+ trials logged "
        "to MLflow at `models/mlruns/`. Logistic Regression is the baseline (no tuning)."
    )

    st.markdown(" ")
    st.markdown("---")
    st.markdown("##### Held-out test results - XGBoost on all 314,212 2017 loans")

    m1, m2, m3, m4 = st.columns(4)
    with m1: callout("0.7260", "Test AUC-ROC")
    with m2: callout("0.4103", "Test AUC-PR")
    with m3: callout("0.3279", "Test KS", kind="amber")
    with m4: callout("67.3%", "Recall on defaults\n(at Youden threshold 0.495)", kind="red")

    st.markdown(" ")
    st.markdown("---")
    st.markdown("##### Threshold tuner - explore the operating-point trade-off live")
    st.caption(
        "Move the slider to see precision, recall, and the confusion matrix update "
        "across the full 314k test set. Useful for stakeholders who want to choose "
        "an operating point matched to their cost ratio (e.g. a lender willing to "
        "review more loans for higher recall on defaults)."
    )

    y_test, y_score = get_test_scores()
    threshold = st.slider(
        "Operating threshold",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01,
        help="Predict default when the model probability is at least this value.",
    )
    y_pred = (y_score >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    acc  = (tp + tn) / max(len(y_test), 1)
    flag_rate = (y_pred == 1).sum() / len(y_pred)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)

    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
    tm1.metric("Precision", f"{prec:.1%}")
    tm2.metric("Recall (defaults caught)", f"{rec:.1%}")
    tm3.metric("Accuracy", f"{acc:.1%}")
    tm4.metric("F1", f"{f1:.3f}")
    tm5.metric("% of book flagged", f"{flag_rate:.1%}")

    # Find the operating point on the precomputed ROC
    fpr_arr, tpr_arr, thr_arr = get_roc_data()
    mask = thr_arr <= threshold
    if mask.any():
        idx = int(mask.argmax())
    else:
        idx = len(thr_arr) - 1
    op_fpr = float(fpr_arr[idx])
    op_tpr = float(tpr_arr[idx])

    cm_col, roc_col = st.columns(2)
    with cm_col:
        st.plotly_chart(chart_threshold_cm(tn, fp, fn, tp),
                        use_container_width=True,
                        config={"displayModeBar": False})
    with roc_col:
        st.plotly_chart(chart_threshold_roc(threshold, op_fpr, op_tpr),
                        use_container_width=True,
                        config={"displayModeBar": False})

    pr_img = FIG_DIR / "pr_curve.png"
    with st.expander("Also see: precision-recall curve"):
        if pr_img.exists():
            st.image(str(pr_img), use_container_width=True)
        st.caption(
            "AP = 0.4103 vs class prior of 21.0% - the model lifts precision over "
            "the random-guess baseline by roughly 2x at any sensible operating point."
        )


# === TAB 5: Phase 3 - Spark + Macro ==========================================
with tabs[4]:
    tab_header(
        "Spark, Delta Lake, and FRED macro overlay",
        "Same data, same target, rebuilt on Databricks using the medallion architecture and extended with a FRED macro layer.",
        pill="PHASE 3",
    )

    st.markdown("##### Medallion architecture - Bronze / Silver / Gold")
    b, s, g = st.columns(3)
    with b: card("BRONZE", "Raw Kaggle archive landed as-is. Filtered 2014-2017 on Spark. Written to Delta as `bronze_loans`.", kind="amber")
    with s: card("SILVER", "Cleaned and type-fixed. Leakage columns dropped. Schema enforced. -> `silver_loans`.")
    with g: card("GOLD",   "Time-based split + stratified train/val sample. Full 2017 test retained. -> `gold_loans_train/val/test`.")

    st.markdown(" ")
    st.markdown("---")
    st.markdown("##### Secondary data: FRED macro overlay")
    st.caption(
        "The Phase 4 rubric specifically rewards adding recommended secondary "
        "data sources. We pulled 8 series directly from FRED (no auth) and "
        "engineered 4 more, then asked: does macro context improve the model?"
    )

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**National series**")
        st.markdown(
            "- `UNRATE` - civilian unemployment rate\n"
            "- `FEDFUNDS` - effective federal funds rate\n"
            "- `CPIAUCSL` - Consumer Price Index (all urban)\n"
            "- `CPI_YOY` - 12-month CPI change (inflation proxy)"
        )
        st.markdown("**Engineered**")
        st.markdown(
            "- `real_int_rate` = `int_rate` - `CPI_YOY`\n"
            "- `rate_spread` = `int_rate` - `FEDFUNDS`"
        )
    with s2:
        st.markdown("**State unemployment (top-5 LC volume states)**")
        st.markdown(
            "- `CAUR`, `TXUR`, `NYUR`, `FLUR`, `ILUR`\n"
            "- `state_unrate` - per-borrower lookup based on `addr_state`"
        )
        st.markdown("**Production pipeline already uses macro**")
        st.markdown(
            "Phase 2's `MacroJoiner` injects `unemployment_rate` into every "
            "training row; the Phase 3 work tested whether **richer macro** "
            "(7 features) adds further lift."
        )

    st.markdown(" ")
    st.markdown("##### Three macro insights")
    st.caption(
        "Computed locally on the 1.36M-row cleaned dataset, joined to FRED. "
        "Bars / lines reproduce the queries from notebook 05."
    )

    st.markdown("**1. Default rate vs national unemployment at issuance**")
    st.plotly_chart(chart_insight1_unrate(), use_container_width=True,
                    config={"displayModeBar": False})
    st.caption(
        "Bars = volume of loans issued at each unemployment level (left axis). "
        "Red line = charge-off rate among those loans (right axis). The relationship "
        "is positive but modest because issuance volume itself shifts with the cycle."
    )

    st.markdown(" ")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**2. Sub-grade default rate by Fed Funds regime**")
        st.plotly_chart(chart_insight2_regime(), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            "The default-rate gap across sub-grades widens noticeably under "
            "higher Fed Funds rates - tighter monetary policy hits weaker "
            "borrowers harder."
        )
    with cc2:
        st.markdown("**3. State unemployment differential vs default rate**")
        st.plotly_chart(chart_insight3_state(), use_container_width=True,
                        config={"displayModeBar": False})
        st.caption(
            "Among the 5 highest-volume LC states, a positive state-over-national "
            "unemployment gap correlates with higher default rates - local labor "
            "markets carry signal beyond the national series."
        )

    st.markdown(" ")
    st.markdown("---")
    st.markdown("##### Honest finding: did 7 macro features improve the model?")
    e1, e2, e3 = st.columns(3)
    with e1: callout("0.7136", "Phase 3 LR - loan features only", kind="navy")
    with e2: callout("0.7131", "Phase 3 LR - loan + 7 FRED features", kind="red")
    with e3: callout("-0.0005", "Val AUC-ROC delta", kind="amber")
    st.caption(
        "Adding 7 additional FRED features beyond the production `unemployment_rate` did NOT improve LR validation AUC. "
        "The macro signal at issuance time is largely captured by the simpler single feature already in production. "
        "The macro work still earns its place: the three insights above are stakeholder-facing communication wins, and "
        "the engineered `state_unrate` and `real_int_rate` are useful for stress testing even if AUC is flat."
    )


# === TAB 6: Phase 3 - MCP ====================================================
with tabs[5]:
    tab_header(
        "Conversational deployment via MCP",
        "Same trained model, exposed as a Model Context Protocol tool. Claude Desktop can predict default risk from a plain-English loan description.",
        pill="PHASE 3",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### How it works")
        st.markdown(
            """
            ```
            Claude Desktop
                | MCP (stdio transport)
                v
            src/mcp/server.py  (FastMCP)
              - loads preprocessing_pipeline.pkl + best_model.pkl on startup
              - predict_loan_default():
                  1. validates typed inputs
                  2. builds a DataFrame (user inputs + 24 secondary defaults)
                  3. PIPELINE.transform()  - same steps as training
                  4. MODEL.predict_proba() - returns risk + tier + recommendation
            ```
            """
        )
        st.caption(
            "The pickle of the preprocessing pipeline references custom transformer "
            "classes under module path `__main__` (it was pickled while the script ran "
            "as `__main__`). Both this Streamlit app and the MCP server inject those "
            "classes into `__main__` before unpickling - same trick, two consumers."
        )
        st.markdown(" ")
        card(
            "Why this matters",
            "Most ML projects hand off a model file. Ours is reachable two ways: "
            "the visual UI on this page for analysts, and a conversational "
            "interface for everyone else. The same training-time transform "
            "pipeline runs in both consumers - guaranteed parity.",
            kind="amber",
        )

    with c2:
        st.markdown("##### Live demo prompts (paste into Claude Desktop)")
        st.caption(
            "These are the exact prompts to copy-paste during the live demo. "
            "Each one demonstrates a different MCP capability."
        )

        st.markdown("**1. Basic prediction**")
        st.caption("One loan, one tool call, one risk recommendation.")
        st.code(
            "Predict default risk for a $12,000, 36-month loan at 11.5% "
            "interest. Sub-grade B2, $58,000 income, DTI 15%, FICO 720, "
            "renting, debt consolidation.",
            language=None,
        )

        st.markdown("**2. Multi-turn what-if (the wow moment)**")
        st.caption("Claude runs 4-5 variations through the tool autonomously.")
        st.code(
            "How can I get it approved?",
            language=None,
        )

        st.markdown("**3. Side-by-side comparison**")
        st.caption("Claude calls the tool twice and reasons about the contrast.")
        st.code(
            "Compare two loans: (1) $5,000 at 7%, A1 grade, FICO 780, "
            "$90K income vs (2) $25,000 at 22%, E4 grade, FICO 620, $35K "
            "income. Which is riskier and why?",
            language=None,
        )

        st.markdown("**4. Targeted what-if**")
        st.caption("Iterative tuning toward a target outcome.")
        st.code(
            "What FICO score would this same borrower need to drop "
            "below 20% default probability?",
            language=None,
        )


# === TAB 7: Predictor (the live demo) ========================================

def pkey(prefix, name):
    """Prefix a session-state key for the predictor form (supports A/B forms)."""
    return f"pred_{prefix}_{name}"


def render_loan_form(prefix: str, *, show_preset: bool = True):
    """Render the 3-column loan input form keyed under `prefix`.
    Returns the current dict of inputs read from session_state."""
    for k, v in PREDICTOR_DEFAULTS.items():
        st.session_state.setdefault(pkey(prefix, k), v)

    if show_preset:
        preset_select_key = pkey(prefix, "_preset_selector")
        preset_last_key   = pkey(prefix, "_last_preset")
        preset_name = st.selectbox(
            "Quick preset",
            list(PRESETS.keys()),
            key=preset_select_key,
            help="Fill all inputs with a representative borrower profile.",
        )
        if PRESETS.get(preset_name) and st.session_state.get(preset_last_key) != preset_name:
            for k, v in PRESETS[preset_name].items():
                st.session_state[pkey(prefix, k)] = v
            st.session_state[preset_last_key] = preset_name
            st.rerun()

    loan_col, borrower_col, credit_col = st.columns(3)

    with loan_col:
        st.markdown("##### Loan")
        st.number_input("Loan amount ($)", min_value=500.0, max_value=40000.0,
                        step=500.0, key=pkey(prefix, "loan_amnt"))
        st.selectbox("Term (months)", [36, 60], key=pkey(prefix, "term"))
        st.number_input("Interest rate (%)", min_value=5.0, max_value=35.0,
                        step=0.1, key=pkey(prefix, "int_rate"))
        st.selectbox("Sub-grade", SUB_GRADES, key=pkey(prefix, "sub_grade"))
        st.selectbox("Purpose", PURPOSES, key=pkey(prefix, "purpose"))

    with borrower_col:
        st.markdown("##### Borrower")
        st.number_input("Annual income ($)", min_value=1000.0, max_value=500000.0,
                        step=1000.0, key=pkey(prefix, "annual_inc"))
        st.number_input("DTI (%)", min_value=0.0, max_value=60.0,
                        step=0.5, key=pkey(prefix, "dti"))
        st.slider("FICO score", min_value=600, max_value=850,
                  key=pkey(prefix, "fico_score"))
        st.selectbox("Home ownership", HOME_OWNERSHIPS,
                     key=pkey(prefix, "home_ownership"))
        st.selectbox("Income verification", VERIFICATIONS,
                     key=pkey(prefix, "verification_status"))

    with credit_col:
        st.markdown("##### Credit history")
        st.number_input("Open credit lines", min_value=0, max_value=80,
                        step=1, key=pkey(prefix, "open_acc"))
        st.number_input("Revolving utilization (%)", min_value=0.0, max_value=150.0,
                        step=1.0, key=pkey(prefix, "revol_util"))
        st.number_input("Total credit lines (ever)", min_value=0, max_value=150,
                        step=1, key=pkey(prefix, "total_acc"))
        st.number_input("Delinquencies (last 2 yrs)", min_value=0, max_value=20,
                        step=1, key=pkey(prefix, "delinq_2yrs"))
        st.number_input("Public derogatory records", min_value=0, max_value=20,
                        step=1, key=pkey(prefix, "pub_rec"))
        st.number_input("Inquiries (last 6 mo)", min_value=0, max_value=15,
                        step=1, key=pkey(prefix, "inq_last_6mths"))
        st.text_input("Earliest credit line (YYYY-MM)",
                      key=pkey(prefix, "earliest_cr_line"))

    return {k: st.session_state[pkey(prefix, k)] for k in PREDICTOR_DEFAULTS}


def render_prediction_panel(inputs: dict, *, compact: bool = False):
    """Score the inputs and draw the gauge + why-panel. Recomputes on every
    render (cheap enough for a single XGBoost call) so adjusting an input after
    a prediction auto-updates the result - effectively a built-in what-if mode.
    """
    df = build_input_row(inputs)
    try:
        proba, shap_values, bias = predict_with_contribs(df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    level, label, decision, color = risk_tier(proba)
    top_n = 6 if compact else 8

    if compact:
        st.plotly_chart(gauge_chart(proba, color), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(
            f"<div style='text-align:center; font-size:1.2rem; font-weight:600; "
            f"color:{color}; margin-top:-0.6rem;'>{label} risk &mdash; {decision}</div>",
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("Default probability", f"{proba:.1%}")
        m2.metric("Risk tier", f"{level} of 5")
        st.markdown("###### Why this prediction")
        st.plotly_chart(contribs_chart(shap_values, top_n=top_n),
                        use_container_width=True,
                        config={"displayModeBar": False})
    else:
        res_col, why_col = st.columns([1, 1.4])
        with res_col:
            st.markdown("### Prediction")
            st.plotly_chart(gauge_chart(proba, color),
                            use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown(
                f"<div style='text-align:center; font-size:1.4rem; font-weight:600; "
                f"color:{color};'>{label} risk &mdash; {decision}</div>",
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Default probability", f"{proba:.1%}")
            m2.metric("Risk tier", f"{level} of 5")
        with why_col:
            st.markdown("### Why this prediction")
            st.caption(
                "Top features driving this decision (XGBoost SHAP contributions). "
                "Red bars push toward default, green toward repayment."
            )
            st.plotly_chart(contribs_chart(shap_values, top_n=top_n),
                            use_container_width=True,
                            config={"displayModeBar": False})
            with st.expander("Model baseline (bias)"):
                base_proba = 1.0 / (1.0 + np.exp(-bias))
                st.write(
                    f"Average predicted probability before any feature contributions: "
                    f"**{base_proba:.1%}** (log-odds bias: {bias:+.3f}). The bars "
                    "above show how each feature shifts the prediction away from "
                    "this baseline."
                )


with tabs[6]:
    tab_header(
        "Live model demo",
        "Adjust borrower and loan attributes, then click Predict. After a prediction, "
        "any input change re-scores the loan live - effectively a what-if mode.",
    )

    top_l, top_r = st.columns([1.0, 2.4])
    with top_l:
        compare_mode = st.toggle(
            "Compare two loans",
            value=False,
            key="compare_mode",
            help="Show two side-by-side forms, each with its own preset and result.",
        )
    with top_r:
        st.caption(
            f"Production model: **{MODEL_NAME}**, trained on Lending Club 2014-2016 "
            f"with a time-based split. {len(FEATURE_NAMES)} features after "
            "preprocessing. Scoring is identical to the MCP server in "
            "`src/mcp/server.py`."
        )

    st.markdown("---")

    if compare_mode:
        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### Loan A")
            inputs_a = render_loan_form("a")
        with right_col:
            st.markdown("### Loan B")
            inputs_b = render_loan_form("b")

        if st.button("Predict both loans", type="primary",
                     use_container_width=True, key="predict_both"):
            st.session_state["did_predict_compare"] = True

        if st.session_state.get("did_predict_compare"):
            st.markdown("---")
            ra, rb = st.columns(2)
            with ra:
                st.markdown("### Loan A result")
                render_prediction_panel(inputs_a, compact=True)
            with rb:
                st.markdown("### Loan B result")
                render_prediction_panel(inputs_b, compact=True)
        else:
            st.info(
                "Fill both forms (or pick presets) and click **Predict both loans**. "
                "After predicting, changing any input on either side re-scores live."
            )

    else:
        inputs = render_loan_form("a")

        if st.button("Predict default risk", type="primary",
                     use_container_width=True, key="predict_single"):
            st.session_state["did_predict_single"] = True

        if st.session_state.get("did_predict_single"):
            st.markdown("---")
            render_prediction_panel(inputs, compact=False)
        else:
            st.info("Adjust inputs and click **Predict default risk** to score the loan.")


# === TAB 8: Insights + Next ==================================================
with tabs[7]:
    tab_header(
        "What we found - and what's next",
        "Key findings from the model, the three macro insights, honest limitations, and concrete future work.",
    )

    st.markdown("##### Default rate ramps cleanly with sub-grade")
    sgc1, sgc2 = st.columns([2, 1])
    with sgc1:
        st.plotly_chart(chart_default_by_subgrade(), use_container_width=True,
                        config={"displayModeBar": False})
    with sgc2:
        callout("~6%", "default rate at A1 (top grade)")
        st.markdown(" ")
        callout("~45%", "default rate at G5 (bottom grade)", kind="red")
        st.caption(
            "Lending Club's own sub-grade is already a near-sufficient summary "
            "of borrower risk - our pipeline mostly refines it."
        )

    st.markdown("---")
    st.markdown("##### What the model learned (top features by gain)")

    g1, g2 = st.columns([1.3, 1])
    with g1:
        feat_img = FIG_DIR / "feature_importance.png"
        if feat_img.exists():
            st.image(str(feat_img), use_container_width=True)

    with g2:
        card(
            "Sub-grade dominates",
            "<code>sub_grade</code> carries roughly 3x more gain than the next feature - mirroring the chart above.",
        )
        card(
            "Term length is a hidden risk lever",
            "Longer 60-month loans default substantially more often than 36-month loans at the same grade.",
        )
        card(
            "Renting > owning, all else equal",
            "<code>home_ownership = RENT</code> pushes toward default; <code>MORTGAGE</code> pulls away.",
        )

    st.markdown("---")
    st.markdown("##### Limitations and what's next")

    l_col, n_col = st.columns(2)
    with l_col:
        st.markdown("###### Limitations (honest)")
        card(
            "Class imbalance left as-is",
            "We did not resample. Any deployment with a different operating cost-ratio needs threshold re-tuning.",
            kind="red",
        )
        card(
            "Concept drift after 2017",
            "Training cohort spans falling unemployment. Model is not validated on COVID-era loan books.",
            kind="red",
        )
        card(
            "No fairness audit",
            "Lending data has well-documented protected-class disparities. Disparate-impact testing was out of scope.",
            kind="red",
        )
        card(
            "FICO-only credit signal",
            "We use <code>fico_range_low</code>; richer bureau features would likely add lift but were absent from this dataset.",
            kind="red",
        )

    with n_col:
        st.markdown("###### What's next")
        card(
            "Real-time scoring API",
            "Wrap the same pipeline in a FastAPI service; the MCP server already shows the loading pattern.",
            kind="green",
        )
        card(
            "Fairness + adverse-action layer",
            "Disparate-impact metrics by protected group + per-prediction reason codes for compliant denial letters.",
            kind="green",
        )
        card(
            "Retraining cadence",
            "Quarterly re-fit on the latest closed cohort; alarm on AUC drift > 0.02.",
            kind="green",
        )
        card(
            "More macro signals",
            "FRED unemployment is one series; rates, housing, and consumer sentiment likely add lift in nonlinear models.",
            kind="green",
        )

    st.markdown(" ")
    st.markdown("---")
    st.markdown("##### Use case revisited - did we solve it?")
    c1, c2, c3 = st.columns(3)
    with c1:
        card(
            "Approve / decline",
            "The Predict-a-loan tab returns a 5-tier recommendation (Approve / Review / Decline) per borrower. The same recommendation is reachable via Claude Desktop through MCP.",
            kind="amber",
        )
    with c2:
        card(
            "Risk-based pricing",
            "Default probability is continuous, so it directly supports tiered pricing - higher predicted risk maps to higher quoted interest rate.",
            kind="amber",
        )
    with c3:
        card(
            "Portfolio stress testing",
            "Phase 3's macro overlay lets a portfolio manager re-score historical books under different unemployment / Fed Funds regimes - the data path exists, it's a query away.",
            kind="amber",
        )


# === TAB 9: Q&A / closing ====================================================
with tabs[8]:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, var(--t-hero-a) 0%, var(--t-hero-b) 100%);
            color: var(--t-hero-text);
            padding: 3.0rem 2rem;
            border-radius: 14px;
            margin: 0.2rem 0 1.5rem 0;
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);
            border-left: 8px solid var(--t-accent);
            text-align: center;
        ">
          <div style="font-size:3.4rem; font-weight:700; line-height:1.05; color:var(--t-hero-text);">
            Thank you
          </div>
          <div style="font-size:1.7rem; color:var(--t-accent); font-weight:600;
                      margin-top:0.6rem; letter-spacing:0.04em;">
            Questions?
          </div>
          <div style="font-size:0.78rem; color:var(--t-accent); font-weight:700;
                      letter-spacing:0.2em; margin-top:1.4rem;">
            404 TEAM NOT FOUND
          </div>
          <div style="font-size:0.95rem; color:var(--t-hero-sub); font-style:italic;
                      margin-top:0.4rem;">
            Aayush Nepal  &middot;  Junwei Zhang  &middot;  Lusi Zhang
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Three things to remember")
    t1, t2, t3 = st.columns(3)
    with t1:
        card(
            "WHAT WE BUILT",
            "End-to-end loan default predictor: pandas pipeline, Spark + Delta medallion, MCP server, this Streamlit app.",
            kind="amber",
        )
    with t2:
        card(
            "HEADLINE METRIC",
            "<b>AUC-ROC 0.726</b> on 314,212 held-out 2017 loans &mdash; never seen during training or tuning.",
            kind="amber",
        )
    with t3:
        card(
            "WHY IT MATTERS",
            "The same model is reachable two ways: a visual UI for analysts, a conversational MCP tool for everyone else.",
            kind="amber",
        )

    st.markdown("---")
    rep_col, q_col = st.columns([1, 1.5])
    with rep_col:
        st.markdown("##### Repository")
        st.markdown(
            "[github.com/Aayushnepal09/Eas587_project]"
            "(https://github.com/Aayushnepal09/Eas587_project)"
        )
        st.caption(
            "Includes the full pipeline, the trained model artifact, the MCP "
            "server code, and the source for this app."
        )
    with q_col:
        st.markdown("##### Likely questions, brief answers")
        st.markdown(
            "- **Why XGBoost over HGB?** Tied on AUC-ROC at the 4th decimal; "
            "XGBoost was 5x faster to fit and exposes native SHAP "
            "contributions.\n"
            "- **Why didn't the extra macro features improve the model?** "
            "`MacroJoiner` already adds `unemployment_rate`; the additional "
            "FRED features were correlated and added little marginal signal "
            "to a linear model.\n"
            "- **Can this be deployed?** The MCP server is the deployment "
            "blueprint. A FastAPI version is the natural next step."
        )
