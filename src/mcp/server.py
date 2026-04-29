"""
Phase 2: MCP Deployment Server
This script wraps our trained XGBoost model in a Model Context Protocol (FastMCP) server,
exposing a prediction tool that Claude Desktop can consume via natural language.
"""

import __main__
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
PIPELINE_PATH = ROOT / "data" / "processed" / "preprocessing_pipeline.pkl"

# The pipeline was pickled while 06_data_processing_pipeline.py ran as __main__,
# we need to inject the custom transformer classes into __main__ before loading.
# If not, pickle.load() will throw an AttributeError on the custom classes.
_spec = importlib.util.spec_from_file_location(
    "_pipeline_module",
    ROOT / "src" / "06_data_processing_pipeline.py",
)
_pipeline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pipeline_module)

for name, obj in vars(_pipeline_module).items():
    if not name.startswith("__"):
        setattr(__main__, name, obj)

# load model and pipeline once at startup
with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

with open(PIPELINE_PATH, "rb") as f:
    PIPELINE = pickle.load(f)

MODEL_NAME = type(MODEL).__name__

mcp = FastMCP("lending-default-predictor")

# population medians/modes from the Lending Club dataset,
# used to fill in columns that the user doesn't need to provide
SECONDARY_DEFAULTS = {
    # credit history
    "acc_open_past_24mths": 4,
    "mo_sin_old_rev_tl_op": 130,
    "mo_sin_rcnt_rev_tl_op": 12,
    "mo_sin_rcnt_tl": 8,
    "mort_acc": 1,
    "mths_since_recent_bc": 18,
    "mths_since_recent_inq": np.nan,  # NaN indicates missing
    "num_actv_rev_tl": 5,
    "num_il_tl": 8,
    "num_rev_accts": 12,
    "num_tl_op_past_12m": 2,
    "pct_tl_nvr_dlq": 94.0,
    "total_il_high_credit_limit": 25000.0,
    # balances
    "bc_open_to_buy": 5000.0,
    "revol_bal": 10000.0,
    "tot_cur_bal": 60000.0,
    "total_rev_hi_lim": 25000.0,
    # derogatory marks
    "tot_coll_amt": 0,
    "num_tl_90g_dpd_24m": 0,
    "pub_rec_bankruptcies": 0,
    "num_accts_ever_120_pd": 0,
    # other defaults
    "addr_state": "CA",
    "initial_list_status": "w",
    "emp_length": 5,
}

VALID_SUB_GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
VALID_HOME_OWN = {"RENT", "OWN", "MORTGAGE", "OTHER", "NONE", "ANY"}
VALID_VERIF = {"Not Verified", "Verified", "Source Verified"}
VALID_PURPOSES = {
    "debt_consolidation", "credit_card", "home_improvement", "other",
    "major_purchase", "medical", "small_business", "car", "vacation",
    "moving", "house", "wedding", "renewable_energy", "educational",
}


@mcp.tool()
def predict_loan_default(
    loan_amnt: float,
    term: int,
    int_rate: float,
    sub_grade: str,
    annual_inc: float,
    dti: float,
    fico_score: int,
    home_ownership: str,
    purpose: str,
    open_acc: int = 10,
    revol_util: float = 30.0,
    total_acc: int = 20,
    delinq_2yrs: int = 0,
    pub_rec: int = 0,
    inq_last_6mths: int = 0,
    verification_status: str = "Not Verified",
    earliest_cr_line: str = "2010-01",
) -> dict:

    errors = []

    if loan_amnt <= 0:
        errors.append("loan_amnt must be a positive number.")
    if term not in (36, 60):
        errors.append("term must be 36 or 60 months.")
    if not (0 < int_rate < 100):
        errors.append("int_rate must be between 0 and 100 (exclusive).")
    if annual_inc <= 0:
        errors.append("annual_inc must be a positive number.")
    if dti < 0:
        errors.append("dti must be non-negative.")
    if not (300 <= fico_score <= 850):
        errors.append("fico_score must be between 300 and 850.")
    if sub_grade.upper() not in VALID_SUB_GRADES:
        errors.append(f"sub_grade '{sub_grade}' is invalid. Use A1-G5 (e.g. B3).")
    if home_ownership.upper() not in VALID_HOME_OWN:
        errors.append(f"home_ownership must be one of {sorted(VALID_HOME_OWN)}.")
    if verification_status not in VALID_VERIF:
        errors.append(f"verification_status must be one of {sorted(VALID_VERIF)}.")
    if open_acc < 0:
        errors.append("open_acc must be non-negative.")
    if total_acc < 0:
        errors.append("total_acc must be non-negative.")
    if not (0 <= revol_util <= 200):
        errors.append("revol_util must be between 0 and 200.")
    if delinq_2yrs < 0:
        errors.append("delinq_2yrs must be non-negative.")
    if pub_rec < 0:
        errors.append("pub_rec must be non-negative.")
    if inq_last_6mths < 0:
        errors.append("inq_last_6mths must be non-negative.")

    if errors:
        return {"error": " | ".join(errors)}

    issue_d = pd.Timestamp.now().strftime("%Y-%m")

    row = {
        "loan_amnt": float(loan_amnt),
        "term": int(term),
        "int_rate": float(int_rate),
        "sub_grade": sub_grade.upper(),
        "annual_inc": float(annual_inc),
        "dti": float(dti),
        "fico_range_low": int(fico_score),
        "home_ownership": home_ownership.upper(),
        "purpose": purpose.lower().replace(" ", "_"),
        "verification_status": verification_status,
        "open_acc": int(open_acc),
        "revol_util": float(revol_util),
        "total_acc": int(total_acc),
        "delinq_2yrs": int(delinq_2yrs),
        "pub_rec": int(pub_rec),
        "inq_last_6mths": int(inq_last_6mths),
        "issue_d": issue_d,
        "earliest_cr_line": earliest_cr_line,
        **SECONDARY_DEFAULTS,
    }

    df = pd.DataFrame([row])
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].astype(float)

    try:
        X = PIPELINE.transform(df)
    except Exception as exc:
        return {"error": f"Preprocessing failed: {exc}"}

    try:
        proba = float(MODEL.predict_proba(X)[0, 1])
    except Exception as exc:
        return {"error": f"Prediction failed: {exc}"}

    if proba < 0.15:
        risk_level, tier_label, decision = 1, "Very Low", "Approve"
    elif proba < 0.25:
        risk_level, tier_label, decision = 2, "Low", "Approve"
    elif proba < 0.35:
        risk_level, tier_label, decision = 3, "Moderate", "Review"
    elif proba < 0.50:
        risk_level, tier_label, decision = 4, "High", "Decline"
    else:
        risk_level, tier_label, decision = 5, "Very High", "Decline"

    return {
        "default_probability": round(proba, 4),
        "prediction": "HIGH RISK" if proba >= 0.35 else "LOW RISK",
        "risk_level": risk_level,
        "risk_description": f"{tier_label} risk ({proba:.1%}) — Recommendation: {decision}",
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    mcp.run()
