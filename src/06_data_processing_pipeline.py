"""
Phase 2: Data Processing Pipeline
This script builds the final Scikit-Learn preprocessing pipeline (imputation, scaling, encoding)
and saves the transformed feature matrices for model training.
"""


import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Module-level helpers 

# Convert to binary: 1 if value > 0, else 0
def _binarize(X):
    return (X > 0).astype(float)


# Constants

TARGET = "charged_off"

# drop the less informative member of each high-correlation pair
CORR_DROP_COLS = [
    "fico_range_high",      # r≈1.0  with fico_range_low
    "total_bal_ex_mort",    # r≈0.99 with tot_cur_bal  (excludes mortgage → less info)
    "tot_hi_cred_lim",      # r≈0.95 with total_rev_hi_lim
    "num_bc_tl",            # subset of num_rev_accts  (bank-card only)
    "num_rev_tl_bal_gt_0",  # r≈0.97 with num_actv_rev_tl
    "num_bc_sats",          # r≈0.94 with num_actv_bc_tl
    "percent_bc_gt_75",     # binary proxy for bc_util  (continuous preferred)
    "mo_sin_old_il_acct",   # r≈0.91 with mo_sin_old_rev_tl_op
    "avg_cur_bal",          # derived: tot_cur_bal / open_acc
    "total_bc_limit",       # r≈0.93 with bc_open_to_buy + bc_util combined
    "bc_util",              # r≈0.84 with revol_util  (revol_util is broader)
    "num_op_rev_tl",        # r≈0.83 with open_acc  (open_acc is broader)
    "num_actv_bc_tl",       # r≈0.80 with num_actv_rev_tl  (rev is broader)
]

# Ordinal grade: A1 (best) → G5 (worst)
SUB_GRADE_ORDER = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]

# Columns with high right-skew that respond well to log1p
# dti: cap at 99th percentile first (max=999 is a data error), then log1p
SKEWED_COLS = [
    "bc_open_to_buy",
    "annual_inc",
    "revol_bal",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "dti",                  # capped before log1p
    "inq_last_6mths",
    "num_accts_ever_120_pd",
    "pub_rec_bankruptcies",
    # engineered features (also right-skewed)
    "loan_to_income",
    "monthly_payment_to_income",
    "delinq_rate",
    "credit_history_months",
]

# Columns where log1p still leaves high skew — binarize to 0 vs >0
BINARIZE_COLS = [
    "tot_coll_amt",       
    "num_tl_90g_dpd_24m",  
    "pub_rec",
    "delinq_2yrs"
]

# Columns to cap at upper_quantile before any transform (extreme outliers)
DTI_CAP_COLS = ["dti"]      # max=999 vs median=18.84 — data entry error

# Columns where missingness itself is informative → add binary flag before imputing
MISSING_INDICATOR_COLS = ["mths_since_recent_inq"]

ORDINAL_COLS = ["sub_grade"]
ORDINAL_CATEGORIES = [
    SUB_GRADE_ORDER,        # sub_grade: A1 … G5
]

OHE_COLS = ["home_ownership", "verification_status", "purpose", "addr_state",
            "initial_list_status"]


# Custom Transformer 1 — DropCorrelated

class DropCorrelated(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        to_drop = [c for c in self.cols if c in X.columns]
        logger.debug("DropCorrelated: removing %d columns", len(to_drop))
        return X.drop(columns=to_drop)


# Custom Transformer 2 — DateExtractor

class DateExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        issue = pd.to_datetime(X["issue_d"], errors="coerce")
        cr    = pd.to_datetime(X["earliest_cr_line"], errors="coerce")

        X["issue_year"]  = issue.dt.year.astype(float)
        X["issue_month"] = issue.dt.month.astype(float)
        X["credit_history_months"] = (
            (issue.dt.year  - cr.dt.year) * 12 +
            (issue.dt.month - cr.dt.month)
        ).clip(lower=0).astype(float)

        X.drop(columns=["issue_d", "earliest_cr_line"], inplace=True)
        logger.debug("DateExtractor: added issue_year, issue_month, credit_history_months")
        return X


# Custom Transformer 3 — FeatureConstructor

class FeatureConstructor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["loan_to_income"] = X["loan_amnt"] / (X["annual_inc"] + 1.0)

        monthly_rate = X["int_rate"] / 100.0 / 12.0
        n = X["term"].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = (1.0 + monthly_rate) ** n
            pmt = np.where(
                monthly_rate == 0,
                X["loan_amnt"] / n,
                X["loan_amnt"] * monthly_rate * factor / (factor - 1.0),
            )
        X["monthly_payment_to_income"] = pmt / (X["annual_inc"] / 12.0 + 1.0)

        X["delinq_rate"] = X["delinq_2yrs"] / (X["total_acc"] + 1.0)

        logger.debug("FeatureConstructor: added loan_to_income, monthly_payment_to_income, delinq_rate")
        return X


# Custom Transformer 4 — OutlierCapper

class OutlierCapper(BaseEstimator, TransformerMixin):

    def __init__(self, cols, upper_quantile= 0.99):
        self.cols = cols
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.caps_ = {
            col: X[col].quantile(self.upper_quantile)
            for col in self.cols
            if col in X.columns
        }
        for col, cap in self.caps_.items():
            print("OutlierCapper: '%s' capped at %.2f (p%.0f)",
                        col, cap, self.upper_quantile * 100)
        return self

    def transform(self, X):
        X = X.copy()
        for col, cap in self.caps_.items():
            if col in X.columns:
                X[col] = X[col].clip(upper=cap)
        return X


# Custom Transformer 5 — MissingIndicatorAdder

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        self.cols_present_ = [c for c in self.cols if c in X.columns]
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols_present_:
            X[f"{col}_missing"] = X[col].isna().astype(float)
            logger.debug("MissingIndicatorAdder: added '%s_missing'", col)
        return X


# Custom Transformer 6 — MacroJoiner

class MacroJoiner(BaseEstimator, TransformerMixin):

    def __init__(self, start= "2013-01-01", end= "today"):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        end = pd.Timestamp.today().strftime("%Y-%m-%d") if self.end == "today" else self.end
        try:
            raw = web.DataReader("UNRATE", "fred", self.start, end).reset_index()
            raw.columns = ["date", "unemployment_rate"]
            raw["year"]  = raw["date"].dt.year
            raw["month"] = raw["date"].dt.month
            self.macro_df_ = raw[["year", "month", "unemployment_rate"]].copy()
            print("MacroJoiner: fetched %d monthly UNRATE records from FRED",
                        len(self.macro_df_))
        except Exception as exc:
            print("MacroJoiner: FRED fetch failed (%s). unemployment_rate will be NaN.", exc)
            self.macro_df_ = None
        return self

    def transform(self, X):
        if self.macro_df_ is None:
            X = X.copy()
            X["unemployment_rate"] = np.nan
            return X

        X = X.merge(
            self.macro_df_,
            left_on=["issue_year", "issue_month"],
            right_on=["year", "month"],
            how="left",
        ).drop(columns=["year", "month"])

        if X["unemployment_rate"].isna().any():
            latest = self.macro_df_["unemployment_rate"].iloc[-1]
            X["unemployment_rate"] = X["unemployment_rate"].fillna(latest)

        return X


# Custom Transformer 7 — RareCategoryMerger

class RareCategoryMerger(BaseEstimator, TransformerMixin):

    def __init__(self, cols, threshold= 50):
        self.cols = cols
        self.threshold = threshold

    def fit(self, X, y=None):
        self.rare_ = {}
        for col in self.cols:
            if col not in X.columns:
                continue
            counts = X[col].value_counts()
            rare = counts[counts < self.threshold].index.tolist()
            self.rare_[col] = rare
            if rare:
                print("RareCategoryMerger: col='%s' rare categories %s → 'other'",
                            col, rare)
        return self

    def transform(self, X):
        X = X.copy()
        for col, rare in self.rare_.items():
            if rare and col in X.columns:
                X[col] = X[col].replace(rare, "other")
        return X


# Custom Transformer 8 — ColumnPreprocessor

class ColumnPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        skewed_cols,
        binarize_cols,
        ordinal_cols,
        ordinal_categories,
        ohe_cols,
        target= TARGET,
    ):
        self.skewed_cols = skewed_cols
        self.binarize_cols = binarize_cols
        self.ordinal_cols = ordinal_cols
        self.ordinal_categories = ordinal_categories
        self.ohe_cols = ohe_cols
        self.target = target

    def fit(self, X, y=None):
        skewed   = [c for c in self.skewed_cols   if c in X.columns]
        binarize = [c for c in self.binarize_cols  if c in X.columns]
        ordinal  = [c for c in self.ordinal_cols   if c in X.columns]
        ohe      = [c for c in self.ohe_cols       if c in X.columns]
        ordinal_cats = [
            self.ordinal_categories[i]
            for i, c in enumerate(self.ordinal_cols)
            if c in X.columns
        ]

        exclude = set(skewed + binarize + ordinal + ohe + [self.target])
        normal = [
            c for c in X.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(X[c])
        ]

        self.skewed_   = skewed
        self.binarize_ = binarize
        self.normal_   = normal
        self.ordinal_  = ordinal
        self.ohe_      = ohe

        print(
            "ColumnPreprocessor: skewed=%d  binarize=%d  normal=%d  ordinal=%d  ohe=%d",
            len(skewed), len(binarize), len(normal), len(ordinal), len(ohe),
        )

        skewed_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p",   FunctionTransformer(np.log1p, validate=False,
                                             feature_names_out="one-to-one")),
            ("scaler",  StandardScaler()),
        ])

        binarize_pipe = Pipeline([
            ("imputer",   SimpleImputer(strategy="constant", fill_value=0)),
            ("binarize",  FunctionTransformer(_binarize, validate=False,
                                               feature_names_out="one-to-one")),
        ])

        normal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        ordinal_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=ordinal_cats,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
            ("scaler", StandardScaler()),
        ])

        ohe_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                drop="first",
                sparse_output=False,
                handle_unknown="ignore",
            )),
        ])

        self.ct_ = ColumnTransformer(
            transformers=[
                ("skewed",   skewed_pipe,   skewed),
                ("binarize", binarize_pipe, binarize),
                ("normal",   normal_pipe,   normal),
                ("ordinal",  ordinal_pipe,  ordinal),
                ("ohe",      ohe_pipe,      ohe),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.ct_.fit(X)
        self.feature_names_out_ = self.ct_.get_feature_names_out().tolist()
        return self

    def transform(self, X):
        return self.ct_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


# Pipeline builder

def build_pipeline() -> Pipeline:

    return Pipeline([
        ("drop_corr",    DropCorrelated(cols=CORR_DROP_COLS)),
        ("dates",        DateExtractor()),
        ("features",     FeatureConstructor()),
        ("outlier_cap",  OutlierCapper(cols=DTI_CAP_COLS, upper_quantile=0.99)),
        ("missing_flags", MissingIndicatorAdder(cols=MISSING_INDICATOR_COLS)),
        ("macro",        MacroJoiner(start="2013-01-01", end="today")),
        ("rare_merge",   RareCategoryMerger(
                             cols=["purpose", "home_ownership"],
                             threshold=50,
                         )),
        ("preprocessor", ColumnPreprocessor(
            skewed_cols=SKEWED_COLS,
            binarize_cols=BINARIZE_COLS,
            ordinal_cols=ORDINAL_COLS,
            ordinal_categories=ORDINAL_CATEGORIES,
            ohe_cols=OHE_COLS,
        )),
    ])


# Training entry point

def train(data_dir, artifact_dir):

    os.makedirs(artifact_dir, exist_ok=True)

    # Load 
    print("Loading data from %s", data_dir)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
    val_df   = pd.read_csv(os.path.join(data_dir, "val.csv"),   low_memory=False)
    test_df  = pd.read_csv(os.path.join(data_dir, "test.csv"),  low_memory=False)
    print("Loaded — train %s  val %s  test %s",
                train_df.shape, val_df.shape, test_df.shape)

    # Split features / target
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].values.astype(int)
    X_val   = val_df.drop(columns=[TARGET])
    y_val   = val_df[TARGET].values.astype(int)
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET].values.astype(int)

    # Build and fit pipeline
    print("Building and fitting preprocessing pipeline on train set")
    pipeline = build_pipeline()
    pipeline.fit(X_train)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    print("Pipeline fitted — %d features output", len(feature_names))

    # Transform
    print("Transforming all splits")
    Xt_train = pipeline.transform(X_train)
    Xt_val   = pipeline.transform(X_val)
    Xt_test  = pipeline.transform(X_test)

    df_train_out = pd.DataFrame(Xt_train, columns=feature_names)
    df_train_out[TARGET] = y_train

    df_val_out = pd.DataFrame(Xt_val, columns=feature_names)
    df_val_out[TARGET] = y_val

    df_test_out = pd.DataFrame(Xt_test, columns=feature_names)
    df_test_out[TARGET] = y_test

    print("train_features %s  val_features %s  test_features %s",
                df_train_out.shape, df_val_out.shape, df_test_out.shape)
    print("Class balance (train) — 0: %.1f%%  1: %.1f%%",
                (y_train == 0).mean() * 100, (y_train == 1).mean() * 100)

    # Save CSVs
    df_train_out.to_csv(os.path.join(artifact_dir, "train_features.csv"), index=False)
    df_val_out.to_csv(  os.path.join(artifact_dir, "val_features.csv"),   index=False)
    df_test_out.to_csv( os.path.join(artifact_dir, "test_features.csv"),  index=False)
    print("Feature CSVs saved to %s", artifact_dir)

    # Save pipeline artifact
    artifact_path = os.path.join(artifact_dir, "preprocessing_pipeline.pkl")
    with open(artifact_path, "wb") as fh:
        pickle.dump(pipeline, fh)
    print("Pipeline artifact saved → %s", artifact_path)
    print("Pipeline complete. Next: 07_model_training.py")



if __name__ == "__main__":
    _src_dir  = os.path.dirname(os.path.abspath(__file__))
    _root_dir = os.path.join(_src_dir, "..")

    DATA_DIR     = os.getenv("DATA_DIR",     os.path.join(_root_dir, "data", "processed"))
    ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", DATA_DIR)

    train(DATA_DIR, ARTIFACT_DIR)
