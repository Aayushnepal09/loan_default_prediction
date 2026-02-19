"""
data_cleaning.py

Data cleaning pipeline (pre-EDA stage).
Input : data/processed/optimized_accepted_data.csv  (150k raw rows)
Output: data/processed/cleaned_data.csv

Steps:
  1. Filter loan_status and create binary target variable (charged_off)
  2. Drop columns with >30% missing values
  3. Drop post-loan leakage columns
  4. Drop uninformative columns (IDs, free-text, near-duplicates)
  5. Type conversions (term, int_rate, revol_util, emp_length)
  6. Impute remaining missing values
  7. Label-encode remaining categorical columns

Prerequisites:
  - Run preliminary_eda.py first to understand the data.
  - Adjust cleaning logic below based on EDA findings before running.

Next step: data_splitting.py.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# SECTION 1: TARGET VARIABLE
# ============================================================

def prepare_target(df):
    """
    Filter the dataset to completed loans only and create a binary target.

    Keep:
      - 'Fully Paid'   -> charged_off = 0
      - 'Charged Off'  -> charged_off = 1
    Drop all other statuses (Current, Late, In Grace Period, Default, etc.)
    because their outcome is not yet known and they cannot be used for
    supervised classification.
    """
    completed_statuses = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(completed_statuses)].copy()
    print(f"  Rows after filtering to completed loans: {df.shape[0]}")

    df['charged_off'] = (df['loan_status'] == 'Charged Off').astype(np.uint8)
    df.drop('loan_status', axis=1, inplace=True)

    dist = df['charged_off'].value_counts(normalize=True)
    print(f"  Class distribution -> Fully Paid: {dist[0]:.1%}  |  Charged Off: {dist[1]:.1%}")
    return df


# ============================================================
# SECTION 2: COLUMN FILTERING
# ============================================================

def drop_high_missing_cols(df, threshold=0.3):
    """
    Drop columns where more than `threshold` fraction of values are missing.
    The reference notebook identified ~58 such columns using a 30% threshold,
    including joint-application fields, hardship fields, and settlement fields.
    """
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped {len(cols_to_drop)} columns with >{threshold * 100:.0f}% missing values.")
    return df


def drop_leakage_cols(df):
    """
    Remove post-origination columns that would leak the loan outcome.

    These fields are only populated AFTER a loan has been repaid or defaulted,
    so including them in a predictive model would allow the model to learn from
    information unavailable at the time of the original lending decision.

    Key leakage groups:
      - Post-payment totals    : total_pymnt, total_rec_*, out_prncp, etc.
      - Post-default recovery  : recoveries, collection_recovery_fee
      - Updated credit data    : last_fico_range_*, last_credit_pull_d
      - Debt settlement data   : debt_settlement_flag, settlement_*
    """
    leakage_cols = [
        # Post-payment data
        'last_pymnt_amnt', 'last_pymnt_d',
        'total_pymnt', 'total_pymnt_inv',
        'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'out_prncp', 'out_prncp_inv',
        # Post-default recovery data
        'recoveries', 'collection_recovery_fee',
        # Post-origination credit data (updated after loan is issued)
        'last_credit_pull_d',
        'last_fico_range_low', 'last_fico_range_high',
        # Debt settlement (only occurs after charge-off)
        'debt_settlement_flag', 'debt_settlement_flag_date',
        'settlement_status', 'settlement_date',
        'settlement_amount', 'settlement_percentage', 'settlement_term',
    ]
    existing = [c for c in leakage_cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    print(f"  Dropped {len(existing)} post-loan leakage columns.")
    return df


def drop_uninformative_cols(df):
    """
    Drop columns that carry no predictive signal for loan default:
      - Unique identifiers   : id, member_id, url
      - Free-text fields     : desc, title, emp_title  (too high cardinality)
      - Near-duplicates      : funded_amnt / funded_amnt_inv (≈ loan_amnt)
      - Administrative flags : policy_code (constant=1), pymnt_plan (99%+ 'n')
      - Partial geography    : zip_code (3-digit prefix, covered by addr_state)
      - Future-dated field   : next_pymnt_d (not available at origination)
    """
    drop_list = [
        'id', 'member_id', 'url',
        'desc', 'title', 'emp_title',
        'zip_code',
        'policy_code',
        'pymnt_plan',
        'funded_amnt', 'funded_amnt_inv',
        'next_pymnt_d',
    ]
    existing = [c for c in drop_list if c in df.columns]
    df.drop(columns=existing, inplace=True)
    print(f"  Dropped {len(existing)} uninformative columns.")
    return df


# ============================================================
# SECTION 3: TYPE CONVERSIONS
# ============================================================

def convert_types(df):
    """
    Fix columns whose values are stored in an unusable string format.
    This is pure data parsing, not feature engineering:
      - term        : "36 months" -> 36 (integer)
      - int_rate    : "12.5%"     -> 12.5 (float)
      - revol_util  : "45.3%"     -> 45.3 (float)
      - emp_length  : "5 years"   -> 5 (integer), NaN -> -1

    Deferred to feature_engineering.py (after full EDA):
      - FICO merge, log(annual_inc), credit_history_years, issue_year,
        installment_to_income, loan_to_income, grade/sub_grade ordinal encoding
    """

    # -- Term: "36 months" -> 36 (integer) --
    if 'term' in df.columns and df['term'].dtype == object:
        df['term'] = df['term'].str.strip().str.split().str[0].astype(int)
        print("  'term': converted to integer.")

    # -- int_rate and revol_util: strip '%' if stored as string --
    for col in ['int_rate', 'revol_util']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.rstrip('%').astype(float)
            print(f"  '{col}': stripped '%' and converted to float.")

    # -- Employment length: text -> integer (0-10), NaN -> -1 (unknown) --
    if 'emp_length' in df.columns:
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years':  4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years':  8, '9 years': 9, '10+ years': 10,
        }
        df['emp_length'] = df['emp_length'].map(emp_map).fillna(-1).astype(int)
        print("  'emp_length': text -> integer (unknown -> -1).")

    return df


# ============================================================
# SECTION 4: MISSING VALUE IMPUTATION
# ============================================================

def impute_missing_values(df, target_col='charged_off'):
    """
    Impute remaining missing values after column drops and feature engineering.
      - Numeric columns   : fill with column median (robust to outliers)
      - Categorical cols  : fill with column mode (most frequent value)
    The target column is excluded from imputation.
    """
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    num_imputed, cat_imputed = 0, 0

    for col in num_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
            num_imputed += 1

    for col in cat_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
            cat_imputed += 1

    print(f"  Imputed {num_imputed} numeric columns (median) "
          f"and {cat_imputed} categorical columns (mode).")
    return df


# ============================================================
# SECTION 5: CATEGORICAL ENCODING
# ============================================================

def encode_categorical(df, target_col='charged_off'):
    """
    Label-encode all remaining string (object) columns.

    Covers nominal categoricals: grade, sub_grade, home_ownership,
    verification_status, purpose, addr_state, application_type,
    initial_list_status, disbursement_method, earliest_cr_line, etc.

    Note: grade and sub_grade are label-encoded here with arbitrary integer
    codes. Proper ordinal encoding (A=1...G=7) will be applied in
    feature_engineering.py after EDA confirms the monotonic relationship
    with charge-off rate.

    issue_d is excluded because it is a date used for train/test splitting,
    not a model feature.
    """
    cat_cols = [
        c for c in df.select_dtypes(include=['object']).columns
        if c not in (target_col, 'issue_d')
    ]

    if not cat_cols:
        print("  No remaining categorical columns to encode.")
        return df

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    print(f"  Label-encoded {len(cat_cols)} columns: {cat_cols}")
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == '__main__':
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    input_path   = os.path.join(processed_dir, 'optimized_accepted_data.csv')
    cleaned_path = os.path.join(processed_dir, 'cleaned_data.csv')

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Load data")
    print("=" * 60)
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Prepare target variable")
    print("=" * 60)
    df = prepare_target(df)

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Drop high-missing columns  (threshold = 30%)")
    print("=" * 60)
    df = drop_high_missing_cols(df, threshold=0.3)
    print(f"  Remaining columns: {df.shape[1]}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Drop post-loan leakage columns")
    print("=" * 60)
    df = drop_leakage_cols(df)
    print(f"  Remaining columns: {df.shape[1]}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Drop uninformative columns")
    print("=" * 60)
    df = drop_uninformative_cols(df)
    print(f"  Remaining columns: {df.shape[1]}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Type conversions")
    print("=" * 60)
    df = convert_types(df)
    print(f"  Columns after type conversion: {df.shape[1]}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Impute remaining missing values")
    print("=" * 60)
    df = impute_missing_values(df)
    print(f"  Total nulls remaining: {df.isnull().sum().sum()}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: Encode categorical variables")
    print("=" * 60)
    df = encode_categorical(df)

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9: Retained columns overview")
    print("=" * 60)
    col_info = df.dtypes.reset_index()
    col_info.columns = ['column', 'dtype']
    col_info['null_pct'] = (df.isnull().mean() * 100).values
    print(f"  Total columns retained: {df.shape[1]}\n")
    print(f"  {'#':<4} {'Column':<35} {'Dtype':<12} {'Null%'}")
    print(f"  {'-'*4} {'-'*35} {'-'*12} {'-'*6}")
    for i, row in col_info.iterrows():
        print(f"  {i+1:<4} {row['column']:<35} {str(row['dtype']):<12} {row['null_pct']:.1f}%")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 10: Save output")
    print("=" * 60)
    df.to_csv(cleaned_path, index=False)
    print(f"  Cleaned dataset saved  ->  {cleaned_path}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  cleaned_data.csv : {df.shape}")
    print("\nNext: run full EDA on cleaned_data.csv, then feature_engineering.py")
