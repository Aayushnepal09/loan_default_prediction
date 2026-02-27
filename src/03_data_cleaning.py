"""
data_cleaning.py

Data cleaning pipeline (pre-split stage).
Input : data/processed/optimized_data_14_17.csv
Output: data/processed/cleaned_data.csv

Steps:
  1. Filter loan_status and create binary target variable
  2. Drop columns with >30% missing values
  3. Drop post-loan leakage columns
  4. Drop uninformative columns
  5. Type conversions
  6. Retained columns overview
  7. Save output

Imputation and encoding are deferred to feature_engineering.py
(after train/val/test split) to prevent data leakage.

Next step: data_splitting.py -> EDA (train only) -> feature_engineering.py.
"""

import os
import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype


def prepare_target(df):
    # only keep loans that have actually finished - fully paid or charged off
    # "current", "late" etc dont have a final label yet so we skip those
    completed_statuses = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(completed_statuses)].copy()
    print(f"  rows after filtering to completed loans: {df.shape[0]}")

    df['charged_off'] = (df['loan_status'] == 'Charged Off').astype(np.uint8)
    df.drop('loan_status', axis=1, inplace=True)

    dist = df['charged_off'].value_counts(normalize=True)
    print(f"  class split -> fully paid: {dist[0]:.1%}  |  charged off: {dist[1]:.1%}")
    return df


def drop_high_missing_cols(df, threshold=0.3):
    """
    Drop columns where more than `threshold` fraction of values are missing.
    """
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  dropped {len(cols_to_drop)} columns with >{threshold * 100:.0f}% missing values.")
    return df


def drop_leakage_cols(df):
    # anything that gets filled in after the loan closes leaks the outcome
    # e.g. total payments received, recovery amounts, settlement info
    leakage_cols = [
        'last_pymnt_amnt', 'last_pymnt_d',
        'total_pymnt', 'total_pymnt_inv',
        'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'out_prncp', 'out_prncp_inv',
        'hardship_flag',
        'recoveries', 'collection_recovery_fee',
        'last_credit_pull_d',
        'last_fico_range_low', 'last_fico_range_high',
        'debt_settlement_flag', 'debt_settlement_flag_date',
        'settlement_status', 'settlement_date',
        'settlement_amount', 'settlement_percentage', 'settlement_term',
    ]
    existing = [c for c in leakage_cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    print(f"  dropped {len(existing)} post-loan leakage columns.")
    return df


def drop_uninformative_cols(df):
    # IDs, free-text fields, near-duplicate cols, zero variance cols - none of these help
    drop_list = [
        'id', 'member_id', 'url',
        'desc', 'title', 'emp_title',
        'zip_code',
        'policy_code',
        'pymnt_plan',
        'funded_amnt', 'funded_amnt_inv',
        'next_pymnt_d',
        'installment',
        'disbursement_method',
        'application_type',
        'Unnamed: 0',
        'grade',        # almost perfectly correlated with int_rate, keeping int_rate instead
        'tax_liens',
        'acc_now_delinq',
        'num_sats',
        'delinq_amnt',
        'num_tl_30dpd',
        'chargeoff_within_12_mths',
        'collections_12_mths_ex_med',
        'num_tl_120dpd_2m',
    ]
    existing = [c for c in drop_list if c in df.columns]
    df.drop(columns=existing, inplace=True)
    print(f"  dropped {len(existing)} uninformative columns.")
    return df


def fix_invalid_values(df):
    """
    Set physically impossible values to NaN so they can be imputed later
    in feature_engineering.py (fit on train only).

    Rules derived from domain knowledge -- no target involved:
      - dti < 0         : debt-to-income ratio cannot be negative (2 rows in raw data)
      - annual_inc < 0  : income cannot be negative
      - open_acc < 0    : account count cannot be negative
      - revol_util > 200: revolving utilization above 200% is a data-entry error
    """
    fixes = 0
    checks = [
        ('dti',        df['dti'] < 0,          'impossible value(s) (<0)'),
        ('annual_inc', df['annual_inc'] < 0,    'impossible value(s) (<0)'),
        ('open_acc',   df['open_acc'] < 0,      'impossible value(s) (<0)'),
        ('revol_util', df['revol_util'] > 200,  'impossible value(s) (>200)'),
    ]
    for col, condition, label in checks:
        if col in df.columns:
            n = condition.sum()
            if n > 0:
                df.loc[condition, col] = np.nan
                print(f"  '{col}': set {n} {label} to NaN.")
                fixes += n
    if fixes == 0:
        print("  no impossible values found.")
    return df


def strip_numeric_strings(df):
    """
    Numeric string normalization.
    Convert columns where numeric values are stored as strings with non-numeric
      - term       : " 36 months" -> 36   (strip unit suffix, cast to int)
      - int_rate   : "12.5%"      -> 12.5 (strip '%', cast to float)
      - revol_util : "45.3%"      -> 45.3 (strip '%', cast to float)
    """
    if 'term' in df.columns and not pd.api.types.is_numeric_dtype(df['term']):
        df['term'] = df['term'].str.strip().str.split().str[0].astype(int)
        print("  'term': stripped unit suffix, converted to integer.")

    for col in ['int_rate', 'revol_util']:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].str.rstrip('%').astype(float)
            print(f"  '{col}': stripped '%', converted to float.")

    return df


def parse_structured_text(df):
    """
    Structured text and date parsing.

    Convert columns whose string values encode structured information that
    requires either a domain-ordered lookup table or a date-format parser:
      - emp_length       : "5 years" / "10+ years" / NaN
                           -> ordinal integer 0-10, unknown -> -1
      - earliest_cr_line : "Oct-1981" -> datetime
      - issue_d          : "2014-01-01" -> datetime
    """
    if 'emp_length' in df.columns and not pd.api.types.is_numeric_dtype(df['emp_length']):
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years':  4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years':  8, '9 years': 9, '10+ years': 10,
        }
        df['emp_length'] = df['emp_length'].map(emp_map).fillna(-1).astype(int)
        print("  'emp_length': ordinal text -> integer (unknown -> -1).")

    if 'earliest_cr_line' in df.columns and not pd.api.types.is_numeric_dtype(df['earliest_cr_line']):
        df['earliest_cr_line'] = pd.to_datetime(
            df['earliest_cr_line'], format='%b-%Y', errors='coerce'
        )
        print("  'earliest_cr_line': parsed to datetime (format='%b-%Y').")

    if 'issue_d' in df.columns and not pd.api.types.is_numeric_dtype(df['issue_d']):
        df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
        print("  'issue_d': parsed to datetime.")

    return df


if __name__ == '__main__':
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    input_path   = os.path.join(processed_dir, 'optimized_data_14_17.csv')
    cleaned_path = os.path.join(processed_dir, 'cleaned_data.csv')

    print("\n" + "=" * 60)
    print("STEP 1: Load data")
    print("=" * 60)
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n" + "=" * 60)
    print("STEP 2: Prepare target variable")
    print("=" * 60)
    df = prepare_target(df)

    print("\n" + "=" * 60)
    print("STEP 3: Drop high-missing columns  (threshold = 30%)")
    print("=" * 60)
    df = drop_high_missing_cols(df, threshold=0.3)
    print(f"  remaining columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 4: Drop post-loan leakage columns")
    print("=" * 60)
    df = drop_leakage_cols(df)
    print(f"  remaining columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 5: Drop uninformative columns")
    print("=" * 60)
    df = drop_uninformative_cols(df)
    print(f"  remaining columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 6: Numeric string normalization (strip suffixes/symbols)")
    print("=" * 60)
    df = strip_numeric_strings(df)

    print("\n" + "=" * 60)
    print("STEP 7: Structured text and date parsing")
    print("=" * 60)
    df = parse_structured_text(df)
    print(f"  columns after type conversions: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 8: Fix impossible values (domain constraints)")
    print("=" * 60)
    df = fix_invalid_values(df)

    print("\n" + "=" * 60)
    print("STEP 9: Retained columns overview")
    print("=" * 60)
    col_info = df.dtypes.reset_index()
    col_info.columns = ['column', 'dtype']
    col_info['null_pct'] = (df.isnull().mean() * 100).values
    print(f"  total columns retained: {df.shape[1]}")
    print(f"  total nulls remaining : {df.isnull().sum().sum()}\n")
    print(f"  {'#':<4} {'Column':<35} {'Dtype':<12} {'Null%'}")
    print(f"  {'-'*4} {'-'*35} {'-'*12} {'-'*6}")
    for i, row in col_info.iterrows():
        print(f"  {i+1:<4} {row['column']:<35} {str(row['dtype']):<12} {row['null_pct']:.1f}%")

    print("\n" + "=" * 60)
    print("STEP 10: Save output")
    print("=" * 60)
    df.to_csv(cleaned_path, index=False)
    print(f"  saved -> {cleaned_path}")

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"  cleaned_data.csv : {df.shape}")
    print(f"\nnext: data_splitting.py -> EDA (train only) -> feature_engineering.py")
