"""
data_cleaning.py

Cleans the raw data and gets it ready for splitting/modeling.
Reads optimized_data_14_17.csv, outputs cleaned_data.csv
"""

import os
import numpy as np
import pandas as pd


def prepare_target(df):
    # keep only fully paid and charged off loans, make a binary column
    completed_statuses = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(completed_statuses)].copy()
    print(f"  Rows after filtering to completed loans: {df.shape[0]}")

    df['charged_off'] = (df['loan_status'] == 'Charged Off').astype(np.uint8)
    df.drop('loan_status', axis=1, inplace=True)

    dist = df['charged_off'].value_counts(normalize=True)
    print(f"  Class distribution -> Fully Paid: {dist[0]:.1%}  |  Charged Off: {dist[1]:.1%}")
    return df


def drop_high_missing_cols(df, threshold=0.3):
    # drop any column where more than 30% of values are missing
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  Dropped {len(cols_to_drop)} columns with >{threshold * 100:.0f}% missing values.")
    return df


def drop_leakage_cols(df):
    # these columns contain info that wouldnt be available at the time of lending
    # so they would leak the outcome if we kept them
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
    print(f"  Dropped {len(existing)} post-loan leakage columns.")
    return df


def drop_uninformative_cols(df):
    # columns that wont help predict defaults - IDs, free text, near-duplicates, etc.
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
        'grade',
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
    print(f"  Dropped {len(existing)} uninformative columns.")
    return df


def fix_invalid_values(df):
    # set negative values to NaN where they dont make sense (dti, income, accounts)
    fixes = 0
    for col, condition in [('dti', df['dti'] < 0),
                            ('annual_inc', df['annual_inc'] < 0),
                            ('open_acc', df['open_acc'] < 0)]:
        if col in df.columns:
            n = condition.sum()
            if n > 0:
                df.loc[condition, col] = np.nan
                print(f"  '{col}': set {n} impossible value(s) (<0) to NaN.")
                fixes += n
    if fixes == 0:
        print("  No impossible values found.")
    return df


def convert_types(df):
    # fix columns stored as strings that should be numeric
    # term: "36 months" -> 36
    if 'term' in df.columns and df['term'].dtype == object:
        df['term'] = df['term'].str.strip().str.split().str[0].astype(int)
        print("  'term': converted to integer.")

    # int_rate and revol_util: strip the % sign
    for col in ['int_rate', 'revol_util']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.rstrip('%').astype(float)
            print(f"  '{col}': stripped '%' and converted to float.")

    # emp_length: text to number
    if 'emp_length' in df.columns:
        emp_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
            '4 years':  4, '5 years': 5, '6 years': 6, '7 years': 7,
            '8 years':  8, '9 years': 9, '10+ years': 10,
        }
        df['emp_length'] = df['emp_length'].map(emp_map).fillna(-1).astype(int)
        print("  'emp_length': text -> integer (unknown -> -1).")

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
    print(f"  Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n" + "=" * 60)
    print("STEP 2: Prepare target variable")
    print("=" * 60)
    df = prepare_target(df)

    print("\n" + "=" * 60)
    print("STEP 3: Drop high-missing columns  (threshold = 30%)")
    print("=" * 60)
    df = drop_high_missing_cols(df, threshold=0.3)
    print(f"  Remaining columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 4: Drop post-loan leakage columns")
    print("=" * 60)
    df = drop_leakage_cols(df)
    print(f"  Remaining columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 5: Drop uninformative columns")
    print("=" * 60)
    df = drop_uninformative_cols(df)
    print(f"  Remaining columns: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 6: Fix impossible values")
    print("=" * 60)
    df = fix_invalid_values(df)

    print("\n" + "=" * 60)
    print("STEP 7: Type conversions")
    print("=" * 60)
    df = convert_types(df)
    print(f"  Columns after type conversion: {df.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 8: Retained columns overview")
    print("=" * 60)
    col_info = df.dtypes.reset_index()
    col_info.columns = ['column', 'dtype']
    col_info['null_pct'] = (df.isnull().mean() * 100).values
    print(f"  Total columns retained: {df.shape[1]}")
    print(f"  Total nulls remaining : {df.isnull().sum().sum()}\n")
    print(f"  {'#':<4} {'Column':<35} {'Dtype':<12} {'Null%'}")
    print(f"  {'-'*4} {'-'*35} {'-'*12} {'-'*6}")
    for i, row in col_info.iterrows():
        print(f"  {i+1:<4} {row['column']:<35} {str(row['dtype']):<12} {row['null_pct']:.1f}%")

    print("\n" + "=" * 60)
    print("STEP 9: Save output")
    print("=" * 60)
    df.to_csv(cleaned_path, index=False)
    print(f"  Cleaned dataset saved  ->  {cleaned_path}")

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)
    print(f"  cleaned_data.csv : {df.shape}")
