"""
Phase 1: Data Inspection
This script reads the filtered data and prints out basic statistics, missing value counts,
and class distribution to help us understand what needs to be cleaned.
"""


import os
import numpy as np
import pandas as pd


def raw_data_inspection(df):
    SEP = "=" * 60
    findings = {}

    # 1 - basic shape and date range
    print(f"\n{SEP}")
    print("EDA [1/6]  DATA STRUCTURE & TIME RANGE")
    print(SEP)

    print(f"  Rows    : {df.shape[0]:,}")
    print(f"  Columns : {df.shape[1]}")

    issue_d_parsed = None
    if 'issue_d' in df.columns:
        issue_d_parsed = pd.to_datetime(df['issue_d'], errors='coerce')
        date_min  = issue_d_parsed.min()
        date_max  = issue_d_parsed.max()
        is_sorted = issue_d_parsed.is_monotonic_increasing
        print(f"\n  Date range : {date_min.strftime('%b-%Y')} -> {date_max.strftime('%b-%Y')}")
        print(f"  Sorted     : {'yes' if is_sorted else 'no'}")
        findings['date_range'] = f"{date_min.strftime('%b-%Y')} to {date_max.strftime('%b-%Y')}"
        findings['is_sorted'] = is_sorted
    else:
        print("  WARNING: issue_d column not found")
        findings['date_range'] = 'N/A'
        findings['is_sorted']  = 'N/A'

    # 2 - missing values
    print(f"\n{SEP}")
    print("EDA [2/6]  MISSING VALUES")
    print(SEP)

    missing = df.isnull().mean().sort_values(ascending=False)
    total_missing_cells = df.isnull().sum().sum()
    cols_over_30 = int((missing > 0.3).sum())
    cols_with_any = int((missing > 0).sum())

    print(f"  total missing cells  : {total_missing_cells:,}  ({total_missing_cells / df.size * 100:.1f}% of all values)")
    print(f"  cols with any NaN    : {cols_with_any} / {df.shape[1]}")
    print(f"  cols with >30% NaN   : {cols_over_30}  (will be dropped)")

    print(f"\n  top 25 columns by missing rate:")
    print(f"  {'Column':<45} {'Rate':>7}  Action")
    print(f"  {'-'*45} {'-'*7}  {'-'*8}")
    for col, rate in missing[missing > 0].head(25).items():
        action = "DROP" if rate > 0.3 else "impute"
        print(f"  {col:<45} {rate:>7.1%}  {action}")

    findings['cols_over_30pct']   = cols_over_30
    findings['cols_with_any_nan'] = cols_with_any

    # 3 - duplicates and sanity checks
    print(f"\n{SEP}")
    print("EDA [3/6]  DUPLICATE & CONSISTENCY CHECK")
    print(SEP)

    n_dupes = int(df.duplicated().sum())
    print(f"  Duplicate rows : {n_dupes:,}")
    findings['duplicate_rows'] = n_dupes

    if 'id' in df.columns:
        n_dup_ids = int(df['id'].duplicated().sum())
        print(f"  Duplicate loan IDs : {n_dup_ids:,}")

    print(f"\n  Consistency checks:")
    checks = {
        'funded_amnt > loan_amnt': (
            lambda d: (d['funded_amnt'] > d['loan_amnt']).sum()
            if {'funded_amnt', 'loan_amnt'} <= set(d.columns) else None
        ),
        'fico_range_low > fico_range_high': (
            lambda d: (d['fico_range_low'] > d['fico_range_high']).sum()
            if {'fico_range_low', 'fico_range_high'} <= set(d.columns) else None
        ),
        'dti < 0': (
            lambda d: (d['dti'] < 0).sum()
            if 'dti' in d.columns else None
        ),
        'annual_inc < 0': (
            lambda d: (d['annual_inc'] < 0).sum()
            if 'annual_inc' in d.columns else None
        ),
        'open_acc < 0': (
            lambda d: (d['open_acc'] < 0).sum()
            if 'open_acc' in d.columns else None
        ),
    }

    for label, check_fn in checks.items():
        result = check_fn(df)
        if result is not None:
            status = 'OK' if result == 0 else f'WARNING  ({result:,} rows affected)'
            print(f"    {label:<40} {status}")

    # 4 - distributions and outliers for key columns
    print(f"\n{SEP}")
    print("EDA [4/6]  DISTRIBUTION & OUTLIER CHECK  (key numeric columns)")
    print(SEP)

    key_cols = [
        c for c in ['loan_amnt', 'annual_inc', 'dti', 'int_rate',
                    'revol_util', 'fico_range_low', 'installment',
                    'revol_bal', 'open_acc', 'total_acc', 'delinq_2yrs']
        if c in df.columns
    ]

    print(f"  {'Column':<22} {'Min':>10} {'Median':>12} {'Mean':>12} {'Max':>14}  {'Outliers(3s)':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*12} {'-'*14}  {'-'*12}")

    for col in key_cols:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s.astype(str).str.rstrip('%'), errors='coerce')
        s = s.dropna()
        if len(s) == 0:
            continue
        mean, std  = s.mean(), s.std()
        n_outliers = int(((s < mean - 3 * std) | (s > mean + 3 * std)).sum())
        print(f"  {col:<22} {s.min():>10,.1f} {s.median():>12,.1f} {mean:>12,.1f} {s.max():>14,.1f}  {n_outliers:>12,}")

    # 5 - target variable class imbalance
    print(f"\n{SEP}")
    print("EDA [5/6]  TARGET VARIABLE & CLASS IMBALANCE")
    print(SEP)

    if 'loan_status' in df.columns:
        counts = df['loan_status'].value_counts(dropna=False)
        pcts   = counts / len(df) * 100
        print("  Full loan_status distribution (before filtering):")
        for status, cnt in counts.items():
            print(f"    {str(status):<55} {cnt:>7,}  ({pcts[status]:5.1f}%)")

        df_model = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
        n_total  = len(df_model)
        if n_total > 0:
            n_co = int((df_model['loan_status'] == 'Charged Off').sum())
            n_fp = int((df_model['loan_status'] == 'Fully Paid').sum())
            pct_co    = n_co / n_total
            imbalance = n_fp / n_co if n_co > 0 else float('inf')

            print(f"\n  After filtering to completed loans ({n_total:,} rows):")
            print(f"    Fully Paid  : {n_fp:>7,}  ({n_fp / n_total:.1%})")
            print(f"    Charged Off : {n_co:>7,}  ({pct_co:.1%})")
            print(f"    Imbalance   : {imbalance:.1f}:1")
            severity = 'Moderate' if imbalance < 5 else 'Severe'
            print(f"    Severity    : {severity}")
            print(f"    will need SMOTE or class_weight=balanced")

            findings['pct_charged_off'] = pct_co
            findings['imbalance_ratio'] = imbalance

    # 5b - temporal drift
    print(f"\n{SEP}")
    print("EDA [5b]   TEMPORAL DRIFT  (charged-off rate by year)")
    print(SEP)

    if issue_d_parsed is not None and 'loan_status' in df.columns:
        df_tmp = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
        df_tmp['_year'] = issue_d_parsed[df_tmp.index].dt.year
        df_tmp['_co']   = (df_tmp['loan_status'] == 'Charged Off').astype(int)

        drift = (df_tmp.groupby('_year')
                       .agg(total=('_co', 'count'), n_co=('_co', 'sum'))
                       .assign(rate=lambda x: x['n_co'] / x['total']))

        print(f"  {'Year':<6} {'Loans':>8} {'Charged Off':>13} {'Rate':>7}  Chart")
        print(f"  {'-'*6} {'-'*8} {'-'*13} {'-'*7}  {'-'*24}")
        for year, row in drift.iterrows():
            bar = '█' * int(row['rate'] * 50)
            print(f"  {year:<6} {int(row['total']):>8,} {int(row['n_co']):>13,} {row['rate']:>7.1%}  {bar}")

        rate_range = drift['rate'].max() - drift['rate'].min()
        drift_flag = rate_range > 0.05
        print(f"\n  rate range across years : {rate_range:.1%}")
        print(f"  temporal drift : {'DETECTED -> use time-based split' if drift_flag else 'not significant'}")

        findings['temporal_drift']   = drift_flag
        findings['drift_rate_range'] = rate_range
    else:
        print("  cant compute - issue_d or loan_status not found")

    # 6 - data types
    print(f"\n{SEP}")
    print("EDA [6/6]  DATA TYPE CLASSIFICATION")
    print(SEP)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    object_cols  = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # check which string columns are actually numeric stored as text (e.g. "12.5%")
    numeric_as_str = []
    for col in object_cols:
        sample = df[col].dropna().head(200)
        try:
            cleaned = sample.str.rstrip('%').str.split().str[0]
            pd.to_numeric(cleaned, errors='raise')
            numeric_as_str.append(col)
        except (ValueError, AttributeError):
            pass

    true_cat_cols = [c for c in object_cols if c not in numeric_as_str]

    print(f"  Numeric columns        : {len(numeric_cols)}")
    print(f"  Categorical columns    : {len(true_cat_cols)}")
    print(f"  Numeric-as-string cols : {len(numeric_as_str)}  (need type conversion)")
    if numeric_as_str:
        print(f"    -> {numeric_as_str}")

    print(f"\n  Categorical columns and cardinality:")
    print(f"  {'Column':<45} {'Unique':>7}  Top value")
    print(f"  {'-'*45} {'-'*7}  {'-'*20}")
    for col in sorted(true_cat_cols):
        n_unique = df[col].nunique()
        top_val  = str(df[col].value_counts().index[0])[:22] if n_unique > 0 else 'N/A'
        print(f"  {col:<45} {n_unique:>7}  {top_val}")

    findings['n_numeric']      = len(numeric_cols)
    findings['n_categorical']  = len(true_cat_cols)
    findings['numeric_as_str'] = numeric_as_str

    # summary at the end
    print(f"\n{SEP}")
    print("EDA SUMMARY")
    print(SEP)

    print(f"  Dataset        : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Date range     : {findings.get('date_range', 'N/A')}")
    print(f"  Time-sorted    : {findings.get('is_sorted', 'N/A')}")
    print(f"  Duplicate rows : {findings.get('duplicate_rows', 0):,}")

    print(f"\n  Missing values:")
    print(f"    Cols >30% NaN  : {findings.get('cols_over_30pct', 0)}  -> will drop")
    print(f"    Cols any NaN   : {findings.get('cols_with_any_nan', 0)}  -> will impute later")

    if 'pct_charged_off' in findings:
        print(f"\n  Class imbalance:")
        print(f"    Charged-Off rate  : {findings['pct_charged_off']:.1%}")
        print(f"    Imbalance ratio   : {findings['imbalance_ratio']:.1f}:1")
        print(f"    need to balance training data")

    if 'temporal_drift' in findings:
        if findings['temporal_drift']:
            print(f"\n  Temporal drift : DETECTED (rate varies {findings['drift_rate_range']:.1%} across years)")
            print(f"    -> use time-based split for train/test")
        else:
            print(f"\n  Temporal drift : not significant (range {findings['drift_rate_range']:.1%})")

    if findings.get('numeric_as_str'):
        print(f"\n  Need type conversion for : {findings['numeric_as_str']}")

    print(f"\n  Column types:")
    print(f"    Numeric     : {findings.get('n_numeric', 0)}")
    print(f"    Categorical : {findings.get('n_categorical', 0)}")

    print(f"\n{SEP}")
    print("END OF RAW DATA INSPECTION")
    print(SEP)


if __name__ == '__main__':
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')
    input_path    = os.path.join(processed_dir, 'optimized_data_14_17.csv')

    print(f"loading: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    raw_data_inspection(df)
