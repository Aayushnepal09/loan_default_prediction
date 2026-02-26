"""
data_splitting.py

Time-based train / validation / test split.
Input : data/processed/cleaned_data.csv   (must contain 'issue_d' column)
Output: data/processed/train.csv
        data/processed/val.csv
        data/processed/test.csv

Split strategy (time-based):
  - Train+Val : 2014–2016, sorted by issue_d ascending
  - Train     : first 80% of Train+Val rows (chronological order)
  - Validation: last  20% of Train+Val rows (chronological order)
  - Test      : 2017 (full year, all loans matured by 2020)

2007-2013 excluded: too sparse (~8% of data), different lending era.
2018-2020 excluded: loans not fully matured → unreliable labels.

Using chronological order for the train/val cut prevents look-ahead bias:
the model never sees future data during training or validation tuning.
"""

import os
import pandas as pd


def time_based_split(df, target_col='charged_off', val_fraction=0.2):
    """
    Split a DataFrame into train, validation, and test sets
    using the 'issue_d' column for time-based partitioning.

    Strategy (Option A)
    -------------------
    - Test      : 2017 (full year)
    - Train+Val : 2014–2016, sorted by issue_d ascending
    - Val       : last `val_fraction` rows of the sorted Train+Val block
    - Train     : remaining first rows

    Parameters
    ----------
    df           : DataFrame containing 'issue_d' (datetime or string)
    target_col   : binary target column name
    val_fraction : fraction of 2014-2016 data reserved for validation

    Returns
    -------
    train_df, val_df, test_df : DataFrames with all columns (features + target)
    """
    df = df.copy()
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')

    # --- Test: 2017 (full year) ---
    test_df = (df[df['issue_d'].dt.year == 2017]
               .sort_values('issue_d')
               .reset_index(drop=True))

    # --- Train+Val: 2014-2016, sorted chronologically ---
    tv_df = (df[df['issue_d'].dt.year.isin([2014, 2015, 2016])]
             .sort_values('issue_d')
             .reset_index(drop=True))

    cutoff = int(len(tv_df) * (1 - val_fraction))
    train_df = tv_df.iloc[:cutoff].reset_index(drop=True)
    val_df   = tv_df.iloc[cutoff:].reset_index(drop=True)

    # Print summary
    for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if target_col in split.columns:
            rate = split[target_col].mean()
            rate_str = f"  |  {target_col} rate: {rate:.1%}"
        else:
            rate_str = ""
        print(f"  {name:<5}: {len(split):>9,} rows  |  "
              f"date range: {split['issue_d'].min().date()} – "
              f"{split['issue_d'].max().date()}{rate_str}")

    return train_df, val_df, test_df


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

    input_path  = os.path.join(processed_dir, 'cleaned_data.csv')
    train_path  = os.path.join(processed_dir, 'train.csv')
    val_path    = os.path.join(processed_dir, 'val.csv')
    test_path   = os.path.join(processed_dir, 'test.csv')

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Load cleaned data (2014-2017)")
    print("=" * 60)
    df = pd.read_csv(input_path)
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Show year distribution
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    year_dist = df['issue_d'].dt.year.value_counts().sort_index()
    print(f"  Year distribution:")
    for year, count in year_dist.items():
        print(f"    {year}: {count:>9,} rows ({count/len(df)*100:.1f}%)")

    if 'charged_off' in df.columns:
        dist = df['charged_off'].value_counts(normalize=True)
        print(f"  Class distribution -> "
              f"Fully Paid: {dist[0]:.1%}  |  Charged Off: {dist[1]:.1%}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Time-based split  (train/val: 2014-16 80/20 chrono | test: 2017)")
    print("=" * 60)
    train_df, val_df, test_df = time_based_split(df)

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Save outputs")
    print("=" * 60)
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"  train.csv saved  ->  {train_path}")
    print(f"  val.csv   saved  ->  {val_path}")
    print(f"  test.csv  saved  ->  {test_path}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"  train.csv  : {train_df.shape}")
    print(f"  val.csv    : {val_df.shape}")
    print(f"  test.csv   : {test_df.shape}")
    print("\nNext: EDA (train.csv only) -> feature_engineering.py")
