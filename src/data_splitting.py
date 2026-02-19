"""
data_splitting.py

Time-based train / validation / test split.
Input : data/processed/cleaned_data.csv   (must contain 'issue_d' column)
Output: data/processed/train.csv
        data/processed/val.csv
        data/processed/test.csv

Split strategy:
  - Test       : Jan 2016 - Jun 2016  (loans with mature labels)
  - Train+Val  : all rows from 2014-2015, sorted by issue_d ascending
  - Validation : last 20% of train+val rows (most recent portion)
  - Train      : first 80% of train+val rows

2017 data is excluded because most loans hadn't matured yet, causing an
artificially low charge-off rate (immature labels). Using 2016 H1 as the
test set ensures the held-out data has had sufficient time to resolve.

Using chronological order for the train/val cut prevents look-ahead bias:
the model never sees future data during training or validation tuning.
"""

import os
import pandas as pd


def time_based_split(df, target_col='charged_off', val_fraction=0.2):
    """
    Split a cleaned DataFrame into train, validation, and test sets
    using the 'issue_d' column for time-based partitioning.

    Strategy
    --------
    - Test      : Jan 2016 - Jun 2016
    - Train+Val : year in [2014, 2015], sorted by issue_d ascending
    - Val       : last `val_fraction` rows of the sorted Train+Val block
    - Train     : remaining first rows

    Parameters
    ----------
    df           : cleaned DataFrame containing 'issue_d' (datetime)
    target_col   : binary target column name
    val_fraction : fraction of 2014-2015 data reserved for validation

    Returns
    -------
    train_df, val_df, test_df : DataFrames with all columns (features + target)
    """
    df = df.copy()
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')

    # --- Test: Jan 2016 – Jun 2016 ---
    test_df = df[
        (df['issue_d'].dt.year == 2016) & (df['issue_d'].dt.month <= 6)
    ].reset_index(drop=True)

    # --- Train+Val: 2014-2015, sorted chronologically ---
    tv_df = (df[df['issue_d'].dt.year.isin([2014, 2015])]
             .sort_values('issue_d')
             .reset_index(drop=True))

    cutoff = int(len(tv_df) * (1 - val_fraction))
    train_df = tv_df.iloc[:cutoff].reset_index(drop=True)
    val_df   = tv_df.iloc[cutoff:].reset_index(drop=True)

    for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        rate = split[target_col].mean()
        print(f"  {name:<5}: {len(split):>7,} rows  |  "
              f"date range: {split['issue_d'].min().date()} – "
              f"{split['issue_d'].max().date()}  |  "
              f"Charged Off rate: {rate:.1%}")

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
    print("STEP 1: Load cleaned data")
    print("=" * 60)
    df = pd.read_csv(input_path)
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    dist = df['charged_off'].value_counts(normalize=True)
    print(f"  Class distribution -> "
          f"Fully Paid: {dist[0]:.1%}  |  Charged Off: {dist[1]:.1%}")

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Time-based split  (train 2014-15 | val last 20% of 2014-15 | test 2016 H1)")
    print("=" * 60)
    train_df, val_df, test_df = time_based_split(df)

    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Save outputs")
    print("=" * 60)
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
