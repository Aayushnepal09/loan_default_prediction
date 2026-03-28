"""
Phase 1: Data Splitting
This script chronologically splits the dataset into training (2014-2016), validation (2014-2016 20%),
and testing (2017) sets to prevent data leakage from future events.
"""


import os
import pandas as pd


def time_based_split(df, target_col='charged_off', val_fraction=0.2):
    # test = all of 2017, train+val = 2014-2016 split 80/20 by date order
    df = df.copy()
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')

    test_df = (df[df['issue_d'].dt.year == 2017]
               .sort_values('issue_d')
               .reset_index(drop=True))

    tv_df = (df[df['issue_d'].dt.year.isin([2014, 2015, 2016])]
             .sort_values('issue_d')
             .reset_index(drop=True))

    cutoff   = int(len(tv_df) * (1 - val_fraction))
    train_df = tv_df.iloc[:cutoff].reset_index(drop=True)
    val_df   = tv_df.iloc[cutoff:].reset_index(drop=True)

    for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if target_col in split.columns:
            rate = split[target_col].mean()
            rate_str = f"  |  {target_col} rate: {rate:.1%}"
        else:
            rate_str = ""
        print(f"  {name:<5}: {len(split):>9,} rows  |  "
              f"date range: {split['issue_d'].min().date()} - "
              f"{split['issue_d'].max().date()}{rate_str}")

    return train_df, val_df, test_df


if __name__ == '__main__':
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(current_dir, '..', 'data', 'processed')

    input_path  = os.path.join(processed_dir, 'cleaned_data.csv')
    train_path  = os.path.join(processed_dir, 'train.csv')
    val_path    = os.path.join(processed_dir, 'val.csv')
    test_path   = os.path.join(processed_dir, 'test.csv')

    print("\n" + "=" * 60)
    print("STEP 1: Load cleaned data (2014-2017)")
    print("=" * 60)
    df = pd.read_csv(input_path)
    print(f"  loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    year_dist = df['issue_d'].dt.year.value_counts().sort_index()
    print(f"  year breakdown:")
    for year, count in year_dist.items():
        print(f"    {year}: {count:>9,} rows ({count/len(df)*100:.1f}%)")

    if 'charged_off' in df.columns:
        dist = df['charged_off'].value_counts(normalize=True)
        print(f"  class split -> fully paid: {dist[0]:.1%}  |  charged off: {dist[1]:.1%}")

    print("\n" + "=" * 60)
    print("STEP 2: Time-based split  (train/val: 2014-16 80/20 | test: 2017)")
    print("=" * 60)
    train_df, val_df, test_df = time_based_split(df)

    print("\n" + "=" * 60)
    print("STEP 3: Save")
    print("=" * 60)
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,     index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"  train.csv -> {train_path}")
    print(f"  val.csv   -> {val_path}")
    print(f"  test.csv  -> {test_path}")

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"  train : {train_df.shape}")
    print(f"  val   : {val_df.shape}")
    print(f"  test  : {test_df.shape}")
