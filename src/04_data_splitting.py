"""
data_splitting.py

Splits cleaned_data.csv into train/val/test using time-based approach.
Train+Val = 2014-2016 (80/20 chronological), Test = 2017.
"""

import os
import pandas as pd


def time_based_split(df, target_col='charged_off', val_fraction=0.2):
    # split into train/val/test by date
    # test = all of 2017, train+val = 2014-2016 split 80/20 chronologically
    df = df.copy()
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')

    # test set is 2017
    test_df = (df[df['issue_d'].dt.year == 2017]
               .sort_values('issue_d')
               .reset_index(drop=True))

    # train+val is 2014-2016 sorted by date
    tv_df = (df[df['issue_d'].dt.year.isin([2014, 2015, 2016])]
             .sort_values('issue_d')
             .reset_index(drop=True))

    cutoff = int(len(tv_df) * (1 - val_fraction))
    train_df = tv_df.iloc[:cutoff].reset_index(drop=True)
    val_df   = tv_df.iloc[cutoff:].reset_index(drop=True)

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
    print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # show year breakdown
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
    year_dist = df['issue_d'].dt.year.value_counts().sort_index()
    print(f"  Year distribution:")
    for year, count in year_dist.items():
        print(f"    {year}: {count:>9,} rows ({count/len(df)*100:.1f}%)")

    if 'charged_off' in df.columns:
        dist = df['charged_off'].value_counts(normalize=True)
        print(f"  Class distribution -> "
              f"Fully Paid: {dist[0]:.1%}  |  Charged Off: {dist[1]:.1%}")

    print("\n" + "=" * 60)
    print("STEP 2: Time-based split  (train/val: 2014-16 80/20 chrono | test: 2017)")
    print("=" * 60)
    train_df, val_df, test_df = time_based_split(df)

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

    print("\n" + "=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)
    print(f"  train.csv  : {train_df.shape}")
    print(f"  val.csv    : {val_df.shape}")
    print(f"  test.csv   : {test_df.shape}")
