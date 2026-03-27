"""
Phase 1: Data Loading
This script extracts the raw Lending Club dataset from the archive and filters it to loans issued between 2014 and 2017,
saving the output for inspection and cleaning.
"""

import pandas as pd
import os
import zipfile


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"cant find file: {file_path}")
    return pd.read_csv(file_path)


def get_year_distribution(file_path, chunk_size=100_000):
    # scan just the issue_d column to see how many rows per year
    # doing it in chunks so we dont load the whole thing
    year_counts = {}
    for chunk in pd.read_csv(file_path, usecols=['issue_d'],
                              chunksize=chunk_size, compression=None,
                              low_memory=False):
        chunk['issue_d'] = pd.to_datetime(chunk['issue_d'], format='%b-%Y', errors='coerce')
        for year, count in chunk['issue_d'].dt.year.value_counts().items():
            year_counts[year] = year_counts.get(year, 0) + count
    return pd.Series(year_counts).sort_index()


def print_year_distribution(year_counts, title="Year distribution"):
    total = year_counts.sum()
    print(f"\n{title}:")
    print(f"  {'Year':<6} {'Rows':>10}   {'%':>6}")
    print(f"  {'-'*28}")
    for year, count in year_counts.items():
        pct = count / total * 100
        print(f"  {int(year):<6} {count:>10,}   {pct:>5.1f}%")
    print(f"  {'-'*28}")
    print(f"  {'Total':<6} {total:>10,}   100.0%")


if __name__ == '__main__':
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir  = os.path.join(current_dir, '..', 'data', 'raw')
    processed_data_dir = os.path.join(current_dir, '..', 'data', 'processed')

    # dataset from kaggle: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1
    # download archive.zip and drop it in data/raw/
    raw_data_path = os.path.join(raw_data_dir, 'Loan_status_2007-2020Q3.gzip')
    zip_file_path = os.path.join(raw_data_dir, 'archive.zip')
    processed_data_path = os.path.join(processed_data_dir, 'optimized_data_14_17.csv')

    # unzip it if we havent already
    if not os.path.exists(processed_data_path):
        if os.path.exists(zip_file_path):
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extract('Loan_status_2007-2020Q3.gzip', raw_data_dir)
                print("extracted zip ok")
            except zipfile.BadZipFile:
                print("zip file is broken")
                raise
        else:
            print(f"put archive.zip in data/raw/ first")

    print("scanning full dataset for year distribution...")
    year_counts = get_year_distribution(raw_data_path)
    print_year_distribution(year_counts, title="all years - rows per year")

    # we only want 2014-2017
    # before 2014 - too few rows, not representative
    # after 2017 - loans not finished yet so labels are unreliable
    start_date  = pd.Timestamp('2014-01-01')
    end_date    = pd.Timestamp('2017-12-31')
    sample_size = None

    chunk_size = 100_000
    chunks = []

    try:
        for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size, compression=None, low_memory=False):
            chunk['issue_d'] = pd.to_datetime(chunk['issue_d'], format='%b-%Y', errors='coerce')
            chunk = chunk[(chunk['issue_d'] >= start_date) & (chunk['issue_d'] <= end_date)]
            if len(chunk):
                chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values('issue_d').reset_index(drop=True)
        print(f"\nrows after date filter: {len(df):,}")

        if sample_size is not None:
            df = df.tail(sample_size).reset_index(drop=True)
            print(f"sampled {sample_size:,} most recent rows")

        print(f"date range: {df['issue_d'].min().date()} to {df['issue_d'].max().date()}")
        print(f"shape: {df.shape}")

        sample_dist = df['issue_d'].dt.year.value_counts().sort_index()
        print_year_distribution(sample_dist, title="filtered sample - rows per year")

        os.makedirs(processed_data_dir, exist_ok=True)
        processed_file_path = os.path.join(processed_data_dir, 'optimized_data_14_17.csv')
        df.to_csv(processed_file_path, index=False)
        print(f"\nsaved to {processed_file_path}")

    except FileNotFoundError:
        print(f"file not found: {raw_data_path}")
    except Exception as e:
        print(f"something went wrong: {e}")
