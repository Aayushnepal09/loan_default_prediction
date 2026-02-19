import pandas as pd
import os
import zipfile


def load_data(file_path):
    """
    Loads data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def get_year_distribution(file_path, chunk_size=100_000):
    """
    Lightweight first pass: read only the 'issue_d' column to count
    rows per year across the full raw dataset.

    Args:
        file_path (str): Path to the raw CSV file.
        chunk_size (int): Rows per chunk while scanning.

    Returns:
        pd.Series: Row counts indexed by year (int), sorted ascending.
    """
    year_counts = {}
    for chunk in pd.read_csv(file_path, usecols=['issue_d'],
                              chunksize=chunk_size, low_memory=False):
        chunk['issue_d'] = pd.to_datetime(chunk['issue_d'], format='%b-%Y', errors='coerce')
        for year, count in chunk['issue_d'].dt.year.value_counts().items():
            year_counts[year] = year_counts.get(year, 0) + count
    return pd.Series(year_counts).sort_index()


def print_year_distribution(year_counts, title="Year distribution"):
    """Print a formatted table of row counts per year."""
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
    raw_data_path = os.path.join(raw_data_dir, 'accepted_2007_to_2017.csv')
    zip_file_path = os.path.join(raw_data_dir, 'datasetzip.zip')

    # Check if the CSV file exists; if not, try to unzip it
    if not os.path.exists(raw_data_path):
        if os.path.exists(zip_file_path):
            print(f"CSV file not found. Extracting from {zip_file_path}...")
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(raw_data_dir)
                print("Extraction complete.")
            except zipfile.BadZipFile:
                print("Error: The zip file is corrupted.")
                raise
        else:
            print(f"Error: Neither the CSV file nor the zip file were found at {raw_data_dir}")

    # ----------------------------------------------------------------
    # STEP 1: Lightweight scan — understand full dataset time coverage
    # ----------------------------------------------------------------
    # This pass reads ONLY the 'issue_d' column so it is fast even for
    # the ~450 k-row raw file.  Use the printed table to decide the
    # start_year / end_year / sample_size parameters below.
    print("Scanning full dataset for year distribution (lightweight pass)...")
    year_counts = get_year_distribution(raw_data_path)
    print_year_distribution(year_counts, title="Full dataset — rows per year")

    # ----------------------------------------------------------------
    # STEP 2: Configure sampling based on distribution above
    # ----------------------------------------------------------------
    # 2017 data is excluded because most loans hadn't matured yet,
    # resulting in an artificially low charge-off rate (immature labels).
    # We use the following time-based split strategy:
    #   - Train      : 2014-2015  (first 80% by row order used for training)
    #   - Validation : last 20% of 2014-2015 rows (cut in data_splitting.py)
    #   - Test       : Jan 2016 – Jun 2016
    #
    # So here we load 2014-01 through 2016-06 inclusive.
    start_date  = pd.Timestamp('2014-01-01')
    end_date    = pd.Timestamp('2016-06-30')
    sample_size = None   # e.g. 200_000; None = load all rows in range

    # ----------------------------------------------------------------
    # STEP 3: Load the full data within the chosen date range
    # ----------------------------------------------------------------
    chunk_size = 100_000
    chunks = []

    try:
        for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size, low_memory=False):
            chunk['issue_d'] = pd.to_datetime(chunk['issue_d'], format='%b-%Y', errors='coerce')
            chunk = chunk[(chunk['issue_d'] >= start_date) & (chunk['issue_d'] <= end_date)]
            if len(chunk):
                chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values('issue_d', ascending=True).reset_index(drop=True)
        print(f"\nRows after date filter ({start_date.date()} – {end_date.date()}): {len(df):,}")

        if sample_size is not None:
            df = df.tail(sample_size).reset_index(drop=True)
            print(f"Sampled most recent {sample_size:,} rows.")

        print(f"Date range in sample : {df['issue_d'].min().date()} to {df['issue_d'].max().date()}")
        print(f"Final shape          : {df.shape}")

        # Show year distribution of the final sample so we can verify coverage
        sample_dist = df['issue_d'].dt.year.value_counts().sort_index()
        print_year_distribution(sample_dist, title="Final sample — rows per year")

        # ----------------------------------------------------------------
        # STEP 4: Save
        # ----------------------------------------------------------------
        processed_data_dir = os.path.join(current_dir, '..', 'data', 'processed')
        os.makedirs(processed_data_dir, exist_ok=True)
        processed_file_path = os.path.join(processed_data_dir, 'optimized_accepted_data.csv')
        df.to_csv(processed_file_path, index=False)
        print(f"\nData saved to {processed_file_path}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {raw_data_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
