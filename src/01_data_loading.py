import pandas as pd
import os
import zipfile


def load_data(file_path):
    # loads a csv file and returns a dataframe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


def get_year_distribution(file_path, chunk_size=100_000):
    # reads only the issue_d column in chunks to count how many rows per year
    year_counts = {}
    for chunk in pd.read_csv(file_path, usecols=['issue_d'],
                              chunksize=chunk_size, compression=None,
                              low_memory=False):
        chunk['issue_d'] = pd.to_datetime(chunk['issue_d'], format='%b-%Y', errors='coerce')
        for year, count in chunk['issue_d'].dt.year.value_counts().items():
            year_counts[year] = year_counts.get(year, 0) + count
    return pd.Series(year_counts).sort_index()


def print_year_distribution(year_counts, title="Year distribution"):
    # prints a table showing rows per year
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

    # Dataset from Kaggle: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1
    # download archive.zip and put it in data/raw/
    raw_data_path = os.path.join(raw_data_dir, 'Loan_status_2007-2020Q3.gzip')
    zip_file_path = os.path.join(raw_data_dir, 'archive.zip')
    processed_data_path = os.path.join(processed_data_dir, 'optimized_data_14_17.csv')

    # extract from zip if we haven't already
    if not os.path.exists(processed_data_path):
        if os.path.exists(zip_file_path):
            try:
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extract('Loan_status_2007-2020Q3.gzip', raw_data_dir)
                print("Extraction complete.")
            except zipfile.BadZipFile:
                print("Error: zip file is corrupted.")
                raise
        else:
            print(f"Data file not found. Please download archive.zip from kaggle and place it in data/raw/")

    # first do a quick scan to see how many rows per year
    print("Scanning full dataset for year distribution...")
    year_counts = get_year_distribution(raw_data_path)
    print_year_distribution(year_counts, title="Full dataset - rows per year")

    # we only want 2014-2017 because:
    # - 2007-2013 is too sparse
    # - 2018-2020 loans havent fully matured yet so labels arent reliable
    start_date  = pd.Timestamp('2014-01-01')
    end_date    = pd.Timestamp('2017-12-31')
    sample_size = None   # set to like 200000 if you want a smaller sample

    # load in chunks and filter by date
    chunk_size = 100_000
    chunks = []

    try:
        for chunk in pd.read_csv(raw_data_path, chunksize=chunk_size, compression=None, low_memory=False):
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

        # check the year distribution of what we loaded
        sample_dist = df['issue_d'].dt.year.value_counts().sort_index()
        print_year_distribution(sample_dist, title="Final sample - rows per year")

        # save it out
        os.makedirs(processed_data_dir, exist_ok=True)
        processed_file_path = os.path.join(processed_data_dir, 'optimized_data_14_17.csv')
        df.to_csv(processed_file_path, index=False)
        print(f"\nData saved to {processed_file_path}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {raw_data_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
