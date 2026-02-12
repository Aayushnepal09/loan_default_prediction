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

if __name__ == '__main__':
    # Define the path to the raw data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '..', 'data', 'raw')
    raw_data_path = os.path.join(raw_data_dir, 'accepted_2007_to_2017.csv')
    zip_file_path = os.path.join(raw_data_dir, 'datasetzip.zip')

    # Check if the CSV file exists, if not, try to unzip it
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
             # Let the subsequent code fail natively or raise here, but the original code expects raw_data_path to exist or handles it in try/except block below.
             # actually the original code has a try-except block for FileNotFoundError later, but we should probably handle it here or let it fall through.
             # The existing code has `pd.read_csv` inside a try block. checking existence explicitly here is better.
        
    # Define columns to keep and their data types for optimization
    columns_to_keep = {
        'loan_amnt': 'float32',
        'term': 'str',
        'int_rate': 'float32',
        'installment': 'float32',
        'grade': 'str',
        'sub_grade': 'str',
        'emp_length': 'str',
        'home_ownership': 'str',
        'annual_inc': 'float32',
        'verification_status': 'str',
        'issue_d': 'str',
        'loan_status': 'str',
        'purpose': 'str',
        'addr_state': 'str',
        'dti': 'float32',
        'delinq_2yrs': 'float32',
        'earliest_cr_line': 'str',
        'inq_last_6mths': 'float32',
        'open_acc': 'float32',
        'pub_rec': 'float32',
        'revol_bal': 'float32',
        'revol_util': 'float32',
        'total_acc': 'float32',
        'last_pymnt_d': 'str',
        'last_credit_pull_d': 'str',
        'application_type': 'str',
        'tot_coll_amt': 'float32',
        'tot_cur_bal': 'float32',
        'total_rev_hi_lim': 'float32'
    }

    # As the file is too large to be loaded into memory, we will process it in chunks.
    chunk_size = 100000  # Process 100,000 rows at a time
    chunks = []

    try:
        # Create a text parser to read the CSV in chunks
        for chunk in pd.read_csv(raw_data_path, usecols=columns_to_keep.keys(), dtype=columns_to_keep, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            print(f"Loaded chunk of size {len(chunk)}")

        # Concatenate all chunks into a single DataFrame
        df_optimized = pd.concat(chunks, ignore_index=True)
        print("Optimized data loading complete.")
        print(df_optimized.info())

        # Define the path for the processed data
        processed_data_dir = os.path.join(current_dir, '..', 'data', 'processed')
        os.makedirs(processed_data_dir, exist_ok=True)  # Create the directory if it doesn't exist
        
        # Save the optimized DataFrame to a new CSV file
        processed_file_path = os.path.join(processed_data_dir, 'optimized_accepted_data.csv')
        df_optimized.to_csv(processed_file_path, index=False)
        print(f"Optimized data saved to {processed_file_path}")

    except FileNotFoundError:
        print(f"Error: The file was not found at {raw_data_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

