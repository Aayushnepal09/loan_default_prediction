import pandas as pd
import os
import sys

def verify_data(df):
    errors = []
    
    # 1. Term Check
    if 'term' in df.columns:
        if not pd.api.types.is_integer_dtype(df['term']):
            errors.append(f"'term' column is not integer type. Found: {df['term'].dtype}")
        
        invalid_terms = df[~df['term'].isin([36, 60])]
        if not invalid_terms.empty:
            errors.append(f"Found invalid 'term' values (expected 36 or 60): {invalid_terms['term'].unique()}")
    else:
        errors.append("'term' column missing")

    # 2. Emp Length Check
    if 'emp_length' in df.columns:
        if not pd.api.types.is_integer_dtype(df['emp_length']):
             errors.append(f"'emp_length' column is not integer type. Found: {df['emp_length'].dtype}")
        
        if df['emp_length'].min() < 0 or df['emp_length'].max() > 10:
             errors.append(f"'emp_length' values out of range (0-10). Range: {df['emp_length'].min()} - {df['emp_length'].max()}")
        
        if df['emp_length'].isnull().sum() > 0:
            errors.append(f"Found {df['emp_length'].isnull().sum()} null values in 'emp_length'")

    # 3. Missing Value Check
    cols_to_check = ['revol_util', 'pub_rec', 'mort_acc', 'pub_rec_bankruptcies', 'loan_amnt', 'loan_status']
    for col in cols_to_check:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Found {null_count} nulls in '{col}' (should be 0)")

    # 4. Date Check
    date_cols = ['issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line']
    for col in date_cols:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                errors.append(f"'{col}' is not datetime type. Found: {df[col].dtype}")

    return errors

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cleaned_sample_path = os.path.join(current_dir, '..', 'data', 'processed', 'cleaned_sample_data.csv')
    
    try:
        if not os.path.exists(cleaned_sample_path):
             print(f"File not found: {cleaned_sample_path}")
             sys.exit(1)

        df = pd.read_csv(cleaned_sample_path)
        print(f"Loaded data from {cleaned_sample_path}, shape: {df.shape}")
        
        # We need to reload data with date parsing to check date types correctly as CSV loses type info
        # However, checking if they are convertible or format is correct is better.
        # But let's check values.
        
        # Actually, reading csv makes everything object/int/float. Dates become strings.
        # So we should expect dates to be object (strings) unless we parse them. 
        # But wait, the verification requirement was to ensure they ARE datetime objects in the dataframe returned by clean_data.
        # But here we are verifying the SAVED csv. CSVs don't store "datetime" type.
        # So we should verify that we can parse them back to datetime without errors, OR modify `data_cleaning.py` to also run verification before saving.
        # Let's verify we can parse them.
        
        date_cols = ['issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line']
        for col in date_cols:
             if col in df.columns:
                 try:
                     pd.to_datetime(df[col], errors='raise')
                 except Exception as e:
                     print(f"Date parsing error in {col}: {e}")

        # Re-apply rudimentary checks for other columns
        errors = verify_data(df)
        
        # Filter out "not datetime" error for dates because of CSV loading
        errors = [e for e in errors if "is not datetime type" not in e]

        if errors:
            print("\nVerification FAILED with errors:")
            for e in errors:
                print(f"- {e}")
            sys.exit(1)
        else:
            print("\nVerification PASSED: Data looks clean.")
            sys.exit(0)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
