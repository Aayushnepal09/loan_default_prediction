import pandas as pd
import os

def clean_data(df):
    """
    Performs data cleaning on the dataframe.
    Args:
        df (pd.DataFrame): The input dataframe to clean.
    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    print("Starting data cleaning...")
    print(f"Initial shape: {df.shape}")

    # --- 1. Handle Missing Values ---
    
    # Impute missing values for specific columns
    # revol_util: Fill with mean
    if 'revol_util' in df.columns:
        mean_revol_util = df['revol_util'].mean()
        df['revol_util'] = df['revol_util'].fillna(mean_revol_util)
        print(f"Filled missing 'revol_util' with mean: {mean_revol_util:.2f}")

    # pub_rec, mort_acc, pub_rec_bankruptcies: Fill with 0 (assuming null means none)
    cols_to_zero_fill = ['pub_rec', 'mort_acc', 'pub_rec_bankruptcies']
    for col in cols_to_zero_fill:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            print(f"Filled missing '{col}' with 0")

    # Drop rows with missing critical values (e.g., loan_amnt, loan_status) if any remain
    # Although the inspection showed 0 nulls for loan_amnt, it's good practice.
    critical_cols = ['loan_amnt', 'loan_status']
    before_drop = df.shape[0]
    df.dropna(subset=[c for c in critical_cols if c in df.columns], inplace=True)
    if df.shape[0] < before_drop:
        print(f"Dropped {before_drop - df.shape[0]} rows with missing critical values.")


    # --- 2. Handle Duplicates ---
    before_dedup = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < before_dedup:
         print(f"Dropped {before_dedup - df.shape[0]} duplicate rows.")


    # --- 3. Feature Engineering & Standardization ---

    # Term: Remove " months" and convert to integer
    if 'term' in df.columns:
        # Ensure we work with strings first if it's object
        if df['term'].dtype == 'object':
             df['term'] = df['term'].str.replace(' months', '', regex=False).str.strip()
        
        # Convert to numeric, coercing errors to NaN
        df['term'] = pd.to_numeric(df['term'], errors='coerce')
        
        # Fill NaN with a default (e.g., 36) or drop? 
        # Given this is critical, let's fill with 0 so it fails validation if something is wrong, 
        # or drop. But for now, let's just ensure it's int.
        # Actually, let's filter rows where term is NaN after coercion if we want to be strict.
        # But for 'cleaning', let's just cast to Int64 (nullable int) or fill 0.
        df['term'] = df['term'].fillna(0).astype(int)
        print("Converted 'term' to integer.")

    # Emp Length: Extract numeric years. 
    # Logic: < 1 year -> 0, 10+ years -> 10, NaN -> 0 (and maybe add a flag?)
    if 'emp_length' in df.columns:
        # Create a missing flag
        df['emp_length_missing'] = df['emp_length'].isna().astype(int)
        
        # Replace string values
        replace_dict = {
            '< 1 year': 0,
            '1 year': 1,
            '2 years': 2,
            '3 years': 3,
            '4 years': 4,
            '5 years': 5,
            '6 years': 6,
            '7 years': 7,
            '8 years': 8,
            '9 years': 9,
            '10+ years': 10
        }
        # First map known strings, then fillna with 0. 
        # Note: We need to handle the case where it might be mixed types if not loaded strictly as string.
        # The `map` function is good here.
        df['emp_length_num'] = df['emp_length'].map(replace_dict)
        
        # Fill NaN in the new numeric column with 0
        df['emp_length_num'] = df['emp_length_num'].fillna(0).astype(int)
        
        # Drop the original text column if preferred, or keep it. 
        # For ML, we usually want the numeric one. Let's replace 'emp_length' with the numeric version?
        # Or keep both? Let's replace to keep it clean, as per "cleaning" goal.
        df['emp_length'] = df['emp_length_num']
        df.drop(columns=['emp_length_num'], inplace=True)
        print("Converted 'emp_length' to numeric (0-10), filled NaNs with 0.")


    # --- 4. Correct Data Types ---

    # Convert date columns to datetime objects
    date_cols = ['issue_d', 'last_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line']
    for col in date_cols:
        if col in df.columns:
            # The `errors='coerce'` will turn unparseable dates into NaT (Not a Time)
            df[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce')
            
    print(f"Final shape: {df.shape}")
    return df


if __name__ == '__main__':
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(current_dir, '..', 'data', 'processed', 'optimized_accepted_data.csv')
    cleaned_sample_path = os.path.join(current_dir, '..', 'data', 'processed', 'cleaned_sample_data.csv')

    try:
        # Load the processed data
        df = pd.read_csv(processed_data_path)

        # --- Create a Sample for Development ---
        # Taking a 10% random sample to speed up development.
        # `random_state` is set for reproducibility.
        sample_df = df.sample(frac=0.1, random_state=42)

        print("Created a sample of size:", sample_df.shape)

        # --- Initial Data Cleaning on the Sample ---
        cleaned_sample_df = clean_data(sample_df.copy()) # Use a copy to avoid SettingWithCopyWarning


        print("\nData after cleaning:")
        # Show info to see the effect of cleaning
        cleaned_sample_df.info()


        # --- Save the Cleaned Sample Data ---
        # This allows for faster iteration during EDA and modeling
        cleaned_sample_df.to_csv(cleaned_sample_path, index=False)
        print(f"\nCleaned sample data saved to: {cleaned_sample_path}")


    except FileNotFoundError:
        print(f"Error: The file was not found at {processed_data_path}")
        print("Please run `src/data_loading.py` first to generate the processed data.")
    except Exception as e:
        print(f"An error occurred: {e}")

