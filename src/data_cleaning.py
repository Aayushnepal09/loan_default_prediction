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

    # Remove empty rows (rows where all values are NaN)
    df.dropna(how='all', inplace=True)
    print(f"Removed {df.shape[0] - len(df)} empty rows.")

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

