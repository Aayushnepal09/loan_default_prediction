# Eas587_project - Lending Club Analysis

This project analyzes Lending Club loan data from 2007 to 2017.

## Prerequisites

- Python 3.x
- pip

## Installation

1.  **Clone the repository** (if applicable).
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Setup & Processing

The project includes scripts to automatically handle large datasets.

### 1. Load and Optimize Data

Run the data loading script first. This script will:
- Automatically check for the raw CSV file.
- If missing, it will **automatically extract** `data/raw/accepted_2007_to_2017.csv` from `data/raw/datasetzip.zip`.
- Process the large CSV in chunks to optimize memory usage.
- Save an optimized version to `data/processed/optimized_accepted_data.csv`.

```bash
python src/data_loading.py
```

### 2. Clean and Sample Data

Run the cleaning script to prepare data for analysis. This script will:
- Load the optimized data.
- **Handle Missing Values**: 
    - Imputes `revol_util` with mean.
    - Fills `emp_length` missing values and converts to numeric.
    - Zero-fills other relevant columns.
    - Drops rows with critical missing data.
- **Standardization**: Converts `term` to integer (36/60).
- **Remove Duplicates**: Ensures data integrity.
- **Fix Data Types**: Converts date columns to proper datetime objects.
- **Create a Sample**: Generates a 10% random sample for faster development.
- Save the cleaned sample to `data/processed/cleaned_sample_data.csv`.

```bash
python src/data_cleaning.py
```

## Project Structure

- `data/raw/`: Contains the original dataset (CSV or ZIP).
- `data/processed/`: Contains the optimized and cleaned data files.
- `src/`: Source code directory.
    - `data_loading.py`: Handles data extraction and initial optimization.
    - `data_cleaning.py`: Performs advanced cleaning (imputation, standardization) and sampling.
    - `verify_cleaning.py`: Verifies the quality and integrity of the cleaned data.
    - `eda.py`: (Placeholder) For Exploratory Data Analysis.
- `requirements.txt`: List of Python dependencies.
