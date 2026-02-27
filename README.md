# Lending Club Loan Default Prediction

## Overview

This project analyzes Lending Club loan data (2014-2017) to predict whether a borrower will **charge off** (default) or **fully pay** their loan. The pipeline covers the full data science workflow: loading raw data, cleaning, splitting, exploratory data analysis (EDA), and preparation for modeling.

**Dataset:** [Lending Club 2007-2020Q3 (Kaggle)](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)

**Key decisions:**
- **Date range:** 2014-2017 (2007-2013 too sparse; 2018-2020 loans not matured)
- **Target variable:** `charged_off` (binary: 0 = Fully Paid, 1 = Charged Off)
- **Split strategy:** Time-based (train/val: 2014-2016, test: 2017) to prevent look-ahead bias
- **Class imbalance:** ~4:1 ratio (80% Fully Paid, 20% Charged Off)

---

## Project Structure

```
DIC_project/
├── data/
│   ├── raw/                          # Raw data files (archive.zip)
│   └── processed/                    # Pipeline outputs
│       ├── optimized_data_14_17.csv  # Step 1 output
│       ├── cleaned_data.csv          # Step 3 output
│       ├── train.csv                 # Step 4 output (2014-2016, 80%)
│       ├── val.csv                   # Step 4 output (2014-2016, 20%)
│       └── test.csv                  # Step 4 output (2017)
├── reports/
│   └── eda/
│       └── eda_report.html           # Step 5 output (self-contained HTML)
├── src/
│   ├── 01_data_loading.py            # Load and filter raw data
│   ├── 02_data_inspection.py         # Raw data EDA (before cleaning)
│   ├── 03_data_cleaning.py           # Clean and prepare data
│   ├── 04_data_splitting.py          # Time-based train/val/test split
│   └── 05_data_eda.py                # EDA on training set (HTML report)
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (recommended) or pip

### 1. Clone the Repository

```bash
git clone https://github.com/Aayushnepal09/Eas587_project.git
cd Eas587_project
```

### 2. Create a Virtual Environment

**Using Conda:**
```bash
conda create -n DIC_Project python=3.10
conda activate DIC_Project
```

**Using pip:**
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

1. Download `archive.zip` from [Kaggle](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)
2. Place `archive.zip` in the `data/raw/` directory

---

## How to Run

Run the scripts **in order** from the project root directory. Each step depends on the output of the previous one.

### Step 1: Load Data

```bash
python src/01_data_loading.py
```

- Extracts `Loan_status_2007-2020Q3.gzip` from `archive.zip`
- Filters to loans issued between 2014-2017
- Saves `data/processed/optimized_data_14_17.csv` (~1.5M rows)

### Step 2: Inspect Raw Data (Optional)

```bash
python src/02_data_inspection.py
```

- Prints a comprehensive report of the raw data **before** cleaning
- Covers: data structure, missing values, duplicates, distributions, class imbalance, temporal drift, and data type classification
- Use findings to guide cleaning decisions

### Step 3: Clean Data

```bash
python src/03_data_cleaning.py
```

- Filters to completed loans (Fully Paid / Charged Off) and creates binary target
- Drops columns with >30% missing values
- Removes post-loan leakage columns and uninformative features
- Converts string types (term, int_rate, revol_util, emp_length) to numeric
- Saves `data/processed/cleaned_data.csv`

### Step 4: Split Data

```bash
python src/04_data_splitting.py
```

- Time-based split to prevent look-ahead bias:
  - **Train:** 2014-2016, first 80% chronologically
  - **Validation:** 2014-2016, last 20% chronologically
  - **Test:** 2017 (full year)
- Saves `train.csv`, `val.csv`, `test.csv` to `data/processed/`

### Step 5: Exploratory Data Analysis

```bash
python src/05_data_eda.py
```

- Runs EDA on the **training set only** (no data leakage)
- Generates a self-contained HTML report at `reports/eda/eda_report.html`
- Sections: overview, missing values, target distribution, numeric statistics, distributions & outliers, categorical features, charge-off rates by category, correlation analysis
- Open the report in any browser to view

---

## Pipeline Summary

```
archive.zip
    │
    ▼
01_data_loading.py ──► optimized_data_14_17.csv (1.5M rows, 142 cols)
    │
    ▼
02_data_inspection.py  (optional, stdout report)
    │
    ▼
03_data_cleaning.py ──► cleaned_data.csv (1.4M rows, ~61 cols)
    │
    ▼
04_data_splitting.py ──► train.csv / val.csv / test.csv
    │
    ▼
05_data_eda.py ──► reports/eda/eda_report.html
    │
    ▼
feature_engineering.py (TODO)
```
