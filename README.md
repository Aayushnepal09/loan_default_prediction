# Lending Club Loan Default Prediction



## What This Is

A data science project that tries to predict if a borrower will default on their Lending Club loan. We use loan data from 2014-2017 and build a pipeline that goes from raw data all the way through cleaning, splitting, and EDA.

The dataset is from Kaggle: https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1

Some important choices we made:
- Only using 2014-2017 data (older data is too sparse, newer loans havent matured)
- Target variable is `charged_off` (1 = defaulted, 0 = fully paid)
- Time-based splitting so we dont accidentally use future info during training
- Theres a ~4:1 class imbalance (way more paid loans than defaults)

## Project Structure

```
Eas587_project/
├── data/
│   ├── raw/                   # put archive.zip here
│   └── processed/             # pipeline outputs go here
├── reports/
│   └── eda/
│       └── eda_report.html    # EDA report (generated)
├── src/
│   ├── 01_data_loading.py
│   ├── 02_data_inspection.py
│   ├── 03_data_cleaning.py
│   ├── 04_data_splitting.py
│   ├── 05_data_eda.py
│   ├── 06_data_processing_pipeline.py
│   ├── mcp/
│   │   ├── server.py
│   │   └── README.md
│   └── models/
│       ├── 07_model_knn.py
│       ├── 08_model_svm.py
│       ├── 09_model_lr.py
│       ├── 10_model_dt.py
│       ├── 11_model_xgb.py
│       ├── 12_model_hgb.py
│       ├── 13_model_selection.py
│       └── 14_final_evaluation.py
├── requirements.txt
└── README.md
```

### MCP Server Deployment
This repository includes a fully functioning Model Context Protocol (MCP) server that exposes the trained loan default model to AI assistants like Claude Desktop.
Setup instructions and documentation can be found in `src/mcp/README.md`.


## Setup

Need Python 3.10+ and these packages:

```bash
pip install -r requirements.txt
```

Then download `archive.zip` from the Kaggle link above and put it in `data/raw/`.

## Running the Pipeline

Run scripts in order:
### i have created a new file run_pipeline.py that runs all the scripts in order you can run it using the following command:
```bash
python run_pipeline.py
```
## if you want to run the scripts manually, you can run the following commands:  
```bash
python src/01_data_loading.py      # loads raw data, filters to 2014-2017
python src/02_data_inspection.py   # prints a report about the raw data (optional)
python src/03_data_cleaning.py     # cleans data, drops bad columns, fixes types
python src/04_data_splitting.py    # splits into train/val/test by date
python src/05_data_eda.py          # generates HTML EDA report
python src/06_data_processing_pipeline.py  # builds ML pipeline, scales/encodes features

# (Optional) Individual model training algorithms
python src/models/07_model_knn.py
python src/models/08_model_svm.py
python src/models/09_model_lr.py
python src/models/10_model_dt.py
python src/models/11_model_xgb.py
python src/models/12_model_hgb.py

# Model Selection & Final Evaluation
python src/models/13_model_selection.py  # hyperparameter tuning, compares models, saves best
python src/models/14_final_evaluation.py # evaluates the best model on the hidden test set
```

Each script reads the output of the previous one so they have to be run in order.

## What Each Script Does

**01_data_loading.py** - Extracts the gzip from archive.zip, filters to 2014-2017, saves as csv

**02_data_inspection.py** - Looks at missing values, duplicates, class balance, outliers etc. Prints everything to console. Good to run before cleaning to understand whats going on.

**03_data_cleaning.py** - Drops columns with too many missing values (>30%), removes columns that would leak the outcome (like payment totals), drops useless columns (IDs, free text), fixes data types

**04_data_splitting.py** - Train = 2014-2016 first 80%, Val = 2014-2016 last 20%, Test = all of 2017. Uses chronological order to avoid data leakage.

**05_data_eda.py** - Runs on training set only. Makes an HTML report with histograms, correlation heatmaps, outlier analysis, etc.

**06_data_processing_pipeline.py** - Builds the Scikit-Learn preprocessing pipeline (imputation, scaling, encoding) and saves transformed features for model training.

**models/07_model_knn.py** through **models/12_model_hgb.py** - Individual scripts exploring baseline models and various algorithms.

**models/13_model_selection.py** - Runs multiple models with Optuna hyperparameter tuning, validates them against the validation set, logs results to MLflow, and saves the single best model.

**models/14_final_evaluation.py** - Loads the best saved model and performs a final unbiased evaluation on the holdout 2017 test set, plotting the ROC curve and confusion matrix.
