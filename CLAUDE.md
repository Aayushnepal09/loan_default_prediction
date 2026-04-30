# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup (Python 3.10+):
```bash
pip install -r requirements.txt
```

Before any local Phase 1-2 run, drop the Kaggle archive (`archive.zip` from https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1) into `data/raw/`. Phase 3 Databricks notebooks pull the data themselves.

Run the full local pipeline (data prep → model selection → final eval). It auto-skips the expensive `13_model_selection.py` step if `models/best_model.pkl` already exists — delete that file to force re-tuning:
```bash
python src/run_pipeline.py
```

Run individual stages (must be in order — each script consumes the previous one's output in `data/processed/`):
```bash
python src/01_data_loading.py            # extract gzip from archive.zip, filter 2014-2017
python src/02_data_inspection.py         # console-only EDA report (optional)
python src/03_data_cleaning.py           # drops >30%-missing, leakage, and uninformative cols
python src/04_data_splitting.py          # chronological train/val/test split
python src/05_data_eda.py                # writes HTML report to reports/eda/ (gitignored)
python src/06_data_processing_pipeline.py  # fits sklearn pipeline, writes *_features.csv + preprocessing_pipeline.pkl
python src/models/07_model_knn.py … 12_model_hgb.py   # individual baselines (optional)
python src/models/13_model_selection.py  # Optuna tuning + MLflow logging, writes models/best_model.pkl
python src/models/14_final_evaluation.py # holdout eval on 2017 test set
```

Override pipeline I/O paths via env vars (read by `06_data_processing_pipeline.py`): `DATA_DIR`, `ARTIFACT_DIR`.

View MLflow runs from `13_model_selection.py`:
```bash
mlflow ui --backend-store-uri models/mlruns
```

MCP server (after a full pipeline run produces `data/processed/preprocessing_pipeline.pkl` and `models/best_model.pkl`):
```bash
python src/mcp/server.py    # FastMCP stdio server, normally launched by Claude Desktop config
```
See `src/mcp/README.md` for the `claude_desktop_config.json` entry; the template is at `mcp_config_template.json`.

## Architecture

**Phase 1-2 (local, scripts in `src/` numbered 01–14)** is a strictly sequential file-on-disk pipeline. Each script reads the previous step's CSV from `data/processed/`, transforms it, and writes the next CSV. There is no orchestration framework — `run_pipeline.py` just `subprocess.run`s each script in order. If you change one stage's output schema, every downstream stage breaks.

**Target and split policy.** The target `charged_off` is built in `03_data_cleaning.py` by filtering `loan_status` to `{Fully Paid, Charged Off}` only. The split in `04_data_splitting.py` is time-based: train = first 80% of 2014-2016 by `issue_d`, val = last 20% of 2014-2016, test = all of 2017. Do not introduce random splits or stratified shuffles — they would leak future information given the macroeconomic-driven default rate. Class imbalance is ~4:1 (paid:default) and is intentional, not corrected by resampling.

**Leakage discipline (`03_data_cleaning.py`).** A specific list of post-loan columns is dropped (`total_pymnt*`, `recoveries`, `last_pymnt_*`, settlement/hardship fields, `last_fico_range_*`, etc.) because they only get values after the outcome is known. When adding features, check `drop_leakage_cols` and `drop_uninformative_cols` — if a column was dropped, there is a reason recorded inline.

**Preprocessing pipeline (`06_data_processing_pipeline.py`).** A single sklearn `Pipeline` of eight custom `BaseEstimator/TransformerMixin` stages, fit on train only, then applied to val and test. Order matters: `DropCorrelated → DateExtractor → FeatureConstructor → OutlierCapper → MissingIndicatorAdder → MacroJoiner → RareCategoryMerger → ColumnPreprocessor`. The final `ColumnPreprocessor` partitions numeric columns into four buckets (skewed→log1p+scale, binarize→>0 flag, normal→scale, ordinal `sub_grade`→OrdinalEncoder, categorical→OneHotEncoder). Whether a column lands in `SKEWED_COLS`, `BINARIZE_COLS`, `ORDINAL_COLS`, `OHE_COLS`, or the implicit "normal" bucket is decided by module-level constants — adding a new feature usually means deciding which bucket it belongs to and (if categorical) whether `RareCategoryMerger` should fold its tail.

**MacroJoiner / FRED.** The pipeline calls out to FRED (`UNRATE` series via `pandas-datareader`) at fit time and merges monthly unemployment by `(issue_year, issue_month)`. If FRED is unreachable, it falls back to a constant 5.0% so SimpleImputer doesn't drop the column — be aware of this when running offline.

**Pickled pipeline gotcha.** `preprocessing_pipeline.pkl` is pickled while `06_data_processing_pipeline.py` runs as `__main__`, so the custom transformer classes (`DropCorrelated`, `DateExtractor`, `FeatureConstructor`, `OutlierCapper`, `MissingIndicatorAdder`, `MacroJoiner`, `RareCategoryMerger`, `ColumnPreprocessor`, `_binarize`) are pickled with module path `__main__`. To unpickle anywhere else (e.g. `src/mcp/server.py`), you must inject those classes into `__main__` before `pickle.load` — see how `server.py` does it via `importlib.util`. If you rename or move any of those classes, retrain the pipeline.

**Model selection (`src/models/13_model_selection.py`).** Compares LogisticRegression, DecisionTree, HistGradientBoosting, and XGBoost using Optuna (50 trials each) on the val set, logs everything to MLflow at `models/mlruns/`, and writes the best model (XGBoost in current artifacts) to `models/best_model.pkl`. The selection metric and threshold logic live in `evaluate()` near the top of the file. `14_final_evaluation.py` then picks Youden-J on val to set the operating threshold and reports test-set metrics.

**Phase 3 (Databricks)** is an independent rebuild on Spark + Delta Lake using the Medallion (bronze/silver/gold) pattern in `notebooks/databricks/`. Notebooks 01-03 are pure Spark and write to `workspace.default.{bronze,silver,gold}_loans` tables. Notebooks 04-05 deliberately drop down to pandas + sklearn for the model-fit step because Databricks Free Edition's serverless Py4J whitelist blocks `Imputer`, `StringIndexer`, and `OneHotEncoder`. The bronze notebook expects a Unity Catalog volume at `/Volumes/workspace/default/raw_data/` to exist before it runs.

## Conventions

- Generated artifacts (`data/raw/`, `data/processed/`, `reports/`, `models/mlruns/`) are gitignored. Don't commit pipeline outputs; only the curated artifacts in `models/` (`best_model.pkl`, `model_results.csv`) and `data/raw/LCDataDictionary.csv` are tracked.
- Numeric prefixes in `src/` (`01_…` … `14_…`) encode pipeline order. Keep new pipeline stages in this scheme; helper modules go in subpackages like `src/mcp/`.
- This is a school project — strip the default `Co-Authored-By: Claude` trailer from any commits you make.
