"""
One-shot precompute of the three Phase 3 macro insights so the Streamlit app
can render them as Plotly charts without depending on FRED at runtime.

Reproduces what notebook 05 (notebooks/databricks/05_macro_integration.ipynb)
did on Databricks, but on the local cleaned_data.csv. Outputs three small
parquet files into presentation/data/.

Run with:
    python presentation/precompute_macro_insights.py
"""

from pathlib import Path

import pandas as pd
import pandas_datareader.data as web


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLEANED = ROOT / "data" / "processed" / "cleaned_data.csv"

NATIONAL = {"UNRATE": "unemployment_rate",
            "FEDFUNDS": "fed_funds_rate"}
STATE = {"CAUR": "CA", "TXUR": "TX", "NYUR": "NY", "FLUR": "FL", "ILUR": "IL"}


def fetch_series(series_id, start, end):
    df = web.DataReader(series_id, "fred", start, end).reset_index()
    df.columns = ["date", series_id]
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df.drop(columns=["date"])


def main():
    print("Loading cleaned loans...")
    loans = pd.read_csv(
        CLEANED,
        usecols=["issue_d", "charged_off", "sub_grade", "addr_state"],
        parse_dates=["issue_d"],
        low_memory=False,
    )
    loans["year"] = loans["issue_d"].dt.year
    loans["month"] = loans["issue_d"].dt.month
    print(f"  {len(loans):,} rows  |  range {loans.issue_d.min().date()} to {loans.issue_d.max().date()}")

    print("Fetching FRED national series (UNRATE, FEDFUNDS)...")
    unrate = fetch_series("UNRATE",  "2013-01-01", "2018-01-01")
    fed    = fetch_series("FEDFUNDS","2013-01-01", "2018-01-01")
    macro  = unrate.merge(fed, on=["year", "month"], how="outer")
    print(f"  national series rows: {len(macro)}")

    loans_m = loans.merge(macro, on=["year", "month"], how="left")

    # Insight 1: default rate vs UNRATE binned to nearest 0.1
    print("Computing Insight 1: default rate by UNRATE bin...")
    i1 = (
        loans_m.assign(unrate_bin=lambda d: d["UNRATE"].round(1))
               .dropna(subset=["unrate_bin"])
               .groupby("unrate_bin")
               .agg(n_loans=("charged_off", "size"),
                    default_rate=("charged_off", "mean"))
               .reset_index()
               .sort_values("unrate_bin")
    )
    i1.to_parquet(OUT_DIR / "insight1_unrate.parquet", index=False)
    print(f"  -> insight1_unrate.parquet  ({len(i1)} bins)")

    # Insight 2: sub_grade x FEDFUNDS regime
    print("Computing Insight 2: sub_grade default rate by Fed Funds regime...")
    i2 = (
        loans_m.dropna(subset=["sub_grade", "FEDFUNDS"])
               .assign(rate_regime=lambda d:
                       d["FEDFUNDS"].apply(lambda x: "low rate (<=0.5%)"
                                           if x <= 0.5 else "high rate (>0.5%)"))
               .groupby(["sub_grade", "rate_regime"])
               .agg(default_rate=("charged_off", "mean"),
                    n=("charged_off", "size"))
               .reset_index()
    )
    i2.to_parquet(OUT_DIR / "insight2_subgrade_regime.parquet", index=False)
    print(f"  -> insight2_subgrade_regime.parquet  ({len(i2)} rows)")

    # Insight 3: state UR differential vs default rate (top-5 states)
    print("Computing Insight 3: state UR differential by state (CA/TX/NY/FL/IL)...")
    state_dfs = []
    for fred_id, st in STATE.items():
        s = fetch_series(fred_id, "2013-01-01", "2018-01-01")
        s = s.rename(columns={fred_id: "state_unrate"})
        s["addr_state"] = st
        state_dfs.append(s)
    state_macro = pd.concat(state_dfs, ignore_index=True)

    loans_st = (
        loans[loans["addr_state"].isin(STATE.values())]
        .merge(state_macro, on=["addr_state", "year", "month"], how="left")
        .merge(unrate, on=["year", "month"], how="left")
    )
    loans_st["excess_unemp"] = (loans_st["state_unrate"] - loans_st["UNRATE"]).round(1)
    i3 = (
        loans_st.dropna(subset=["excess_unemp"])
                .groupby(["addr_state", "excess_unemp"])
                .agg(n=("charged_off", "size"),
                     default_rate=("charged_off", "mean"))
                .reset_index()
                .query("n >= 200")  # drop tiny bins for cleaner viz
    )
    i3.to_parquet(OUT_DIR / "insight3_state_diff.parquet", index=False)
    print(f"  -> insight3_state_diff.parquet  ({len(i3)} rows)")

    # Bonus precomputes for tab 8 visuals
    print("Bonus: default rate by sub_grade...")
    sg = (
        loans.groupby("sub_grade")
             .agg(n=("charged_off", "size"),
                  default_rate=("charged_off", "mean"))
             .reset_index()
             .sort_values("sub_grade")
    )
    sg.to_parquet(OUT_DIR / "default_by_subgrade.parquet", index=False)
    print(f"  -> default_by_subgrade.parquet  ({len(sg)} rows)")

    print("Bonus: default rate over time (issue month)...")
    tm = (
        loans.assign(year_month=loans["issue_d"].dt.to_period("M").astype(str))
             .groupby("year_month")
             .agg(n=("charged_off", "size"),
                  default_rate=("charged_off", "mean"))
             .reset_index()
             .sort_values("year_month")
    )
    tm.to_parquet(OUT_DIR / "default_over_time.parquet", index=False)
    print(f"  -> default_over_time.parquet  ({len(tm)} rows)")

    print("Bonus: rows per year (Phase 1 split visual)...")
    yr = (
        loans.groupby("year")
             .agg(n=("charged_off", "size"),
                  default_rate=("charged_off", "mean"))
             .reset_index()
    )
    yr.to_parquet(OUT_DIR / "rows_per_year.parquet", index=False)
    print(f"  -> rows_per_year.parquet  ({len(yr)} rows)")

    print()
    print("All artifacts in", OUT_DIR)


if __name__ == "__main__":
    main()
