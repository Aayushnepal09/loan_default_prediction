"""
05_data_eda.py

EDA on the training set. Generates a self-contained HTML report
with embedded plots at reports/eda/eda_report.html.

Run after data_splitting.py.
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# paths and settings
current_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(current_dir, '..', 'data', 'processed', 'train.csv')
REPORT_DIR  = os.path.join(current_dir, '..', 'reports', 'eda')
TARGET_COL  = 'charged_off'

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.max_open_warning': 0})


def fig_to_base64(fig):
    # saves a matplotlib figure as a base64 string for embedding in html
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"<img src='data:image/png;base64,{b64}' style='max-width:100%;'>"


def heading(title, level=2):
    return f"<h{level}>{title}</h{level}>"


def text(content):
    return f"<p style='font-family:monospace; white-space:pre-wrap;'>{content}</p>"


def table_html(df, max_rows=50):
    return df.head(max_rows).to_html(
        classes='styled-table', index=True, border=0,
        float_format=lambda x: f'{x:,.4f}' if abs(x) < 1 else f'{x:,.2f}'
    )


# --- section 1: basic overview ---

def section_basic_overview(df):
    parts = [heading("Section 1: Basic Overview")]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    info_lines = [
        f"Shape           : {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Memory usage    : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB",
        f"Numeric columns : {len(num_cols)}",
        f"Categorical cols: {len(cat_cols)}",
        "",
        "Column dtypes:",
    ]
    for dtype, count in df.dtypes.value_counts().items():
        info_lines.append(f"  {str(dtype):<15} : {count}")

    parts.append(text("\n".join(info_lines)))
    print("  Section 1: Basic Overview [OK]")
    return parts


# --- section 2: missing values ---

def section_missing_values(df):
    parts = [heading("Section 2: Missing Values")]

    missing = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing Count': missing.values,
        'Missing %': missing_pct.values
    }).sort_values('Missing %', ascending=False)

    has_missing = missing_df[missing_df['Missing Count'] > 0]

    if has_missing.empty:
        parts.append(text("No missing values found!"))
    else:
        parts.append(text(f"Columns with missing values: {len(has_missing)} / {len(df.columns)}"))
        parts.append(table_html(has_missing.set_index('Column')))

        top = has_missing.head(20).sort_values('Missing %', ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.4)))
        bars = ax.barh(top['Column'], top['Missing %'], color=plt.cm.Reds(top['Missing %'] / top['Missing %'].max()))
        ax.set_xlabel('Missing %')
        ax.set_title('Top 20 Columns with Missing Values')
        for bar, val in zip(bars, top['Missing %']):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)
        fig.tight_layout()
        parts.append(fig_to_base64(fig))

    print("  Section 2: Missing Values [OK]")
    return parts


# --- section 3: target variable ---

def section_target_variable(df):
    parts = [heading("Section 3: Target Variable Distribution")]

    if TARGET_COL not in df.columns:
        parts.append(text(f"Target column '{TARGET_COL}' not found."))
        return parts

    counts = df[TARGET_COL].value_counts()
    labels_map = {0: 'Fully Paid', 1: 'Charged Off'}
    labels = [labels_map.get(v, str(v)) for v in counts.index]
    values = counts.values
    pcts = (values / values.sum() * 100).round(1)

    summary = pd.DataFrame({'Label': labels, 'Count': values, 'Percentage': pcts})
    parts.append(table_html(summary.set_index('Label')))

    imbalance = counts.get(0, 0) / counts.get(1, 1)
    parts.append(text(f"Imbalance ratio: {imbalance:.1f}:1 (Fully Paid : Charged Off)"))

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)
    ax.set_title('Target Variable Distribution', fontsize=13)
    ax.set_ylabel('Count')
    for bar, val, pct in zip(bars, values, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{int(val):,}\n({pct}%)', ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    parts.append(fig_to_base64(fig))

    print("  Section 3: Target Variable [OK]")
    return parts


# --- section 4: numeric stats ---

def section_numeric_stats(df):
    parts = [heading("Section 4: Numeric Features - Descriptive Statistics")]

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != TARGET_COL]

    if not num_cols:
        parts.append(text("No numeric columns found."))
        return parts, num_cols

    desc = df[num_cols].describe().T
    desc['skew'] = df[num_cols].skew()
    desc['kurtosis'] = df[num_cols].kurtosis()

    parts.append(text(f"{len(num_cols)} numeric features"))
    parts.append(table_html(desc[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew']]))

    print("  Section 4: Numeric Stats [OK]")
    return parts, num_cols


# --- section 5: distributions and outliers ---

def section_numeric_distributions(df, num_cols):
    parts = [heading("Section 5: Numeric Distributions & Outliers")]

    if num_cols is None or len(num_cols) == 0:
        return parts

    # outlier detection using IQR
    outlier_rows = []
    for col in num_cols:
        if df[col].isnull().all():
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        if n_outliers > 0:
            outlier_rows.append({
                'Column': col, 'Outliers': n_outliers, 'Outlier %': round(pct, 1),
                'Min': round(df[col].min(), 1), 'Max': round(df[col].max(), 1),
                'Q1': round(q1, 1), 'Q3': round(q3, 1)
            })

    if outlier_rows:
        parts.append(heading("Outlier Summary (IQR Method)", level=3))
        outlier_df = pd.DataFrame(outlier_rows).sort_values('Outlier %', ascending=False)
        parts.append(table_html(outlier_df.set_index('Column'), max_rows=40))

        top_outliers = outlier_df.head(20).sort_values('Outlier %', ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(5, len(top_outliers) * 0.4)))
        bars = ax.barh(top_outliers['Column'], top_outliers['Outlier %'],
                       color=plt.cm.YlOrRd(top_outliers['Outlier %'] / top_outliers['Outlier %'].max()))
        ax.set_xlabel('Outlier %')
        ax.set_title('Top 20 Columns by Outlier Percentage (IQR)')
        for bar, val in zip(bars, top_outliers['Outlier %']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=9)
        fig.tight_layout()
        parts.append(fig_to_base64(fig))

    # histograms and boxplots
    parts.append(heading("Distribution Plots", level=3))
    plot_cols = [c for c in num_cols if df[c].nunique() > 1]

    for i in range(0, min(len(plot_cols), 32), 4):
        batch = plot_cols[i:i+4]
        fig, axes = plt.subplots(len(batch), 2, figsize=(14, 3.5 * len(batch)))
        if len(batch) == 1:
            axes = axes.reshape(1, -1)
        for row, col in enumerate(batch):
            data = df[col].dropna()
            axes[row, 0].hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
            axes[row, 0].set_title(f'{col} - Histogram', fontsize=11)
            axes[row, 1].boxplot(data, vert=False, widths=0.7,
                                 boxprops=dict(color='steelblue'),
                                 medianprops=dict(color='red'))
            axes[row, 1].set_title(f'{col} - Box Plot', fontsize=11)
        fig.suptitle(f'Distributions (Page {i//4 + 1})', fontsize=13, y=1.01)
        fig.tight_layout()
        parts.append(fig_to_base64(fig))

    print("  Section 5: Distributions & Outliers [OK]")
    return parts


# --- section 6: categorical features ---

def section_categorical_stats(df):
    parts = [heading("Section 6: Categorical Features - Value Counts")]

    cat_cols = [c for c in df.select_dtypes(include=['object', 'string']).columns
                if c not in (TARGET_COL, 'issue_d')]

    if not cat_cols:
        parts.append(text("No categorical columns found."))
        return parts, cat_cols

    for col in cat_cols:
        counts = df[col].value_counts()
        n_unique = df[col].nunique()
        parts.append(heading(f"{col}  ({n_unique} unique values)", level=4))

        top_vals = counts.head(15)
        val_df = pd.DataFrame({
            'Value': top_vals.index.astype(str),
            'Count': top_vals.values,
            'Percentage': (top_vals.values / len(df) * 100).round(1)
        })
        parts.append(table_html(val_df.set_index('Value')))

        if n_unique >= 4 and n_unique <= 20:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(val_df['Value'], val_df['Count'], color=plt.cm.Blues(
                np.linspace(0.4, 0.9, len(val_df))))
            ax.set_title(f'{col} - Value Counts', fontsize=12)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            for bar, val in zip(bars, val_df['Count']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(val_df['Count'])*0.01,
                        f'{int(val):,}', ha='center', va='bottom', fontsize=8)
            fig.tight_layout()
            parts.append(fig_to_base64(fig))

    print("  Section 6: Categorical Stats [OK]")
    return parts, cat_cols


# --- section 7: charge-off rate by category ---

def section_target_vs_features(df, cat_cols):
    parts = [heading("Section 7: Charge-Off Rate by Category")]

    if TARGET_COL not in df.columns or not cat_cols:
        parts.append(text("Skipped - no target or no categorical columns."))
        return parts

    for col in cat_cols:
        if df[col].nunique() > 20:
            continue

        grouped = df.groupby(col)[TARGET_COL].agg(['mean', 'count']).reset_index()
        grouped.columns = [col, 'Charge-Off Rate', 'Count']
        grouped = grouped.sort_values('Charge-Off Rate', ascending=False)

        parts.append(heading(f"{col}", level=4))

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.RdYlGn_r(grouped['Charge-Off Rate'] / grouped['Charge-Off Rate'].max())
        bars = ax.bar(grouped[col].astype(str), grouped['Charge-Off Rate'], color=colors)
        ax.set_title(f'Charge-Off Rate by {col}', fontsize=12)
        ax.set_ylabel('Charge-Off Rate')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        plt.xticks(rotation=45, ha='right')
        for bar, val in zip(bars, grouped['Charge-Off Rate']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.1%}', ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        parts.append(fig_to_base64(fig))

    print("  Section 7: Target vs Features [OK]")
    return parts


# --- section 8: correlation ---

def section_correlation(df):
    parts = [heading("Section 8: Correlation Analysis")]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        parts.append(text("Not enough numeric columns for correlation."))
        return parts

    corr = df[num_cols].corr()

    # how each feature correlates with the target
    if TARGET_COL in corr.columns:
        target_corr = corr[TARGET_COL].drop(TARGET_COL)
        target_corr_abs = target_corr.abs().sort_values(ascending=False)

        parts.append(heading("Top 20 Features Correlated with Target", level=3))
        top20 = target_corr_abs.head(20)
        corr_df = pd.DataFrame({
            'Feature': top20.index,
            'Correlation': [target_corr[c] for c in top20.index],
            '|Correlation|': top20.values
        })
        parts.append(table_html(corr_df.set_index('Feature')))

        plot_data = corr_df.sort_values('Correlation', ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in plot_data['Correlation']]
        ax.barh(plot_data['Feature'], plot_data['Correlation'], color=colors)
        ax.set_xlabel('Correlation with Charge-Off')
        ax.set_title('Top 20 Features Correlated with Charge-Off', fontsize=13)
        ax.axvline(x=0, color='black', linewidth=0.5)
        fig.tight_layout()
        parts.append(fig_to_base64(fig))

    # find pairs with really high correlation
    parts.append(heading("Highly Correlated Feature Pairs (|r| > 0.8)", level=3))
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.8:
                pairs.append({
                    'Feature 1': corr.columns[i],
                    'Feature 2': corr.columns[j],
                    'Correlation': round(corr.iloc[i, j], 4)
                })

    if pairs:
        pairs_df = pd.DataFrame(pairs).sort_values('Correlation', key=abs, ascending=False)
        parts.append(table_html(pairs_df.set_index('Feature 1')))
    else:
        parts.append(text("No feature pairs with |r| > 0.8 found."))

    # heatmap of top features
    if TARGET_COL in corr.columns:
        top_features = target_corr_abs.head(15).index.tolist() + [TARGET_COL]
        sub_corr = df[top_features].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(sub_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, ax=ax, annot_kws={'size': 8})
        ax.set_title('Correlation Heatmap (Top 15 Features + Target)', fontsize=13)
        fig.tight_layout()
        parts.append(fig_to_base64(fig))

    print("  Section 8: Correlation Analysis [OK]")
    return parts


# --- build the html report ---

def build_html_report(sections):
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px 40px;
            background: #fafafa;
            color: #333;
        }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 40px; }
        h3 { color: #34495e; }
        h4 { color: #7f8c8d; font-size: 1em; }
        img { margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .styled-table {
            border-collapse: collapse; font-size: 0.85em;
            font-family: 'Segoe UI', sans-serif; min-width: 400px; margin: 15px 0;
        }
        .styled-table thead tr { background-color: #3498db; color: #fff; text-align: left; }
        .styled-table th, .styled-table td { padding: 8px 12px; border: 1px solid #ddd; }
        .styled-table tbody tr:nth-of-type(even) { background-color: #f3f3f3; }
        .styled-table tbody tr:hover { background-color: #e8f4fd; }
        p { line-height: 1.6; }
    </style>
    """

    html = [
        "<!DOCTYPE html>", "<html lang='en'>", "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>LendingClub EDA Report</title>",
        css, "</head>", "<body>",
        "<h1>LendingClub EDA Report - Training Set Only</h1>",
        f"<p><em>Generated from: {os.path.basename(TRAIN_PATH)}</em></p>",
    ]

    for section in sections:
        for part in section:
            html.append(part)

    html.extend(["</body>", "</html>"])
    return "\n".join(html)


if __name__ == '__main__':
    print("\n" + "#" * 60)
    print("#  EDA - Generating HTML Report")
    print("#" * 60)

    df = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f"\n  Loaded: {TRAIN_PATH}")
    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    sections = []
    sections.append(section_basic_overview(df))
    sections.append(section_missing_values(df))
    sections.append(section_target_variable(df))

    parts4, num_cols = section_numeric_stats(df)
    sections.append(parts4)

    sections.append(section_numeric_distributions(df, num_cols))

    parts6, cat_cols = section_categorical_stats(df)
    sections.append(parts6)

    sections.append(section_target_vs_features(df, cat_cols))
    sections.append(section_correlation(df))

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, 'eda_report.html')

    html = build_html_report(sections)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    abs_path = os.path.abspath(report_path)
    print(f"\n  Report saved: {abs_path}")
    print(f"  Open in browser: file:///{abs_path.replace(os.sep, '/')}")
