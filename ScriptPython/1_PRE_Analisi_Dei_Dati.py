import os
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DATA_DIR = './Dataset_Sporchi'
INPUT_FILES = {
    'Online Sales': './online_sales_dataset.csv',
    'GDP': f'{DATA_DIR}/GDP.csv',
    'GDP Per Capita': f'{DATA_DIR}/GDP_PER_CAPITA.csv'
}
OUTPUT_DIR = '../Visualizations/Pre'
RECENT_YEARS = ['2020', '2021', '2022', '2023', '2024']

GDP_CONFIG = {year: {'type': 'numeric', 'consistency': lambda x: x > 0} for year in RECENT_YEARS}
GDP_CONFIG.update({
    'Country Name': {'type': 'string', 'consistency': r'^[A-Za-z\s]+$'},
    'Country Code': {'type': 'string'}
})

SALES_CONFIG = {
    'InvoiceNo':        {'type': 'string',  'consistency': r'^\d{6}$'},
    'Quantity':         {'type': 'numeric', 'precision': 'positive_int', 'consistency': lambda x: x > 0},
    'InvoiceDate':      {'type': 'datetime', 'precision': 'hourly'},
    'UnitPrice':        {'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: x > 0},
    'Country':          {'type': 'string',  'consistency': r'^[A-Za-z\s]+$'},
    'StockCode':        {'type': 'string',  'consistency': r'^SKU_\d+$'},
    'Description':      {'type': 'string',  'consistency': r'^.{3,}$'},
    'PaymentMethod':    {'type': 'string',  'consistency': r'^(PayPal|Credit Card|Bank Transfer)$'},
    'Category':         {'type': 'string',  'consistency': r'^[A-Za-z\s&-]+$'},
    'SalesChannel':     {'type': 'string',  'consistency': r'^(Online|In-store)$'},
    'ReturnStatus':     {'type': 'string',  'consistency': r'^(Not Returned|Returned)$'},
    'ShipmentProvider': {'type': 'string',  'consistency': r'^[A-Za-z\s]+$'},
    'WarehouseLocation':{'type': 'string',  'consistency': r'^[A-Za-z\s]+$'},
    'OrderPriority':    {'type': 'string',  'consistency': r'^(Low|Medium|High)$'},
    'Discount':         {'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: x >= 0},
    'ShippingCost':     {'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: x >= 0},
    'CustomerID':       {'type': 'numeric', 'precision': 'positive_int', 'consistency': lambda x: x > 0}
}

def calculate_quality(series, dtype, precision=None, consistency=None):
    n = len(series)
    mask_notna = series.notna()
    if dtype == 'string':
        mask_notna &= series.astype(str).str.strip().ne('')
    completeness = mask_notna.sum() / n * 100

    if dtype == 'numeric':
        converted = pd.to_numeric(series, errors='coerce')
        mask_type = converted.notna()
        if precision == 'decimal_2':
            mask_fmt = series.dropna().astype(str).str.match(r'^\d+(\.\d{2})$')
            mask_valid = mask_type & mask_fmt.reindex(series.index, fill_value=False)
        elif precision == 'positive_int':
            mask_valid = series.dropna().apply(lambda x: isinstance(x, (int, float)) and x > 0 and x == int(x))
            mask_valid = mask_valid.reindex(series.index, fill_value=False)
        else:
            mask_valid = mask_type
        if precision == 'decimal_2':
            prec = series.dropna().astype(str).str.split('.').str[-1].str.len().eq(2).sum()
        elif precision == 'positive_int':
            prec = series.dropna().apply(lambda x: isinstance(x, (int, float)) and x > 0 and x == int(x)).sum()
        else:
            prec = mask_type.sum()
        precision_score = prec / n * 100

    elif dtype == 'datetime':
        parsed = pd.to_datetime(series, errors='coerce')
        mask_type = parsed.notna()
        if precision == 'hourly':
            mask_fmt = parsed.dt.minute.eq(0) & parsed.dt.second.eq(0)
            mask_valid = mask_type & mask_fmt
            prec = mask_fmt.sum()
            precision_score = prec / n * 100
        else:
            mask_valid = mask_type
            precision_score = np.nan

    else:
        mask_type = mask_notna
        mask_valid = mask_type
        precision_score = np.nan

    validity = mask_valid.sum() / n * 100

    if callable(consistency):
        cons_count = series.dropna().apply(consistency).sum()
        consistency_score = cons_count / n * 100
    elif isinstance(consistency, str):
        cons_count = series.dropna().astype(str).str.match(consistency).sum()
        consistency_score = cons_count / n * 100
    else:
        consistency_score = np.nan

    uniqueness_score = series.nunique(dropna=False) / n * 100

    def _r(x): return round(x, 2) if not np.isnan(x) else np.nan
    return {
        'validity':     _r(validity),
        'completeness': _r(completeness),
        'precision':    _r(precision_score),
        'consistency':  _r(consistency_score),
        'uniqueness':   _r(uniqueness_score),
    }

def analyze_df(df, config):
    results = {}
    for col, cfg in config.items():
        if col in df.columns:
            results[col] = calculate_quality(
                df[col],
                dtype=cfg.get('type'),
                precision=cfg.get('precision'),
                consistency=cfg.get('consistency')
            )
    return pd.DataFrame(results).T


def plot_heatmap(qdf, title, fname):
    plt.figure(figsize=(12, 6))
    sns.heatmap(qdf.T, annot=True, cmap='RdYlGn', center=50,
                fmt='.1f', cbar_kws={'label': '% score'})
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/PRE_{fname}", dpi=300)
    plt.close()


def plot_combined_heatmap(results_dict):
    n = len(results_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 6 * n))
    if n == 1: axes = [axes]
    for ax, (name, qdf) in zip(axes, results_dict.items()):
        sns.heatmap(qdf.T, annot=True, cmap='RdYlGn', center=50,
                    fmt='.1f', cbar_kws={'label': '% score'}, ax=ax)
        ax.set_title(f'{name} Data Quality')
        ax.tick_params(axis='x', rotation=45)
    plt.suptitle(' ', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/PRE_combined_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

results = {}
for name, path in INPUT_FILES.items():
    df = pd.read_csv(path, skiprows=4 if 'GDP' in name else 0)
    if 'GDP' in name:
        for yr in RECENT_YEARS:
            df[yr] = pd.to_numeric(df.get(yr), errors='coerce')
        cfg = GDP_CONFIG
    else:
        cfg = SALES_CONFIG
    qdf = analyze_df(df, cfg)
    results[name] = qdf

for name, qdf in results.items():
    fname = name.lower().replace(' ', '_') + '_heatmap.png'
    plot_heatmap(qdf, f"{name} Data Quality", fname)

plot_combined_heatmap(results)
