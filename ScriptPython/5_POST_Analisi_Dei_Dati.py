import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil, os, stat, errno

CSV_FILE   = 'READY_OSD_plus_GDP.csv'
OUTPUT_DIR = Path('../Visualizations/Post')
FIGSIZE    = (12, 8)
CMAP       = 'RdYlGn'
CENTER     = 50

def _rm_readonly(func, path, exc_info):
    if exc_info[1].errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def reset_output(path: Path = OUTPUT_DIR):
    if path.exists():
        shutil.rmtree(path, onerror=_rm_readonly)
    path.mkdir(parents=True, exist_ok=True)

CONFIG = {
    'InvoiceNo':        {'type': 'string',  'consistency': r'^\d{6}$'},
    'StockCode':        {'type': 'string',  'consistency': r'^SKU_\d+$'},
    'ArticleName':      {'type': 'string'},
    'Quantity':         {'type': 'numeric', 'precision': 'positive_int', 'consistency': lambda x: x > 0},
    'InvoiceDate':      {'type': 'datetime','precision': 'hourly'},
    'UnitPrice':        {'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: x > 0},
    'CustomerID':       {'type': 'numeric', 'consistency': lambda x: x > 0},
    'Country':          {'type': 'string',  'consistency': r'^[A-Za-z\s]+$'},
    'Discount':         {'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: 0 <= x <= 100},
    'PaymentMethod':    {'type': 'string',  'consistency': r'^(Credit Card|Paypal|Bank Transfer)$'},
    'ShippingCost':     {'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: x >= 0},
    'Category':         {'type': 'string'},
    'SalesChannel':     {'type': 'string',  'consistency': r'^(Online|In-store)$'},
    'ReturnStatus':     {'type': 'string',  'consistency': r'^(Not Returned|Returned)$'},
    'ShipmentProvider': {'type': 'string'},
    'WarehouseLocation':{'type':'string'},
    'OrderPriority':    {'type': 'string',  'consistency': r'^(Low|Medium|High)$'},
    'GDP':              {'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: x >= 0},
    'GDP_per_capita':   {'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: x >= 0},
    'EstimatedUnitCost':{'type': 'numeric', 'precision': 'decimal_2', 'consistency': lambda x: x >= 0}
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
            mask_fmt   = converted.round(2).eq(converted)
            mask_valid = mask_type & mask_fmt
        elif precision == 'positive_int':
            mask_valid = series.dropna().apply(
                lambda x: isinstance(x, (int, float)) and x > 0 and x == int(x)
            ).reindex(series.index, fill_value=False)
        else:
            mask_valid = mask_type

    elif dtype == 'datetime':
        parsed = pd.to_datetime(series, errors='coerce')
        mask_type = parsed.notna()
        if precision == 'hourly':
            mask_fmt = (parsed.dt.minute.eq(0) & parsed.dt.second.eq(0))
            mask_valid = mask_type & mask_fmt
        else:
            mask_valid = mask_type

    else:  
        mask_type = series.notna() & series.astype(str).str.strip().ne('')
        if isinstance(consistency, str):
            mask_fmt = series.astype(str).str.match(consistency)
            mask_valid = mask_type & mask_fmt
        else:
            mask_valid = mask_type

    validity = mask_valid.sum() / n * 100

    if dtype == 'numeric' and precision is not None:
        if precision == 'decimal_2':
            prec = mask_fmt.sum()
        elif precision == 'positive_int':
            prec = series.dropna().apply(
                lambda x: isinstance(x, (int, float)) and x > 0 and x == int(x)
            ).sum()
        else:
            prec = series.notna().sum()
        precision_score = prec / n * 100

    elif dtype == 'datetime' and precision == 'hourly':
        dt = pd.to_datetime(series, errors='coerce')
        prec = dt.dropna().apply(lambda x: x.minute == 0 and x.second == 0).sum()
        precision_score = prec / n * 100

    else:
        precision_score = np.nan

    if isinstance(consistency, str):
        cons = series.dropna().astype(str).str.match(consistency).sum()
        consistency_score = cons / n * 100
    elif callable(consistency):
        cons = series.dropna().apply(consistency).sum()
        consistency_score = cons / n * 100
    else:
        consistency_score = np.nan

    uniqueness_score = series.nunique(dropna=False) / n * 100

    def _r(x): return round(x, 2) if not np.isnan(x) else np.nan

    return {
        'validity':     _r(validity),
        'completeness': _r(completeness),
        'precision':    _r(precision_score),
        'consistency':  _r(consistency_score),
        'uniqueness':   _r(uniqueness_score)
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
    if qdf.empty:
        print(f"Warning: DataFrame vuoto per {title}. Saltando la creazione del grafico.")
        return
    
    plt.figure(figsize=FIGSIZE)
    sns.heatmap(qdf.T, annot=True, cmap=CMAP, center=CENTER,
                fmt='.1f', cbar_kws={'label': '% score'})
    plt.title(title, weight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/POST_{fname}", dpi=300)
    plt.close()

def move_files():
    files_to_move = [CSV_FILE]
    
    for file in files_to_move:
        source = Path(file)
        destination = Path(f"../{file}")
        
        if source.exists():
            try:
                shutil.move(str(source), str(destination))
                print(f"Spostato: {source} -> {destination}")
            except Exception as e:
                print(f"Errore nello spostamento di {source}: {e}")
        else:
            print(f"Warning: File {source} non trovato")

def main():
    reset_output()
    
    try:
        df = pd.read_csv(CSV_FILE, sep=';')
        print(f"CSV caricato correttamente. Shape: {df.shape}")
        print(f"Colonne trovate: {list(df.columns)}")
        
        qdf = analyze_df(df, CONFIG)
        print("\nRisultati analisi qualità:")
        print(qdf)
        
        if not qdf.empty:
            plot_heatmap(qdf, 'READY_OSD_plus_GDP Data Quality', 'OSD_plus_GDP_heatmap.png')
            print("Heatmap creata con successo!")
        else:
            print("Nessuna colonna configurata trovata nel dataset")
            
    except FileNotFoundError:
        print(f"Errore: File {CSV_FILE} non trovato")
    except Exception as e:
        print(f"Errore durante il caricamento del CSV: {e}")
    
    move_files()

if __name__ == '__main__':
    main()