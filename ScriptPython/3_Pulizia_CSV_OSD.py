import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random

INPUT_PATH = Path('online_sales_dataset.csv')
OUTPUT_PATH = Path('OSD_cleaned.csv')
VALID_CHANNELS = ['In-store', 'Online']
DEFAULT_VALUES = {
    'Description': 'Unknown Product',
    'Country': 'Unknown',
    'ReturnStatus': 'Not Returned',
    'SalesChannel': 'Online',
    'OrderPriority': 'Medium',
    'PaymentMethod': 'Unknown',
    'Category': 'Unknown',
    'ShipmentProvider': 'Unknown',
    'WarehouseLocation': 'Unknown'
}
TODAY = pd.Timestamp('2025-07-03')

def log_step(step, before_count, after_count, description):
    removed = before_count - after_count
    print(f"Step {step:02d}: {description}\n"
          f"    → Righe prima: {before_count}, dopo: {after_count}, rimosse: {removed}\n")

def analyze_duplicates(df, stage):
    dups = df[df.duplicated('InvoiceNo', keep=False)]
    count = len(dups)
    print(f"--- DUPLICATI {stage} ({count} righe) ---")
    if count:
        vc = dups['InvoiceNo'].value_counts()
        dup_ids = vc[vc > 1]
        print(f"InvoiceNo duplicati ({len(dup_ids)} ID); primi 10:")
        for inv, c in dup_ids.head(10).items():
            print(f"  - {inv}: {c} occorrenze")
    print()

def enforce_shipment_provider_rules(df):
    print("=== APPLICAZIONE REGOLE SHIPMENT PROVIDER ===")
    
    print("Regola 1: Coerenza ShipmentProvider per InvoiceDate...")
    
    date_provider_issues = []
    for invoice_date in df['InvoiceDate'].unique():
        date_records = df[df['InvoiceDate'] == invoice_date]
        unique_providers = date_records['ShipmentProvider'].dropna().unique()
        if len(unique_providers) > 1:
            date_provider_issues.append({
                'InvoiceDate': invoice_date,
                'Providers': list(unique_providers),
                'Count': len(date_records)
            })
    
    if date_provider_issues:
        print(f"  → Trovate {len(date_provider_issues)} date con ShipmentProvider inconsistenti")
        print("  → Esempi di inconsistenze:")
        for i, issue in enumerate(date_provider_issues[:5]):
            print(f"    {i+1}. {issue['InvoiceDate']}: {issue['Providers']} ({issue['Count']} record)")
        
        for issue in date_provider_issues:
            date_mask = df['InvoiceDate'] == issue['InvoiceDate']
            date_records = df[date_mask]
            
            provider_counts = date_records['ShipmentProvider'].value_counts()
            most_common_provider = provider_counts.index[0]
            
            df.loc[date_mask, 'ShipmentProvider'] = most_common_provider
            
        print(f"  → Risolte {len(date_provider_issues)} inconsistenze InvoiceDate-ShipmentProvider")
    else:
        print("  → Nessuna inconsistenza trovata per InvoiceDate-ShipmentProvider")
    
    print("\nRegola 2: SalesChannel 'In-store' → ShipmentProvider 'Store'...")
    
    instore_mask = df['SalesChannel'] == 'In-store'
    instore_records = df[instore_mask]
    
    if len(instore_records) > 0:
        incorrect_provider_mask = instore_mask & (df['ShipmentProvider'] != 'Store')
        incorrect_count = incorrect_provider_mask.sum()
        
        if incorrect_count > 0:
            print(f"  → Trovati {incorrect_count} record In-store con ShipmentProvider scorretto")
            
            incorrect_providers = df[incorrect_provider_mask]['ShipmentProvider'].value_counts()
            print("  → Provider scorretti trovati:")
            for provider, count in incorrect_providers.items():
                print(f"    - {provider}: {count} record")
            
            df.loc[instore_mask, 'ShipmentProvider'] = 'Store'
            print(f"  → Corretti {incorrect_count} record: ShipmentProvider → 'Store'")
        else:
            print("  → Tutti i record In-store hanno già ShipmentProvider = 'Store'")
        
        print(f"  → Totale record In-store: {len(instore_records)}")
    else:
        print("  → Nessun record In-store trovato")
    
    print("\n--- VERIFICA FINALE REGOLE ---")
    
    final_date_issues = []
    for invoice_date in df['InvoiceDate'].unique():
        date_records = df[df['InvoiceDate'] == invoice_date]
        unique_providers = date_records['ShipmentProvider'].dropna().unique()
        if len(unique_providers) > 1:
            final_date_issues.append(invoice_date)
    
    if final_date_issues:
        print(f"ERRORE: {len(final_date_issues)} date hanno ancora ShipmentProvider inconsistenti!")
    else:
        print("Regola 1 verificata: Tutti i record con stesso InvoiceDate hanno stesso ShipmentProvider")
    
    instore_with_wrong_provider = df[
        (df['SalesChannel'] == 'In-store') & 
        (df['ShipmentProvider'] != 'Store')
    ]
    
    if len(instore_with_wrong_provider) > 0:
        print(f"ERRORE: {len(instore_with_wrong_provider)} record In-store hanno ShipmentProvider ≠ 'Store'!")
    else:
        print("Regola 2 verificata: Tutti i record In-store hanno ShipmentProvider = 'Store'")
    
    print()
    return df

def consolidate_stockcode_data(df):
    print("=== CONSOLIDAMENTO DATI STOCKCODE ===")
    stockcode_fields = ['Description', 'Category', 'UnitPrice']
    grouped = df.groupby('StockCode')
    inconsistent_stockcodes = []
    for stock_code, group in grouped:
        for field in stockcode_fields:
            if field in group.columns:
                unique_values = group[field].dropna().unique()
                if len(unique_values) > 1:
                    inconsistent_stockcodes.append({
                        'StockCode': stock_code,
                        'Field': field,
                        'Values': unique_values,
                        'Count': len(unique_values)
                    })
    if inconsistent_stockcodes:
        print(f"Trovate {len(inconsistent_stockcodes)} inconsistenze nei dati StockCode")
        print("\n--- ESEMPI DI INCONSISTENZE ---")
        for i, issue in enumerate(inconsistent_stockcodes[:10]):
            print(f"{i+1:2d}. StockCode {issue['StockCode']} - {issue['Field']}:")
            print(f"     Valori trovati: {list(issue['Values'])}")
        print()
    else:
        print("Nessuna inconsistenza trovata nei dati StockCode")
        return df
    consolidated_data = {}
    for stock_code, group in grouped:
        consolidated_values = {}
        for field in stockcode_fields:
            if field in group.columns:
                if field == 'UnitPrice':
                    valid_prices = pd.to_numeric(group[field], errors='coerce').dropna()
                    if len(valid_prices) > 0:
                        consolidated_values[field] = valid_prices.median()
                    else:
                        consolidated_values[field] = group[field].iloc[0]
                else:
                    mode_values = group[field].dropna().mode()
                    if len(mode_values) > 0:
                        consolidated_values[field] = mode_values[0]
                    else:
                        valid_values = group[field].dropna()
                        if len(valid_values) > 0:
                            consolidated_values[field] = valid_values.iloc[0]
                        else:
                            consolidated_values[field] = group[field].iloc[0]
        consolidated_data[stock_code] = consolidated_values
    df_consolidated = df.copy()
    for stock_code, values in consolidated_data.items():
        mask = df_consolidated['StockCode'] == stock_code
        for field, value in values.items():
            df_consolidated.loc[mask, field] = value
    verification_issues = []
    grouped_after = df_consolidated.groupby('StockCode')
    for stock_code, group in grouped_after:
        for field in stockcode_fields:
            if field in group.columns:
                unique_values = group[field].dropna().unique()
                if len(unique_values) > 1:
                    verification_issues.append({
                        'StockCode': stock_code,
                        'Field': field,
                        'Values': unique_values
                    })
    if verification_issues:
        print(f"ATTENZIONE: Rimangono {len(verification_issues)} inconsistenze dopo la consolidazione")
        for issue in verification_issues[:5]:
            print(f"  - {issue['StockCode']}.{issue['Field']}: {list(issue['Values'])}")
    else:
        print("Consolidamento completato con successo - tutti i StockCode hanno dati consistenti")
    print(f"\nStockCode unici processati: {len(consolidated_data)}")
    print(f"Inconsistenze risolte: {len(inconsistent_stockcodes)}")
    if inconsistent_stockcodes:
        print("\n--- ESEMPI DI CONSOLIDAMENTO ---")
        for i, issue in enumerate(inconsistent_stockcodes[:5]):
            stock_code = issue['StockCode']
            field = issue['Field']
            if stock_code in consolidated_data:
                final_value = consolidated_data[stock_code].get(field, 'N/A')
                print(f"{i+1}. {stock_code}.{field}: {list(issue['Values'])} → {final_value}")
    print()
    return df_consolidated

def consolidate_orders(df):
    print("=== CONSOLIDAMENTO ORDINI ===")
    order_fields = ['InvoiceDate', 'CustomerID', 'Country', 'PaymentMethod', 
                   'SalesChannel', 'ShipmentProvider', 'WarehouseLocation', 'OrderPriority']
    grouped = df.groupby('InvoiceNo')
    consolidated_rows = []
    multi_item_invoices = []
    for invoice_no, group in grouped:
        if len(group) > 1:
            consolidated_values = {}
            for field in order_fields:
                if field in group.columns:
                    mode_values = group[field].mode()
                    if len(mode_values) > 0:
                        consolidated_values[field] = mode_values[0]
                    else:
                        consolidated_values[field] = group[field].iloc[0]
            for _, row in group.iterrows():
                consolidated_row = row.copy()
                for field, value in consolidated_values.items():
                    consolidated_row[field] = value
                consolidated_rows.append(consolidated_row)
            multi_item_invoices.append({
                'InvoiceNo': invoice_no,
                'Items': len(group),
                'StockCodes': list(group['StockCode'].unique()),
                'CustomerID': consolidated_values.get('CustomerID', 'N/A'),
                'Country': consolidated_values.get('Country', 'N/A')
            })
        else:
            consolidated_rows.append(group.iloc[0])
    df_consolidated = pd.DataFrame(consolidated_rows)
    print(f"Ordini consolidati: {len(df)} → {len(df_consolidated)} righe")
    print(f"Ordini multi-articolo trovati: {len(multi_item_invoices)}")
    if multi_item_invoices:
        print("\n--- ESEMPI DI ORDINI MULTI-ARTICOLO ---")
        multi_item_invoices.sort(key=lambda x: x['Items'], reverse=True)
        for i, order in enumerate(multi_item_invoices[:10]):
            print(f"{i+1:2d}. Invoice {order['InvoiceNo']}: {order['Items']} articoli")
            print(f"     Customer: {order['CustomerID']}, Country: {order['Country']}")
            print(f"     StockCodes: {', '.join(order['StockCodes'][:5])}")
            if len(order['StockCodes']) > 5:
                print(f"     ... e altri {len(order['StockCodes']) - 5} articoli")
            print()
    return df_consolidated

def create_realistic_multi_item_orders(df, target_multi_percentage=0.15):
    print(f"=== CREAZIONE ORDINI MULTI-ARTICOLO (target: {target_multi_percentage:.1%}) ===")
    current_invoices = df['InvoiceNo'].nunique()
    target_multi_orders = int(current_invoices * target_multi_percentage)
    compatibility_groups = df.groupby(['CustomerID', 'Country', 'PaymentMethod', 
                                     'SalesChannel', 'ShipmentProvider', 
                                     'WarehouseLocation', 'OrderPriority'])
    combinable_groups = []
    for group_key, group_df in compatibility_groups:
        if len(group_df) >= 2:
            combinable_groups.append(group_df)
    if not combinable_groups:
        print("Nessun gruppo compatibile trovato per la combinazione")
        return df
    random.seed(42)
    combined_count = 0
    df_result = df.copy()
    for group_df in combinable_groups:
        if combined_count >= target_multi_orders:
            break
        if len(group_df) >= 2:
            n_items = min(random.randint(2, 4), len(group_df))
            selected_rows = group_df.sample(n=n_items, random_state=42+combined_count)
            base_invoice = selected_rows.iloc[0]['InvoiceNo']
            indices_to_update = selected_rows.index
            df_result.loc[indices_to_update, 'InvoiceNo'] = base_invoice
            base_date = selected_rows.iloc[0]['InvoiceDate']
            df_result.loc[indices_to_update, 'InvoiceDate'] = base_date
            combined_count += 1
    print(f"Ordini multi-articolo creati: {combined_count}")
    return df_result

def clean_dataset():
    df = pd.read_csv(INPUT_PATH)
    print(f"Inizio pulizia dataset: {len(df)} righe, colonne: {list(df.columns)}\n")
    analyze_duplicates(df, 'INIZIALI')
    step = 0
    prev_count = len(df)

    # 0) Filtro ShipmentProvider e Category
    valid_shipment_providers = ['DHL', 'FedEx', 'Royal Mail', 'UPS', 'Store']  # Aggiunto 'Store'
    valid_categories         = ['Accessories', 'Apparel', 'Electronics', 'Furniture', 'Stationery']
    df = df[
        df['ShipmentProvider'].isin(valid_shipment_providers) &
        df['Category'].isin(valid_categories)
    ]
    log_step(step, prev_count, len(df), 
             f"Filter ShipmentProvider in {valid_shipment_providers} e Category in {valid_categories}")
    prev_count = len(df)
    step += 1

    # 1) Filtra SalesChannel
    df = df[df['SalesChannel'].isin(VALID_CHANNELS)]
    log_step(step, prev_count, len(df), f"Filter SalesChannel in {VALID_CHANNELS}")
    prev_count = len(df); step += 1

    # 2) ShippingCost >= 0 
    df['ShippingCost'] = pd.to_numeric(df['ShippingCost'], errors='coerce').fillna(0)
    df = df[df['ShippingCost'] >= 0]
    log_step(step, prev_count, len(df), "ShippingCost >= 0")
    prev_count = len(df); step += 1

    # 3) Normalize PaymentMethod 
    df['PaymentMethod'] = (
        df['PaymentMethod'].astype(str)
          .replace(r'(?i)paypall', 'PayPal', regex=True)
          .str.strip().str.title()
    )
    log_step(step, prev_count, len(df), "Normalize PaymentMethod")
    prev_count = len(df); step += 1

    # 4) Quantity per ReturnStatus
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    ret_mask = df['ReturnStatus'] == 'Returned'
    df.loc[ret_mask & (df['Quantity'] < 0), 'Quantity'] = df.loc[ret_mask & (df['Quantity'] < 0), 'Quantity'].abs()
    df = df[~(~ret_mask & (df['Quantity'] < 0))]
    log_step(step, prev_count, len(df), "Fix Quantity per ReturnStatus")
    prev_count = len(df); step += 1

    # 5) UnitPrice per returns
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce').fillna(0)
    ret_mask = df['ReturnStatus'] == 'Returned'
    mask_bad_price = (df['UnitPrice'] < 0) & (~ret_mask)
    df = df.loc[~mask_bad_price]
    log_step(step, prev_count, len(df), "Filter UnitPrice < 0 non-returns")
    prev_count = len(df); step += 1

    # 6) CustomerID numerico
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(int)
    log_step(step, prev_count, len(df), "Valid CustomerID")
    prev_count = len(df); step += 1
    df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce').fillna(0).clip(upper=1).round(2)
    log_step(step, prev_count, len(df), "Normalize Discount <= 1 e round(2)")
    prev_count = len(df); step += 1
    digits = df['StockCode'].astype(str).str.extract(r'(\d+)', expand=False)
    df = df[digits.notna()]
    df['StockCode'] = 'SKU_' + digits.str.zfill(4)
    log_step(step, prev_count, len(df), "Format StockCode SKU_xxxx")
    prev_count = len(df); step += 1

    # 9) OrderPriority valido
    df = df[df['OrderPriority'].isin(['Low', 'Medium', 'High'])]
    log_step(step, prev_count, len(df), "Filter OrderPriority in [Low, Medium, High]")
    prev_count = len(df); step += 1

    # 10) ShippingCost = 0 per In-store
    mask_store = df['SalesChannel'] == 'In-store'
    df.loc[mask_store, 'ShippingCost'] = 0.00
    log_step(step, prev_count, len(df), "Set ShippingCost=0.00 per In-store")
    prev_count = len(df); step += 1

    # 11) InvoiceNo formato esatto
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.strip()
    df = df[df['InvoiceNo'].str.fullmatch(r'\d{6}')]
    log_step(step, prev_count, len(df), "Validate InvoiceNo (6 digits)")
    prev_count = len(df); step += 1

    # 12) InvoiceDate tra 2000 e oggi
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df[df['InvoiceDate'].between('2000-01-01', TODAY)]
    log_step(step, prev_count, len(df), "Filter InvoiceDate tra 2000-01-01 e oggi")
    prev_count = len(df); step += 1

    # 13) Rimuovi CountryClean se identico a Country
    if 'CountryClean' in df.columns:
        if (df['CountryClean'] == df['Country']).all():
            df = df.drop(columns=['CountryClean'])
            print("CountryClean rimosso (identico a Country)\n")
        else:
            print("CountryClean mantenuto (discrepanze presenti)\n")

    # 14) Drop missing critical
    critical = ['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']
    df = df.dropna(subset=[c for c in critical if c in df.columns])
    log_step(step, prev_count, len(df), "Drop righe con campi critici mancanti")
    prev_count = len(df); step += 1
    for col, val in DEFAULT_VALUES.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    print("\n" + "="*60)
    
    df = enforce_shipment_provider_rules(df)
    print("="*60 + "\n")

    df = consolidate_stockcode_data(df)
    print("="*60 + "\n")

    df = create_realistic_multi_item_orders(df, target_multi_percentage=0.15)
    print("="*60 + "\n")
    
    df = consolidate_orders(df)
    print("="*60 + "\n")

    # 15) Analisi finale duplicati (ora dovrebbero essere logici)
    analyze_duplicates(df, 'DOPO CONSOLIDAMENTO')
    print("=== VERIFICA FINALE COERENZA STOCKCODE ===")
    stockcode_fields = ['Description', 'Category', 'UnitPrice']
    final_issues = []
    for stock_code in df['StockCode'].unique():
        stock_data = df[df['StockCode'] == stock_code]
        for field in stockcode_fields:
            if field in df.columns:
                unique_values = stock_data[field].dropna().unique()
                if len(unique_values) > 1:
                    final_issues.append(f"{stock_code}.{field}: {list(unique_values)}")
    if final_issues:
        print(f"ATTENZIONE: {len(final_issues)} inconsistenze rilevate:")
        for issue in final_issues[:10]:
            print(f"  - {issue}")
    else:
        print("Tutti i StockCode hanno dati consistenti")
    print()
    
    print("=== VERIFICA FINALE REGOLE SHIPMENT PROVIDER ===")
    
    date_provider_violations = 0
    for invoice_date in df['InvoiceDate'].unique():
        date_records = df[df['InvoiceDate'] == invoice_date]
        unique_providers = date_records['ShipmentProvider'].dropna().unique()
        if len(unique_providers) > 1:
            date_provider_violations += 1
    
    if date_provider_violations > 0:
        print(f"ERRORE: {date_provider_violations} date violano la regola InvoiceDate-ShipmentProvider")
    else:
        print("Regola 1 rispettata: Stesso InvoiceDate → Stesso ShipmentProvider")
    
    instore_violations = df[
        (df['SalesChannel'] == 'In-store') & 
        (df['ShipmentProvider'] != 'Store')
    ]
    
    if len(instore_violations) > 0:
        print(f"ERRORE: {len(instore_violations)} record In-store violano la regola ShipmentProvider")
    else:
        print("Regola 2 rispettata: In-store → ShipmentProvider = 'Store'")
    
    print()
    
    total_invoices = df['InvoiceNo'].nunique()
    total_rows = len(df)
    multi_item_invoices = df[df.duplicated('InvoiceNo', keep=False)]['InvoiceNo'].nunique()
    unique_stockcodes = df['StockCode'].nunique()
    instore_count = (df['SalesChannel'] == 'In-store').sum()
    
    print(f"=== RIEPILOGO FINALE ===")
    print(f"Righe totali: {total_rows}")
    print(f"Ordini unici (InvoiceNo): {total_invoices}")
    print(f"Ordini multi-articolo: {multi_item_invoices} ({multi_item_invoices/total_invoices:.1%})")
    print(f"Ordini mono-articolo: {total_invoices - multi_item_invoices}")
    print(f"StockCode unici: {unique_stockcodes}")
    print(f"Record In-store: {instore_count}")
    print(f"Colonne: {list(df.columns)}")
    
    print(f"\n=== DISTRIBUZIONE SHIPMENT PROVIDER ===")
    provider_stats = df['ShipmentProvider'].value_counts()
    for provider, count in provider_stats.items():
        percentage = (count / len(df)) * 100
        print(f"{provider}: {count} ({percentage:.1f}%)")

    # 16) Aggiunta campo EstimatedCost
    rs = np.random.RandomState(42)
    stock_unit = df[['StockCode', 'UnitPrice']].drop_duplicates()
    variations = rs.uniform(0.2, 0.7, size=len(stock_unit))
    stock_unit['EstimatedCost'] = (stock_unit['UnitPrice'] * variations).round(2)
    mapping = dict(zip(stock_unit['StockCode'], stock_unit['EstimatedCost']))
    df['EstimatedUnitCost'] = df['StockCode'].map(mapping)

    
    df.to_csv(OUTPUT_PATH, index=False, float_format="%.2f")
    print(f"\nDataset consolidato salvato in: {OUTPUT_PATH}")
    return df

if __name__ == '__main__':
    clean_dataset()