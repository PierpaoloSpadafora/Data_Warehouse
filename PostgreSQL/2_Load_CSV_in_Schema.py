import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
import numpy as np

PATH_CSV = '../READY_OSD_plus_GDP.csv'

def connect_to_db():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="DataW",
            user="postgres",
            password="postgres"
        )
        return conn
    except Exception as e:
        print(f"Errore connessione database: {e}")
        return None

def clean_data(df):
    df = df.fillna({
        'ArticleName': 'Unknown',
        'Category': 'Unknown',
        'PaymentMethod': 'Unknown',
        'SalesChannel': 'Unknown',
        'ReturnStatus': 'No Return',
        'ShipmentProvider': 'Unknown',
        'WarehouseLocation': 'Unknown',
        'OrderPriority': 'Normal',
        'Discount': 0.0,
        'ShippingCost': 0.0,
        'EstimatedUnitCost': 0.0,
        'GDP': 0.0,
        'GDP_per_capita': 0.0
    })
    
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    df = df.dropna(subset=['InvoiceNo', 'StockCode', 'CustomerID', 'Country'])
    
    return df

def insert_countries(conn, df):
    cursor = conn.cursor()
    
    countries = df[['Country']].drop_duplicates()
    countries['CountryCode'] = countries['Country'].str[:3].str.upper()
    
    countries_data = [
        (row['CountryCode'], row['Country'])
        for _, row in countries.iterrows()
    ]
    
    insert_query = """
    INSERT INTO COUNTRY (CountryCode, CountryName)
    VALUES %s
    ON CONFLICT (CountryCode) DO NOTHING
    """
    
    execute_values(cursor, insert_query, countries_data)
    conn.commit()
    print(f"Inseriti {len(countries_data)} paesi")
    
    return dict(zip(countries['Country'], countries['CountryCode']))

def insert_economic_measurements(conn, df, country_mapping):
    cursor = conn.cursor()
    
    # Aggiungi l'anno dal campo InvoiceDate
    df['Year'] = df['InvoiceDate'].dt.year
    
    # SOLUZIONE: Rimuovi duplicati PRIMA di inserire
    economic_data = df[['Country', 'Year', 'GDP', 'GDP_per_capita']].drop_duplicates(subset=['Country', 'Year'])
    
    # Filtra solo i record con Year valido
    economic_data = economic_data[economic_data['Year'].notna()]
    
    print(f"Record economici unici da inserire: {len(economic_data)}")
    
    economic_measurements = [
        (
            country_mapping[row['Country']],
            int(row['Year']),
            float(row['GDP']),
            float(row['GDP_per_capita'])
        )
        for _, row in economic_data.iterrows()
    ]
    
    insert_query = """
    INSERT INTO ECONOMIC_MEASUREMENT (CountryCode, Year, GDP, GDP_per_capita)
    VALUES %s
    ON CONFLICT (CountryCode, Year) DO UPDATE SET
        GDP = EXCLUDED.GDP,
        GDP_per_capita = EXCLUDED.GDP_per_capita
    """
    
    execute_values(cursor, insert_query, economic_measurements)
    conn.commit()
    print(f"Inseriti {len(economic_measurements)} record economici")

def insert_customers(conn, df, country_mapping):
    cursor = conn.cursor()
    
    customers = df[['CustomerID', 'Country']].drop_duplicates()
    
    customers_data = [
        (
            int(row['CustomerID']),
            country_mapping[row['Country']]
        )
        for _, row in customers.iterrows()
    ]
    
    insert_query = """
    INSERT INTO CUSTOMER (CustomerID, CountryCode)
    VALUES %s
    ON CONFLICT (CustomerID) DO NOTHING
    """
    
    execute_values(cursor, insert_query, customers_data)
    conn.commit()
    print(f"Inseriti {len(customers_data)} clienti")

def insert_products(conn, df):
    cursor = conn.cursor()
    
    products = df[['StockCode', 'ArticleName', 'UnitPrice', 'Category', 'EstimatedUnitCost']].drop_duplicates('StockCode')
    
    products_data = [
        (
            row['StockCode'],
            row['ArticleName'],
            float(row['UnitPrice']),
            row['Category'],
            float(row['EstimatedUnitCost'])
        )
        for _, row in products.iterrows()
    ]
    
    insert_query = """
    INSERT INTO PRODUCT (StockCode, ArticleName, UnitPrice, Category, EstimatedUnitCost)
    VALUES %s
    ON CONFLICT (StockCode) DO UPDATE SET
        ArticleName = EXCLUDED.ArticleName,
        UnitPrice = EXCLUDED.UnitPrice,
        Category = EXCLUDED.Category,
        EstimatedUnitCost = EXCLUDED.EstimatedUnitCost
    """
    
    execute_values(cursor, insert_query, products_data)
    conn.commit()
    print(f"Inseriti {len(products_data)} prodotti")

def insert_orders(conn, df):
    cursor = conn.cursor()
    
    orders = df[['InvoiceNo', 'CustomerID', 'InvoiceDate', 'PaymentMethod', 
                'ShippingCost', 'SalesChannel', 'ShipmentProvider', 
                'WarehouseLocation', 'OrderPriority']].drop_duplicates('InvoiceNo')
    
    orders_data = [
        (
            int(row['InvoiceNo']),
            int(row['CustomerID']),
            row['InvoiceDate'].date() if pd.notna(row['InvoiceDate']) else None,
            row['PaymentMethod'],
            float(row['ShippingCost']),
            row['SalesChannel'],
            row['ShipmentProvider'],
            row['WarehouseLocation'],
            row['OrderPriority']
        )
        for _, row in orders.iterrows()
    ]
    
    insert_query = """
    INSERT INTO "ORDER" (InvoiceNo, CustomerID, InvoiceDate, PaymentMethod, 
                        ShippingCost, SalesChannel, ShipmentProvider, 
                        WarehouseLocation, OrderPriority)
    VALUES %s
    ON CONFLICT (InvoiceNo) DO NOTHING
    """
    
    execute_values(cursor, insert_query, orders_data)
    conn.commit()
    print(f"Inseriti {len(orders_data)} ordini")

def insert_sales(conn, df):
    cursor = conn.cursor()
    
    sales = df[['InvoiceNo', 'StockCode', 'Quantity', 'Discount', 'ReturnStatus']]
    
    sales_data = [
        (
            int(row['InvoiceNo']),
            row['StockCode'],
            int(row['Quantity']),
            float(row['Discount']),
            row['ReturnStatus']
        )
        for _, row in sales.iterrows()
    ]
    
    insert_query = """
    INSERT INTO SALE (InvoiceNo, StockCode, Quantity, Discount, ReturnStatus)
    VALUES %s
    ON CONFLICT (InvoiceNo, StockCode) DO UPDATE SET
        Quantity = EXCLUDED.Quantity,
        Discount = EXCLUDED.Discount,
        ReturnStatus = EXCLUDED.ReturnStatus
    """
    
    execute_values(cursor, insert_query, sales_data)
    conn.commit()
    print(f"Inseriti {len(sales_data)} record di vendita")

def main():
    print("Lettura del file CSV...")
    df = pd.read_csv(PATH_CSV, sep=';', encoding='utf-8')
    print(f"Letti {len(df)} record dal CSV")
    
    print("Pulizia e preparazione dei dati...")
    df = clean_data(df)
    print(f"Dopo la pulizia: {len(df)} record validi")
    
    print("Connessione al database...")
    conn = connect_to_db()
    if conn is None:
        return
    
    try:
        print("Inserimento paesi...")
        country_mapping = insert_countries(conn, df)
        
        print("Inserimento dati economici...")
        insert_economic_measurements(conn, df, country_mapping)
        
        print("Inserimento clienti...")
        insert_customers(conn, df, country_mapping)
        
        print("Inserimento prodotti...")
        insert_products(conn, df)
        
        print("Inserimento ordini...")
        insert_orders(conn, df)
        
        print("Inserimento vendite...")
        insert_sales(conn, df)
        
        print("Inserimento completato con successo!")
        
    except Exception as e:
        print(f"Errore durante l'inserimento: {e}")
        conn.rollback()
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()