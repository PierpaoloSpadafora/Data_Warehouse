import pandas as pd
from difflib import get_close_matches
from typing import Optional
import shutil
import os

INPUT_GDP   = "./GDP_Country_Data.csv"
INPUT_OSD   = "./OSD_cleaned.csv"
OUTPUT_CSV  = "./READY_OSD_plus_GDP.csv"

gdp = pd.read_csv(INPUT_GDP)
osd = pd.read_csv(INPUT_OSD)

print("=== ANALISI GDP DATA ===")
print(f"GDP records: {len(gdp)}")
print(f"GDP countries: {gdp['Country Name'].nunique()}")
print(f"GDP years: {sorted(gdp['Year'].unique())}")

mapping_df = gdp[["Country Name", "Country Code"]].drop_duplicates()
name_list  = mapping_df["Country Name"].tolist()

def match_country(name: str) -> Optional[str]:
    exact = mapping_df.loc[mapping_df["Country Name"].str.lower() == name.lower(), "Country Code"]
    if not exact.empty:
        return exact.iloc[0]
    close = get_close_matches(name, name_list, n=1, cutoff=0.8)
    if close:
        return mapping_df.loc[mapping_df["Country Name"] == close[0], "Country Code"].iloc[0]
    return None 

country_map = {c: match_country(c) for c in osd["Country"].unique()}
osd["Country"] = osd["Country"].map(country_map)

print("\n=== COUNTRY MAPPING ===")
valid_countries = [c for c in country_map.values() if c is not None]
print(f"OSD countries mapped: {len(valid_countries)}")
print(f"Valid countries: {valid_countries}")

invoice_dates = osd["InvoiceDate"]  
parsed_dates = pd.to_datetime(invoice_dates, format="%Y-%m-%d %H:%M:%S", errors="coerce")
year_series = parsed_dates.dt.year.copy()

needs_fallback = year_series.isna()
if needs_fallback.any():
    fallback_years = (
        invoice_dates[needs_fallback]
        .str.extract(r"(\d{4})")[0]              
        .astype("float")                         
    )
    year_series = year_series.mask(needs_fallback, fallback_years)

osd["Year"] = year_series.astype("Int64") 

print("\n=== YEAR EXTRACTION ===")
print(f"Years in OSD: {sorted(osd['Year'].dropna().unique())}")

print("\n=== CALCOLO GDP AGGREGATO PER PAESE ===")

gdp_latest = gdp.loc[gdp.groupby('Country Code')['Year'].idxmax()]
gdp_country_only = gdp_latest[['Country Code', 'GDP', 'GDP_per_capita']].copy()

print("GDP aggregato per paese (dati più recenti):")
print(gdp_country_only.head())

merged = osd.merge(
    gdp_country_only,
    how="left",
    left_on="Country",
    right_on="Country Code"
).drop(columns=["Country Code", "Year"])  

print(f"\n=== MERGE RESULTS ===")
print(f"Records after merge: {len(merged)}")
print(f"Countries with GDP data: {merged['GDP'].notna().sum()}")
print(f"Countries without GDP data: {merged['GDP'].isna().sum()}")

for col in ["GDP", "GDP_per_capita"]:
    missing_before = merged[col].isna().sum()
    merged[col] = merged[col].fillna(merged[col].median())
    missing_after = merged[col].isna().sum()
    print(f"{col}: {missing_before} → {missing_after} missing values")

print("\n=== VERIFICA COERENZA GDP ===")
gdp_check = merged.groupby('Country')[['GDP', 'GDP_per_capita']].nunique()
inconsistent_countries = gdp_check[(gdp_check['GDP'] > 1) | (gdp_check['GDP_per_capita'] > 1)]

if len(inconsistent_countries) > 0:
    print(f"ERRORE: {len(inconsistent_countries)} paesi hanno GDP inconsistenti!")
    print(inconsistent_countries)
    
    for country in inconsistent_countries.index[:5]:
        country_data = merged[merged['Country'] == country][['Country', 'GDP', 'GDP_per_capita']].drop_duplicates()
        print(f"\n{country}:")
        print(country_data)
else:
    print("✓ Tutti i paesi hanno GDP coerenti")

print(f"\n=== STATISTICHE FINALI ===")
unique_gdp_values = merged.groupby('Country')[['GDP', 'GDP_per_capita']].first()
print(f"Paesi unici nel dataset: {len(unique_gdp_values)}")
print(f"Paesi con GDP data: {unique_gdp_values['GDP'].notna().sum()}")

print("\nTop 10 paesi per GDP:")
top_gdp = unique_gdp_values.nlargest(10, 'GDP')[['GDP', 'GDP_per_capita']]
print(top_gdp)

merged.rename(columns={"Description": "ArticleName"}, inplace=True)

merged.to_csv(OUTPUT_CSV, index=False, sep=';', float_format="%.2f")
print(f"\nFile completo creato: {OUTPUT_CSV}")
print("Elaborazione completata!")

print("\n=== VERIFICA FINALE FILE SALVATO ===")
final_check = pd.read_csv(OUTPUT_CSV, sep=';')
final_gdp_check = final_check.groupby('Country')[['GDP', 'GDP_per_capita']].nunique()
final_inconsistent = final_gdp_check[(final_gdp_check['GDP'] > 1) | (final_gdp_check['GDP_per_capita'] > 1)]

if len(final_inconsistent) > 0:
    print(f"ERRORE FINALE: {len(final_inconsistent)} paesi hanno ancora GDP inconsistenti nel file salvato!")
else:
    print("✓ File salvato correttamente - tutti i paesi hanno GDP coerenti")