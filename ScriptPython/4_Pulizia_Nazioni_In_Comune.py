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

invoice_dates = osd["InvoiceDate"]  
parsed_dates = pd.to_datetime( invoice_dates, format="%Y-%m-%d %H:%M:%S", errors="coerce")
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

merged = osd.merge(
    gdp[["Country Code", "Year", "GDP", "GDP_per_capita"]],
    how="left",
    left_on=["Country", "Year"],
    right_on=["Country Code", "Year"]
).drop(columns=["Country Code"])           

for col in ["GDP", "GDP_per_capita"]:
    merged[col] = (
        merged.groupby("Country")[col]
              .transform(lambda s: s.interpolate().bfill().ffill())
    )
    merged[col] = merged[col].fillna(merged[col].median())

merged.drop(columns=["Year"], inplace=True)
merged.rename(columns={"Description": "ArticleName"}, inplace=True)

merged.to_csv(OUTPUT_CSV, index=False, sep=';', float_format="%.2f")
print(f"File completo creato: {OUTPUT_CSV}")

print("Elaborazione completata!")