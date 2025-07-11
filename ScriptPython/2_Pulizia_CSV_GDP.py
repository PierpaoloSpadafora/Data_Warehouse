#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import os

if OUTPUT := Path("GDP_Country_Data.csv"):
    try:
        os.remove(OUTPUT)
    except FileNotFoundError:
        pass

GDP_FILE = Path("./Dataset_Sporchi/GDP.csv")
GDP_PC_FILE = Path("./Dataset_Sporchi/GDP_PER_CAPITA.csv")
OUTPUT_FILE = Path("GDP_Country_Data.csv")
YEARS = ["2020", "2021", "2022", "2023", "2024"]

def read_wb(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, skiprows=4)
    except pd.errors.ParserError:
        return pd.read_csv(path)

def reshape(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    cols = ["Country Name", "Country Code"] + YEARS
    df = df[cols]
    return df.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=YEARS,
        var_name="Year",
        value_name=value_name,
    )

def main() -> None:
    gdp = reshape(read_wb(GDP_FILE), "GDP")
    gdp_pc = reshape(read_wb(GDP_PC_FILE), "GDP_per_capita")
    df = (
        gdp.merge(
            gdp_pc,
            on=["Country Name", "Country Code", "Year"],
            how="outer",
            validate="one_to_one"
        )
        .sort_values(["Country Name", "Year"])
        .reset_index(drop=True)
    )

    def impute_or_drop(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("Year").reindex(YEARS)
        group[["GDP", "GDP_per_capita"]] = group[["GDP", "GDP_per_capita"]].astype(float)
        n_missing = group[["GDP", "GDP_per_capita"]].isna().sum().sum()
        if n_missing == 0:
            return group.reset_index()
        if n_missing <= 2:
            group[["GDP", "GDP_per_capita"]] = group[["GDP", "GDP_per_capita"]].interpolate(
                method="linear", limit_direction="both"
            )
            return group.reset_index() if not group.isna().any().any() else pd.DataFrame()
        return pd.DataFrame()

    filled = (
        df.groupby(["Country Name", "Country Code"], group_keys=False)
          .apply(impute_or_drop)
    )

    filled.to_csv(OUTPUT_FILE, index=False, lineterminator="\n")
    print(f"Creato {OUTPUT_FILE} con {len(filled):,} righe.")

if __name__ == "__main__":
    main()
