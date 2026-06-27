"""
clean_data.py
--------------
Cleans the raw Kaggle "Car Price Prediction" dataset
(data/raw_car_data.csv) and saves an analysis-ready CSV to
data/car_data_clean.csv.

Run this BEFORE preprocessing.py / main.py when using the real dataset.

Usage:
    python clean_data.py
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw_car_data.csv"
CLEAN_PATH = "data/car_data_clean.csv"


def load_raw(path=RAW_PATH):
    return pd.read_csv(path)


def clean(df):
    data = df.copy()
    print(f"Starting rows: {len(data)}")

    # ---------------------------------------------------------------
    # 1. Drop columns that aren't useful predictors
    # ---------------------------------------------------------------
    data.drop(columns=["ID"], inplace=True)

    # 'Model' has 1500+ unique values -> too high-cardinality to one-hot.
    # We keep Manufacturer (65 categories) as the brand signal instead.
    data.drop(columns=["Model"], inplace=True)

    # ---------------------------------------------------------------
    # 2. Remove exact duplicate rows
    # ---------------------------------------------------------------
    before = len(data)
    data.drop_duplicates(inplace=True)
    print(f"Dropped {before - len(data)} duplicate rows")

    # ---------------------------------------------------------------
    # 3. Fix 'Levy': "-" means missing, rest are numeric strings
    # ---------------------------------------------------------------
    data["Levy"] = data["Levy"].replace("-", np.nan).astype(float)
    # Flag whether Levy was originally missing (often meaningful itself)
    data["Levy_missing"] = data["Levy"].isna().astype(int)
    # Impute missing Levy with the median (robust to outliers)
    data["Levy"] = data["Levy"].fillna(data["Levy"].median())

    # ---------------------------------------------------------------
    # 4. Fix 'Mileage': strip " km" and convert to int
    # ---------------------------------------------------------------
    data["Mileage"] = (
        data["Mileage"].astype(str).str.replace(" km", "", regex=False).astype(float)
    )

    # ---------------------------------------------------------------
    # 5. Fix 'Engine volume': split numeric size from Turbo flag
    # ---------------------------------------------------------------
    data["Is_Turbo"] = data["Engine volume"].astype(str).str.contains("Turbo").astype(int)
    data["Engine volume"] = (
        data["Engine volume"].astype(str).str.replace(" Turbo", "", regex=False).astype(float)
    )

    # ---------------------------------------------------------------
    # 6. Fix 'Doors': Excel mangled "2-3"/"4-5" door ranges into dates
    #    "02-Mar" -> 2-3 doors, "04-May" -> 4-5 doors, ">5" stays ">5"
    # ---------------------------------------------------------------
    doors_map = {"02-Mar": "2-3", "04-May": "4-5", ">5": ">5"}
    data["Doors"] = data["Doors"].map(doors_map)

    # ---------------------------------------------------------------
    # 7. Remove unrealistic / outlier prices
    #    Use the IQR method on Price to drop extreme outliers
    #    (data errors like price=1 or price=26 million)
    # ---------------------------------------------------------------
    q1, q3 = data["Price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = max(100, q1 - 1.5 * iqr)   # also enforce a sane absolute floor
    upper = q3 + 1.5 * iqr
    before = len(data)
    data = data[(data["Price"] >= lower) & (data["Price"] <= upper)]
    print(f"Dropped {before - len(data)} price outlier rows "
          f"(kept range: {lower:.0f} - {upper:.0f})")

    # ---------------------------------------------------------------
    # 7b. Remove corrupted Mileage values
    #    This dataset has a known data-corruption bug: some rows show
    #    mileage values in the billions (e.g. 2,147,483,647 km - an
    #    integer overflow artifact). No real used car has done more
    #    than ~1,000,000 km, so anything above that is bad data.
    # ---------------------------------------------------------------
    before = len(data)
    data = data[data["Mileage"] <= 1_000_000]
    print(f"Dropped {before - len(data)} rows with corrupted Mileage values "
          f"(over 1,000,000 km)")

    # ---------------------------------------------------------------
    # 8. Engineer Car_Age from Prod. year
    # ---------------------------------------------------------------
    current_year = 2026
    data["Car_Age"] = current_year - data["Prod. year"]
    data.drop(columns=["Prod. year"], inplace=True)

    # ---------------------------------------------------------------
    # 9. Group rare Manufacturers into "Other" (anything under 1% of rows)
    # ---------------------------------------------------------------
    threshold = len(data) * 0.01
    counts = data["Manufacturer"].value_counts()
    rare = counts[counts < threshold].index
    data["Manufacturer"] = data["Manufacturer"].replace(rare, "Other")

    # ---------------------------------------------------------------
    # 10. Tidy column names (remove spaces/periods for easier coding)
    # ---------------------------------------------------------------
    data.rename(columns={
        "Leather interior": "Leather_Interior",
        "Fuel type": "Fuel_Type",
        "Engine volume": "Engine_Volume",
        "Gear box type": "Gear_Box_Type",
        "Drive wheels": "Drive_Wheels",
    }, inplace=True)

    print(f"Final rows: {len(data)}")
    print(f"Final columns: {list(data.columns)}")
    return data


def main():
    os.makedirs("data", exist_ok=True)
    df = load_raw()
    cleaned = clean(df)
    cleaned.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset to {CLEAN_PATH}")
    print(cleaned.head())


if __name__ == "__main__":
    main()
