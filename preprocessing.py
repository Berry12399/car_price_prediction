"""
preprocessing.py
-----------------
Loads the CLEANED dataset (produced by clean_data.py) and turns it into
clean, numeric, model-ready features.
"""

import pandas as pd

CLEAN_PATH = "data/car_data_clean.csv"


def load_data(path=CLEAN_PATH):
    """Load the cleaned CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


def preprocess(df):
    """
    Turn the cleaned dataframe into X (features) and y (target).

    - One-hot encodes all categorical/text columns.
    - Target column is 'Price'.
    """
    data = df.copy()

    categorical_cols = [
        "Manufacturer", "Category", "Leather_Interior", "Fuel_Type",
        "Gear_Box_Type", "Drive_Wheels", "Doors", "Wheel", "Color",
    ]
    categorical_cols = [c for c in categorical_cols if c in data.columns]

    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    target_col = "Price"
    X = data.drop(columns=[target_col])
    y = data[target_col]

    return X, y


if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    print(f"Number of features after encoding: {X.shape[1]}")
    print("Sample columns:", list(X.columns)[:15], "...")
    print("\nSample rows:")
    print(X.head())
