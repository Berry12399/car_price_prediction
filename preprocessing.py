import pandas as pd

CLEAN_PATH = "data/car_data_clean.csv"


def load_data(path=CLEAN_PATH):
    
    df = pd.read_csv(path)
    return df


def preprocess(df):

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
