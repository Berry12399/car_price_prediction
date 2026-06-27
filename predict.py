"""
predict.py
-----------
Loads the saved models (trained on the real cleaned dataset) and lets you
predict the selling price of a car by entering its details in the terminal.

Usage (AFTER running clean_data.py and main.py once):
    python predict.py
"""

import joblib
import pandas as pd

CURRENT_YEAR = 2026


def get_user_input():
    print("Enter the car details below:\n")
    return {
        "Levy": float(input("Levy / import tax (e.g. 800, enter 0 if unknown): ")),
        "Manufacturer": input("Manufacturer (e.g. TOYOTA, FORD, Other): ").strip().upper(),
        "Category": input("Category [Sedan/Jeep/Hatchback/...]: ").strip().capitalize(),
        "Leather_Interior": input("Leather interior? [Yes/No]: ").strip().capitalize(),
        "Fuel_Type": input("Fuel type [Petrol/Diesel/Hybrid/CNG/LPG]: ").strip().capitalize(),
        "Engine_Volume": float(input("Engine volume in liters (e.g. 2.0): ")),
        "Is_Turbo": int(input("Turbo engine? [1=yes, 0=no]: ")),
        "Mileage": float(input("Mileage in km (e.g. 120000): ")),
        "Cylinders": float(input("Number of cylinders (e.g. 4): ")),
        "Gear_Box_Type": input("Gear box type [Automatic/Manual/Tiptronic/Variator]: ").strip().capitalize(),
        "Drive_Wheels": input("Drive wheels [Front/Rear/4x4]: ").strip().capitalize(),
        "Doors": input("Doors [2-3/4-5/>5]: ").strip(),
        "Wheel": input("Wheel [Left wheel/Right-hand drive]: ").strip(),
        "Color": input("Color (e.g. Black, White, Silver): ").strip().capitalize(),
        "Airbags": int(input("Number of airbags (e.g. 6): ")),
        "Prod_year": int(input("Production year (e.g. 2015): ")),
    }


def build_feature_row(raw_input, feature_columns):
    """Turn the raw user input into the exact one-hot encoded row the model expects."""
    row = {col: 0 for col in feature_columns}

    # Numeric fields go in directly
    row["Levy"] = raw_input["Levy"]
    row["Levy_missing"] = 0
    row["Engine_Volume"] = raw_input["Engine_Volume"]
    row["Is_Turbo"] = raw_input["Is_Turbo"]
    row["Mileage"] = raw_input["Mileage"]
    row["Cylinders"] = raw_input["Cylinders"]
    row["Airbags"] = raw_input["Airbags"]
    row["Car_Age"] = CURRENT_YEAR - raw_input["Prod_year"]

    # Categorical fields -> map to their one-hot column
    cat_map = {
        "Manufacturer": raw_input["Manufacturer"],
        "Category": raw_input["Category"],
        "Leather_Interior": raw_input["Leather_Interior"],
        "Fuel_Type": raw_input["Fuel_Type"],
        "Gear_Box_Type": raw_input["Gear_Box_Type"],
        "Drive_Wheels": raw_input["Drive_Wheels"],
        "Doors": raw_input["Doors"],
        "Wheel": raw_input["Wheel"],
        "Color": raw_input["Color"],
    }
    for prefix, value in cat_map.items():
        col_name = f"{prefix}_{value}"
        if col_name in row:
            row[col_name] = 1
        # if the category was the "dropped" baseline (drop_first=True) or
        # an unseen category, all its dummy columns stay 0 -> that's fine.

    return pd.DataFrame([row])[feature_columns]


def main():
    lr_model = joblib.load("models/linear_regression.pkl")
    dt_model = joblib.load("models/decision_tree.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")

    raw_input = get_user_input()
    X_new = build_feature_row(raw_input, feature_columns)
    X_new_scaled = scaler.transform(X_new)

    lr_price = lr_model.predict(X_new_scaled)[0]
    dt_price = dt_model.predict(X_new)[0]  # tree was trained on unscaled features

    print("\n--- Predicted Price ---")
    print(f"Linear Regression : {lr_price:,.0f}")
    print(f"Decision Tree      : {dt_price:,.0f}")


if __name__ == "__main__":
    main()
