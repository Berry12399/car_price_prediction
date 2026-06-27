"""
train_models.py
-----------------
Trains a Linear Regression model and a Decision Tree Regressor on the
car price data, evaluates both, and saves them to disk with joblib.
"""

import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import load_data, preprocess


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=6, random_state=42):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, name="Model"):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"\n--- {name} Performance ---")
    print(f"MAE  (Mean Absolute Error) : {mae:,.0f}")
    print(f"RMSE (Root Mean Sq. Error) : {rmse:,.0f}")
    print(f"R^2 Score                  : {r2:.3f}")

    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2, "preds": preds}


def main():
    print("1. Loading data...")
    df = load_data()

    print("2. Preprocessing...")
    X, y = preprocess(df)

    print("3. Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("3b. Scaling features (helps Linear Regression stay numerically stable)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("4. Training Linear Regression...")
    lr_model = train_linear_regression(X_train_scaled, y_train)

    print("5. Training Decision Tree Regressor...")
    dt_model = train_decision_tree(X_train, y_train)  # trees don't need scaling

    print("\n6. Evaluating both models on the TEST set:")
    lr_results = evaluate_model(lr_model, X_test_scaled, y_test, name="Linear Regression")
    dt_results = evaluate_model(dt_model, X_test, y_test, name="Decision Tree")

    # 7. Pick the winner based on R^2 score
    winner = "Linear Regression" if lr_results["r2"] >= dt_results["r2"] else "Decision Tree"
    print(f"\n>>> Best model on this dataset: {winner} <<<")

    # 8. Save models + the column order/scaler used for training (needed for predict.py)
    os.makedirs("models", exist_ok=True)
    joblib.dump(lr_model, "models/linear_regression.pkl")
    joblib.dump(dt_model, "models/decision_tree.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(list(X.columns), "models/feature_columns.pkl")
    print("\nModels saved to the models/ folder.")

    return lr_results, dt_results, X_test, y_test


if __name__ == "__main__":
    main()
