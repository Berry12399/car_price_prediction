# Car Price Prediction — Linear Regression vs Decision Tree

A complete, runnable Python project that predicts car selling prices
using two algorithms — **Linear Regression** and **Decision Tree Regressor**
— trained on the real Kaggle "Car Price Prediction" dataset.

```
car_price_prediction/
├── data/
│   ├── raw_car_data.csv      ← put your downloaded Kaggle CSV here
│   └── car_data_clean.csv    (created by clean_data.py)
├── models/                   (created automatically after training)
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   └── feature_columns.pkl
├── outputs/                  (created automatically — charts go here)
│   ├── actual_vs_predicted.png
│   └── metric_comparison.png
├── clean_data.py             # Step 1: cleans the messy raw CSV
├── preprocessing.py          # Step 2: encodes the cleaned data for ML
├── train_models.py           # Step 3: trains + evaluates both models
├── main.py                   # Runs the full pipeline + saves charts
├── predict.py                # Step 4: predict price for a new car
└── requirements.txt
```

---

## What was wrong with the raw dataset (and how `clean_data.py` fixes it)

| Problem | Fix applied |
|---|---|
| 313+ duplicate rows | Dropped with `drop_duplicates()` |
| `Levy` column uses `"-"` for missing values, stored as text | Converted to numeric, missing values filled with the median, plus a new `Levy_missing` flag column so the model can learn from "was this missing" too |
| `Mileage` has `" km"` glued onto every value | Stripped the text, converted to a clean number |
| `Engine volume` mixes engine size and turbo status (e.g. `"2.0 Turbo"`) | Split into `Engine_Volume` (numeric) and a separate `Is_Turbo` (0/1) flag |
| `Doors` got corrupted by Excel into date-like strings (`"04-May"`, `"02-Mar"`) | Mapped back to their real meaning: `"2-3"`, `"4-5"`, `">5"` |
| Extreme price outliers (prices of 1, or up to 26 million) | Removed using the IQR (interquartile range) method |
| `Model` column has 1,590 unique values — too many for one-hot encoding | Dropped (kept `Manufacturer`, which is more generalizable) |
| `Manufacturer` has 65 categories, several very rare | Categories under 1% of rows grouped into `"Other"` |
| `ID` column | Dropped — it's just a record number, not predictive |
| `Prod. year` isn't directly useful | Converted to `Car_Age` (current year − production year) |

---

## ROADMAP: How to use this in VS Code

### Step 0 — Install prerequisites
1. Install **Python 3.10+** and **VS Code**.
2. In VS Code, install the **Python** extension (Microsoft).

### Step 1 — Open the project folder
`File → Open Folder...` → select `car_price_prediction`.

### Step 2 — Create + activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Adding dataset
Downloaded dataset from Kaggle and place it at:
data/raw_car_data.csv



### Step 5 — Clean the data
```bash
python clean_data.py
```
This reads `data/raw_car_data.csv`, fixes all the issues listed above, and
writes a clean file to `data/car_data_clean.csv`. It also prints a summary
of how many duplicate/outlier rows were removed.

### Step 6 — (Optional) Inspect the encoded features
```bash
python preprocessing.py
```

### Step 7 — Train both models & see results
```bash
python main.py
```
Trains Linear Regression and Decision Tree, prints MAE/RMSE/R² for each,
saves models to `models/`, and saves comparison charts to `outputs/`.

### Step 8 — Predict the price of a new car
```bash
python predict.py
```
Answer the prompts (manufacturer, mileage, engine volume, etc.) and get a
predicted price from both models.



## Current results on this dataset

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~6,390 | ~8,514 | ~0.41 |
| Decision Tree | ~5,416 | ~7,529 | ~0.54 |

Decision Tree currently wins — likely because price depends on non-linear
interactions (e.g. luxury brand + low mileage + recent year compounds price
in a way a straight line can't capture). Your exact numbers may vary
slightly depending on random splits.

