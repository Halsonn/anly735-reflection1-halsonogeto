import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 1. Acquire & Load UCI Bike Sharing Hourly Dataset
url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as response:
    zip_data = zipfile.ZipFile(io.BytesIO(response.read()))
    df = pd.read_csv(zip_data.open("hour.csv"))

# 2. Prevent Data Leakage & Engineer Realistic Temporal Features
df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

# Engineer lag features available at prediction time
df["cnt_lag_24h"] = df["cnt"].shift(24)
df["cnt_lag_48h"] = df["cnt"].shift(48)
df = df.dropna().reset_index(drop=True)

target = "cnt"
leakage_and_id_cols = ["instant", "dteday", "casual", "registered", target]
feature_cols = [col for col in df.columns if col not in leakage_and_id_cols]

categorical_features = ["season", "yr", "mnth", "hr", "weekday", "weathersit"]
numeric_features = [col for col in feature_cols if col not in categorical_features]

X = df[feature_cols]
y = df[target]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

results = []

# Evaluation Design 1: Temporally Structured Evaluation (Earliest 80% train, latest 20% test)
split_idx = int(len(df) * 0.8)
X_train_temp, X_test_temp = X.iloc[:split_idx], X.iloc[split_idx:]
y_train_temp, y_test_temp = y.iloc[:split_idx], y.iloc[split_idx:]

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    pipeline.fit(X_train_temp, y_train_temp)
    y_pred = np.clip(pipeline.predict(X_test_temp), a_min=0, a_max=None)

    results.append({
        "evaluation_design": "Temporally Structured",
        "model": name,
        "rmse": np.sqrt(mean_squared_error(y_test_temp, y_pred)),
        "mae": mean_absolute_error(y_test_temp, y_pred)
    })

# Evaluation Design 2: Random Evaluation (80% train, 20% test without temporal order)
X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for name, model in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])
    pipeline.fit(X_train_rand, y_train_rand)
    y_pred = np.clip(pipeline.predict(X_test_rand), a_min=0, a_max=None)

    results.append({
        "evaluation_design": "Random Split",
        "model": name,
        "rmse": np.sqrt(mean_squared_error(y_test_rand, y_pred)),
        "mae": mean_absolute_error(y_test_rand, y_pred)
    })

# Write output to the required location relative to root
os.makedirs("analysis", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("analysis/lab01_results.csv", index=False)
print("Analysis complete. Saved to analysis/lab01_results.csv")
print(results_df)
