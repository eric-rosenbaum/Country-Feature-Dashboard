import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import pickle
import os
from data_loader import load_data
import json

# === Load Data ===
df = load_data()

# === Preprocess ===
drop_cols = ['Country', 'Country Name']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])
all_features = df.columns.tolist()

# === Store all importances ===
importance_json = {}

for target in all_features:
    X = df.drop(columns=[target])
    y = df[target]

    # Drop rows with NaNs
    temp_df = pd.concat([X, y], axis=1).dropna()
    X_clean = temp_df.drop(columns=[target])
    y_clean = temp_df[target]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate (optional)
    y_pred = model.predict(X_test)
    print(f"[{target}] R^2: {r2_score(y_test, y_pred):.3f}, RMSE: {root_mean_squared_error(y_test, y_pred):.2f}")

    # Get Importances
    importances = model.feature_importances_
    feature_names = X_clean.columns.tolist()

    # Save for JSON
    importance_json[target] = dict(zip(feature_names, importances))

# === Save to JSON ===
import json
with open("models/all_feature_importances.json", "w") as f:
    json.dump(importance_json, f, indent=2)
