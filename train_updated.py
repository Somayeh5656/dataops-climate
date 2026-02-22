import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
from deltalake import DeltaTable
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ClimateForecast")

# Load Gold version 3 using version parameter in constructor
gold_version = 3
gold_path = "data/delta/gold"
dt = DeltaTable(gold_path, version=gold_version)
df = dt.to_pandas()

df = df.sort_values('date')

feature_cols = [col for col in df.columns if col not in ['date', 'target']]
X = df[feature_cols]
y = df['target']

split_idx = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

with mlflow.start_run(run_name="updated_model_v3"):
    params = {
        "model": "XGBoost",
        "gold_version": gold_version,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    mlflow.log_params(params)

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    mlflow.log_metrics({"rmse": rmse, "mae": mae})
    mlflow.xgboost.log_model(model, "model")

    print(f"Updated model (Gold v{gold_version}) - RMSE: {rmse:.4f}, MAE: {mae:.4f}")