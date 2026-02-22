import pandas as pd
import numpy as np
from deltalake import DeltaTable
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("ClimateForecast")


# 1. Load Silver data (raw cleaned variables)

print("Loading Silver data...")
silver_path = "data/delta/silver"
silver_dt = DeltaTable(silver_path)
silver = silver_dt.to_pandas()

# Keep only the raw columns (no metadata)
silver = silver[['date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure']]
silver = silver.sort_values('date').reset_index(drop=True)
print(f"Silver rows: {len(silver)}")


# 2. Load test.csv

print("Loading test.csv...")
test_raw = pd.read_csv("data/raw/DailyDelhiClimateTest.csv", parse_dates=['date'])
test_raw = test_raw.sort_values('date').reset_index(drop=True)
print(f"Test rows: {len(test_raw)}")


# 3. Combine and engineer features
print("Engineering features...")
combined = pd.concat([silver, test_raw], ignore_index=True)
combined = combined.sort_values('date').reset_index(drop=True)

# Create lag and rolling features 
for col in ['meantemp', 'humidity', 'wind_speed', 'meanpressure']:
    combined[f'{col}_lag1'] = combined[col].shift(1)
    combined[f'{col}_lag7'] = combined[col].shift(7)
    combined[f'{col}_roll7_avg'] = combined[col].rolling(7).mean()

# Create target: next day's meantemp (needed for test evaluation)
combined['target'] = combined['meantemp'].shift(-1)

# Drop rows with NaN (first 7 rows and last row)
combined = combined.dropna().reset_index(drop=True)


# 4. Split back into train and test sets

train_dates = set(silver['date'])
test_dates = set(test_raw['date'])

# Feature columns (all engineered columns except raw and target)
feature_cols = [col for col in combined.columns if col not in ['date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure', 'target']]

train_features = combined[combined['date'].isin(train_dates)]
test_features = combined[combined['date'].isin(test_dates)]

X_test = test_features[feature_cols]
y_test = test_features['target']

print(f"Test features shape: {X_test.shape}")


# 5. Load the latest model from MLflow

print("Loading latest model from MLflow...")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("ClimateForecast")
if experiment is None:
    raise Exception("Experiment 'ClimateForecast' not found. Run training first.")

runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
if len(runs) == 0:
    raise Exception("No runs found. Run training first.")

latest_run = runs[0]
run_id = latest_run.info.run_id
model_uri = f"runs:/{run_id}/model"
model = mlflow.xgboost.load_model(model_uri)
print(f"Loaded model from run {run_id}")

#  Reorder features to match model's expected order ---
expected_features = model.get_booster().feature_names
print(f"Model expects features: {expected_features}")
X_test = X_test[expected_features]

# 6. Predict and evaluate

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")


# 7. Log results to MLflow 

with mlflow.start_run(run_name="test_evaluation"):
    mlflow.log_params({
        "model_run_id": run_id,
        "model_gold_version": latest_run.data.params.get("gold_version", "unknown"),
        "test_size": len(X_test)
    })
    mlflow.log_metrics({"test_rmse": rmse, "test_mae": mae})
    print("Test results logged to MLflow.")