# DataOps Climate Pipeline

This project implements a Bronze–Silver–Gold data pipeline for incremental climate data using Delta Lake (via `deltalake`). It ingests five time-ordered batches, performs cleaning and validation, and produces an ML-ready dataset for time-series forecasting.

---

## Prerequisites

- Python 3.10+
- Install dependencies:


pip install -r requirements.txt

---

## Dataset
Original file: `DailyDelhiClimateTrain.csv`

Source: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data

The dataset is split into 5 chronological batches using `split_batches.py`.

Pre-split batches are available in:

data/batches/


---

## DataOps Pipeline Steps

1. **Split the dataset** (if not already split)

   python split_batches.py

   This creates:

   data/batches/batch1.csv
   data/batches/batch2.csv
   data/batches/batch3.csv
   data/batches/batch4.csv
   data/batches/batch5.csv


2. **Run the pipeline**

   python src/pipeline.py

   This will:
   - Ingest any new batch (append to Bronze layer)
   - Rebuild Silver layer (cleaned + validated data)
   - Rebuild Gold layer (feature-engineered dataset)

3. **Add the next batch** (simulate incremental ingestion)
   - Move the next batch file into `data/batches/`
   - Re-run:

     python src/pipeline.py

   Only new batches will be appended to Bronze.

4. **Check Delta version history**
  
   from deltalake import DeltaTable
   print(DeltaTable("data/delta/bronze").history())

   This prints the full transaction history of the Bronze table.

---

## Outputs
After running the pipeline, the following Delta tables are created:

- **Bronze** > `data/delta/bronze`  
  Raw ingested data with `batch_id` and `ingestion_time`
- **Silver** > `data/delta/silver`  
  Cleaned and validated dataset
- **Gold** > `data/delta/gold`  
  Feature-engineered dataset for time-series forecasting

---

## Assumptions
- Batches are added in correct chronological order.
- Pipeline runs sequentially (no concurrent writes).
- Silver and Gold layers are fully rebuilt after each ingestion.
- Validation includes basic range checks and null handling.

---

## Reproducibility
- Delta Lake provides versioning and time travel.
- All processing logic is version-controlled.
- Running the pipeline from scratch recreates all layers.

---

## ModelOps Pipeline (Assignment 4)

This project also implements a **ModelOps** pipeline for time‑series forecasting using the Gold dataset. The pipeline trains XGBoost models on different Gold versions, tracks experiments with MLflow, and evaluates the final model on a held‑out test set.

### Additional Prerequisites for ModelOps
Install the following packages (already included in `requirements.txt`):

pip install mlflow xgboost scikit-learn


### Training

1. **Initial model** (trained on Gold version 1):

   python train_initial.py

2. **Updated model** (trained on Gold version 3):

   python train_updated.py


Each training script:
- Loads the specified Gold version using Delta Lake time travel.
- Splits data chronologically (80% train, 20% validation).
- Trains an XGBoost regressor with fixed hyperparameters.
- Logs parameters, metrics (RMSE, MAE), and the model to MLflow.

### Evaluation on Test Set

The test set (`DailyDelhiClimateTest.csv`) contains unseen data from 2017. To evaluate the latest model:

python evaluate_test.py


This script:
- Loads Silver data (cleaned raw variables) and the test CSV.
- Engineers the same features (lags, rolling averages) on the combined series.
- Extracts test rows and predicts using the most recent model from MLflow.
- Computes RMSE and MAE, and logs them as a new MLflow run.

### MLflow UI

To view experiment results, start the MLflow UI:

python -m mlflow ui

Then open http://localhost:5000 in your browser. You will see:
- Two training runs (`initial_model_v1`, `updated_model_v3`) with their parameters and metrics.
- One test evaluation run (`test_evaluation`) with `test_rmse` and `test_mae`.

### Model Versioning and Data‑Model Linkage

- Each MLflow run logs a `gold_version` parameter, explicitly linking the model to the Gold dataset version used for training.
- The Gold Delta table retains full version history (versions 0–3), ensuring reproducibility.

### Automation (Bonus)

A simple continuous update script (`continuous_update.py`) is provided. It checks the latest Gold version against the last trained version and triggers retraining if a newer version exists. This script can be scheduled to run periodically (e.g., via cron or Task Scheduler).

