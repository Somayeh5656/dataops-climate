# DataOps Climate Pipeline

This project implements a Bronze–Silver–Gold data pipeline for incremental climate data using Delta Lake (via `deltalake`). It ingests five time-ordered batches, performs cleaning and validation, and produces an ML-ready dataset for time-series forecasting.

---

## Prerequisites

- Python 3.10+
- Install dependencies:


pip install -r requirements.txt

---

## Dataset
Original file: DailyDelhiClimateTrain.csv

Source: Kaggle – Daily Climate Time Series Data
https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data

The dataset is split into 5 chronological batches using split_batches.py.

Pre-split batches are available in:

data/batches/

---

## Pipeline Steps
1. Split the dataset (if not already split)
python split_batches.py
This creates:

data/batches/batch1.csv
data/batches/batch2.csv
data/batches/batch3.csv
data/batches/batch4.csv
data/batches/batch5.csv

2. Run the pipeline
python src/pipeline.py
This will:

Ingest any new batch (append to Bronze layer)

Rebuild Silver layer (cleaned + validated data)

Rebuild Gold layer (feature-engineered dataset)

3. Add the next batch
To simulate incremental ingestion:

Move the next batch file into data/batches/

Re-run:

python src/pipeline.py
Only new batches will be appended to Bronze.

4. Check Delta version history
from deltalake import DeltaTable

print(DeltaTable("data/delta/bronze").history())
This prints the full transaction history of the Bronze table.

---

## Outputs
After running the pipeline, the following Delta tables are created:

Bronze → data/delta/bronze
Raw ingested data with batch_id and ingestion_time

Silver → data/delta/silver
Cleaned and validated dataset

Gold → data/delta/gold
Feature-engineered dataset for time-series forecasting

---

## Assumptions
Batches are added in correct chronological order.

Pipeline runs sequentially (no concurrent writes).

Silver and Gold layers are fully rebuilt after each ingestion.

Validation includes basic range checks and null handling.

---

## Reproducibility
Delta Lake provides versioning and time travel.

All processing logic is version-controlled.

Running the pipeline from scratch recreates all layers.

