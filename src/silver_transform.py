import pandas as pd
from deltalake import DeltaTable, write_deltalake
import os

def transform_to_silver():
    bronze_path = "data/delta/bronze"
    silver_path = "data/delta/silver"

    # Read entire Bronze table
    bronze_df = DeltaTable(bronze_path).to_pandas()

    # Data cleaning
    bronze_df['date'] = pd.to_datetime(bronze_df['date'])
    bronze_df = bronze_df.dropna(subset=['date'])
    bronze_df = bronze_df.drop_duplicates(subset=['date'])

    # Value range checks (optional)
    valid_rows = bronze_df[
        (bronze_df['meantemp'].between(-10, 50)) &
        (bronze_df['humidity'].between(0, 100)) &
        (bronze_df['wind_speed'].between(0, 200)) &
        (bronze_df['meanpressure'].between(900, 1100))
    ]
    print(f"Total rows: {len(bronze_df)}, valid rows: {len(valid_rows)}")

    # Drop rows with nulls in key columns
    silver_df = valid_rows.dropna(subset=['meantemp', 'humidity', 'wind_speed', 'meanpressure']).copy()

    # 🔥 FIX: Reset index to avoid it being written as an extra column
    silver_df = silver_df.reset_index(drop=True)

    # Write to Silver (overwrite)
    write_deltalake(silver_path, silver_df, mode='overwrite')
    print(f"Silver table updated with {len(silver_df)} rows")