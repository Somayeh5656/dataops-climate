import pandas as pd
from deltalake import write_deltalake, DeltaTable
import os
from datetime import datetime

def ingest_batch(batch_path, batch_id):
    bronze_path = "data/delta/bronze"
    
    # Read batch CSV
    df = pd.read_csv(batch_path, parse_dates=['date'])
    
    # Add metadata columns
    df['batch_id'] = batch_id
    df['ingestion_time'] = datetime.now()
    
    # Append to Bronze Delta table
    if os.path.exists(bronze_path):
        # Existing table: append
        write_deltalake(bronze_path, df, mode='append')
    else:
        # New table: create with schema
        write_deltalake(bronze_path, df)
    
    print(f"Ingested {batch_path} into Bronze as batch {batch_id}")