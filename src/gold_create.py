import pandas as pd
from deltalake import DeltaTable, write_deltalake

def create_gold():
    silver_path = "data/delta/silver"
    gold_path = "data/delta/gold"
    
    silver_df = DeltaTable(silver_path).to_pandas()
    silver_df = silver_df.sort_values('date')
    
    # Target: next day's meantemp
    silver_df['target'] = silver_df['meantemp'].shift(-1)
    
    # Lag features
    for col in ['meantemp', 'humidity', 'wind_speed', 'meanpressure']:
        silver_df[f'{col}_lag1'] = silver_df[col].shift(1)
        silver_df[f'{col}_lag7'] = silver_df[col].shift(7)
    
    # Rolling averages (7-day)
    for col in ['meantemp', 'humidity', 'wind_speed', 'meanpressure']:
        silver_df[f'{col}_roll7_avg'] = silver_df[col].rolling(7).mean()
    
    # Drop rows with NaN (first 7 days)
    gold_df = silver_df.dropna().reset_index(drop=True)
    
    # Select relevant columns
    gold_columns = ['date', 'target'] + \
                   [c for c in gold_df.columns if '_lag' in c or '_roll7_avg' in c]
    gold_df = gold_df[gold_columns]
    
    # Write Gold
    write_deltalake(gold_path, gold_df, mode='overwrite')
    print("Gold table created")
    print(gold_df.head())