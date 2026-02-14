import pandas as pd
import numpy as np
import os

# Create batches folder
os.makedirs('data/batches', exist_ok=True)

# Read full dataset
df = pd.read_csv('data/raw/DailyDelhiClimateTrain.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Split into 5 roughly equal parts
total_rows = len(df)
rows_per_batch = total_rows // 5
batches = []
for i in range(5):
    start = i * rows_per_batch
    end = (i + 1) * rows_per_batch if i < 4 else total_rows
    batch = df.iloc[start:end].copy()
    batches.append(batch)

# --- Simulate data quality issues in some batches ---
np.random.seed(42)

# Batch 2: remove one row, add duplicate date, add null in humidity
batch2 = batches[1].copy()
# Remove one random row
drop_idx = np.random.choice(batch2.index, size=1, replace=False)
batch2 = batch2.drop(drop_idx)
# Add a duplicate (clone first row, shift date by 1 day)
dup = batch2.iloc[0].copy()
dup['date'] = dup['date'] + pd.Timedelta(days=1)
batch2 = pd.concat([batch2, dup.to_frame().T], ignore_index=True)
# Sort again after adding duplicate
batch2 = batch2.sort_values('date').reset_index(drop=True)
# Introduce a null in humidity at random row
null_idx = np.random.choice(batch2.index, size=1, replace=False)
batch2.loc[null_idx, 'humidity'] = np.nan
batches[1] = batch2

# Batch 4: remove two rows
batch4 = batches[3].copy()
drop_idx = np.random.choice(batch4.index, size=2, replace=False)
batch4 = batch4.drop(drop_idx).reset_index(drop=True)
batches[3] = batch4

# Save all batches
for i, batch in enumerate(batches, start=1):
    batch.to_csv(f'data/batches/batch{i}.csv', index=False)

print("Batches created in data/batches/")