import os
from bronze_ingest import ingest_batch
from silver_transform import transform_to_silver
from gold_create import create_gold

# Track processed batches
processed_file = "processed_batches.txt"
processed = set()
if os.path.exists(processed_file):
    with open(processed_file, "r") as f:
        processed = set(line.strip() for line in f)

batch_folder = "data/batches"
batches = [f for f in os.listdir(batch_folder) if f.endswith(".csv") and f not in processed]

if batches:
    print(f"Found new batches: {batches}")
    for batch_file in batches:
        batch_id = batch_file.replace(".csv", "")
        ingest_batch(os.path.join(batch_folder, batch_file), batch_id)
        with open(processed_file, "a") as f:
            f.write(batch_file + "\n")
    
    # Rebuild Silver and Gold from all Bronze data
    transform_to_silver()
    create_gold()
else:
    print("No new batches found.")