import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deltalake import DeltaTable
import pandas as pd

def test_bronze():
    """Test Bronze layer: schema, row count, nulls."""
    print("\n--- Testing Bronze ---")
    bronze = DeltaTable("data/delta/bronze").to_pandas()
    
    # Expected columns
    expected_cols = {'date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure', 'batch_id', 'ingestion_time'}
    assert expected_cols.issubset(set(bronze.columns)), f"Bronze missing columns: {expected_cols - set(bronze.columns)}"
    print(" Bronze has all expected columns.")
    
    # No nulls in date
    assert bronze['date'].notna().all(), "Bronze has null dates!"
    print(" Bronze has no null dates.")
    
    # Row count (should be around 1460)
    total_rows = len(bronze)
    assert total_rows >= 1450, f"Bronze row count too low: {total_rows}"
    print(f" Bronze row count: {total_rows} (expected ~1460).")
    
    # Batch_id present and non-null
    assert bronze['batch_id'].notna().all(), "Bronze has null batch_id!"
    print(" Bronze batch_id present.")
    
    return bronze

def test_silver():
    """Test Silver layer: cleaned data."""
    print("\n--- Testing Silver ---")
    silver = DeltaTable("data/delta/silver").to_pandas()
    
    # Expected columns (should be the original four + date, no metadata)
    expected_cols = {'date', 'meantemp', 'humidity', 'wind_speed', 'meanpressure'}
    assert expected_cols.issubset(set(silver.columns)), f"Silver missing columns: {expected_cols - set(silver.columns)}"
    print(" Silver has all expected columns.")
    
    # No duplicates in date
    assert silver['date'].is_unique, "Silver has duplicate dates!"
    print(" Silver has unique dates.")
    
    # No nulls in core columns
    for col in ['meantemp', 'humidity', 'wind_speed', 'meanpressure']:
        assert silver[col].notna().all(), f"Silver has nulls in {col}!"
    print(" Silver has no nulls in core columns.")
    
    # Value ranges
    assert silver['meantemp'].between(-10, 50).all(), "meantemp out of range"
    assert silver['humidity'].between(0, 100).all(), "humidity out of range"
    assert silver['wind_speed'].between(0, 200).all(), "wind_speed out of range"
    assert silver['meanpressure'].between(900, 1100).all(), "meanpressure out of range"
    print(" Silver values are within expected ranges.")
    
    # Row count (should be <= Bronze, around 1450)
    bronze_rows = len(DeltaTable("data/delta/bronze").to_pandas())
    silver_rows = len(silver)
    assert silver_rows <= bronze_rows, f"Silver rows > Bronze rows: {silver_rows} > {bronze_rows}"
    print(f" Silver row count: {silver_rows} (Bronze: {bronze_rows})")
    
    return silver

def test_gold():
    """Test Gold layer: features and target."""
    print("\n--- Testing Gold ---")
    gold = DeltaTable("data/delta/gold").to_pandas()
    
    # Expected columns (target + lags + rolling)
    expected_prefixes = ['date', 'target', '_lag1', '_lag7', '_roll7_avg']

    assert 'target' in gold.columns, "Gold missing target column"
    assert any('_lag1' in col for col in gold.columns), "Gold missing lag1 features"
    assert any('_lag7' in col for col in gold.columns), "Gold missing lag7 features"
    assert any('_roll7_avg' in col for col in gold.columns), "Gold missing rolling averages"
    print(" Gold has expected feature columns.")
    
    # No nulls in target or lags (except first rows which were dropped)
    assert gold['target'].notna().all(), "Gold has null target!"
    print(" Gold target has no nulls.")
    
    # Date should be unique
    assert gold['date'].is_unique, "Gold has duplicate dates!"
    print(" Gold dates are unique.")
    
    # Row count: should be less than Silver (due to lags)
    silver_rows = len(DeltaTable("data/delta/silver").to_pandas())
    gold_rows = len(gold)
    assert gold_rows < silver_rows, f"Gold rows not less than Silver: {gold_rows} >= {silver_rows}"
    print(f" Gold row count: {gold_rows} (Silver: {silver_rows})")
    
    return gold

if __name__ == "__main__":
    try:
        test_bronze()
        test_silver()
        test_gold()
        print("\n All tests passed!")
    except AssertionError as e:
        print(f"\n Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        sys.exit(1)