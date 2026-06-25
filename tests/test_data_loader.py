import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta

# Add current directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data.data_loader import load_crypto_data

def test_loader_cache():
    # Test a small period: 7 days ago to today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Testing loader with CACHE for BTCUSDT from {start_str} to {end_str} with 4h interval...\n")
    
    try:
        # First run: should download and cache
        print("--- RUN 1: Fetching and Caching ---")
        t0 = time.time()
        df1 = load_crypto_data(
            symbol="BTCUSDT",
            start_date=start_str,
            end_date=end_str,
            interval="4h",
            use_cache=True
        )
        t1 = time.time()
        print(f"Run 1 completed in {t1-t0:.2f} seconds. Rows: {len(df1)}")
        
        # Second run: should load from cache
        print("\n--- RUN 2: Loading from Cache ---")
        t2 = time.time()
        df2 = load_crypto_data(
            symbol="BTCUSDT",
            start_date=start_str,
            end_date=end_str,
            interval="4h",
            use_cache=True
        )
        t3 = time.time()
        print(f"Run 2 completed in {t3-t2:.4f} seconds. Rows: {len(df2)}")
        
        if (t3-t2) > 0:
            print(f"\nPerformance: Loading from cache was {(t1-t0)/(t3-t2):.1f}x faster!")
            
        print("\nData Integrity Check (df1 == df2):", df1.equals(df2))
        
    except Exception as e:
        import traceback
        print(f"ERROR: Failed to load data. Exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_loader_cache()
