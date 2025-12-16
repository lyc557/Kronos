import os
import sys
import pickle
import pandas as pd
from config import Config
from dataset import QlibDataset

def check_dataset():
    print("=" * 50)
    print("Checking Dataset Size and Content")
    print("=" * 50)
    
    config = Config()
    dataset_path = config.dataset_path
    print(f"Dataset Path Configured: {dataset_path}")
    
    abs_dataset_path = os.path.abspath(dataset_path)
    print(f"Absolute Dataset Path: {abs_dataset_path}")
    
    if not os.path.exists(abs_dataset_path):
        print(f"ERROR: Dataset directory does not exist: {abs_dataset_path}")
        return

    # Check raw pickle files
    for split in ['train', 'val', 'test']:
        filename = f"{split}_data.pkl"
        file_path = os.path.join(abs_dataset_path, filename)
        
        print(f"\n--- Checking {split.upper()} Data ({filename}) ---")
        if not os.path.exists(file_path):
            print(f"ERROR: File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Raw Data Loaded. Number of symbols: {len(data)}")
            
            total_rows = 0
            empty_symbols = 0
            short_symbols = 0
            required_len = config.lookback_window + config.predict_window + 1
            
            print(f"Required sequence length per sample: {required_len} (Lookback {config.lookback_window} + Predict {config.predict_window} + 1)")
            global_min_dt, global_max_dt = None, None
            sample_printed = 0
            
            for symbol, df in data.items():
                rows = len(df)
                total_rows += rows
                if rows == 0:
                    empty_symbols += 1
                elif rows < required_len:
                    short_symbols += 1
                    if short_symbols <= 5: # Print first 5 short symbols
                        print(f"  [Warning] Symbol {symbol} has length {rows} < {required_len}")
                # Date range collection
                if rows > 0:
                    if 'datetime' in df.index.names:
                        min_dt = df.index.min()
                        max_dt = df.index.max()
                    else:
                        # If index lost, try to infer datetime column
                        if 'datetime' in df.columns:
                            min_dt = pd.to_datetime(df['datetime']).min()
                            max_dt = pd.to_datetime(df['datetime']).max()
                        else:
                            min_dt, max_dt = None, None
                    if min_dt is not None and max_dt is not None:
                        global_min_dt = min_dt if global_min_dt is None else min(global_min_dt, min_dt)
                        global_max_dt = max_dt if global_max_dt is None else max(global_max_dt, max_dt)
                        if sample_printed < 3:
                            print(f"  [Example] {symbol} date range: {min_dt} → {max_dt} (rows: {rows})")
                            sample_printed += 1
            
            print(f"Total rows across all symbols: {total_rows}")
            if empty_symbols > 0:
                print(f"Symbols with 0 rows: {empty_symbols}")
            if short_symbols > 0:
                print(f"Symbols with insufficient length (<{required_len}): {short_symbols}")
            if global_min_dt is not None and global_max_dt is not None:
                print(f"[{split.upper()}] Global date range across symbols: {global_min_dt} → {global_max_dt}")
            else:
                print(f"[{split.upper()}] Could not determine global date range (no valid datetime found).")
                
        except Exception as e:
            print(f"Error loading pickle file: {e}")

    # Check QlibDataset logic
    print("\n" + "=" * 50)
    print("Checking QlibDataset Logic")
    print("=" * 50)
    
    for split in ['train', 'val']:
        print(f"\nInitializing QlibDataset('{split}')...")
        try:
            dataset = QlibDataset(data_type=split)
            print(f"[{split.upper()}] Dataset Length (samples per epoch): {len(dataset)}")
            
            # Access internal indices to see actual available samples
            actual_samples = len(dataset.indices)
            print(f"[{split.upper()}] Total valid samples available (indices): {actual_samples}")
            
            if len(dataset) == 0:
                print(f"[{split.upper()}] WARNING: Dataset is empty!")
            else:
                print(f"[{split.upper()}] Dataset seems OK.")
                
        except Exception as e:
            print(f"[{split.upper()}] Error initializing dataset: {e}")

if __name__ == "__main__":
    check_dataset()
