import pandas as pd
import numpy as np

def create_purged_train_val_split(
    dfs_dict: dict[str, pd.DataFrame], 
    val_slices: dict[str, tuple[str, str]], 
    embargo_candles: int = 42
) -> tuple[list[pd.DataFrame], dict[str, dict[str, pd.DataFrame]]]:
    """
    Creates Purged Combinatorial Cross-Validation split for multi-asset data.
    
    For each asset:
    - Extracts val_slices as validation data.
    - 'Punches holes' in the main DataFrame for these slices AND an embargo period 
      before and after each slice to prevent leakage of smoothed indicators.
    - The remaining continuous chunks are collected into a single list of train_dfs.
    
    Args:
        dfs_dict: Dict mapping symbol to its full DataFrame.
        val_slices: Dict mapping slice name to (start_date, end_date) strings.
        embargo_candles: Number of candles to drop before and after each validation slice.
        
    Returns:
        train_dfs: A flat list of continuous pandas DataFrames for training.
        val_dfs_dict: Dict mapping symbol -> {slice_name: val_df}.
    """
    train_dfs = []
    val_dfs_dict = {sym: {} for sym in dfs_dict.keys()}
    
    for sym, df in dfs_dict.items():
        df = df.copy()
        if 'timestamp' not in df.columns:
            df = df.reset_index(names=['timestamp'])
            
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        is_tz_aware = df['timestamp'].dt.tz is not None
            
        is_val = np.zeros(len(df), dtype=bool)
        is_embargo = np.zeros(len(df), dtype=bool)
        
        for slice_name, (start_str, end_str) in val_slices.items():
            start_date = pd.to_datetime(start_str)
            end_date = pd.to_datetime(end_str)
            
            if is_tz_aware and start_date.tzinfo is None:
                start_date = start_date.tz_localize('UTC')
                end_date = end_date.tz_localize('UTC')
            elif not is_tz_aware and start_date.tzinfo is not None:
                start_date = start_date.tz_localize(None)
                end_date = end_date.tz_localize(None)
                
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            
            if not mask.any():
                print(f"⚠️ Slice {slice_name} not found in {sym} data.")
                continue
                
            # Extract val slice
            val_df = df[mask].copy()
            val_dfs_dict[sym][slice_name] = val_df
            
            # Mark validation and embargo indices
            idx_true = np.where(mask)[0]
            if len(idx_true) > 0:
                first_idx = idx_true[0]
                last_idx = idx_true[-1]
                
                is_val[first_idx:last_idx+1] = True
                
                # Embargo before
                embargo_start = max(0, first_idx - embargo_candles)
                is_embargo[embargo_start:first_idx] = True
                
                # Embargo after
                embargo_end = min(len(df), last_idx + 1 + embargo_candles)
                is_embargo[last_idx+1:embargo_end] = True
                
        # The remaining valid training rows
        is_train = ~(is_val | is_embargo)
        
        # Split is_train into continuous chunks
        padded = np.concatenate([[False], is_train, [False]])
        diffs = np.diff(padded.astype(int))
        
        chunk_starts = np.where(diffs == 1)[0]
        chunk_ends = np.where(diffs == -1)[0]
        
        for start, end in zip(chunk_starts, chunk_ends):
            chunk = df.iloc[start:end].copy()
            # Only add chunks that are long enough to be useful
            if len(chunk) > embargo_candles * 2: 
                train_dfs.append(chunk.reset_index(drop=True))
            else:
                print(f"⚠️ Discarded a {sym} chunk of length {len(chunk)} (too short, needs > {embargo_candles * 2} to cover embargoes).")
                
    print(f"✅ Generated {len(train_dfs)} purged training chunks.")
    return train_dfs, val_dfs_dict
