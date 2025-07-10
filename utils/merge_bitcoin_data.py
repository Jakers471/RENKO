#!/usr/bin/env python3
"""
Specialized Bitcoin CSV merger for TradingView exports
"""

import pandas as pd
import os
import glob
from typing import List

def merge_bitcoin_csvs():
    """Merge Bitcoin CSV files from TradingView"""
    
    print("=== BITCOIN CSV MERGER ===")
    
    # Find all Bitcoin CSV files
    data_dir = "data"
    bitcoin_files = glob.glob(os.path.join(data_dir, "CRYPTO_BTCUSD*.csv"))
    
    if not bitcoin_files:
        print("No Bitcoin CSV files found!")
        return None
    
    print(f"Found {len(bitcoin_files)} Bitcoin CSV files:")
    for file in bitcoin_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load and process each file
    all_dataframes = []
    
    for i, file_path in enumerate(bitcoin_files):
        print(f"\nProcessing file {i+1}/{len(bitcoin_files)}: {os.path.basename(file_path)}")
        
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            print(f"  Original columns: {list(df.columns)}")
            print(f"  Original rows: {len(df)}")
            
            # Handle time column (rename to date)
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'date'})
            
            # Convert date format and handle timezones
            # Check if we have Unix timestamps (numeric) or ISO dates (string)
            if df['date'].dtype in ['int64', 'float64']:
                # Unix timestamp - convert from seconds to datetime
                df['date'] = pd.to_datetime(df['date'], unit='s', utc=True)
            else:
                # ISO8601 date string
                df['date'] = pd.to_datetime(df['date'], utc=True)
            
            # Remove rows with invalid data
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Ensure valid OHLC relationships
            df = df[
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ]
            
            all_dataframes.append(df)
            print(f"  Cleaned rows: {len(df)}")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    if not all_dataframes:
        print("No valid data found!")
        return None
    
    # Merge all dataframes
    print(f"\nMerging {len(all_dataframes)} dataframes...")
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by date
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    # Remove duplicates
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['date'], keep='first')
    final_count = len(merged_df)
    
    if initial_count != final_count:
        print(f"Removed {initial_count - final_count} duplicate timestamps")
    
    # Save merged data
    output_file = os.path.join(data_dir, "bitcoin_merged_data.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n✅ MERGE COMPLETE!")
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Price range: ${merged_df['low'].min():.2f} - ${merged_df['high'].max():.2f}")
    
    # Show sample data
    print(f"\nFirst 5 rows:")
    print(merged_df[['date', 'open', 'high', 'low', 'close']].head())
    
    return merged_df

if __name__ == "__main__":
    merge_bitcoin_csvs() 