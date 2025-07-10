#!/usr/bin/env python3
"""
Simple test script to debug CSV merging
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.csv_merger import CSVMerger

def test_csv_merger():
    print("=== CSV MERGER TEST ===")
    
    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Creating {data_dir} directory...")
        os.makedirs(data_dir)
    
    # List files in data directory
    print(f"\nFiles in {data_dir}/ directory:")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("  No CSV files found!")
        print("  Please place your CSV files in the data/ directory")
        return
    
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        file_size = os.path.getsize(file_path)
        print(f"  - {file} ({file_size} bytes)")
    
    # Try to merge
    print(f"\nAttempting to merge {len(csv_files)} CSV files...")
    
    merger = CSVMerger()
    
    try:
        merged_data = merger.load_and_merge_csvs(
            file_pattern="*.csv",
            output_file="test_merged.csv",
            sort_by_date=True,
            remove_duplicates=True,
            fill_gaps=False
        )
        
        if not merged_data.empty:
            print(f"\n✅ SUCCESS! Merged data:")
            print(f"  - Total rows: {len(merged_data)}")
            print(f"  - Date range: {merged_data['date'].min()} to {merged_data['date'].max()}")
            print(f"  - Columns: {list(merged_data.columns)}")
            
            # Show first few rows
            print(f"\nFirst 5 rows:")
            print(merged_data.head())
            
        else:
            print("❌ No data after merging")
            
    except Exception as e:
        print(f"❌ Error during merging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_csv_merger() 