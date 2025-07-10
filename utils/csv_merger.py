import pandas as pd
import os
import glob
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

class CSVMerger:
    """Merges multiple CSV files with OHLC data, handling overlaps and gaps"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.merged_data = None
        
    def load_and_merge_csvs(self, 
                           file_pattern: str = "*.csv",
                           output_file: str = "merged_data.csv",
                           sort_by_date: bool = True,
                           remove_duplicates: bool = True,
                           fill_gaps: bool = False) -> pd.DataFrame:
        """
        Load and merge multiple CSV files
        
        Args:
            file_pattern: Pattern to match CSV files (e.g., "*.csv", "data_*.csv")
            output_file: Name of output merged file
            sort_by_date: Whether to sort by date
            remove_duplicates: Whether to remove duplicate timestamps
            fill_gaps: Whether to fill gaps with interpolation
            
        Returns:
            Merged DataFrame
        """
        print(f"Searching for CSV files matching pattern: {file_pattern}")
        
        # Find all CSV files
        search_path = os.path.join(self.data_dir, file_pattern)
        csv_files = glob.glob(search_path)
        
        if not csv_files:
            print(f"No CSV files found matching pattern: {search_path}")
            return pd.DataFrame()
        
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  - {os.path.basename(file)}")
        
        # Load and process each file
        all_dataframes = []
        
        for i, file_path in enumerate(csv_files):
            print(f"\nProcessing file {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
            
            try:
                df = self._load_single_csv(file_path)
                if not df.empty:
                    all_dataframes.append(df)
                    print(f"  Loaded {len(df)} rows")
                else:
                    print(f"  Warning: Empty or invalid file")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        
        if not all_dataframes:
            print("No valid data found in any CSV files")
            return pd.DataFrame()
        
        # Merge all dataframes
        print(f"\nMerging {len(all_dataframes)} dataframes...")
        merged_df = self._merge_dataframes(all_dataframes, sort_by_date, remove_duplicates)
        
        if merged_df.empty:
            print("No data after merging")
            return merged_df
        
        # Fill gaps if requested
        if fill_gaps:
            print("Filling gaps in data...")
            merged_df = self._fill_gaps(merged_df)
        
        # Save merged data
        output_path = os.path.join(self.data_dir, output_file)
        merged_df.to_csv(output_path, index=False)
        print(f"\nMerged data saved to: {output_path}")
        print(f"Final dataset: {len(merged_df)} rows")
        print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
        
        self.merged_data = merged_df
        return merged_df
    
    def _load_single_csv(self, file_path: str) -> pd.DataFrame:
        """Load a single CSV file with proper column standardization"""
        df = pd.read_csv(file_path)
        
        # Standardize column names
        column_mapping = {
            'Date': 'date', 'Time': 'time', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close',
            'VOLUME': 'volume', 'datetime': 'date', 'timestamp': 'date'
        }
        
        df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Remove rows with invalid data
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Ensure high >= low, high >= open, high >= close, etc.
        df = df[
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        ]
        
        return df
    
    def _merge_dataframes(self, 
                         dataframes: List[pd.DataFrame], 
                         sort_by_date: bool = True,
                         remove_duplicates: bool = True) -> pd.DataFrame:
        """Merge multiple dataframes, handling overlaps"""
        
        if not dataframes:
            return pd.DataFrame()
        
        # Concatenate all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        if merged_df.empty:
            return merged_df
        
        # Sort by date if requested
        if sort_by_date and 'date' in merged_df.columns:
            merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        # Remove duplicates if requested
        if remove_duplicates and 'date' in merged_df.columns:
            initial_count = len(merged_df)
            merged_df = merged_df.drop_duplicates(subset=['date'], keep='first')
            final_count = len(merged_df)
            if initial_count != final_count:
                print(f"Removed {initial_count - final_count} duplicate timestamps")
        
        return merged_df
    
    def _fill_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in data using interpolation"""
        if df.empty or 'date' not in df.columns:
            return df
        
        # Set date as index for resampling
        df = df.set_index('date')
        
        # Resample to regular intervals (assuming hourly data)
        # You can change this based on your data frequency
        resampled = df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum' if 'volume' in df.columns else 'first'
        })
        
        # Forward fill missing values
        resampled = resampled.fillna(method='ffill')
        
        # Reset index
        resampled = resampled.reset_index()
        
        return resampled
    
    def get_data_summary(self) -> Dict:
        """Get summary of merged data"""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        df = self.merged_data
        
        summary = {
            'total_rows': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['date'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'price_range': {
                'min_low': df['low'].min(),
                'max_high': df['high'].max(),
                'avg_close': df['close'].mean()
            },
            'data_quality': {
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_dates': len(df) - len(df.drop_duplicates(subset=['date']))
            }
        }
        
        return summary
    
    def validate_data_continuity(self, max_gap_hours: int = 24) -> Dict:
        """Validate data continuity and identify gaps"""
        if self.merged_data is None or self.merged_data.empty:
            return {}
        
        df = self.merged_data.copy()
        df = df.sort_values('date')
        
        # Calculate time differences
        df['time_diff'] = df['date'].diff()
        
        # Find gaps larger than max_gap_hours
        gaps = df[df['time_diff'] > pd.Timedelta(hours=max_gap_hours)]
        
        gap_info = []
        for idx, row in gaps.iterrows():
            gap_info.append({
                'gap_start': (row['date'] - row['time_diff']).strftime('%Y-%m-%d %H:%M:%S'),
                'gap_end': row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                'gap_duration_hours': row['time_diff'].total_seconds() / 3600
            })
        
        return {
            'total_gaps': len(gap_info),
            'gaps': gap_info,
            'max_gap_hours': max_gap_hours
        }
    
    def export_subset(self, 
                     start_date: str = None, 
                     end_date: str = None,
                     output_file: str = "subset_data.csv") -> pd.DataFrame:
        """Export a subset of the merged data"""
        if self.merged_data is None or self.merged_data.empty:
            print("No merged data available")
            return pd.DataFrame()
        
        df = self.merged_data.copy()
        
        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['date'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['date'] <= end_dt]
        
        if df.empty:
            print("No data in specified date range")
            return df
        
        # Save subset
        output_path = os.path.join(self.data_dir, output_file)
        df.to_csv(output_path, index=False)
        print(f"Subset exported to: {output_path}")
        print(f"Subset contains {len(df)} rows")
        
        return df

def main():
    """Example usage of CSV merger"""
    merger = CSVMerger()
    
    # Merge all CSV files in data directory
    merged_data = merger.load_and_merge_csvs(
        file_pattern="*.csv",
        output_file="merged_ohlc_data.csv",
        sort_by_date=True,
        remove_duplicates=True,
        fill_gaps=False
    )
    
    if not merged_data.empty:
        # Print summary
        summary = merger.get_data_summary()
        print("\nData Summary:")
        print(f"Total rows: {summary['total_rows']}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Price range: {summary['price_range']['min_low']:.2f} - {summary['price_range']['max_high']:.2f}")
        
        # Check for gaps
        gaps = merger.validate_data_continuity()
        if gaps['total_gaps'] > 0:
            print(f"\nFound {gaps['total_gaps']} gaps in data:")
            for gap in gaps['gaps'][:5]:  # Show first 5 gaps
                print(f"  Gap: {gap['gap_start']} to {gap['gap_end']} ({gap['gap_duration_hours']:.1f} hours)")
        else:
            print("\nNo significant gaps found in data")

if __name__ == "__main__":
    main() 