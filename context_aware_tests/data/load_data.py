"""
Data loading utilities for context-aware market state testing
"""

import pandas as pd
import os
import sys
from typing import Optional, List
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.data_handler import DataHandler
from config import get_data_path, DATA_FILES

class DataLoader:
    """
    Data loading utilities for market state testing
    """
    
    def __init__(self):
        """Initialize the data loader"""
        self.data_handler = DataHandler()
    
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame with OHLC data
        """
        filepath = get_data_path(filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"Loading data from {filepath}...")
        data = self.data_handler.load_csv(filepath)
        print(f"Loaded {len(data)} bars of data")
        
        return data
    
    def get_available_data_files(self) -> List[str]:
        """
        Get list of available data files
        
        Returns:
            List of available data file names
        """
        available_files = []
        
        for filename in DATA_FILES:
            filepath = get_data_path(filename)
            if os.path.exists(filepath):
                available_files.append(filename)
        
        return available_files
    
    def create_data_chunks(self, data: pd.DataFrame, 
                          window_size: int = 2000, 
                          step_size: int = 500) -> List[pd.DataFrame]:
        """
        Create overlapping data chunks for rolling window testing
        
        Args:
            data: Full dataset
            window_size: Number of bars per chunk
            step_size: Bars to slide forward between chunks
            
        Returns:
            List of data chunks
        """
        chunks = []
        total_bars = len(data)
        
        if total_bars < window_size:
            print(f"Warning: Data has {total_bars} bars, less than window size {window_size}")
            return [data]
        
        start_idx = 0
        chunk_count = 0
        
        while start_idx + window_size <= total_bars:
            end_idx = start_idx + window_size
            chunk = data.iloc[start_idx:end_idx].copy()
            chunk['chunk_id'] = chunk_count
            chunk['chunk_start'] = start_idx
            chunk['chunk_end'] = end_idx
            
            chunks.append(chunk)
            
            start_idx += step_size
            chunk_count += 1
        
        print(f"Created {len(chunks)} data chunks:")
        print(f"  Window size: {window_size} bars")
        print(f"  Step size: {step_size} bars")
        print(f"  Total data: {total_bars} bars")
        
        return chunks
    
    def get_chunk_by_index(self, data: pd.DataFrame, 
                          chunk_index: int,
                          window_size: int = 2000,
                          step_size: int = 500) -> Optional[pd.DataFrame]:
        """
        Get a specific chunk by index
        
        Args:
            data: Full dataset
            chunk_index: Index of the chunk to retrieve
            window_size: Number of bars per chunk
            step_size: Bars to slide forward between chunks
            
        Returns:
            DataFrame chunk or None if index out of range
        """
        total_bars = len(data)
        
        if total_bars < window_size:
            if chunk_index == 0:
                return data
            else:
                return None
        
        start_idx = chunk_index * step_size
        
        if start_idx + window_size > total_bars:
            return None
        
        end_idx = start_idx + window_size
        chunk = data.iloc[start_idx:end_idx].copy()
        chunk['chunk_id'] = chunk_index
        chunk['chunk_start'] = start_idx
        chunk['chunk_end'] = end_idx
        
        return chunk
    
    def get_chunk_by_time_range(self, data: pd.DataFrame,
                               start_date: str,
                               end_date: str) -> Optional[pd.DataFrame]:
        """
        Get a chunk by time range
        
        Args:
            data: Full dataset
            start_date: Start date (string format)
            end_date: End date (string format)
            
        Returns:
            DataFrame chunk or None if no data in range
        """
        # Convert dates if needed
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            mask = (data['date'] >= start_date) & (data['date'] <= end_date)
            chunk = data.loc[mask].copy()
            
            if len(chunk) > 0:
                chunk['chunk_id'] = 0
                chunk['chunk_start'] = chunk.index[0]
                chunk['chunk_end'] = chunk.index[-1]
                return chunk
        
        return None
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close']
        
        for col in required_columns:
            if col not in data.columns:
                print(f"Error: Missing required column '{col}'")
                return False
        
        if len(data) == 0:
            print("Error: Data is empty")
            return False
        
        print(f"Data validation passed: {len(data)} bars with required columns")
        return True
    
    def convert_to_renko(self, data: pd.DataFrame, 
                        use_atr: bool = True,
                        atr_period: int = 14,
                        atr_multiplier: float = 1.0,
                        fixed_brick_size: Optional[float] = None) -> pd.DataFrame:
        """
        Convert OHLC data to Renko bricks
        """
        # Import here to avoid circular import
        from .renko_converter import RenkoConverter
        renko_converter = RenkoConverter()
        print("🔄 Converting OHLC data to Renko bricks...")
        
        renko_data = renko_converter.convert_to_renko(
            data=data,
            brick_size=fixed_brick_size,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            use_atr=use_atr
        )
        
        if not renko_data.empty:
            # Get and display Renko statistics
            stats = renko_converter.get_renko_statistics(renko_data)
            print(f"📊 Renko Statistics:")
            print(f"  Total bricks: {stats['total_bricks']}")
            print(f"  Up bricks: {stats['up_bricks']} ({stats['up_percentage']:.1f}%)")
            print(f"  Down bricks: {stats['down_bricks']} ({stats['down_percentage']:.1f}%)")
            print(f"  Avg brick size: {stats['avg_brick_size']:.4f}")
        
        return renko_data 