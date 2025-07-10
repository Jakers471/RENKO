"""
DataHandler: Handles loading of OHLCV and Renko data, including multi-timeframe support.

Multi-Timeframe Support:
- Place all CSVs for a symbol in data/{symbol}/
- Name each file {symbol}_{timeframe}.csv (e.g., BTCUSD_1h.csv)
- Use load_symbol_timeframes(symbol) to load all timeframes as a dict: {'1m': df1m, '1h': df1h, ...}
"""
import pandas as pd
import numpy as np
from typing import Optional, List
import os
import glob

class DataHandler:
    """Handles loading and preprocessing of OHLC data"""
    
    def __init__(self):
        self.data = None
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load OHLC data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            # Try different date parsing strategies
            df = pd.read_csv(file_path)
            
            # Check if we have a timestamp column
            timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                
                # Try parsing as Unix timestamp first
                try:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
                except (ValueError, TypeError):
                    # Try parsing as ISO format
                    try:
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                    except (ValueError, TypeError):
                        # Try parsing as regular date format
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                
                # Set as index
                df.set_index(timestamp_col, inplace=True)
                df.index.name = 'date'
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            # Convert to numeric
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values
            df.dropna(inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return pd.DataFrame()
    
    def load_symbol_timeframes(self, symbol: str):
        """Load all available timeframes for a symbol from data/{symbol}/ directory."""
        import os
        base_dir = os.path.join('data', symbol)
        if not os.path.isdir(base_dir):
            print(f"[WARN] Directory not found: {base_dir}")
            return {}
        
        timeframes = {}
        for file in glob.glob(os.path.join(base_dir, f"{symbol}_*.csv")):
            tf = file.split('_')[-1].replace('.csv', '')
            try:
                df = pd.read_csv(file, parse_dates=['date'])
                timeframes[tf] = df
                print(f"Loaded {file} ({len(df)} rows) as timeframe '{tf}'")
            except Exception as e:
                print(f"[ERROR] Could not load {file}: {e}")
        if not timeframes:
            print(f"[WARN] No timeframe CSVs found in {base_dir}")
        return timeframes
    
    def get_data_info(self) -> dict:
        """Get information about loaded data"""
        if self.data is None:
            return {}
        
        info = {
            'rows': len(self.data),
            'columns': list(self.data.columns),
            'date_range': f"{self.data.index.min()} to {self.data.index.max()}",
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        return info
    
    def resample_data(self, frequency: str = '1H') -> pd.DataFrame:
        """
        Resample data to different frequency
        
        Args:
            frequency: Pandas frequency string (e.g., '1H', '1D', '15T')
            
        Returns:
            Resampled DataFrame
        """
        if self.data is None:
            return pd.DataFrame()
        
        resampled = self.data.resample(frequency).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        
        return resampled.dropna()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators to the data
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate OHLC data for consistency
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        if df.empty:
            return False
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # Check for logical consistency
        logical_errors = (
            (df['high'] < df['low']).any() or
            (df['open'] > df['high']).any() or
            (df['open'] < df['low']).any() or
            (df['close'] > df['high']).any() or
            (df['close'] < df['low']).any()
        )
        
        if logical_errors:
            return False
        
        # Check for negative prices
        if (df[required_cols] < 0).any().any():
            return False
        
        return True 