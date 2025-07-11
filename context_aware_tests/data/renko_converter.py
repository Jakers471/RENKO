"""
Renko Chart Converter
Converts OHLC data to Renko bricks for context-aware testing
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple

class RenkoConverter:
    """
    Converts OHLC data to Renko bricks
    """
    
    def __init__(self):
        """Initialize the Renko converter"""
        pass
    
    def calculate_atr_based_brick_size(self, data: pd.DataFrame, atr_period: int = 14, 
                                     atr_multiplier: float = 1.0) -> float:
        """
        Calculate ATR-based brick size
        
        Args:
            data: DataFrame with OHLC data
            atr_period: ATR lookback period
            atr_multiplier: Multiplier for ATR value
            
        Returns:
            Brick size based on ATR
        """
        # Calculate ATR
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=atr_period, min_periods=1).mean()
        
        # Use average ATR as brick size
        brick_size = atr.mean() * atr_multiplier
        
        return brick_size
    
    def convert_to_renko(self, data: pd.DataFrame, 
                        brick_size: Optional[float] = None,
                        atr_period: int = 14,
                        atr_multiplier: float = 1.0,
                        use_atr: bool = True) -> pd.DataFrame:
        """
        Convert OHLC data to Renko bricks
        
        Args:
            data: DataFrame with OHLC data (must have open, high, low, close)
            brick_size: Fixed brick size (if not using ATR)
            atr_period: ATR lookback period for dynamic brick size
            atr_multiplier: Multiplier for ATR-based brick size
            use_atr: Whether to use ATR-based brick size
            
        Returns:
            DataFrame with Renko bricks
        """
        if data.empty:
            return pd.DataFrame()
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate brick size
        if use_atr:
            brick_size = self.calculate_atr_based_brick_size(data, atr_period, atr_multiplier)
        elif brick_size is None:
            # Use 1% of average price as default
            brick_size = data['close'].mean() * 0.01
        
        print(f"Renko brick size: {brick_size:.4f}")
        
        # Initialize Renko data
        renko_data = []
        current_open = data['close'].iloc[0]
        current_close = current_open
        current_direction = 0  # 0: neutral, 1: up, -1: down
        brick_count = 0
        
        # Process each OHLC bar
        for i, row in data.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            
            # Determine if we need to create new bricks
            while True:
                if current_direction == 0:  # Neutral - determine direction
                    if close > current_close + brick_size:
                        # Create up brick
                        renko_data.append({
                            'timestamp': i,
                            'renko_open': current_close,
                            'renko_close': current_close + brick_size,
                            'direction': 1,
                            'brick_count': brick_count,
                            'original_bar_index': i
                        })
                        current_open = current_close
                        current_close = current_close + brick_size
                        current_direction = 1
                        brick_count += 1
                    elif close < current_close - brick_size:
                        # Create down brick
                        renko_data.append({
                            'timestamp': i,
                            'renko_open': current_close,
                            'renko_close': current_close - brick_size,
                            'direction': -1,
                            'brick_count': brick_count,
                            'original_bar_index': i
                        })
                        current_open = current_close
                        current_close = current_close - brick_size
                        current_direction = -1
                        brick_count += 1
                    else:
                        # No new brick needed
                        break
                
                elif current_direction == 1:  # Up trend
                    if close > current_close + brick_size:
                        # Continue up trend
                        renko_data.append({
                            'timestamp': i,
                            'renko_open': current_close,
                            'renko_close': current_close + brick_size,
                            'direction': 1,
                            'brick_count': brick_count,
                            'original_bar_index': i
                        })
                        current_open = current_close
                        current_close = current_close + brick_size
                        brick_count += 1
                    elif close < current_close - brick_size:
                        # Reverse to down trend
                        renko_data.append({
                            'timestamp': i,
                            'renko_open': current_close,
                            'renko_close': current_close - brick_size,
                            'direction': -1,
                            'brick_count': brick_count,
                            'original_bar_index': i
                        })
                        current_open = current_close
                        current_close = current_close - brick_size
                        current_direction = -1
                        brick_count += 1
                    else:
                        # No new brick needed
                        break
                
                elif current_direction == -1:  # Down trend
                    if close < current_close - brick_size:
                        # Continue down trend
                        renko_data.append({
                            'timestamp': i,
                            'renko_open': current_close,
                            'renko_close': current_close - brick_size,
                            'direction': -1,
                            'brick_count': brick_count,
                            'original_bar_index': i
                        })
                        current_open = current_close
                        current_close = current_close - brick_size
                        brick_count += 1
                    elif close > current_close + brick_size:
                        # Reverse to up trend
                        renko_data.append({
                            'timestamp': i,
                            'renko_open': current_close,
                            'renko_close': current_close + brick_size,
                            'direction': 1,
                            'brick_count': brick_count,
                            'original_bar_index': i
                        })
                        current_open = current_close
                        current_close = current_close + brick_size
                        current_direction = 1
                        brick_count += 1
                    else:
                        # No new brick needed
                        break
        
        # Create DataFrame
        if not renko_data:
            print("Warning: No Renko bricks generated")
            return pd.DataFrame()
        
        renko_df = pd.DataFrame(renko_data)
        
        # Add OHLC columns for compatibility with existing indicators
        renko_df['open'] = renko_df['renko_open']
        renko_df['high'] = renko_df[['renko_open', 'renko_close']].max(axis=1)
        renko_df['low'] = renko_df[['renko_open', 'renko_close']].min(axis=1)
        renko_df['close'] = renko_df['renko_close']
        
        # Add volume if available (use 1 as default)
        if 'volume' in data.columns:
            # Map volume from original bars to Renko bricks
            renko_df['volume'] = 1  # Placeholder - could implement volume mapping
        else:
            renko_df['volume'] = 1
        
        print(f"Generated {len(renko_df)} Renko bricks from {len(data)} OHLC bars")
        print(f"Brick directions: Up={len(renko_df[renko_df['direction']==1])}, "
              f"Down={len(renko_df[renko_df['direction']==-1])}")
        
        return renko_df
    
    def get_renko_statistics(self, renko_data: pd.DataFrame) -> dict:
        """
        Get statistics about the Renko conversion
        
        Args:
            renko_data: DataFrame with Renko bricks
            
        Returns:
            Dictionary with Renko statistics
        """
        if renko_data.empty:
            return {}
        
        up_bricks = len(renko_data[renko_data['direction'] == 1])
        down_bricks = len(renko_data[renko_data['direction'] == -1])
        total_bricks = len(renko_data)
        
        # Calculate brick size statistics
        brick_sizes = abs(renko_data['renko_close'] - renko_data['renko_open'])
        
        return {
            'total_bricks': total_bricks,
            'up_bricks': up_bricks,
            'down_bricks': down_bricks,
            'up_percentage': up_bricks / total_bricks * 100 if total_bricks > 0 else 0,
            'down_percentage': down_bricks / total_bricks * 100 if total_bricks > 0 else 0,
            'avg_brick_size': brick_sizes.mean(),
            'min_brick_size': brick_sizes.min(),
            'max_brick_size': brick_sizes.max(),
            'brick_size_std': brick_sizes.std()
        } 