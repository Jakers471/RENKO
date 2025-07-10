import pandas as pd
import numpy as np
from typing import List, Tuple

class RenkoConverter:
    """Converts OHLC candlestick data to Renko bricks"""
    
    def __init__(self, brick_size: float):
        """
        Initialize Renko converter
        
        Args:
            brick_size: Size of each Renko brick (in price units)
        """
        self.brick_size = brick_size
        
    def convert_to_renko(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert OHLC data to Renko bricks
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with Renko brick data
        """
        if df.empty:
            return pd.DataFrame()
        
        # Start with first close price
        current_price = df['close'].iloc[0]
        renko_data = []
        
        # Track current brick direction and price
        current_direction = 0  # 0: neutral, 1: up, -1: down
        brick_open = current_price
        brick_high = current_price
        brick_low = current_price
        
        for idx, row in df.iterrows():
            high = row['high']
            low = row['low']
            close = row['close']
            date = row.get('date', idx)
            
            # Check if price moved enough to create new bricks
            while True:
                if current_direction >= 0:  # Neutral or up trend
                    # Check for upward brick
                    if close >= current_price + self.brick_size:
                        # Create up brick
                        renko_data.append({
                            'date': date,
                            'open': current_price,
                            'high': current_price + self.brick_size,
                            'low': current_price,
                            'close': current_price + self.brick_size,
                            'direction': 1,
                            'brick_size': self.brick_size
                        })
                        current_price += self.brick_size
                        current_direction = 1
                        continue
                    
                    # Check for downward brick (reversal)
                    if close <= current_price - self.brick_size:
                        # Create down brick
                        renko_data.append({
                            'date': date,
                            'open': current_price,
                            'high': current_price,
                            'low': current_price - self.brick_size,
                            'close': current_price - self.brick_size,
                            'direction': -1,
                            'brick_size': self.brick_size
                        })
                        current_price -= self.brick_size
                        current_direction = -1
                        continue
                
                else:  # Down trend
                    # Check for downward brick
                    if close <= current_price - self.brick_size:
                        # Create down brick
                        renko_data.append({
                            'date': date,
                            'open': current_price,
                            'high': current_price,
                            'low': current_price - self.brick_size,
                            'close': current_price - self.brick_size,
                            'direction': -1,
                            'brick_size': self.brick_size
                        })
                        current_price -= self.brick_size
                        continue
                    
                    # Check for upward brick (reversal)
                    if close >= current_price + self.brick_size:
                        # Create up brick
                        renko_data.append({
                            'date': date,
                            'open': current_price,
                            'high': current_price + self.brick_size,
                            'low': current_price,
                            'close': current_price + self.brick_size,
                            'direction': 1,
                            'brick_size': self.brick_size
                        })
                        current_price += self.brick_size
                        current_direction = 1
                        continue
                
                # No new brick created, break
                break
        
        if renko_data:
            return pd.DataFrame(renko_data)
        else:
            return pd.DataFrame()
    
    def get_optimal_brick_size(self, df: pd.DataFrame, method: str = 'atr') -> float:
        """
        Calculate optimal brick size based on price volatility
        
        Args:
            df: OHLC DataFrame
            method: 'atr' (Average True Range) or 'std' (Standard Deviation)
            
        Returns:
            Optimal brick size
        """
        if method == 'atr':
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            return atr * 0.5  # Use 50% of ATR as brick size
            
        elif method == 'std':
            # Use standard deviation of returns
            returns = df['close'].pct_change().dropna()
            std = returns.std()
            return df['close'].iloc[-1] * std * 2  # 2x standard deviation
            
        else:
            raise ValueError("Method must be 'atr' or 'std'")
    
    def analyze_renko_stats(self, renko_df: pd.DataFrame) -> dict:
        """
        Analyze Renko brick statistics
        
        Args:
            renko_df: Renko DataFrame
            
        Returns:
            Dictionary with statistics
        """
        if renko_df.empty:
            return {}
        
        stats = {
            'total_bricks': len(renko_df),
            'up_bricks': len(renko_df[renko_df['direction'] == 1]),
            'down_bricks': len(renko_df[renko_df['direction'] == -1]),
            'avg_brick_size': renko_df['brick_size'].mean(),
            'max_consecutive_up': 0,
            'max_consecutive_down': 0
        }
        
        # Calculate consecutive bricks
        current_up = 0
        current_down = 0
        max_up = 0
        max_down = 0
        
        for direction in renko_df['direction']:
            if direction == 1:
                current_up += 1
                current_down = 0
                max_up = max(max_up, current_up)
            else:
                current_down += 1
                current_up = 0
                max_down = max(max_down, current_down)
        
        stats['max_consecutive_up'] = max_up
        stats['max_consecutive_down'] = max_down
        
        return stats 