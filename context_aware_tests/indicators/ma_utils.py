"""
Moving Average utilities and related calculations
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

def calculate_moving_averages(data: pd.DataFrame, ma_periods: Tuple[int, int, int]) -> pd.DataFrame:
    """
    Calculate multiple moving averages
    
    Args:
        data: DataFrame with OHLC data
        ma_periods: Tuple of (short_ma, medium_ma, long_ma) periods
        
    Returns:
        DataFrame with MA columns added
    """
    result = data.copy()
    short_ma, medium_ma, long_ma = ma_periods
    
    result[f'ma_{short_ma}'] = data['close'].rolling(window=short_ma, min_periods=1).mean()
    result[f'ma_{medium_ma}'] = data['close'].rolling(window=medium_ma, min_periods=1).mean()
    result[f'ma_{long_ma}'] = data['close'].rolling(window=long_ma, min_periods=1).mean()
    
    return result

def calculate_ma_spread(data: pd.DataFrame, short_period: int, long_period: int) -> pd.Series:
    """
    Calculate moving average spread percentage
    
    Args:
        data: DataFrame with OHLC data
        short_period: Short MA period
        long_period: Long MA period
        
    Returns:
        Series with MA spread percentages
    """
    short_ma = data['close'].rolling(window=short_period, min_periods=1).mean()
    long_ma = data['close'].rolling(window=long_period, min_periods=1).mean()
    
    ma_spread = (short_ma - long_ma) / long_ma
    return ma_spread

def calculate_price_slope(data: pd.DataFrame, periods: int = 5) -> pd.Series:
    """
    Calculate price slope (rate of change)
    
    Args:
        data: DataFrame with OHLC data
        periods: Number of periods for slope calculation
        
    Returns:
        Series with price slope values
    """
    slope = data['close'].pct_change(periods=periods)
    return slope

def calculate_range_compression(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Calculate range compression/expansion analysis
    
    Args:
        data: DataFrame with OHLC data
        lookback: Range lookback period
        
    Returns:
        Series with range compression values
    """
    rolling_high = data['high'].rolling(window=lookback, min_periods=1).max()
    rolling_low = data['low'].rolling(window=lookback, min_periods=1).min()
    
    range_compression = (rolling_high - rolling_low) / data['close']
    
    return range_compression

def get_range_percentile(data: pd.DataFrame, lookback: int = 20, percentile: int = 70) -> pd.Series:
    """
    Calculate range percentile for dynamic threshold comparison
    
    Args:
        data: DataFrame with OHLC data
        lookback: Range lookback period
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Series with range percentile values
    """
    range_compression = calculate_range_compression(data, lookback)
    
    # Calculate rolling percentile
    range_percentile = range_compression.rolling(window=20, min_periods=1).quantile(percentile / 100)
    
    return range_percentile

def is_range_expanding(data: pd.DataFrame, lookback: int = 20, threshold_percentile: int = 70) -> pd.Series:
    """
    Check if current range is expanding above threshold percentile
    
    Args:
        data: DataFrame with OHLC data
        lookback: Range lookback period
        threshold_percentile: Percentile threshold for range expansion
        
    Returns:
        Boolean Series indicating if range is expanding
    """
    range_compression = calculate_range_compression(data, lookback)
    range_threshold = get_range_percentile(data, lookback, threshold_percentile)
    
    return range_compression > range_threshold

def calculate_ma_alignment(data: pd.DataFrame, ma_periods: Tuple[int, int, int]) -> pd.Series:
    """
    Calculate moving average alignment score
    
    Args:
        data: DataFrame with OHLC data
        ma_periods: Tuple of (short_ma, medium_ma, long_ma) periods
        
    Returns:
        Series with MA alignment scores (0-1, higher = better aligned)
    """
    result = calculate_moving_averages(data, ma_periods)
    short_ma, medium_ma, long_ma = ma_periods
    
    # Calculate alignment based on MA order
    short_ma_vals = result[f'ma_{short_ma}']
    medium_ma_vals = result[f'ma_{medium_ma}']
    long_ma_vals = result[f'ma_{long_ma}']
    
    # Check if MAs are in proper order (short > medium > long for uptrend, reverse for downtrend)
    uptrend_aligned = (short_ma_vals > medium_ma_vals) & (medium_ma_vals > long_ma_vals)
    downtrend_aligned = (short_ma_vals < medium_ma_vals) & (medium_ma_vals < long_ma_vals)
    
    # Calculate alignment score
    alignment_score = pd.Series(0.0, index=data.index)
    alignment_score[uptrend_aligned | downtrend_aligned] = 1.0
    
    return alignment_score

def get_slope_threshold(data: pd.DataFrame, lookback: int = 20, multiplier: float = 0.25) -> pd.Series:
    """
    Calculate dynamic slope threshold based on price volatility
    
    Args:
        data: DataFrame with OHLC data
        lookback: Lookback period for volatility calculation
        multiplier: Multiplier for standard deviation
        
    Returns:
        Series with dynamic slope thresholds
    """
    price_std = data['close'].rolling(window=lookback, min_periods=1).std()
    slope_threshold = multiplier * price_std
    
    return slope_threshold 