"""
Average True Range (ATR) calculation utilities
"""

import pandas as pd
import numpy as np
from typing import Union

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        data: DataFrame with OHLC data
        period: ATR lookback period
        
    Returns:
        Series with ATR values
    """
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period, min_periods=1).mean()
    
    return atr

def get_atr_percentile(data: pd.DataFrame, period: int = 14, percentile: int = 50) -> pd.Series:
    """
    Calculate ATR percentile for dynamic threshold comparison
    
    Args:
        data: DataFrame with OHLC data
        period: ATR lookback period
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Series with ATR percentile values
    """
    atr = calculate_atr(data, period)
    
    # Calculate rolling percentile
    atr_percentile = atr.rolling(window=50, min_periods=1).quantile(percentile / 100)
    
    return atr_percentile

def is_volatile(data: pd.DataFrame, period: int = 14, threshold_percentile: int = 50) -> pd.Series:
    """
    Check if current ATR is above the threshold percentile
    
    Args:
        data: DataFrame with OHLC data
        period: ATR lookback period
        threshold_percentile: Percentile threshold for volatility
        
    Returns:
        Boolean Series indicating if current ATR is above threshold
    """
    atr = calculate_atr(data, period)
    atr_threshold = get_atr_percentile(data, period, threshold_percentile)
    
    return atr > atr_threshold 