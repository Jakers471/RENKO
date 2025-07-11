"""
Average Directional Index (ADX) calculation utilities
"""

import pandas as pd
import numpy as np
from typing import Tuple

def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX)
    
    Args:
        data: DataFrame with OHLC data
        period: ADX lookback period
        
    Returns:
        Series with ADX values
    """
    # Calculate +DM and -DM
    high_diff = data['high'] - data['high'].shift()
    low_diff = data['low'].shift() - data['low']
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate smoothed values
    tr_smooth = pd.Series(true_range).rolling(window=period, min_periods=1).mean()
    plus_di = pd.Series(plus_dm).rolling(window=period, min_periods=1).mean() / tr_smooth * 100
    minus_di = pd.Series(minus_dm).rolling(window=period, min_periods=1).mean() / tr_smooth * 100
    
    # Calculate DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=period, min_periods=1).mean()
    
    return adx

def calculate_di_components(data: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate +DI and -DI components
    
    Args:
        data: DataFrame with OHLC data
        period: ADX lookback period
        
    Returns:
        Tuple of (+DI, -DI) Series
    """
    # Calculate +DM and -DM
    high_diff = data['high'] - data['high'].shift()
    low_diff = data['low'].shift() - data['low']
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate smoothed values
    tr_smooth = pd.Series(true_range).rolling(window=period, min_periods=1).mean()
    plus_di = pd.Series(plus_dm).rolling(window=period, min_periods=1).mean() / tr_smooth * 100
    minus_di = pd.Series(minus_dm).rolling(window=period, min_periods=1).mean() / tr_smooth * 100
    
    return plus_di, minus_di

def is_trending(data: pd.DataFrame, period: int = 14, threshold: float = 25.0) -> pd.Series:
    """
    Check if ADX indicates trending market
    
    Args:
        data: DataFrame with OHLC data
        period: ADX lookback period
        threshold: ADX threshold for trend detection
        
    Returns:
        Boolean Series indicating if market is trending
    """
    adx = calculate_adx(data, period)
    return adx > threshold

def get_trend_strength(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Get trend strength based on ADX values
    
    Args:
        data: DataFrame with OHLC data
        period: ADX lookback period
        
    Returns:
        Series with trend strength categories: 'weak', 'moderate', 'strong'
    """
    adx = calculate_adx(data, period)
    
    trend_strength = pd.Series('weak', index=adx.index)
    trend_strength[adx > 25] = 'strong'
    trend_strength[(adx > 15) & (adx <= 25)] = 'moderate'
    
    return trend_strength 