"""
Market State Classifier for detecting trend vs. consolidation periods
Uses ATR, ADX, Moving Averages, and Range analysis to classify market states
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators.atr import calculate_atr, get_atr_percentile
from indicators.adx import calculate_adx
from indicators.ma_utils import (
    calculate_moving_averages, 
    calculate_ma_spread, 
    calculate_price_slope,
    calculate_range_compression,
    get_range_percentile,
    get_slope_threshold
)
from config import (
    ADX_TREND_THRESHOLD,
    ATR_PERCENTILE,
    MA_SPREAD_THRESHOLD,
    SLOPE_MULTIPLIER,
    RANGE_PERCENTILE,
    TREND_SCORE_MIN
)

class MarketStateClassifier:
    """
    Market State Classifier for detecting trend vs. consolidation periods
    Uses ATR, ADX, Moving Averages, and Range analysis to classify market states
    """
    
    def __init__(self):
        """Initialize the market state classifier"""
        pass
    
    def classify_market_state(self, 
                            data: pd.DataFrame,
                            atr_period: int = 14,
                            adx_period: int = 14,
                            ma_periods: Tuple[int, int, int] = (20, 50, 200),
                            range_lookback: int = 20,
                            debug: bool = False) -> pd.Series:
        """
        Classify market state as 'trend' or 'consolidation' using trend score system
        
        Args:
            data: DataFrame with OHLC data
            atr_period: ATR lookback period
            adx_period: ADX lookback period
            ma_periods: Tuple of MA periods (short, medium, long)
            range_lookback: Range analysis lookback period
            debug: Whether to print debug information
            
        Returns:
            Series with market state classifications ('trend' or 'consolidation')
        """
        if data.empty:
            return pd.Series()
        
        result = data.copy()
        
        # Calculate indicators using modular functions
        result['atr'] = calculate_atr(data, atr_period)
        result['adx'] = calculate_adx(data, adx_period)
        result = calculate_moving_averages(result, ma_periods)
        result['range_compression'] = calculate_range_compression(data, range_lookback)
        
        # Calculate additional trend indicators
        short_ma, medium_ma, long_ma = ma_periods
        result['ma_spread'] = calculate_ma_spread(data, short_ma, long_ma)
        result['price_slope'] = calculate_price_slope(data, periods=5)
        
        # Initialize market state
        result['market_state'] = 'consolidation'
        
        # Trend Score System (looser thresholds)
        for i in range(len(result)):
            if i < max(atr_period, adx_period, range_lookback):
                continue
                
            trend_score = 0
            
            # 1. ADX trend strength (looser threshold)
            if result['adx'].iloc[i] > ADX_TREND_THRESHOLD:
                trend_score += 1
            
            # 2. ATR volatility (use percentile instead of fixed threshold)
            atr_percentile = get_atr_percentile(data.iloc[max(0, i-50):i+1], atr_period, ATR_PERCENTILE)
            if len(atr_percentile) > 0 and result['atr'].iloc[i] > atr_percentile.iloc[-1]:
                trend_score += 1
            
            # 3. MA alignment (trending MAs)
            if abs(result['ma_spread'].iloc[i]) > MA_SPREAD_THRESHOLD:
                trend_score += 1
            
            # 4. Price slope (directional movement)
            slope_threshold = get_slope_threshold(data.iloc[max(0, i-20):i+1], 20, SLOPE_MULTIPLIER)
            if len(slope_threshold) > 0 and abs(result['price_slope'].iloc[i]) > slope_threshold.iloc[-1]:
                trend_score += 1
            
            # 5. Range expansion (not in tight consolidation)
            range_percentile = get_range_percentile(data.iloc[max(0, i-20):i+1], range_lookback, RANGE_PERCENTILE)
            if len(range_percentile) > 0 and result['range_compression'].iloc[i] > range_percentile.iloc[-1]:
                trend_score += 1
            
            # Classify based on trend score (need at least 2 out of 5)
            if trend_score >= TREND_SCORE_MIN:
                result.loc[result.index[i], 'market_state'] = 'trend'
            
            # Debug logging for last few bars
            if debug and i >= len(result) - 5:
                adx_val = result['adx'].iloc[i]
                atr_val = result['atr'].iloc[i]
                ma_spread = result['ma_spread'].iloc[i]
                slope = result['price_slope'].iloc[i]
                close = result['close'].iloc[i]
                state = result['market_state'].iloc[i]
                
                print(f"Bar {i}: ADX={adx_val:.2f}, ATR={atr_val:.4f}, "
                      f"MA_Spread={ma_spread:.4f}, Slope={slope:.4f}, "
                      f"Close={close:.2f}, Trend_Score={trend_score}, State={state}")
        
        # Sanity check: Manual trend detection
        if len(result) >= 20:
            price_change = abs(result['close'].iloc[-1] - result['close'].iloc[-20]) / result['close'].iloc[-20]
            if price_change > 0.02:  # 2% price movement
                if debug:
                    print(f"Manual trend movement detected: {price_change:.2%} price change over 20 bars")
        
        return result['market_state']
    
    def get_classification_metrics(self, data: pd.DataFrame, state_series: pd.Series) -> Dict:
        """
        Get metrics about the classification results
        
        Args:
            data: Original data
            state_series: Series with market state classifications
            
        Returns:
            Dictionary with classification metrics
        """
        if state_series.empty:
            return {}
        
        total_bars = len(state_series)
        trend_bars = len(state_series[state_series == 'trend'])
        consolidation_bars = len(state_series[state_series == 'consolidation'])
        
        # Calculate state transitions
        state_changes = (state_series != state_series.shift()).sum()
        transition_ratio = state_changes / total_bars if total_bars > 0 else 0
        
        return {
            'total_bars': total_bars,
            'trend_bars': trend_bars,
            'consolidation_bars': consolidation_bars,
            'trend_pct': trend_bars / total_bars * 100 if total_bars > 0 else 0,
            'consolidation_pct': consolidation_bars / total_bars * 100 if total_bars > 0 else 0,
            'state_changes': state_changes,
            'transition_ratio': transition_ratio
        } 