#!/usr/bin/env python3
"""
Test script for R.T.H.E. strategy debugging
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_handler import DataHandler
from utils.renko_converter import RenkoConverter
from strategies.rthe_strategy import RTHEStrategy
from engine.backtest_engine import BacktestEngine

def test_rthe_strategy():
    print("=== R.T.H.E. STRATEGY DEBUG TEST ===")
    
    # Load data
    data_handler = DataHandler()
    ohlc_data = data_handler.load_csv("data/bitcoin_merged_data.csv")
    print(f"Loaded {len(ohlc_data)} OHLC bars")
    
    # Convert to Renko
    renko_converter = RenkoConverter(brick_size=1.0)
    optimal_brick_size = renko_converter.get_optimal_brick_size(ohlc_data)
    print(f"Optimal brick size: {optimal_brick_size}")
    
    renko_converter = RenkoConverter(optimal_brick_size)
    renko_data = renko_converter.convert_to_renko(ohlc_data)
    print(f"Generated {len(renko_data)} Renko bricks")
    
    # Show first few bricks
    print("\nFirst 10 Renko bricks:")
    print(renko_data[['date', 'open', 'high', 'low', 'close', 'direction']].head(10))
    
    # Test R.T.H.E. strategy
    strategy = RTHEStrategy(
        brick_size=optimal_brick_size,
        tsl_offset=optimal_brick_size * 0.02,
        hedge_size_ratio=0.5,
        min_bricks_for_trend=2
    )
    
    print(f"\nR.T.H.E. Strategy Parameters:")
    print(f"  Brick Size: {strategy.brick_size}")
    print(f"  TSL Offset: {strategy.tsl_offset}")
    print(f"  Min Bricks for Trend: {strategy.min_bricks_for_trend}")
    
    # Run strategy manually for first 20 bricks
    print(f"\nRunning strategy on first 20 bricks...")
    
    for i in range(min(20, len(renko_data))):
        brick = renko_data.iloc[i]
        print(f"Brick {i}: Direction={brick['direction']}, Price={brick['close']:.2f}")
        
        strategy.execute_trade(renko_data, i)
        
        if strategy.core_position != 0:
            print(f"  -> Core position: {strategy.core_position}")
            print(f"  -> Entry price: {strategy.core_entry_price:.2f}")
            print(f"  -> TSL level: {strategy.tsl_level:.2f}")
        
        if strategy.hedge_position != 0:
            print(f"  -> Hedge position: {strategy.hedge_position}")
            print(f"  -> Hedge entry price: {strategy.hedge_entry_price:.2f}")
    
    # Check results
    print(f"\nStrategy Results:")
    print(f"  Core trades: {len(strategy.core_trades)}")
    print(f"  Hedge trades: {len(strategy.hedge_trades)}")
    print(f"  Hedge activations: {strategy.hedge_activations}")
    print(f"  TSL hits: {strategy.tsl_hits}")
    
    if strategy.core_trades:
        print(f"\nFirst core trade:")
        trade = strategy.core_trades[0]
        print(f"  Entry: {trade.entry_price:.2f}")
        print(f"  Exit: {trade.exit_price:.2f}")
        print(f"  PnL: {trade.pnl*100:.2f}%")
        print(f"  TSL Hit: {trade.tsl_hit}")
        print(f"  Hedge Activated: {trade.hedge_activated}")

if __name__ == "__main__":
    test_rthe_strategy() 