#!/usr/bin/env python3
"""
Rolling Window Parameter Optimization Runner
Tests parameter combinations across time chunks to find optimal lookback periods
"""

import os
import sys
import pandas as pd
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chunk_runner.rolling_window_tester import RollingWindowTester
from data.load_data import DataLoader
from config import (
    WINDOW_SIZE, STEP_SIZE, 
    RENKO_USE_ATR, RENKO_ATR_PERIOD, RENKO_ATR_MULTIPLIER, RENKO_FIXED_BRICK_SIZE,
    TEST_CONFIGS, generate_run_id, get_output_dir
)
from grid_search.grid_search_optimizer import GridSearchOptimizer

def main():
    """Main function for running rolling window tests"""
    parser = argparse.ArgumentParser(description='Run rolling window parameter optimization')
    parser.add_argument('--config', type=str, default='default', 
                       choices=list(TEST_CONFIGS.keys()),
                       help='Test configuration to use')
    parser.add_argument('--run-id', type=str, default=None,
                       help='Custom run ID (auto-generated if not provided)')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Data file to use (overrides config)')
    parser.add_argument('--chunk-index', type=int, default=None,
                       help='Test specific chunk index')
    parser.add_argument('--max-combinations', type=int, default=None,
                       help='Maximum combinations to test')
    
    args = parser.parse_args()
    
    # Load configuration
    config = TEST_CONFIGS[args.config]
    run_id = args.run_id or generate_run_id(f"{args.config}_test")
    
    print(f"🚀 ROLLING WINDOW PARAMETER OPTIMIZATION")
    print(f"🆔 Run ID: {run_id}")
    print(f"⚙️  Configuration: {config['name']}")
    print(f"📝 Description: {config['description']}")
    print("="*70)
    
    # Initialize components with run ID
    data_loader = DataLoader()
    optimizer = GridSearchOptimizer(run_id=run_id)
    tester = RollingWindowTester(
        data_loader=data_loader,
        optimizer=optimizer,
        window_size=config['window_size'],
        step_size=config['step_size'],
        use_renko=config['use_renko'],
        renko_atr_period=config['renko_atr_period'],
        renko_atr_multiplier=config['renko_atr_multiplier']
    )
    
    # Load data
    data_file = args.data_file or "data/bitcoin_merged_data.csv"
    print(f"📊 Loading data from: {data_file}")
    
    try:
        data = data_loader.load_csv_data(data_file)
    except FileNotFoundError:
        print(f"❌ Data file not found: {data_file}")
        print("Available data files:")
        available_files = data_loader.get_available_data_files()
        for file in available_files:
            print(f"  - {file}")
        return
    
    # Run test based on arguments
    if args.chunk_index is not None:
        print(f"🎯 Testing specific chunk: {args.chunk_index}")
        tester.test_single_chunk(data, args.chunk_index, max_combinations=args.max_combinations)
    else:
        print("🔄 Running full rolling window test")
        tester.run_rolling_window_test(data, max_combinations=args.max_combinations)
    
    print(f"\n✅ Test completed! Results saved to: {get_output_dir(run_id)}")

if __name__ == "__main__":
    main() 