#!/usr/bin/env python3
"""
Trading Strategy Backtesting System
===================================

A comprehensive system for backtesting Renko-based trading strategies.

Usage:
    python main.py [--data-file DATA_FILE] [--strategy STRATEGY] [--iterations ITERATIONS]

Example:
    python main.py --data-file "data/sample_data.csv" --strategy "breakout" --iterations 100
"""

import argparse
import os
import sys
from typing import Dict, List

# Import our modules
from data_handler import DataHandler
from csv_merger import CSVMerger
from renko_converter import RenkoConverter
from trading_strategy import RenkoBreakoutStrategy, RenkoMeanReversionStrategy
from rthe_strategy import RTHEStrategy
from backtest_engine import BacktestEngine
from results_analyzer import ResultsAnalyzer

def main():
    """Main function to run the trading backtesting system"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trading Strategy Backtesting System')
    parser.add_argument('--data-file', type=str, default='merged_ohlc_data.csv',
                       help='Path to CSV data file (or use --merge-csvs to merge multiple files)')
    parser.add_argument('--merge-csvs', action='store_true',
                       help='Merge multiple CSV files in data directory before backtesting')
    parser.add_argument('--csv-pattern', type=str, default='*.csv',
                       help='Pattern to match CSV files when merging (e.g., "data_*.csv")')
    parser.add_argument('--strategy', type=str, default='rthe',
                       choices=['breakout', 'mean_reversion', 'rthe'],
                       help='Trading strategy to use')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of optimization iterations')
    parser.add_argument('--brick-size', type=float, default=None,
                       help='Renko brick size (if None, will be calculated automatically)')
    parser.add_argument('--initial-capital', type=float, default=10000,
                       help='Initial capital for backtesting')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRADING STRATEGY BACKTESTING SYSTEM")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing data handler...")
    data_handler = DataHandler()
    
    # Handle CSV merging if requested
    if args.merge_csvs:
        print("\n1a. Merging multiple CSV files...")
        merger = CSVMerger()
        merged_data = merger.load_and_merge_csvs(
            file_pattern=args.csv_pattern,
            output_file=args.data_file,
            sort_by_date=True,
            remove_duplicates=True,
            fill_gaps=False
        )
        
        if merged_data.empty:
            print("No data found to merge. Please check your CSV files.")
            return
        
        # Use the merged data directly
        ohlc_data = merged_data
        print(f"Using merged data with {len(ohlc_data)} candlestick bars")
        
    else:
        # Load single data file
        data_file_path = os.path.join("data", args.data_file)
        if not os.path.exists(data_file_path):
            print(f"Data file {data_file_path} not found.")
            print("Use --merge-csvs to merge multiple CSV files, or place your data file in the data/ directory.")
            return
        
        print(f"\n2. Loading data from {args.data_file}...")
        try:
            ohlc_data = data_handler.load_csv(args.data_file)
            print(f"Loaded {len(ohlc_data)} candlestick bars")
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    
    # Convert to Renko
    print("\n3. Converting OHLC to Renko bricks...")
    if args.brick_size is None:
        # Calculate optimal brick size
        renko_converter = RenkoConverter(brick_size=1.0)  # Temporary size
        optimal_brick_size = renko_converter.get_optimal_brick_size(ohlc_data, method='atr')
        print(f"Calculated optimal brick size: {optimal_brick_size:.4f}")
    else:
        optimal_brick_size = args.brick_size
        print(f"Using specified brick size: {optimal_brick_size:.4f}")
    
    renko_converter = RenkoConverter(optimal_brick_size)
    renko_data = renko_converter.convert_to_renko(ohlc_data)
    
    if renko_data.empty:
        print("Error: No Renko bricks generated. Check your data and brick size.")
        return
    
    print(f"Generated {len(renko_data)} Renko bricks")
    
    # Analyze Renko statistics
    renko_stats = renko_converter.analyze_renko_stats(renko_data)
    print(f"Renko stats: {renko_stats}")
    
    # Initialize backtesting engine
    print(f"\n4. Initializing backtesting engine with ${args.initial_capital:,.2f} initial capital...")
    backtest_engine = BacktestEngine(initial_capital=args.initial_capital)
    
    # Select strategy and parameters
    print(f"\n5. Setting up {args.strategy} strategy...")
    
    if args.strategy == 'breakout':
        strategy_class = RenkoBreakoutStrategy
        parameter_ranges = {
            'consecutive_bricks': [2, 3, 4, 5],
            'stop_loss_bricks': [1, 2, 3],
            'take_profit_bricks': [3, 4, 5, 6],
            'min_bricks_for_trend': [2, 3]
        }
    elif args.strategy == 'mean_reversion':
        strategy_class = RenkoMeanReversionStrategy
        parameter_ranges = {
            'lookback_period': [5, 10, 15, 20],
            'oversold_threshold': [2, 3, 4],
            'overbought_threshold': [2, 3, 4]
        }
    else:  # rthe
        strategy_class = RTHEStrategy
        # Use brick size relative to the optimal brick size
        optimal_brick_size = optimal_brick_size
        parameter_ranges = {
            'brick_size': [optimal_brick_size * 0.5, optimal_brick_size * 0.75, optimal_brick_size, optimal_brick_size * 1.25],
            'tsl_offset': [optimal_brick_size * 0.01, optimal_brick_size * 0.02, optimal_brick_size * 0.03, optimal_brick_size * 0.05],
            'hedge_size_ratio': [0.3, 0.5, 0.7],
            'min_bricks_for_trend': [2, 3, 4]
        }
    
    # Run parameter optimization
    print(f"\n6. Running {args.iterations} optimization iterations...")
    results = backtest_engine.run_parameter_optimization(
        strategy_class=strategy_class,
        data=renko_data,
        parameter_ranges=parameter_ranges,
        max_iterations=args.iterations
    )
    
    if not results:
        print("No results generated. Check your strategy parameters.")
        return
    
    # Analyze results
    print("\n7. Analyzing results...")
    analyzer = ResultsAnalyzer()
    
    # Display summary statistics
    analyzer.display_summary_stats(results, top_n=10)
    
    # Get best result
    best_result = max(results, key=lambda x: x.profit_factor if x.profit_factor != float('inf') else 0)
    
    # Display detailed report for best result
    print("\n" + analyzer.generate_detailed_report(best_result))
    
    # Create visualizations
    print("\n8. Generating visualizations...")
    
    # Plot equity curves
    analyzer.plot_equity_curves(results, top_n=5, save_path='equity_curves.png')
    
    # Plot performance distribution
    analyzer.plot_performance_distribution(results, save_path='performance_distribution.png')
    
    # Plot parameter heatmap (for first two parameters)
    param_names = list(parameter_ranges.keys())
    if len(param_names) >= 2:
        analyzer.plot_parameter_heatmap(
            results, 
            param_names[0], 
            param_names[1], 
            metric='profit_factor',
            save_path='parameter_heatmap.png'
        )
    
    # Interactive equity curve for best result
    analyzer.plot_interactive_equity_curve(best_result, save_path='best_result_interactive.html')
    
    # Save results to CSV
    print("\n9. Saving results...")
    analyzer.save_results_to_csv(results, 'backtest_results.csv')
    
    print("\n" + "="*60)
    print("BACKTESTING COMPLETE!")
    print("="*60)
    print(f"Results saved to:")
    print("- backtest_results.csv (all results)")
    print("- equity_curves.png (equity curves)")
    print("- performance_distribution.png (performance distributions)")
    print("- parameter_heatmap.png (parameter heatmap)")
    print("- best_result_interactive.html (interactive equity curve)")
    
    # Quick summary
    print(f"\nQuick Summary:")
    print(f"- Best Profit Factor: {best_result.profit_factor:.3f}")
    print(f"- Best Total Return: {best_result.total_return*100:.2f}%")
    print(f"- Best Sharpe Ratio: {best_result.sharpe_ratio:.3f}")
    print(f"- Best Parameters: {best_result.parameters}")

def run_quick_test():
    """Run a quick test with merged CSV data"""
    print("Running quick test with merged CSV data...")
    
    # Check if we have CSV files to merge
    merger = CSVMerger()
    merged_data = merger.load_and_merge_csvs(
        file_pattern="*.csv",
        output_file="merged_ohlc_data.csv",
        sort_by_date=True,
        remove_duplicates=True,
        fill_gaps=False
    )
    
    if merged_data.empty:
        print("No CSV files found in data/ directory.")
        print("Please place your CSV files in the data/ directory and run again.")
        return
    
    # Convert to Renko
    renko_converter = RenkoConverter(brick_size=1.0)
    optimal_brick_size = renko_converter.get_optimal_brick_size(merged_data)
    renko_converter = RenkoConverter(optimal_brick_size)
    renko_data = renko_converter.convert_to_renko(merged_data)
    
    # Run single backtest
    strategy = RenkoBreakoutStrategy(consecutive_bricks=3, stop_loss_bricks=2, take_profit_bricks=4)
    strategy.set_brick_size(optimal_brick_size)
    
    backtest_engine = BacktestEngine(initial_capital=10000)
    result = backtest_engine.run_single_backtest(strategy, renko_data)
    
    # Display results
    analyzer = ResultsAnalyzer()
    print(analyzer.generate_detailed_report(result))
    
    print("Quick test completed!")

if __name__ == "__main__":
    # Check if running in test mode
    if len(sys.argv) == 1:
        print("No arguments provided. Running quick test...")
        run_quick_test()
    else:
        main() 