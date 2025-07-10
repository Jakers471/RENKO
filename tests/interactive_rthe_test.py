#!/usr/bin/env python3
"""
Interactive R.T.H.E. Strategy Test with Parameter Customization
"""

import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import scipy.stats as stats
from scipy.stats import norm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_handler import DataHandler
from utils.renko_converter import RenkoConverter
from strategies.rthe_strategy import RTHEStrategy
from engine.backtest_engine import BacktestEngine

class InteractiveRTHETest:
    def __init__(self):
        self.data_handler = DataHandler()
        self.renko_converter = None
        self.ohlc_data = None
        self.renko_data = None
        self.optimal_brick_size = None
        
    def load_data(self):
        """Load and prepare data"""
        print("Loading Bitcoin data...")
        
        # Check if merged data exists, if not run merger
        if not os.path.exists("data/bitcoin_merged_data.csv"):
            print("Merged data not found. Running Bitcoin CSV merger...")
            from utils.merge_bitcoin_data import merge_bitcoin_csvs
            merge_bitcoin_csvs()
        
        self.ohlc_data = self.data_handler.load_csv("data/bitcoin_merged_data.csv")
        print(f"Loaded {len(self.ohlc_data)} OHLC bars")
        
        # Calculate optimal brick size
        temp_converter = RenkoConverter(brick_size=1.0)
        self.optimal_brick_size = temp_converter.get_optimal_brick_size(self.ohlc_data)
        print(f"Optimal brick size: {self.optimal_brick_size:.2f}")
        
        # Convert to Renko
        self.renko_converter = RenkoConverter(self.optimal_brick_size)
        self.renko_data = self.renko_converter.convert_to_renko(self.ohlc_data)
        print(f"Generated {len(self.renko_data)} Renko bricks")
        
    def explain_parameters(self):
        """Explain System V1 Rules and Parameters"""
        print("\n" + "="*60)
        print("SYSTEM V1 RULES & PARAMETERS")
        print("="*60)
        
        print("🎯 SYSTEM V1 OVERVIEW:")
        print("A Renko-based trend-following system with moving average confirmation.")
        print("Built for robustness and adaptability across different market conditions.")
        print("Uses 20, 50, and 200-day SMAs for trend alignment and entry confirmation.")
        print()
        
        print("📊 CORE COMPONENTS:")
        print("1. Renko Brick Analysis - Price movement converted to bricks")
        print("2. Moving Average Analysis - 20, 50, 200 SMA trend alignment")
        print("3. MA-Based Entry Logic - Enters on MA crossover confirmation")
        print("4. Trend-Following Focus - Long entries with bullish MA alignment")
        print("5. Counter-Trend Toggle - Optional short entries when enabled")
        print("6. Hedge Protection - Reduces risk during reversals")
        print("7. Trailing Stop Loss - Manages position risk")
        print()
        
        params = {
            "brick_size": {
                "description": "Size of each Renko brick (price movement)",
                "default": f"{self.optimal_brick_size:.2f}",
                "effect": "Larger = fewer signals but stronger trends, Smaller = more signals but noise",
                "range": f"{self.optimal_brick_size * 0.5:.2f} - {self.optimal_brick_size * 2:.2f}",
                "recommendation": "Use optimal size for balance of signals vs noise"
            },
            "tsl_offset": {
                "description": "Trailing stop loss offset from brick boundaries",
                "default": f"{self.optimal_brick_size * 0.02:.2f}",
                "effect": "Larger = wider stops (less risk), Smaller = tighter stops (more risk)",
                "range": f"{self.optimal_brick_size * 0.01:.2f} - {self.optimal_brick_size * 0.05:.2f}",
                "recommendation": "2% of brick size for balanced risk management"
            },
            "hedge_size_ratio": {
                "description": "Ratio of hedge position size to core position",
                "default": "0.5",
                "effect": "Higher = more hedging protection, Lower = less hedging",
                "range": "0.1 - 1.0",
                "recommendation": "0.5 for balanced protection vs cost"
            },
            "min_bricks_for_trend": {
                "description": "Minimum consecutive bricks for trend confirmation",
                "default": "2",
                "effect": "Higher = stronger trend requirement, Lower = more entries",
                "range": "1 - 5",
                "recommendation": "2-3 bricks for trend strength"
            },
            "ma_20_period": {
                "description": "Period for 20-day Simple Moving Average",
                "default": "20",
                "effect": "Shorter = more responsive, Longer = smoother trend",
                "range": "10 - 50",
                "recommendation": "20 for short-term trend confirmation"
            },
            "ma_50_period": {
                "description": "Period for 50-day Simple Moving Average",
                "default": "50",
                "effect": "Medium-term trend filter and crossover signal",
                "range": "30 - 100",
                "recommendation": "50 for medium-term trend alignment"
            },
            "ma_200_period": {
                "description": "Period for 200-day Simple Moving Average",
                "default": "200",
                "effect": "Long-term trend filter and major trend direction",
                "range": "100 - 300",
                "recommendation": "200 for long-term trend confirmation"
            }
        }
        
        print("⚙️  CONFIGURABLE PARAMETERS:")
        for param, info in params.items():
            print(f"\n📊 {param.upper()}:")
            print(f"   Description: {info['description']}")
            print(f"   Default: {info['default']}")
            print(f"   Effect: {info['effect']}")
            print(f"   Range: {info['range']}")
            print(f"   Recommendation: {info['recommendation']}")
        
        print("\n📈 MA-BASED ENTRY RULES:")
        print("LONG ENTRIES (Always Enabled):")
        print("• 20 SMA crosses above 50 SMA (bullish crossover)")
        print("• Both 20 and 50 SMA are upsloping")
        print("• Price above 20 SMA, 20 SMA above 50 SMA, 50 SMA above 200 SMA")
        print("• Renko brick confirms trend direction")
        print()
        print("SHORT ENTRIES (Counter-Trend Toggle):")
        print("• 20 SMA crosses below 50 SMA (bearish crossover)")
        print("• Both 20 and 50 SMA are downsloping")
        print("• Price below 20 SMA, 20 SMA below 50 SMA, 50 SMA below 200 SMA")
        print("• Renko brick confirms trend direction")
        print()
        print("\n🔧 HEDGE LOGIC OPTIONS:")
        print("1. single_brick - Hedge on any reversal brick (original)")
        print("2. multiple_bricks - Hedge after N consecutive reversal bricks")
        print("3. price_level - Hedge only if price breaks key levels")
        print("4. trend_strength - Hedge only if reversal trend is stronger")
        print("5. combined - All conditions must be met")
        
        print("\n📈 SYSTEM V1 ADVANTAGES:")
        print("• Adaptive to market volatility through brick size")
        print("• Built-in risk management with hedge protection")
        print("• Trend-following with momentum confirmation")
        print("• Trailing stops for profit protection")
        print("• Customizable parameters for optimization")
        
        print("\n🎯 TESTING APPROACH:")
        print("• Single Test: Quick parameter validation")
        print("• Parameter Sweep: Find optimal settings")
        print("• Monte Carlo: Robustness testing")
        print("• Walk Forward: Out-of-sample validation")
        print("• In-Sample Excellence: Statistical significance")
        
        input("\nPress Enter to return to menu...")
    
    def get_system_selection(self):
        """Get system selection from user"""
        print("\n" + "="*60)
        print("SYSTEM SELECTION")
        print("="*60)
        
        print("Available Systems:")
        print("1. System V1 - Renko Trend-Hedge Engine with Moving Averages")
        print("   • 20, 50, 200 SMA integration")
        print("   • MA-based entry rules")
        print("   • Counter-trend toggle")
        print("   • Hedge protection")
        
        while True:
            system_choice = input("\nSelect system (1, default: 1): ").strip()
            if not system_choice or system_choice == "1":
                return "System V1"
            else:
                print("Please enter 1 for System V1")
    
    def get_counter_trend_setting(self):
        """Get counter-trend setting from user"""
        print("\n" + "="*60)
        print("COUNTER-TREND SETTING")
        print("="*60)
        
        print("Counter-trend entries allow short positions when:")
        print("• 20 SMA crosses below 50 SMA (bearish)")
        print("• Both MAs are downsloping")
        print("• Price is below all MAs")
        print()
        print("⚠️  WARNING: Counter-trend trades are higher risk!")
        print("   Only enable if you want to test short positions.")
        
        while True:
            enable_ct = input("\nEnable counter-trend entries? (y/n, default: n): ").strip().lower()
            if enable_ct in ['y', 'yes']:
                return True
            elif enable_ct in ['n', 'no', '']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def get_user_input(self, enable_counter_trend=None):
        """Get user input for test parameters. If enable_counter_trend is False, skip all counter-trend prompts and logic."""
        print("\n" + "="*60)
        print("TEST CONFIGURATION")
        print("="*60)
        
        # Number of bars to test
        max_bars = len(self.renko_data)
        while True:
            try:
                num_bars = input(f"\nNumber of Renko bars to test (1-{max_bars}): ").strip()
                if num_bars.lower() == 'max':
                    num_bars = max_bars
                else:
                    num_bars = int(num_bars)
                if 1 <= num_bars <= max_bars:
                    break
                else:
                    print(f"Please enter a number between 1 and {max_bars}")
            except ValueError:
                print("Please enter a valid number or 'max'")
        
        # Strategy parameters
        print("\nEnter strategy parameters (press Enter for defaults):")
        
        brick_size = input(f"Brick size (default: {self.optimal_brick_size:.2f}): ").strip()
        brick_size = float(brick_size) if brick_size else self.optimal_brick_size
        
        tsl_offset = input(f"TSL offset (default: {self.optimal_brick_size * 0.02:.2f}): ").strip()
        tsl_offset = float(tsl_offset) if tsl_offset else self.optimal_brick_size * 0.02
        
        hedge_ratio = input("Hedge size ratio (default: 0.5): ").strip()
        hedge_ratio = float(hedge_ratio) if hedge_ratio else 0.5
        
        min_bricks = input("Min bricks for trend (default: 2): ").strip()
        min_bricks = int(min_bricks) if min_bricks else 2
        
        # Hedge logic parameters
        print("\nHedge Logic Options:")
        print("1. single_brick - Hedge on any reversal brick (original)")
        print("2. multiple_bricks - Hedge after N consecutive reversal bricks")
        print("3. price_level - Hedge only if price breaks key levels")
        print("4. trend_strength - Hedge only if reversal trend is stronger")
        print("5. combined - All conditions must be met")
        
        hedge_logic = input("Hedge logic (1-5, default: 1): ").strip()
        if hedge_logic == "2":
            hedge_logic = "multiple_bricks"
        elif hedge_logic == "3":
            hedge_logic = "price_level"
        elif hedge_logic == "4":
            hedge_logic = "trend_strength"
        elif hedge_logic == "5":
            hedge_logic = "combined"
        else:
            hedge_logic = "single_brick"
        
        if hedge_logic in ["multiple_bricks", "combined"]:
            hedge_threshold = input("Hedge brick threshold (default: 2): ").strip()
            hedge_threshold = int(hedge_threshold) if hedge_threshold else 2
        else:
            hedge_threshold = 2
        
        price_conf = input("Require price confirmation? (y/n, default: y): ").strip().lower()
        price_confirmation = price_conf != "n"
        
        trend_strength = input("Require trend strength confirmation? (y/n, default: n): ").strip().lower()
        trend_strength_confirmation = trend_strength == "y"
        
        # Moving average parameters
        print("\nMoving Average Parameters:")
        ma_20 = input("20-day SMA period (default: 20): ").strip()
        ma_20_period = int(ma_20) if ma_20 else 20
        
        ma_50 = input("50-day SMA period (default: 50): ").strip()
        ma_50_period = int(ma_50) if ma_50 else 50
        
        ma_200 = input("200-day SMA period (default: 200): ").strip()
        ma_200_period = int(ma_200) if ma_200 else 200
        
        # Only include counter-trend if enabled
        params = {
            'num_bars': num_bars,
            'brick_size': brick_size,
            'tsl_offset': tsl_offset,
            'hedge_size_ratio': hedge_ratio,
            'min_bricks_for_trend': min_bricks,
            'hedge_logic': hedge_logic,
            'hedge_brick_threshold': hedge_threshold,
            'hedge_price_confirmation': price_confirmation,
            'hedge_trend_strength': trend_strength_confirmation,
            'ma_20_period': ma_20_period,
            'ma_50_period': ma_50_period,
            'ma_200_period': ma_200_period
        }
        if enable_counter_trend is not None:
            params['enable_counter_trend'] = enable_counter_trend
        return params
    
    def run_single_test(self, params):
        """Run a single test with given parameters"""
        print(f"\n{'='*60}")
        print(f"RUNNING {params['system_name'].upper()} TEST")
        print(f"{'='*60}")
        print(f"Testing on {params['num_bars']} Renko bars")
        print(f"Counter-trend entries: {'ENABLED' if params['enable_counter_trend'] else 'DISABLED'}")
        print(f"MA Periods: {params['ma_20_period']}, {params['ma_50_period']}, {params['ma_200_period']}")
        
        # Create strategy
        strategy = RTHEStrategy(
            brick_size=params['brick_size'],
            tsl_offset=params['tsl_offset'],
            hedge_size_ratio=params['hedge_size_ratio'],
            min_bricks_for_trend=params['min_bricks_for_trend'],
            hedge_logic=params.get('hedge_logic', 'single_brick'),
            hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
            hedge_price_confirmation=params.get('hedge_price_confirmation', True),
            hedge_trend_strength=params.get('hedge_trend_strength', False),
            enable_counter_trend=params['enable_counter_trend'],
            ma_20_period=params['ma_20_period'],
            ma_50_period=params['ma_50_period'],
            ma_200_period=params['ma_200_period']
        )
        
        # Run strategy
        test_data = self.renko_data.head(params['num_bars']).reset_index(drop=True)

        # Calculate moving averages on Renko close prices (not OHLC candles)
        test_data['sma_20'] = test_data['close'].rolling(window=params['ma_20_period'], min_periods=1).mean()
        test_data['sma_50'] = test_data['close'].rolling(window=params['ma_50_period'], min_periods=1).mean()
        test_data['sma_200'] = test_data['close'].rolling(window=params['ma_200_period'], min_periods=1).mean()

        for i in range(len(test_data)):
            strategy.execute_trade(test_data, i)

        # Use BacktestEngine to get result (for equity curve and metrics)
        backtest_engine = BacktestEngine(initial_capital=10000)
        result = backtest_engine.run_single_backtest(strategy, test_data, params)

        # Plot Renko chart with trades and correct MAs
        self._create_renko_plot(test_data, strategy, params['num_bars'], params['system_name'])

        # Plot equity curve for the single test
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(result.equity_curve, color='dodgerblue', linewidth=2, label='Equity Curve')
        plt.title(f'{params["system_name"]} - Equity Curve')
        plt.xlabel('Bar')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return result
    
    def run_parameter_sweep(self):
        """Run parameter sweep with visualization"""
        print(f"\n{'='*60}")
        print("PARAMETER SWEEP MODE")
        print(f"{'='*60}")
        
        # Define parameter ranges
        brick_sizes = [
            self.optimal_brick_size * 0.5,
            self.optimal_brick_size * 0.75,
            self.optimal_brick_size,
            self.optimal_brick_size * 1.25,
            self.optimal_brick_size * 1.5
        ]
        
        tsl_offsets = [
            self.optimal_brick_size * 0.01,
            self.optimal_brick_size * 0.02,
            self.optimal_brick_size * 0.03
        ]
        
        hedge_ratios = [0.3, 0.5, 0.7]
        min_bricks = [2, 3]
        
        # Calculate total combinations
        total_combinations = len(brick_sizes) * len(tsl_offsets) * len(hedge_ratios) * len(min_bricks)
        print(f"Testing {total_combinations} parameter combinations...")
        
        # Get number of bars for sweep
        max_bars = len(self.renko_data)
        while True:
            try:
                num_bars = input(f"Number of Renko bars for sweep (1-{max_bars}): ").strip()
                if num_bars.lower() == 'max':
                    num_bars = max_bars
                else:
                    num_bars = int(num_bars)
                if 1 <= num_bars <= max_bars:
                    break
                else:
                    print(f"Please enter a number between 1 and {max_bars}")
            except ValueError:
                print("Please enter a valid number or 'max'")
        
        # Get system settings for sweep
        system_name = self.get_system_selection()
        enable_counter_trend = self.get_counter_trend_setting()
        
        # Run sweep
        results = []
        equity_curves = []
        test_data_ohlc = self.ohlc_data.copy()  # Always use full OHLC data for Renko conversion
        backtest_engine = BacktestEngine(initial_capital=10000)
        renko_converter = RenkoConverter(1.0)  # Placeholder, will set brick_size per combo
        
        print(f"\nRunning {system_name} parameter sweep...")
        print(f"Counter-trend entries: {'ENABLED' if enable_counter_trend else 'DISABLED'}")
        
        for i, (brick_size, tsl_offset, hedge_ratio, min_brick) in enumerate(
            product(brick_sizes, tsl_offsets, hedge_ratios, min_bricks)
        ):
            if i % 10 == 0:  # Progress update every 10 combinations
                print(f"  Testing combination {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
            params = {
                'system_name': system_name,
                'enable_counter_trend': enable_counter_trend,
                'brick_size': brick_size,
                'tsl_offset': tsl_offset,
                'hedge_size_ratio': hedge_ratio,
                'min_bricks_for_trend': min_brick,
                'hedge_logic': 'single_brick',  # Default for sweep
                'hedge_brick_threshold': 2,
                'hedge_price_confirmation': True,
                'hedge_trend_strength': False,
                'ma_20_period': 20,
                'ma_50_period': 50,
                'ma_200_period': 200
            }
            # Regenerate Renko data for this brick size
            renko_converter = RenkoConverter(brick_size)
            renko_data = renko_converter.convert_to_renko(test_data_ohlc)
            if renko_data.empty:
                print(f"[WARN] No Renko bricks generated for brick_size={brick_size}")
                continue
            # Truncate to num_bars if needed
            renko_data = renko_data.head(num_bars).reset_index(drop=True)
            strategy = RTHEStrategy(**params)
            result = backtest_engine.run_single_backtest(strategy, renko_data, params)
            print(f"Params: {params} | PF: {result.profit_factor} | Return: {result.total_return}")
            results.append({
                'system_name': system_name,
                'enable_counter_trend': enable_counter_trend,
                'brick_size': brick_size,
                'tsl_offset': tsl_offset,
                'hedge_ratio': hedge_ratio,
                'min_bricks': min_brick,
                'profit_factor': result.profit_factor,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': int(result.total_trades),
                'win_rate': result.win_rate
            })
            equity_curves.append(np.array(result.equity_curve))
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{total_combinations} combinations...")
        
        print(f"✅ Parameter sweep complete!")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot all equity curves
        best_idx = results_df['profit_factor'].idxmax()
        self.plot_equity_curves_sweep(equity_curves, best_idx)
        
        # Display best results
        print(f"\n🏆 TOP 10 RESULTS BY PROFIT FACTOR:")
        print("="*80)
        top_results = results_df.nlargest(10, 'profit_factor')
        for i, row in top_results.iterrows():
            trades = int(row['total_trades']) if not pd.isna(row['total_trades']) else 0
            brick_size = float(row['brick_size'])
            tsl_offset = float(row['tsl_offset'])
            hedge_ratio = float(row['hedge_ratio'])
            min_bricks = int(row['min_bricks'])
            print(f"{i+1:2d}. PF: {row['profit_factor']:.3f} | "
                  f"Return: {row['total_return']*100:6.2f}% | "
                  f"Trades: {trades:3d} | "
                  f"Brick: {brick_size:.0f} | "
                  f"TSL: {tsl_offset:.0f} | "
                  f"Hedge: {hedge_ratio:.1f} | "
                  f"Min: {min_bricks}")
        
        # Create visualizations
        self.create_parameter_visualizations(results_df)
        
        return results_df
    
    def plot_renko_trades(self):
        """Plot Renko chart with R.T.H.E. strategy trades"""
        print(f"\n{'='*60}")
        print("RENKO CHART WITH TRADES")
        print(f"{'='*60}")
        
        # Get user input for plotting
        max_bars = len(self.renko_data)
        while True:
            try:
                num_bars = input(f"Number of Renko bars to plot (1-{max_bars}): ").strip()
                if num_bars.lower() == 'max':
                    num_bars = max_bars
                else:
                    num_bars = int(num_bars)
                if 1 <= num_bars <= max_bars:
                    break
                else:
                    print(f"Please enter a number between 1 and {max_bars}")
            except ValueError:
                print("Please enter a valid number or 'max'")
        
        # Get strategy parameters
        print("\nEnter strategy parameters (press Enter for defaults):")
        brick_size = input(f"Brick size (default: {self.optimal_brick_size:.2f}): ").strip()
        brick_size = float(brick_size) if brick_size else self.optimal_brick_size
        
        tsl_offset = input(f"TSL offset (default: {self.optimal_brick_size * 0.02:.2f}): ").strip()
        tsl_offset = float(tsl_offset) if tsl_offset else self.optimal_brick_size * 0.02
        
        hedge_ratio = input("Hedge size ratio (default: 0.5): ").strip()
        hedge_ratio = float(hedge_ratio) if hedge_ratio else 0.5
        
        min_bricks = input("Min bricks for trend (default: 2): ").strip()
        min_bricks = int(min_bricks) if min_bricks else 2
        
        # Create strategy and run
        strategy = RTHEStrategy(
            brick_size=brick_size,
            tsl_offset=tsl_offset,
            hedge_size_ratio=hedge_ratio,
            min_bricks_for_trend=min_bricks
        )
        
        # Get data for plotting
        plot_data = self.renko_data.head(num_bars).reset_index(drop=True)
        
        # Run strategy
        print(f"Running strategy on {len(plot_data)} bars...")
        for i in range(len(plot_data)):
            if i % 500 == 0:  # Progress update every 500 bars
                print(f"  Processing bar {i+1}/{len(plot_data)} ({(i+1)/len(plot_data)*100:.1f}%)")
            strategy.execute_trade(plot_data, i)
        print(f"✅ Strategy execution complete!")
        
        # Create the plot
        self._create_renko_plot(plot_data, strategy, num_bars)
    
    def _create_renko_plot(self, renko_data, strategy, num_bars, system_name="System V1"):
        """Create the actual Renko plot with trades and moving averages"""
        import matplotlib.pyplot as plt
        
        # For large datasets, use more efficient plotting
        if len(renko_data) > 1000:
            print(f"Large dataset detected ({len(renko_data)} bars). Using optimized plotting...")
            fig, ax = plt.subplots(figsize=(20, 10))
            
            # Use more efficient bar plotting for large datasets
            print("Creating Renko chart (optimized for large datasets)...")
            
            # Plot up/down bars separately for better performance
            up_bars = renko_data[renko_data['direction'] == 1]
            down_bars = renko_data[renko_data['direction'] == -1]
            
            if len(up_bars) > 0:
                ax.bar(up_bars.index, up_bars['close'] - up_bars['open'], 
                      bottom=up_bars['open'], width=0.8, color='green', 
                      edgecolor='black', alpha=0.7, linewidth=0.5, label='Up Bars')
            
            if len(down_bars) > 0:
                ax.bar(down_bars.index, down_bars['close'] - down_bars['open'], 
                      bottom=down_bars['open'], width=0.8, color='red', 
                      edgecolor='black', alpha=0.7, linewidth=0.5, label='Down Bars')
        else:
            # Original plotting for smaller datasets
            fig, ax = plt.subplots(figsize=(16, 8))
            
            print("Creating Renko chart...")
            for i, row in renko_data.iterrows():
                if i % 1000 == 0:  # Progress update every 1000 bars
                    print(f"  Plotting bar {i+1}/{len(renko_data)} ({(i+1)/len(renko_data)*100:.1f}%)")
                color = 'green' if row['direction'] == 1 else 'red'
                height = row['close'] - row['open']
                ax.bar(i, height, bottom=row['open'], width=0.8, color=color, 
                       edgecolor='black', alpha=0.7, linewidth=0.5)
        
        # Overlay Core trades (optimized for large datasets)
        print("Adding trade markers...")
        if len(strategy.core_trades) > 0:
            core_entry_x, core_entry_y = [], []
            core_exit_x, core_exit_y = [], []
            for trade in strategy.core_trades:
                entry_idx = renko_data[renko_data['close'] == trade.entry_price].index
                exit_idx = renko_data[renko_data['close'] == trade.exit_price].index
                if len(entry_idx) > 0:
                    core_entry_x.append(entry_idx[0] - 0.15)
                    core_entry_y.append(trade.entry_price + 0.5)
                if len(exit_idx) > 0:
                    core_exit_x.append(exit_idx[0] + 0.15)
                    core_exit_y.append(trade.exit_price - 0.5)
                # Draw colored line from entry to exit (green for profit, red for loss)
                if len(entry_idx) > 0 and len(exit_idx) > 0:
                    color = 'green' if trade.pnl > 0 else 'red'
                    ax.plot([entry_idx[0], exit_idx[0]], [trade.entry_price, trade.exit_price],
                            color=color, alpha=0.5, linewidth=1.5, zorder=2)
            if core_entry_x:
                ax.scatter(core_entry_x, core_entry_y, marker='^', color='lime',
                           s=120, zorder=5, label='Core Entry', edgecolor='black', linewidth=0.7)
                for i in range(min(2, len(core_entry_x))):
                    ax.annotate('Entry', (core_entry_x[i], core_entry_y[i]), xytext=(8, 8), textcoords='offset points',
                                fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            if core_exit_x:
                ax.scatter(core_exit_x, core_exit_y, marker='v', color='red',
                           s=120, zorder=5, label='Core Exit', edgecolor='black', linewidth=0.7)
                for i in range(min(2, len(core_exit_x))):
                    ax.annotate('Exit', (core_exit_x[i], core_exit_y[i]), xytext=(8, -12), textcoords='offset points',
                                fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        # Overlay Hedge trades (optimized for large datasets)
        if len(strategy.hedge_trades) > 0:
            hedge_entry_x, hedge_entry_y = [], []
            hedge_exit_x, hedge_exit_y = [], []
            for trade in strategy.hedge_trades:
                entry_idx = renko_data[renko_data['close'] == trade.entry_price].index
                exit_idx = renko_data[renko_data['close'] == trade.exit_price].index
                if len(entry_idx) > 0:
                    hedge_entry_x.append(entry_idx[0] - 0.15)
                    hedge_entry_y.append(trade.entry_price + 0.5)
                if len(exit_idx) > 0:
                    hedge_exit_x.append(exit_idx[0] + 0.15)
                    hedge_exit_y.append(trade.exit_price - 0.5)
                # Draw colored line from entry to exit (green for profit, red for loss)
                if len(entry_idx) > 0 and len(exit_idx) > 0:
                    color = 'green' if trade.pnl > 0 else 'red'
                    ax.plot([entry_idx[0], exit_idx[0]], [trade.entry_price, trade.exit_price],
                            color=color, alpha=0.5, linewidth=1.5, zorder=2)
            if hedge_entry_x:
                ax.scatter(hedge_entry_x, hedge_entry_y, marker='o', color='orange',
                           s=90, zorder=5, label='Hedge Entry', edgecolor='black', linewidth=0.7)
                for i in range(min(2, len(hedge_entry_x))):
                    ax.annotate('Hedge Entry', (hedge_entry_x[i], hedge_entry_y[i]), xytext=(8, 8), textcoords='offset points',
                                fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            if hedge_exit_x:
                ax.scatter(hedge_exit_x, hedge_exit_y, marker='x', color='purple',
                           s=90, zorder=5, label='Hedge Exit', edgecolor='black', linewidth=0.7)
                for i in range(min(2, len(hedge_exit_x))):
                    ax.annotate('Hedge Exit', (hedge_exit_x[i], hedge_exit_y[i]), xytext=(8, -12), textcoords='offset points',
                                fontsize=8, color='black', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Add moving averages if available
        # Prefer plotting from renko_data columns if present
        if 'sma_20' in renko_data.columns:
            ax.plot(renko_data.index, renko_data['sma_20'], color='blue', linewidth=2, label='20 SMA', alpha=0.8)
        if 'sma_50' in renko_data.columns:
            ax.plot(renko_data.index, renko_data['sma_50'], color='orange', linewidth=2, label='50 SMA', alpha=0.8)
        if 'sma_200' in renko_data.columns:
            ax.plot(renko_data.index, renko_data['sma_200'], color='purple', linewidth=2, label='200 SMA', alpha=0.8)
        
        # Finalize plot
        ax.set_title(f'{system_name} - Renko Chart with Trades\n'
                    f'Bars: {num_bars} | Core Trades: {len(strategy.core_trades)} | Hedge Trades: {len(strategy.hedge_trades)}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Renko Brick Index")
        ax.set_ylabel("Price ($)")
        ax.grid(True, alpha=0.3)
        
        # Custom legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='best')
        
        plt.tight_layout()
        
        # Save the plot
        print("Saving chart...")
        filename = f"rthe_renko_trades_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Renko chart saved as: {filename}")
        
        # Show the plot
        plt.show()
        
        # Print trade summary
        print(f"\n📊 TRADE SUMMARY:")
        print(f"   Core Trades: {len(strategy.core_trades)}")
        print(f"   Hedge Trades: {len(strategy.hedge_trades)}")
        print(f"   Total Trades: {len(strategy.trades)}")
        print(f"   Hedge Activations: {strategy.hedge_activations}")
        print(f"   TSL Hits: {strategy.tsl_hits}")
    
    def refresh_data(self):
        """Refresh Bitcoin data by re-running the merger"""
        print(f"\n{'='*60}")
        print("REFRESHING BITCOIN DATA")
        print(f"{'='*60}")
        
        try:
            from utils.merge_bitcoin_data import merge_bitcoin_csvs
            merge_bitcoin_csvs()
            
            # Reload data
            print("\nReloading data...")
            self.load_data()
            print("✅ Data refreshed successfully!")
            
        except Exception as e:
            print(f"❌ Error refreshing data: {e}")
    
    def create_parameter_visualizations(self, results_df):
        """Create parameter sweep visualizations"""
        print(f"\n📊 Creating visualizations...")
        
        # Filter out non-finite values for plotting
        plot_df = results_df[np.isfinite(results_df['profit_factor']) & np.isfinite(results_df['total_return']) & np.isfinite(results_df['sharpe_ratio'])]
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        system_name = results_df['system_name'].iloc[0] if len(results_df) > 0 else "System V1"
        fig.suptitle(f'{system_name} Parameter Sweep Results', fontsize=16, fontweight='bold')
        
        # 1. Profit Factor Heatmap (Brick Size vs TSL Offset)
        pivot_pf = plot_df.pivot_table(
            values='profit_factor', 
            index='brick_size', 
            columns='tsl_offset', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_pf, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0,0])
        axes[0,0].set_title('Profit Factor Heatmap\n(Brick Size vs TSL Offset)')
        axes[0,0].set_xlabel('TSL Offset')
        axes[0,0].set_ylabel('Brick Size')
        
        # 2. Total Return Heatmap (Brick Size vs Hedge Ratio)
        pivot_return = plot_df.pivot_table(
            values='total_return', 
            index='brick_size', 
            columns='hedge_ratio', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_return, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0,1])
        axes[0,1].set_title('Total Return Heatmap\n(Brick Size vs Hedge Ratio)')
        axes[0,1].set_xlabel('Hedge Ratio')
        axes[0,1].set_ylabel('Brick Size')
        
        # 3. Sharpe Ratio Heatmap (TSL Offset vs Hedge Ratio)
        pivot_sharpe = plot_df.pivot_table(
            values='sharpe_ratio', 
            index='tsl_offset', 
            columns='hedge_ratio', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0,2])
        axes[0,2].set_title('Sharpe Ratio Heatmap\n(TSL Offset vs Hedge Ratio)')
        axes[0,2].set_xlabel('Hedge Ratio')
        axes[0,2].set_ylabel('TSL Offset')
        
        # 4. Profit Factor vs Total Return Scatter
        scatter = axes[1,0].scatter(
            plot_df['total_return'] * 100, 
            plot_df['profit_factor'],
            c=plot_df['total_trades'], 
            cmap='viridis', 
            alpha=0.7,
            s=50
        )
        axes[1,0].set_xlabel('Total Return (%)')
        axes[1,0].set_ylabel('Profit Factor')
        axes[1,0].set_title('Profit Factor vs Total Return\n(Color = Number of Trades)')
        plt.colorbar(scatter, ax=axes[1,0], label='Number of Trades')
        
        # 5. Parameter Distribution
        axes[1,1].hist(plot_df['profit_factor'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,1].set_xlabel('Profit Factor')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Profit Factor Distribution')
        axes[1,1].axvline(plot_df['profit_factor'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {plot_df['profit_factor'].mean():.2f}')
        axes[1,1].legend()
        
        # 6. Trade Count vs Performance
        axes[1,2].scatter(plot_df['total_trades'], plot_df['profit_factor'], alpha=0.7, s=50)
        axes[1,2].set_xlabel('Number of Trades')
        axes[1,2].set_ylabel('Profit Factor')
        axes[1,2].set_title('Trade Count vs Profit Factor')
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"rthe_parameter_sweep_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved as: {filename}")
        
        # Show the plot
        plt.show()
    
    def plot_equity_curves_sweep(self, equity_curves, best_idx):
        """Plot all equity curves from parameter sweep, highlight best in red"""
        import matplotlib.pyplot as plt
        plt.style.use('dark_background')
        plt.figure(figsize=(14, 7))
        for i, curve in enumerate(equity_curves):
            if i == best_idx:
                continue
            plt.plot(curve, color='white', alpha=0.08, linewidth=1)
        # Highlight best
        plt.plot(equity_curves[best_idx], color='red', alpha=1, linewidth=2, label='Best Profit Factor')
        plt.title('Equity Curves for All Parameter Combinations')
        plt.xlabel('Bar')
        plt.ylabel('Equity')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def run_monte_carlo_test(self):
        """Run Monte Carlo permutation test to validate strategy robustness"""
        print(f"\n{'='*60}")
        print("MONTE CARLO PERMUTATION TEST")
        print(f"{'='*60}")
        
        # Get parameters
        params = self.get_user_input()
        
        print("Running Monte Carlo test...")
        print("This test randomly shuffles trade sequences to test if results are due to luck.")
        
        # Run original strategy
        strategy = RTHEStrategy(
            brick_size=params['brick_size'],
            tsl_offset=params['tsl_offset'],
            hedge_size_ratio=params['hedge_size_ratio'],
            min_bricks_for_trend=params['min_bricks_for_trend'],
            hedge_logic=params.get('hedge_logic', 'single_brick'),
            hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
            hedge_price_confirmation=params.get('hedge_price_confirmation', True),
            hedge_trend_strength=params.get('hedge_trend_strength', False)
        )
        
        test_data = self.renko_data.head(params['num_bars']).reset_index(drop=True)
        
        # Run original strategy
        for i in range(len(test_data)):
            strategy.execute_trade(test_data, i)
        
        backtest_engine = BacktestEngine(initial_capital=10000)
        original_result = backtest_engine.run_single_backtest(strategy, test_data, params)
        
        print(f"Original Strategy Results:")
        print(f"  Profit Factor: {original_result.profit_factor:.3f}")
        print(f"  Total Return: {original_result.total_return*100:.2f}%")
        print(f"  Total Trades: {original_result.total_trades}")
        
        # Monte Carlo permutations
        num_permutations = 1000
        print(f"\nRunning {num_permutations} Monte Carlo permutations...")
        
        better_count = 0
        profit_factors = []
        
        for i in range(num_permutations):
            if i % 100 == 0:
                print(f"  Progress: {i}/{num_permutations} ({(i/num_permutations)*100:.1f}%)")
            
            # Create shuffled trade sequence
            shuffled_strategy = RTHEStrategy(
                brick_size=params['brick_size'],
                tsl_offset=params['tsl_offset'],
                hedge_size_ratio=params['hedge_size_ratio'],
                min_bricks_for_trend=params['min_bricks_for_trend'],
                hedge_logic=params.get('hedge_logic', 'single_brick'),
                hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
                hedge_price_confirmation=params.get('hedge_price_confirmation', True),
                hedge_trend_strength=params.get('hedge_trend_strength', False)
            )
            
            # Shuffle the data (simulating random trade sequence)
            shuffled_data = test_data.sample(frac=1).reset_index(drop=True)
            
            # Run strategy on shuffled data
            for j in range(len(shuffled_data)):
                shuffled_strategy.execute_trade(shuffled_data, j)
            
            shuffled_result = backtest_engine.run_single_backtest(shuffled_strategy, shuffled_data, params)
            profit_factors.append(shuffled_result.profit_factor)
            
            if shuffled_result.profit_factor > original_result.profit_factor:
                better_count += 1
        
        # Calculate statistics
        p_value = better_count / num_permutations
        mean_pf = np.mean(profit_factors)
        std_pf = np.std(profit_factors)
        
        print(f"\n📊 MONTE CARLO RESULTS:")
        print(f"  Original Profit Factor: {original_result.profit_factor:.3f}")
        print(f"  Mean Random PF: {mean_pf:.3f}")
        print(f"  Std Random PF: {std_pf:.3f}")
        print(f"  Better Random Results: {better_count}/{num_permutations}")
        print(f"  P-Value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✅ SIGNIFICANT: Strategy results are unlikely due to chance (p < 0.05)")
        else:
            print(f"  ⚠️  NOT SIGNIFICANT: Strategy results may be due to chance (p >= 0.05)")
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        plt.hist(profit_factors, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(original_result.profit_factor, color='red', linewidth=2, 
                   label=f'Original PF: {original_result.profit_factor:.3f}')
        plt.axvline(mean_pf, color='green', linewidth=2, linestyle='--',
                   label=f'Mean Random PF: {mean_pf:.3f}')
        plt.xlabel('Profit Factor')
        plt.ylabel('Frequency')
        plt.title('Monte Carlo Permutation Test Results')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"monte_carlo_test_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Monte Carlo plot saved as: {filename}")
        plt.show()
    
    def run_walk_forward_test(self):
        """Run walk-forward analysis to test out-of-sample performance"""
        print(f"\n{'='*60}")
        print("WALK FORWARD ANALYSIS")
        print(f"{'='*60}")
        
        # Get parameters
        params = self.get_user_input()
        
        print("Running walk-forward test...")
        print("This test splits data into training/validation periods.")
        
        # Split data into periods
        total_bars = len(self.renko_data)
        train_ratio = 0.7
        train_bars = int(total_bars * train_ratio)
        
        print(f"Total bars: {total_bars}")
        print(f"Training bars: {train_bars}")
        print(f"Validation bars: {total_bars - train_bars}")
        
        # Training period
        train_data = self.renko_data.head(train_bars).reset_index(drop=True)
        
        strategy_train = RTHEStrategy(
            brick_size=params['brick_size'],
            tsl_offset=params['tsl_offset'],
            hedge_size_ratio=params['hedge_size_ratio'],
            min_bricks_for_trend=params['min_bricks_for_trend'],
            hedge_logic=params.get('hedge_logic', 'single_brick'),
            hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
            hedge_price_confirmation=params.get('hedge_price_confirmation', True),
            hedge_trend_strength=params.get('hedge_trend_strength', False)
        )
        
        print("Running on training data...")
        for i in range(len(train_data)):
            strategy_train.execute_trade(train_data, i)
        
        backtest_engine = BacktestEngine(initial_capital=10000)
        train_result = backtest_engine.run_single_backtest(strategy_train, train_data, params)
        
        # Validation period
        validation_data = self.renko_data.iloc[train_bars:].reset_index(drop=True)
        
        strategy_val = RTHEStrategy(
            brick_size=params['brick_size'],
            tsl_offset=params['tsl_offset'],
            hedge_size_ratio=params['hedge_size_ratio'],
            min_bricks_for_trend=params['min_bricks_for_trend'],
            hedge_logic=params.get('hedge_logic', 'single_brick'),
            hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
            hedge_price_confirmation=params.get('hedge_price_confirmation', True),
            hedge_trend_strength=params.get('hedge_trend_strength', False)
        )
        
        print("Running on validation data...")
        for i in range(len(validation_data)):
            strategy_val.execute_trade(validation_data, i)
        
        val_result = backtest_engine.run_single_backtest(strategy_val, validation_data, params)
        
        print(f"\n📊 WALK FORWARD RESULTS:")
        print(f"  TRAINING PERIOD:")
        print(f"    Profit Factor: {train_result.profit_factor:.3f}")
        print(f"    Total Return: {train_result.total_return*100:.2f}%")
        print(f"    Total Trades: {train_result.total_trades}")
        print(f"    Sharpe Ratio: {train_result.sharpe_ratio:.3f}")
        
        print(f"  VALIDATION PERIOD:")
        print(f"    Profit Factor: {val_result.profit_factor:.3f}")
        print(f"    Total Return: {val_result.total_return*100:.2f}%")
        print(f"    Total Trades: {val_result.total_trades}")
        print(f"    Sharpe Ratio: {val_result.sharpe_ratio:.3f}")
        
        # Calculate degradation
        pf_degradation = (train_result.profit_factor - val_result.profit_factor) / train_result.profit_factor * 100
        return_degradation = (train_result.total_return - val_result.total_return) / train_result.total_return * 100
        
        print(f"  DEGRADATION ANALYSIS:")
        print(f"    PF Degradation: {pf_degradation:.1f}%")
        print(f"    Return Degradation: {return_degradation:.1f}%")
        
        if pf_degradation < 20:
            print(f"    ✅ GOOD: Low degradation suggests strategy is robust")
        elif pf_degradation < 50:
            print(f"    ⚠️  MODERATE: Some degradation, strategy may be overfitting")
        else:
            print(f"    ❌ HIGH: Significant degradation, strategy is likely overfitting")
        
        # Plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training vs Validation comparison
        metrics = ['Profit Factor', 'Total Return (%)', 'Sharpe Ratio']
        train_values = [train_result.profit_factor, train_result.total_return*100, train_result.sharpe_ratio]
        val_values = [val_result.profit_factor, val_result.total_return*100, val_result.sharpe_ratio]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, train_values, width, label='Training', color='blue', alpha=0.7)
        ax1.bar(x + width/2, val_values, width, label='Validation', color='red', alpha=0.7)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Values')
        ax1.set_title('Training vs Validation Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Degradation plot
        degradations = [pf_degradation, return_degradation]
        colors = ['green' if d < 20 else 'orange' if d < 50 else 'red' for d in degradations]
        ax2.bar(['PF Degradation', 'Return Degradation'], degradations, color=colors, alpha=0.7)
        ax2.set_ylabel('Degradation (%)')
        ax2.set_title('Performance Degradation')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"walk_forward_test_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Walk-forward plot saved as: {filename}")
        plt.show()
    
    def run_walk_forward_monte_carlo(self):
        """Run walk-forward Monte Carlo test combining both approaches"""
        print(f"\n{'='*60}")
        print("WALK FORWARD MONTE CARLO TEST")
        print(f"{'='*60}")
        
        # Get parameters
        params = self.get_user_input()
        
        print("Running walk-forward Monte Carlo test...")
        print("This combines walk-forward analysis with Monte Carlo permutations.")
        
        # Split data
        total_bars = len(self.renko_data)
        train_ratio = 0.7
        train_bars = int(total_bars * train_ratio)
        
        train_data = self.renko_data.head(train_bars).reset_index(drop=True)
        validation_data = self.renko_data.iloc[train_bars:].reset_index(drop=True)
        
        # Run original strategy on both periods
        strategy_train = RTHEStrategy(
            brick_size=params['brick_size'],
            tsl_offset=params['tsl_offset'],
            hedge_size_ratio=params['hedge_size_ratio'],
            min_bricks_for_trend=params['min_bricks_for_trend'],
            hedge_logic=params.get('hedge_logic', 'single_brick'),
            hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
            hedge_price_confirmation=params.get('hedge_price_confirmation', True),
            hedge_trend_strength=params.get('hedge_trend_strength', False)
        )
        
        strategy_val = RTHEStrategy(
            brick_size=params['brick_size'],
            tsl_offset=params['tsl_offset'],
            hedge_size_ratio=params['hedge_size_ratio'],
            min_bricks_for_trend=params['min_bricks_for_trend'],
            hedge_logic=params.get('hedge_logic', 'single_brick'),
            hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
            hedge_price_confirmation=params.get('hedge_price_confirmation', True),
            hedge_trend_strength=params.get('hedge_trend_strength', False)
        )
        
        # Original results
        for i in range(len(train_data)):
            strategy_train.execute_trade(train_data, i)
        
        for i in range(len(validation_data)):
            strategy_val.execute_trade(validation_data, i)
        
        backtest_engine = BacktestEngine(initial_capital=10000)
        original_train = backtest_engine.run_single_backtest(strategy_train, train_data, params)
        original_val = backtest_engine.run_single_backtest(strategy_val, validation_data, params)
        
        print(f"Original Results:")
        print(f"  Training PF: {original_train.profit_factor:.3f}")
        print(f"  Validation PF: {original_val.profit_factor:.3f}")
        
        # Monte Carlo permutations on validation set
        num_permutations = 500
        print(f"\nRunning {num_permutations} Monte Carlo permutations on validation set...")
        
        better_count = 0
        val_profit_factors = []
        
        for i in range(num_permutations):
            if i % 100 == 0:
                print(f"  Progress: {i}/{num_permutations} ({(i/num_permutations)*100:.1f}%)")
            
            # Shuffle validation data
            shuffled_val = validation_data.sample(frac=1).reset_index(drop=True)
            
            shuffled_strategy = RTHEStrategy(
                brick_size=params['brick_size'],
                tsl_offset=params['tsl_offset'],
                hedge_size_ratio=params['hedge_size_ratio'],
                min_bricks_for_trend=params['min_bricks_for_trend'],
                hedge_logic=params.get('hedge_logic', 'single_brick'),
                hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
                hedge_price_confirmation=params.get('hedge_price_confirmation', True),
                hedge_trend_strength=params.get('hedge_trend_strength', False)
            )
            
            for j in range(len(shuffled_val)):
                shuffled_strategy.execute_trade(shuffled_val, j)
            
            shuffled_result = backtest_engine.run_single_backtest(shuffled_strategy, shuffled_val, params)
            val_profit_factors.append(shuffled_result.profit_factor)
            
            if shuffled_result.profit_factor > original_val.profit_factor:
                better_count += 1
        
        # Calculate statistics
        p_value = better_count / num_permutations
        mean_val_pf = np.mean(val_profit_factors)
        std_val_pf = np.std(val_profit_factors)
        
        print(f"\n📊 WALK FORWARD MONTE CARLO RESULTS:")
        print(f"  Training PF: {original_train.profit_factor:.3f}")
        print(f"  Validation PF: {original_val.profit_factor:.3f}")
        print(f"  Mean Random Validation PF: {mean_val_pf:.3f}")
        print(f"  Better Random Results: {better_count}/{num_permutations}")
        print(f"  P-Value: {p_value:.4f}")
        
        # Combined assessment
        pf_degradation = (original_train.profit_factor - original_val.profit_factor) / original_train.profit_factor * 100
        
        print(f"  COMBINED ASSESSMENT:")
        print(f"    PF Degradation: {pf_degradation:.1f}%")
        print(f"    Validation Significance: {'Significant' if p_value < 0.05 else 'Not Significant'}")
        
        if pf_degradation < 20 and p_value < 0.05:
            print(f"    ✅ EXCELLENT: Low degradation + significant results")
        elif pf_degradation < 50 and p_value < 0.05:
            print(f"    ⚠️  GOOD: Some degradation but significant results")
        elif pf_degradation < 20 and p_value >= 0.05:
            print(f"    ⚠️  MODERATE: Low degradation but not significant")
        else:
            print(f"    ❌ POOR: High degradation and/or not significant")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Validation Monte Carlo distribution
        ax1.hist(val_profit_factors, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(original_val.profit_factor, color='red', linewidth=2, 
                   label=f'Original Val PF: {original_val.profit_factor:.3f}')
        ax1.axvline(mean_val_pf, color='green', linewidth=2, linestyle='--',
                   label=f'Mean Random PF: {mean_val_pf:.3f}')
        ax1.set_xlabel('Profit Factor')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Validation Monte Carlo Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training vs Validation comparison
        periods = ['Training', 'Validation']
        pfs = [original_train.profit_factor, original_val.profit_factor]
        colors = ['blue', 'red']
        
        bars = ax2.bar(periods, pfs, color=colors, alpha=0.7)
        ax2.set_ylabel('Profit Factor')
        ax2.set_title('Training vs Validation Performance')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, pf in zip(bars, pfs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{pf:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        filename = f"walk_forward_monte_carlo_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 Walk-forward Monte Carlo plot saved as: {filename}")
        plt.show()
    
    def show_testing_framework(self):
        """Display professional quant testing framework explanation"""
        print(f"\n{'='*80}")
        print("PROFESSIONAL QUANT TESTING FRAMEWORK")
        print(f"{'='*80}")
        
        print("\n📚 HOW PROFESSIONAL QUANTS TEST ALGORITHMIC TRADING STRATEGIES")
        print("="*70)
        
        print("\n🎯 PHASE 1: STRATEGY DEVELOPMENT")
        print("-" * 40)
        print("• Strategy Documentation: Understand logic and parameters")
        print("• Single Backtest: Initial proof-of-concept testing")
        print("• Parameter Optimization: Find optimal parameter combinations")
        print("• Trade Visualization: Analyze entry/exit patterns")
        
        print("\n🔬 PHASE 2: VALIDATION FRAMEWORK")
        print("-" * 40)
        print("Professional quants use a multi-layered validation approach:")
        
        print("\nA. ROBUSTNESS TESTING")
        print("   • Monte Carlo Permutation: Tests if results are due to luck vs skill")
        print("   • Bootstrap Analysis: Resampling techniques to assess stability")
        print("   • Sensitivity Analysis: How results change with parameter variations")
        
        print("\nB. OUT-OF-SAMPLE TESTING")
        print("   • Walk Forward Analysis: Tests on unseen data periods")
        print("   • Cross-Validation: Multiple train/test splits")
        print("   • Time Series Split: Respects temporal order of data")
        
        print("\nC. STATISTICAL VALIDATION")
        print("   • Statistical Significance: Z-tests, t-tests vs random trading")
        print("   • Sharpe Ratio Analysis: Risk-adjusted performance")
        print("   • Maximum Drawdown Analysis: Risk assessment")
        
        print("\nD. MARKET REGIME TESTING")
        print("   • Regime Analysis: How strategy performs in different market conditions")
        print("   • Stress Testing: Extreme market scenarios")
        print("   • Transaction Cost Analysis: Real-world implementation costs")
        
        print("\n📊 OUR SYSTEM vs PROFESSIONAL STANDARDS")
        print("=" * 50)
        
        print("\n✅ WHAT WE HAVE (Good Foundation):")
        print("• Parameter optimization with visualization")
        print("• Basic backtesting engine")
        print("• Trade analysis and metrics")
        print("• Advanced validation tests (Monte Carlo, Walk Forward, etc.)")
        
        print("\n❌ WHAT WE'RE MISSING (Critical Gaps):")
        print("• Market regime analysis")
        print("• Transaction cost modeling")
        print("• Stress testing scenarios")
        print("• Bootstrap analysis")
        
        print("\n💡 INDUSTRY REALITY:")
        print("• Most retail traders skip validation → 90% failure rate")
        print("• Professional quants spend 80% time on validation, 20% on strategy")
        print("• Our validation tests are the MINIMUM required to avoid common pitfalls")
        
        print("\n🎯 RECOMMENDED WORKFLOW:")
        print("1. Develop strategy (Parameter Optimization)")
        print("2. Validate robustness (Monte Carlo Test)")
        print("3. Test out-of-sample (Walk Forward Test)")
        print("4. Check statistical significance (In-Sample Excellence)")
        print("5. Deploy only if ALL tests pass")
        
        print(f"\n{'='*80}")
        input("Press Enter to continue...")
    
    def run_in_sample_excellence_test(self):
        """Run in-sample excellence test to check statistical significance"""
        print(f"\n{'='*60}")
        print("IN-SAMPLE EXCELLENCE TEST")
        print(f"{'='*60}")
        
        # Get parameters
        params = self.get_user_input()
        
        print("Running in-sample excellence test...")
        print("This test checks if strategy results are statistically significant vs random trading.")
        
        # Run strategy
        strategy = RTHEStrategy(
            brick_size=params['brick_size'],
            tsl_offset=params['tsl_offset'],
            hedge_size_ratio=params['hedge_size_ratio'],
            min_bricks_for_trend=params['min_bricks_for_trend'],
            hedge_logic=params.get('hedge_logic', 'single_brick'),
            hedge_brick_threshold=params.get('hedge_brick_threshold', 2),
            hedge_price_confirmation=params.get('hedge_price_confirmation', True),
            hedge_trend_strength=params.get('hedge_trend_strength', False)
        )
        
        test_data = self.renko_data.head(params['num_bars']).reset_index(drop=True)
        
        for i in range(len(test_data)):
            strategy.execute_trade(test_data, i)
        
        backtest_engine = BacktestEngine(initial_capital=10000)
        result = backtest_engine.run_single_backtest(strategy, test_data, params)
        
        print(f"Strategy Results:")
        print(f"  Profit Factor: {result.profit_factor:.3f}")
        print(f"  Total Return: {result.total_return*100:.2f}%")
        print(f"  Total Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate*100:.1f}%")
        
        # Generate random trading results for comparison
        num_simulations = 1000
        print(f"\nRunning {num_simulations} random trading simulations...")
        
        random_returns = []
        random_pfs = []
        
        for i in range(num_simulations):
            if i % 200 == 0:
                print(f"  Progress: {i}/{num_simulations} ({(i/num_simulations)*100:.1f}%)")
            
            # Simulate random trading
            random_trades = []
            num_random_trades = result.total_trades
            
            for _ in range(num_random_trades):
                # Random trade with 50% win rate and random PnL
                if np.random.random() < 0.5:
                    pnl = np.random.uniform(0.01, 0.05)  # Random win
                else:
                    pnl = -np.random.uniform(0.01, 0.03)  # Random loss
                random_trades.append(pnl)
            
            random_return = np.sum(random_trades)
            random_pf = np.sum([t for t in random_trades if t > 0]) / abs(np.sum([t for t in random_trades if t < 0])) if np.sum([t for t in random_trades if t < 0]) != 0 else float('inf')
            
            random_returns.append(random_return)
            random_pfs.append(random_pf if random_pf != float('inf') else 10.0)  # Cap at 10
        
        # Calculate statistics
        mean_random_return = np.mean(random_returns)
        std_random_return = np.std(random_returns)
        mean_random_pf = np.mean(random_pfs)
        std_random_pf = np.std(random_pfs)
        
        # Z-scores
        z_score_return = (result.total_return - mean_random_return) / std_random_return if std_random_return > 0 else 0
        z_score_pf = (result.profit_factor - mean_random_pf) / std_random_pf if std_random_pf > 0 else 0
        
        # P-values (assuming normal distribution)
        p_value_return = 1 - norm.cdf(z_score_return)
        p_value_pf = 1 - norm.cdf(z_score_pf)
        
        print(f"\n📊 IN-SAMPLE EXCELLENCE RESULTS:")
        print(f"  Strategy Return: {result.total_return*100:.2f}%")
        print(f"  Mean Random Return: {mean_random_return*100:.2f}%")
        print(f"  Z-Score (Return): {z_score_return:.3f}")
        print(f"  P-Value (Return): {p_value_return:.4f}")
        
        print(f"  Strategy PF: {result.profit_factor:.3f}")
        print(f"  Mean Random PF: {mean_random_pf:.3f}")
        print(f"  Z-Score (PF): {z_score_pf:.3f}")
        print(f"  P-Value (PF): {p_value_pf:.4f}")
        
        # Significance assessment
        significance_level = 0.05
        return_significant = p_value_return < significance_level
        pf_significant = p_value_pf < significance_level
        
        print(f"  SIGNIFICANCE ASSESSMENT:")
        print(f"    Return Significant: {'Yes' if return_significant else 'No'} (p < {significance_level})")
        print(f"    PF Significant: {'Yes' if pf_significant else 'No'} (p < {significance_level})")
        
        if return_significant and pf_significant:
            print(f"    ✅ EXCELLENT: Both metrics are statistically significant")
        elif return_significant or pf_significant:
            print(f"    ⚠️  MODERATE: One metric is significant")
        else:
            print(f"    ❌ POOR: Neither metric is statistically significant")
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Return distribution
        ax1.hist(random_returns, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(result.total_return, color='red', linewidth=2, 
                   label=f'Strategy Return: {result.total_return*100:.2f}%')
        ax1.axvline(mean_random_return, color='green', linewidth=2, linestyle='--',
                   label=f'Mean Random: {mean_random_return*100:.2f}%')
        ax1.set_xlabel('Total Return')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Return Distribution vs Random Trading')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Profit Factor distribution
        ax2.hist(random_pfs, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(result.profit_factor, color='red', linewidth=2, 
                   label=f'Strategy PF: {result.profit_factor:.3f}')
        ax2.axvline(mean_random_pf, color='green', linewidth=2, linestyle='--',
                   label=f'Mean Random PF: {mean_random_pf:.3f}')
        ax2.set_xlabel('Profit Factor')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Profit Factor Distribution vs Random Trading')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"in_sample_excellence_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📈 In-sample excellence plot saved as: {filename}")
        plt.show()
    
    def run(self):
        """Main interactive test runner"""
        print("🚀 SYSTEM V1 - INTERACTIVE TESTER")
        print("="*60)
        
        # Load data
        self.load_data()
        
        while True:
            print(f"\n{'='*60}")
            print("SYSTEM V1 TESTING MENU")
            print("="*60)
            print("1. System V1 Rules & Parameters")
            print("2. Run Single Test (Custom Parameters)")
            print("3. Run Parameter Sweep (Find Best Settings)")
            print("4. Plot Renko Chart with Trades")
            print("5. Test/Visualize Indicators on Renko Chart")
            print("6. Refresh Bitcoin Data (Re-run Merger)")
            print("7. Show available timeframes for a symbol")
            print("8. Testing Framework Documentation")
            print("9. Run Monte Carlo Test")
            print("10. Run Walk Forward Test")
            print("11. Run Walk Forward Monte Carlo Test")
            print("12. Run In-Sample Excellence Test")
            print("13. Exit")
            
            choice = input("\nSelect option (1-13): ").strip()
            
            if choice == '1':
                self.explain_parameters()
            
            elif choice == '2':
                # Get system selection and counter-trend setting first
                system_name = self.get_system_selection()
                enable_counter_trend = self.get_counter_trend_setting()
                
                # Get other parameters, passing counter-trend setting
                params = self.get_user_input(enable_counter_trend=enable_counter_trend)
                
                # Add system settings to params
                params['system_name'] = system_name
                params['enable_counter_trend'] = enable_counter_trend
                
                self.run_single_test(params)
            
            elif choice == '3':
                self.run_parameter_sweep()
            
            elif choice == '4':
                self.plot_renko_trades()
            
            elif choice == '5':
                self.visualize_indicators_on_renko()
            
            elif choice == '6':
                self.refresh_data()
            
            elif choice == '7':
                symbol = input("Enter symbol (e.g., BTCUSD): ").strip().upper()
                timeframes = self.data_handler.load_symbol_timeframes(symbol)
                if timeframes:
                    print(f"Loaded timeframes for {symbol}:")
                    for tf, df in timeframes.items():
                        print(f"  {tf}: {len(df)} rows")
                else:
                    print(f"No timeframes found for {symbol}.")
            
            elif choice == '8':
                self.show_testing_framework()
            
            elif choice == '9':
                self.run_monte_carlo_test()
            
            elif choice == '10':
                self.run_walk_forward_test()
            
            elif choice == '11':
                self.run_walk_forward_monte_carlo()
            
            elif choice == '12':
                self.run_in_sample_excellence_test()
            
            elif choice == '13':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-13.")

    def visualize_indicators_on_renko(self):
        """Prompt for SMA periods, compute on Renko close, and plot over Renko chart with trades."""
        print(f"\n{'='*60}")
        print("RENKO INDICATOR VISUALIZER")
        print(f"{'='*60}")
        max_bars = len(self.renko_data)
        while True:
            try:
                num_bars = input(f"Number of Renko bars to plot (1-{max_bars}): ").strip()
                if num_bars.lower() == 'max':
                    num_bars = max_bars
                else:
                    num_bars = int(num_bars)
                if 1 <= num_bars <= max_bars:
                    break
                else:
                    print(f"Please enter a number between 1 and {max_bars}")
            except ValueError:
                print("Please enter a valid number or 'max'")
        
        # Prompt for SMA periods
        print("\nEnter SMA periods to plot (comma separated, default: 20,50,200):")
        periods_input = input("Periods: ").strip()
        if periods_input:
            try:
                periods = [int(p.strip()) for p in periods_input.split(",") if p.strip()]
            except Exception:
                print("Invalid input. Using default periods 20, 50, 200.")
                periods = [20, 50, 200]
        else:
            periods = [20, 50, 200]
        
        plot_data = self.renko_data.head(num_bars).reset_index(drop=True)
        
        # Compute SMAs on Renko close
        sma_dict = {}
        for period in periods:
            sma = plot_data['close'].rolling(window=period, min_periods=1).mean()
            sma_dict[period] = sma
        
        # Plot Renko chart
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(16, 8))
        for i, row in plot_data.iterrows():
            color = 'green' if row['direction'] == 1 else 'red'
            height = row['close'] - row['open']
            ax.bar(i, height, bottom=row['open'], width=0.8, color=color, edgecolor='black', alpha=0.7, linewidth=0.5)
        # Plot SMAs
        colors = ['blue', 'orange', 'purple', 'magenta', 'brown']
        for idx, period in enumerate(periods):
            ax.plot(range(len(plot_data)), sma_dict[period], color=colors[idx % len(colors)], linewidth=2, label=f'SMA {period}')
        ax.set_title(f'Renko Chart with SMAs (Periods: {", ".join(str(p) for p in periods)})', fontsize=14, fontweight='bold')
        ax.set_xlabel("Renko Brick Index")
        ax.set_ylabel("Price ($)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    tester = InteractiveRTHETest()
    tester.run() 