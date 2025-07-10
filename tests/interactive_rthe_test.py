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
        """Explain R.T.H.E. strategy parameters"""
        print("\n" + "="*60)
        print("R.T.H.E. STRATEGY PARAMETERS EXPLANATION")
        print("="*60)
        
        params = {
            "brick_size": {
                "description": "Size of each Renko brick (price movement)",
                "default": f"{self.optimal_brick_size:.2f}",
                "effect": "Larger = fewer signals but stronger trends, Smaller = more signals but noise",
                "range": f"{self.optimal_brick_size * 0.5:.2f} - {self.optimal_brick_size * 2:.2f}"
            },
            "tsl_offset": {
                "description": "Trailing stop loss offset from brick boundaries",
                "default": f"{self.optimal_brick_size * 0.02:.2f}",
                "effect": "Larger = wider stops (less risk), Smaller = tighter stops (more risk)",
                "range": f"{self.optimal_brick_size * 0.01:.2f} - {self.optimal_brick_size * 0.05:.2f}"
            },
            "hedge_size_ratio": {
                "description": "Ratio of hedge position size to core position",
                "default": "0.5",
                "effect": "Higher = more hedging protection, Lower = less hedging",
                "range": "0.1 - 1.0"
            },
            "min_bricks_for_trend": {
                "description": "Minimum consecutive bricks for trend confirmation",
                "default": "2",
                "effect": "Higher = stronger trend requirement, Lower = more entries",
                "range": "1 - 5"
            }
        }
        
        for param, info in params.items():
            print(f"\n📊 {param.upper()}:")
            print(f"   Description: {info['description']}")
            print(f"   Default: {info['default']}")
            print(f"   Effect: {info['effect']}")
            print(f"   Recommended Range: {info['range']}")
    
    def get_user_input(self):
        """Get user input for test parameters"""
        print("\n" + "="*60)
        print("TEST CONFIGURATION")
        print("="*60)
        
        # Number of bars to test
        max_bars = len(self.renko_data)
        while True:
            try:
                num_bars = input(f"Number of Renko bars to test (1-{max_bars}): ").strip()
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
        
        return {
            'num_bars': num_bars,
            'brick_size': brick_size,
            'tsl_offset': tsl_offset,
            'hedge_size_ratio': hedge_ratio,
            'min_bricks_for_trend': min_bricks
        }
    
    def run_single_test(self, params):
        """Run a single test with given parameters"""
        print(f"\n{'='*60}")
        print("RUNNING R.T.H.E. STRATEGY TEST")
        print(f"{'='*60}")
        print(f"Testing on {params['num_bars']} Renko bars")
        print(f"Parameters: {params}")
        
        # Create strategy
        strategy = RTHEStrategy(
            brick_size=params['brick_size'],
            tsl_offset=params['tsl_offset'],
            hedge_size_ratio=params['hedge_size_ratio'],
            min_bricks_for_trend=params['min_bricks_for_trend']
        )
        
        # Run strategy
        test_data = self.renko_data.head(params['num_bars']).reset_index(drop=True)
        
        print(f"Running strategy on {len(test_data)} bars...")
        for i in range(len(test_data)):
            if i % 500 == 0:  # Progress update every 500 bars
                print(f"  Processing bar {i+1}/{len(test_data)} ({(i+1)/len(test_data)*100:.1f}%)")
            strategy.execute_trade(test_data, i)
        print(f"✅ Strategy execution complete!")
        
        # Calculate results
        backtest_engine = BacktestEngine(initial_capital=10000)
        result = backtest_engine.run_single_backtest(strategy, test_data, params)
        
        # Display results
        print(f"\n📈 RESULTS:")
        print(f"   Total Trades: {result.total_trades}")
        print(f"   Core Trades: {len(strategy.core_trades)}")
        print(f"   Hedge Trades: {len(strategy.hedge_trades)}")
        print(f"   Hedge Activations: {strategy.hedge_activations}")
        print(f"   TSL Hits: {strategy.tsl_hits}")
        print(f"   Profit Factor: {result.profit_factor:.3f}")
        print(f"   Total Return: {result.total_return*100:.2f}%")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"   Max Drawdown: {result.max_drawdown*100:.2f}%")
        print(f"   Win Rate: {result.win_rate*100:.1f}%")
        
        if strategy.core_trades:
            print(f"\n📊 FIRST CORE TRADE:")
            trade = strategy.core_trades[0]
            print(f"   Entry: {trade.entry_price:.2f}")
            print(f"   Exit: {trade.exit_price:.2f}")
            print(f"   PnL: {trade.pnl*100:.2f}%")
            print(f"   TSL Hit: {trade.tsl_hit}")
            print(f"   Hedge Activated: {trade.hedge_activated}")
        
        # Show Renko chart with trades
        print(f"\n📈 Generating Renko chart with trades...")
        self._create_renko_plot(test_data, strategy, params['num_bars'])
        
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
        
        # Run sweep
        results = []
        equity_curves = []
        test_data_ohlc = self.ohlc_data.copy()  # Always use full OHLC data for Renko conversion
        backtest_engine = BacktestEngine(initial_capital=10000)
        renko_converter = RenkoConverter(1.0)  # Placeholder, will set brick_size per combo
        
        print(f"\nRunning parameter sweep...")
        for i, (brick_size, tsl_offset, hedge_ratio, min_brick) in enumerate(
            product(brick_sizes, tsl_offsets, hedge_ratios, min_bricks)
        ):
            if i % 10 == 0:  # Progress update every 10 combinations
                print(f"  Testing combination {i+1}/{total_combinations} ({(i+1)/total_combinations*100:.1f}%)")
            params = {
                'brick_size': brick_size,
                'tsl_offset': tsl_offset,
                'hedge_size_ratio': hedge_ratio,  # NOTE: Make sure this is used in RTHEStrategy logic
                'min_bricks_for_trend': min_brick
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
    
    def _create_renko_plot(self, renko_data, strategy, num_bars):
        """Create the actual Renko plot with trades"""
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
            # Collect all trade points for batch plotting
            core_entry_x, core_entry_y = [], []
            core_exit_x, core_exit_y = [], []
            core_entry_labels, core_exit_labels = [], []
            
            for trade in strategy.core_trades:
                # Find entry and exit indices
                entry_idx = renko_data[renko_data['close'] == trade.entry_price].index
                exit_idx = renko_data[renko_data['close'] == trade.exit_price].index
                
                if len(entry_idx) > 0:
                    core_entry_x.append(entry_idx[0])
                    core_entry_y.append(trade.entry_price)
                    core_entry_labels.append(f'Core Entry\n${trade.entry_price:.0f}')
                
                if len(exit_idx) > 0:
                    core_exit_x.append(exit_idx[0])
                    core_exit_y.append(trade.exit_price)
                    core_exit_labels.append(f'Core Exit\n${trade.exit_price:.0f}\n{trade.pnl*100:.1f}%')
            
            # Batch plot core trades
            if core_entry_x:
                ax.scatter(core_entry_x, core_entry_y, marker='^', color='blue', 
                          s=120, zorder=5, label='Core Entry')
                # Only add annotations for first few trades to avoid clutter
                for i, (x, y, label) in enumerate(zip(core_entry_x, core_entry_y, core_entry_labels)):
                    if i < 5:  # Limit annotations for large datasets
                        ax.annotate(label, (x, y), xytext=(10, 10), textcoords='offset points', 
                                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            if core_exit_x:
                ax.scatter(core_exit_x, core_exit_y, marker='v', color='navy', 
                          s=120, zorder=5, label='Core Exit')
                # Only add annotations for first few trades to avoid clutter
                for i, (x, y, label) in enumerate(zip(core_exit_x, core_exit_y, core_exit_labels)):
                    if i < 5:  # Limit annotations for large datasets
                        ax.annotate(label, (x, y), xytext=(10, -20), textcoords='offset points', 
                                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # Overlay Hedge trades (optimized for large datasets)
        if len(strategy.hedge_trades) > 0:
            # Collect all hedge trade points for batch plotting
            hedge_entry_x, hedge_entry_y = [], []
            hedge_exit_x, hedge_exit_y = [], []
            hedge_entry_labels, hedge_exit_labels = [], []
            
            for trade in strategy.hedge_trades:
                # Find entry and exit indices
                entry_idx = renko_data[renko_data['close'] == trade.entry_price].index
                exit_idx = renko_data[renko_data['close'] == trade.exit_price].index
                
                if len(entry_idx) > 0:
                    hedge_entry_x.append(entry_idx[0])
                    hedge_entry_y.append(trade.entry_price)
                    hedge_entry_labels.append(f'Hedge Entry\n${trade.entry_price:.0f}')
                
                if len(exit_idx) > 0:
                    hedge_exit_x.append(exit_idx[0])
                    hedge_exit_y.append(trade.exit_price)
                    hedge_exit_labels.append(f'Hedge Exit\n${trade.exit_price:.0f}\n{trade.pnl*100:.1f}%')
            
            # Batch plot hedge trades
            if hedge_entry_x:
                ax.scatter(hedge_entry_x, hedge_entry_y, marker='o', color='orange', 
                          s=100, zorder=5, label='Hedge Entry')
                # Only add annotations for first few trades to avoid clutter
                for i, (x, y, label) in enumerate(zip(hedge_entry_x, hedge_entry_y, hedge_entry_labels)):
                    if i < 5:  # Limit annotations for large datasets
                        ax.annotate(label, (x, y), xytext=(10, 10), textcoords='offset points', 
                                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
            
            if hedge_exit_x:
                ax.scatter(hedge_exit_x, hedge_exit_y, marker='x', color='darkorange', 
                          s=100, zorder=5, label='Hedge Exit')
                # Only add annotations for first few trades to avoid clutter
                for i, (x, y, label) in enumerate(zip(hedge_exit_x, hedge_exit_y, hedge_exit_labels)):
                    if i < 5:  # Limit annotations for large datasets
                        ax.annotate(label, (x, y), xytext=(10, -20), textcoords='offset points', 
                                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
        
        # Finalize plot
        ax.set_title(f'Renko Chart with R.T.H.E. Strategy Trades\n'
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
        fig.suptitle('R.T.H.E. Strategy Parameter Sweep Results', fontsize=16, fontweight='bold')
        
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
    
    def run(self):
        """Main interactive test runner"""
        print("🚀 INTERACTIVE R.T.H.E. STRATEGY TESTER")
        print("="*60)
        
        # Load data
        self.load_data()
        
        while True:
            print(f"\n{'='*60}")
            print("MAIN MENU")
            print("="*60)
            print("1. Explain Strategy Parameters")
            print("2. Run Single Test (Custom Parameters)")
            print("3. Run Parameter Sweep (Multiple Combinations)")
            print("4. Plot Renko Chart with Trades")
            print("5. Refresh Bitcoin Data (Re-run Merger)")
            print("6. Show available timeframes for a symbol")
            print("7. Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                self.explain_parameters()
            
            elif choice == '2':
                params = self.get_user_input()
                self.run_single_test(params)
            
            elif choice == '3':
                self.run_parameter_sweep()
            
            elif choice == '4':
                self.plot_renko_trades()
            
            elif choice == '5':
                self.refresh_data()
            
            elif choice == '6':
                symbol = input("Enter symbol (e.g., BTCUSD): ").strip().upper()
                timeframes = self.data_handler.load_symbol_timeframes(symbol)
                if timeframes:
                    print(f"Loaded timeframes for {symbol}:")
                    for tf, df in timeframes.items():
                        print(f"  {tf}: {len(df)} rows")
                else:
                    print(f"No timeframes found for {symbol}.")
            
            elif choice == '7':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-7.")

if __name__ == "__main__":
    tester = InteractiveRTHETest()
    tester.run() 