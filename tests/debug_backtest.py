import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_handler import DataHandler
from utils.renko_converter import RenkoConverter
from strategies.rthe_strategy import RTHEStrategy
from engine.backtest_engine import BacktestEngine

# Load data
print("Loading data...")
data_handler = DataHandler()
ohlc_data = data_handler.load_csv("data/bitcoin_merged_data.csv")
print(f"Loaded {len(ohlc_data)} OHLC bars")

# Convert to Renko
print("Converting to Renko...")
renko_converter = RenkoConverter(brick_size=1.0)
optimal_brick_size = renko_converter.get_optimal_brick_size(ohlc_data)
print(f"Optimal brick size: {optimal_brick_size}")

renko_converter = RenkoConverter(optimal_brick_size)
renko_data = renko_converter.convert_to_renko(ohlc_data)
print(f"Generated {len(renko_data)} Renko bricks")

# Test strategy directly
print("\n=== Testing R.T.H.E. Strategy Directly ===")
strategy = RTHEStrategy(
    brick_size=optimal_brick_size,
    tsl_offset=optimal_brick_size * 0.02,
    hedge_size_ratio=0.5,
    min_bricks_for_trend=2
)

# Run on first 100 bricks
for i in range(min(100, len(renko_data))):
    strategy.execute_trade(renko_data, i)

print(f"Core trades: {len(strategy.core_trades)}")
print(f"Hedge trades: {len(strategy.hedge_trades)}")
print(f"Total trades: {len(strategy.trades)}")

if strategy.core_trades:
    print(f"First core trade PnL: {strategy.core_trades[0].pnl:.4f}")

# Test with backtest engine
print("\n=== Testing with Backtest Engine ===")
backtest_engine = BacktestEngine(initial_capital=10000)

# Test single backtest
test_data = renko_data.head(100).reset_index(drop=True)
result = backtest_engine.run_single_backtest(strategy, test_data, {
    'brick_size': optimal_brick_size,
    'tsl_offset': optimal_brick_size * 0.02,
    'hedge_size_ratio': 0.5,
    'min_bricks_for_trend': 2
})

print(f"Backtest result - Trades: {result.total_trades}")
print(f"Profit factor: {result.profit_factor}")
print(f"Total return: {result.total_return:.4f}")

# Test parameter optimization
print("\n=== Testing Parameter Optimization ===")
parameter_ranges = {
    'brick_size': [optimal_brick_size * 0.8, optimal_brick_size, optimal_brick_size * 1.2],
    'tsl_offset': [optimal_brick_size * 0.01, optimal_brick_size * 0.02, optimal_brick_size * 0.03],
    'hedge_size_ratio': [0.3, 0.5, 0.7],
    'min_bricks_for_trend': [2, 3]
}

results = backtest_engine.run_parameter_optimization(
    RTHEStrategy, test_data, parameter_ranges, max_iterations=10
)

print(f"Optimization results: {len(results)}")
if results:
    best = max(results, key=lambda x: x.profit_factor)
    print(f"Best profit factor: {best.profit_factor}")
    print(f"Best total trades: {best.total_trades}")
    print(f"Best parameters: {best.parameters}") 