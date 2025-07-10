# Trading Strategy Backtesting System

A comprehensive Python system for backtesting Renko-based trading strategies with parameter optimization and detailed performance analysis.

## Features

- **Data Handling**: Load and manage CSV candlestick data with automatic column standardization
- **Renko Conversion**: Convert OHLC data to Renko bricks with optimal brick size calculation
- **Trading Strategies**: 
  - Renko Breakout Strategy
  - Renko Mean Reversion Strategy
  - Extensible base class for custom strategies
- **Backtesting Engine**: Run hundreds/thousands of parameter optimization iterations
- **Performance Analysis**: 
  - Profit factor, Sharpe ratio, maximum drawdown
  - Win rate, average win/loss, equity curves
  - Interactive visualizations with Plotly
- **Results Visualization**: 
  - Equity curves
  - Performance distributions
  - Parameter heatmaps
  - Interactive charts

## Installation

1. Clone or download the project files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Run with Sample Data (No Arguments)
```bash
python main.py
```

This will:
- Generate sample OHLC data
- Convert to Renko bricks
- Run a single backtest with default parameters
- Display results

### Run Full Optimization
```bash
python main.py --data-file "your_data.csv" --strategy "breakout" --iterations 500
```

### Command Line Options

- `--data-file`: Path to your CSV data file (default: data/sample_data.csv)
- `--strategy`: Strategy type - "breakout" or "mean_reversion" (default: breakout)
- `--iterations`: Number of optimization iterations (default: 100)
- `--brick-size`: Manual Renko brick size (default: auto-calculated)
- `--initial-capital`: Starting capital (default: $10,000)

## Data Format

Your CSV file should have these columns (case-insensitive):
- `date` or `Date`: Timestamp
- `open` or `Open`: Opening price
- `high` or `High`: High price
- `low` or `Low`: Low price
- `close` or `Close`: Closing price
- `volume` or `Volume`: Volume (optional)

Example:
```csv
date,open,high,low,close,volume
2023-01-01 09:00:00,100.50,101.20,100.30,100.90,1500
2023-01-01 10:00:00,100.90,102.10,100.80,101.80,2000
```

## Trading Strategies

### Renko Breakout Strategy
- **Entry**: Enter long/short on consecutive up/down bricks
- **Exit**: Stop loss, take profit, or trend reversal
- **Parameters**:
  - `consecutive_bricks`: Number of consecutive bricks for entry
  - `stop_loss_bricks`: Stop loss in brick units
  - `take_profit_bricks`: Take profit in brick units
  - `min_bricks_for_trend`: Minimum bricks for trend reversal exit

### Renko Mean Reversion Strategy
- **Entry**: Enter long on oversold conditions, short on overbought
- **Exit**: Exit on brick direction reversal
- **Parameters**:
  - `lookback_period`: Period to analyze for oversold/overbought
  - `oversold_threshold`: Number of down bricks for long entry
  - `overbought_threshold`: Number of up bricks for short entry

## Output Files

After running a backtest, you'll get:

1. **backtest_results.csv**: All parameter combinations and their results
2. **equity_curves.png**: Equity curves for top 5 results
3. **performance_distribution.png**: Distribution of performance metrics
4. **parameter_heatmap.png**: Heatmap showing parameter performance
5. **best_result_interactive.html**: Interactive equity curve for best result

## Example Usage

### Basic Usage
```python
from data_handler import DataHandler
from renko_converter import RenkoConverter
from trading_strategy import RenkoBreakoutStrategy
from backtest_engine import BacktestEngine
from results_analyzer import ResultsAnalyzer

# Load data
data_handler = DataHandler()
ohlc_data = data_handler.load_csv("your_data.csv")

# Convert to Renko
renko_converter = RenkoConverter(brick_size=1.0)
optimal_brick_size = renko_converter.get_optimal_brick_size(ohlc_data)
renko_converter = RenkoConverter(optimal_brick_size)
renko_data = renko_converter.convert_to_renko(ohlc_data)

# Run backtest
strategy = RenkoBreakoutStrategy(consecutive_bricks=3, stop_loss_bricks=2)
strategy.set_brick_size(optimal_brick_size)

backtest_engine = BacktestEngine(initial_capital=10000)
result = backtest_engine.run_single_backtest(strategy, renko_data)

# Analyze results
analyzer = ResultsAnalyzer()
print(analyzer.generate_detailed_report(result))
analyzer.plot_equity_curves([result])
```

### Parameter Optimization
```python
# Define parameter ranges
parameter_ranges = {
    'consecutive_bricks': [2, 3, 4, 5],
    'stop_loss_bricks': [1, 2, 3],
    'take_profit_bricks': [3, 4, 5, 6]
}

# Run optimization
results = backtest_engine.run_parameter_optimization(
    RenkoBreakoutStrategy,
    renko_data,
    parameter_ranges,
    max_iterations=1000
)

# Get best results
best_results = backtest_engine.get_best_results(results, metric='profit_factor', top_n=10)
analyzer.display_summary_stats(results)
```

## Performance Metrics

- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall percentage return
- **Average Win/Loss**: Average profit/loss per trade

## Custom Strategies

Create your own strategy by inheriting from `TradingStrategy`:

```python
class MyCustomStrategy(TradingStrategy):
    def __init__(self, my_param=10):
        super().__init__("MyCustomStrategy")
        self.my_param = my_param
    
    def should_enter_long(self, data, index):
        # Your entry logic here
        return False
    
    def should_enter_short(self, data, index):
        # Your entry logic here
        return False
    
    def should_exit_long(self, data, index):
        # Your exit logic here
        return False
    
    def should_exit_short(self, data, index):
        # Your exit logic here
        return False
```

## Troubleshooting

### Common Issues

1. **"CSV file not found"**: Make sure your data file is in the `data/` directory
2. **"No Renko bricks generated"**: Check your data quality and brick size
3. **"No trades executed"**: Adjust strategy parameters or check data
4. **Import errors**: Install all required dependencies with `pip install -r requirements.txt`

### Debug Mode

For debugging, add print statements in your strategy methods or use the quick test:

```bash
python main.py  # Runs quick test with sample data
```

## File Structure

```
trading-backtest/
├── main.py                 # Main execution script
├── data_handler.py         # CSV data loading and management
├── renko_converter.py      # OHLC to Renko conversion
├── trading_strategy.py     # Trading strategy implementations
├── backtest_engine.py      # Backtesting and optimization engine
├── results_analyzer.py     # Results analysis and visualization
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── data/                  # Data directory (created automatically)
    └── sample_data.csv    # Sample data (generated automatically)
```

## Contributing

Feel free to extend the system with:
- New trading strategies
- Additional performance metrics
- More visualization options
- Risk management features
- Portfolio optimization

## License

This project is open source. Use at your own risk for trading decisions. 