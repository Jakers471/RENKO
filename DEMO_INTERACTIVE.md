# 🚀 Interactive R.T.H.E. Strategy Tester - Demo Guide

## Quick Start

```bash
# Run the interactive tester
python tests/interactive_rthe_test.py

# Or use the test runner
python run_tests.py interactive
```

## Features

### 1. 📊 **Parameter Explanation**
- **Option 1**: Learn about all R.T.H.E. strategy parameters
- Understand what each parameter does and its effects
- See recommended ranges and defaults

### 2. 🎯 **Single Test Mode**
- **Option 2**: Test with custom parameters
- Choose number of Renko bars (1 to 5256, or "max")
- Customize all strategy parameters:
  - `brick_size`: Renko brick size (default: 333.82)
  - `tsl_offset`: Trailing stop loss offset (default: 6.68)
  - `hedge_size_ratio`: Hedge position ratio (default: 0.5)
  - `min_bricks_for_trend`: Minimum trend bricks (default: 2)

### 3. 🔄 **Parameter Sweep Mode**
- **Option 3**: Test hundreds of parameter combinations
- Automatically tests 90 combinations (5 brick sizes × 3 TSL offsets × 3 hedge ratios × 2 min bricks)
- Generates comprehensive visualizations
- Shows top 10 results by profit factor

## Example Usage

### Single Test Example:
```
Select option (1-4): 2

Number of Renko bars to test (1-5256): 1000

Enter strategy parameters (press Enter for defaults):
Brick size (default: 333.82): 400
TSL offset (default: 6.68): 8
Hedge size ratio (default: 0.5): 0.7
Min bricks for trend (default: 2): 3

📈 RESULTS:
   Total Trades: 15
   Core Trades: 8
   Hedge Trades: 7
   Profit Factor: 2.847
   Total Return: 12.34%
   Sharpe Ratio: 1.234
   Max Drawdown: 3.45%
   Win Rate: 73.3%
```

### Parameter Sweep Example:
```
Select option (1-4): 3

Number of Renko bars for sweep (1-5256): 2000

Testing 90 parameter combinations...
Completed 10/90 combinations...
Completed 20/90 combinations...
...

🏆 TOP 10 RESULTS BY PROFIT FACTOR:
================================================================================
 1. PF: 4.123 | Return: 18.45% | Trades: 25 | Brick: 250 | TSL: 3 | Hedge: 0.7 | Min: 2
 2. PF: 3.987 | Return: 16.78% | Trades: 22 | Brick: 300 | TSL: 6 | Hedge: 0.5 | Min: 2
 3. PF: 3.845 | Return: 15.23% | Trades: 28 | Brick: 400 | TSL: 8 | Hedge: 0.3 | Min: 3
 ...
```

## Generated Visualizations

The parameter sweep creates a comprehensive 6-panel visualization:

1. **Profit Factor Heatmap** (Brick Size vs TSL Offset)
2. **Total Return Heatmap** (Brick Size vs Hedge Ratio)
3. **Sharpe Ratio Heatmap** (TSL Offset vs Hedge Ratio)
4. **Profit Factor vs Total Return Scatter** (colored by trade count)
5. **Profit Factor Distribution** histogram
6. **Trade Count vs Profit Factor** scatter plot

## Strategy Parameters Explained

### 📊 **BRICK_SIZE**
- **What**: Size of each Renko brick (price movement)
- **Default**: 333.82 (auto-optimized)
- **Effect**: Larger = fewer signals but stronger trends, Smaller = more signals but noise
- **Range**: 167 - 668

### 📊 **TSL_OFFSET**
- **What**: Trailing stop loss offset from brick boundaries
- **Default**: 6.68 (2% of brick size)
- **Effect**: Larger = wider stops (less risk), Smaller = tighter stops (more risk)
- **Range**: 3.34 - 16.69

### 📊 **HEDGE_SIZE_RATIO**
- **What**: Ratio of hedge position size to core position
- **Default**: 0.5
- **Effect**: Higher = more hedging protection, Lower = less hedging
- **Range**: 0.1 - 1.0

### 📊 **MIN_BRICKS_FOR_TREND**
- **What**: Minimum consecutive bricks for trend confirmation
- **Default**: 2
- **Effect**: Higher = stronger trend requirement, Lower = more entries
- **Range**: 1 - 5

## Tips for Testing

1. **Start Small**: Test with 100-500 bars first to get quick results
2. **Use Parameter Sweep**: Let the system find optimal combinations
3. **Focus on Profit Factor**: >2.0 is good, >3.0 is excellent
4. **Check Trade Count**: Too few trades (<10) may be unreliable
5. **Monitor Drawdown**: Keep max drawdown <10% for safety

## Output Files

- **Visualization**: `rthe_parameter_sweep_YYYYMMDD_HHMMSS.png`
- **Results**: Displayed in console with detailed metrics
- **Best Parameters**: Top 10 combinations ranked by profit factor

## Next Steps

After finding good parameters:
1. Run validation suite: `python run_tests.py validation`
2. Test on different time periods
3. Implement in live trading (with proper risk management)
4. Monitor and adjust parameters as needed

Happy testing! 🎉 