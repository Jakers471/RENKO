# System V1 - Renko Trading System with Moving Averages

## 🎯 Overview

System V1 is a focused, robust Renko-based trading system with moving average confirmation. The system uses 20, 50, and 200-day SMAs for trend alignment and entry confirmation, with optional counter-trend trading.

## 🚀 Quick Start

### Launch System V1 Tester
```bash
python tests/interactive_rthe_test.py
```

### Or use the Quick Menu
```bash
python quick_menu.py
```

## 📊 System V1 Components

### Core Strategy
- **Renko Brick Analysis**: Converts price movement to bricks
- **Moving Average Analysis**: 20, 50, 200 SMA trend alignment
- **MA-Based Entry Logic**: Enters on MA crossover confirmation
- **Trend-Following Focus**: Long entries with bullish MA alignment
- **Counter-Trend Toggle**: Optional short entries when enabled
- **Hedge Protection**: Reduces risk during reversals
- **Trailing Stop Loss**: Manages position risk

### MA-Based Entry Rules

#### LONG ENTRIES (Always Enabled)
- 20 SMA crosses above 50 SMA (bullish crossover)
- Both 20 and 50 SMA are upsloping
- Price above 20 SMA, 20 SMA above 50 SMA, 50 SMA above 200 SMA
- Renko brick confirms trend direction

#### SHORT ENTRIES (Counter-Trend Toggle)
- 20 SMA crosses below 50 SMA (bearish crossover)
- Both 20 and 50 SMA are downsloping
- Price below 20 SMA, 20 SMA below 50 SMA, 50 SMA below 200 SMA
- Renko brick confirms trend direction

### Configurable Parameters
1. **Brick Size**: Size of each Renko brick (default: auto-calculated)
2. **TSL Offset**: Trailing stop loss offset (default: 2% of brick size)
3. **Hedge Size Ratio**: Hedge position size ratio (default: 0.5)
4. **Min Bricks for Trend**: Trend confirmation requirement (default: 2)
5. **MA 20 Period**: 20-day SMA period (default: 20)
6. **MA 50 Period**: 50-day SMA period (default: 50)
7. **MA 200 Period**: 200-day SMA period (default: 200)
8. **Counter-Trend Toggle**: Enable/disable short entries (default: disabled)

### Hedge Logic Options
1. **Single Brick**: Hedge on any reversal brick
2. **Multiple Bricks**: Hedge after N consecutive reversals
3. **Price Level**: Hedge only if price breaks key levels
4. **Trend Strength**: Hedge only if reversal trend is stronger
5. **Combined**: All conditions must be met

## 🧪 Testing Framework

### 1. System Selection
- Choose System V1 from available options
- All tests and results are labeled with system name

### 2. Counter-Trend Setting
- Enable or disable counter-trend entries
- Warning about higher risk for short positions

### 3. System V1 Rules & Parameters
- View detailed explanation of all parameters
- Understand MA-based entry rules
- See recommended ranges and effects

### 4. Single Test
- Quick parameter validation
- Customize all parameters including MAs
- See immediate results with system labeling

### 5. Parameter Sweep
- Find optimal settings
- Test multiple combinations
- Identify best parameters
- Results labeled with system name

### 6. Monte Carlo Test
- Robustness testing
- Random data variations
- Statistical significance

### 7. Walk Forward Test
- Out-of-sample validation
- Rolling window analysis
- Real-world performance

### 8. In-Sample Excellence
- Statistical significance testing
- Compare vs random trading
- P-value analysis

## 📈 Results Analysis

### Key Metrics
- **Profit Factor**: Ratio of gross profit to gross loss
- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### Visualizations
- **Renko Charts**: Price action with trade markers and moving averages
- **Equity Curves**: Performance over time with system labeling
- **Parameter Heatmaps**: Sensitivity analysis
- **Distribution Plots**: Statistical analysis

## 🔧 Customization

### Moving Average Parameters
- Adjust MA periods (20, 50, 200-day)
- Test different MA combinations
- Optimize for specific timeframes

### Entry Rules
- Modify MA crossover conditions
- Adjust trend alignment requirements
- Fine-tune entry timing

### Risk Management
- Counter-trend toggle for short positions
- Hedge protection settings
- Trailing stop loss parameters

## 📁 Project Structure

```
├── tests/
│   └── interactive_rthe_test.py    # Main System V1 tester
├── strategies/
│   └── rthe_strategy.py            # System V1 implementation
├── engine/
│   └── backtest_engine.py          # Backtesting engine
├── utils/
│   ├── data_handler.py             # Data management
│   ├── renko_converter.py          # Renko conversion
│   └── merge_bitcoin_data.py       # Data merger
├── data/                           # Market data
├── quick_menu.py                   # Quick access menu
└── main.py                         # Command-line interface
```

## 🎯 Development Approach

### Focus on One Strategy
- Build System V1 thoroughly with MA integration
- Test extensively with all validation methods
- Optimize parameters systematically
- Document all findings

### MA-Based Optimization
- Test different MA periods
- Validate crossover signals
- Optimize trend alignment rules
- Balance signal frequency vs quality

### Statistical Validation
- Use multiple testing methods
- Ensure statistical significance
- Validate out-of-sample performance
- Monitor for overfitting

## 🚨 Best Practices

1. **Start Simple**: Begin with default MA parameters
2. **Test Thoroughly**: Use all validation methods
3. **Document Changes**: Track all modifications
4. **Validate Results**: Ensure statistical significance
5. **Monitor Performance**: Watch for degradation
6. **Use Counter-Trend Carefully**: Higher risk for short positions

## 📞 Support

For questions or issues:
1. Check the parameter explanations in the menu
2. Review the MA-based entry rules
3. Examine the generated charts and results
4. Use the validation tests to verify performance

---

**System V1 - MA-Enhanced, Focused, Robust** 🚀 