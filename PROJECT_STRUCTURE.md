# Trading Backtesting System - Project Structure

## Overview
This is a comprehensive trading strategy backtesting system with Renko chart analysis, featuring the R.T.H.E. (Renko Trend-Hedge Engine) strategy.

## Folder Structure

```
test test/
├── data/                          # Raw and processed CSV data files
│   └── bitcoin_merged_data.csv    # Merged Bitcoin OHLC data
│
├── strategies/                     # Trading strategy implementations
│   ├── trading_strategy.py        # Base strategy class
│   └── rthe_strategy.py           # R.T.H.E. strategy implementation
│
├── engine/                        # Backtesting and analysis engine
│   ├── backtest_engine.py         # Main backtesting engine
│   ├── results_analyzer.py        # Results analysis and visualization
│   ├── validation_suite.py        # Advanced validation tests
│   └── rthe_analyzer.py           # R.T.H.E. specific analyzer
│
├── utils/                         # Utility modules
│   ├── data_handler.py            # CSV loading and preprocessing
│   ├── renko_converter.py         # OHLC to Renko conversion
│   ├── csv_merger.py              # Multiple CSV file merger
│   └── merge_bitcoin_data.py      # Bitcoin data merger
│
├── tests/                         # Test and debug scripts
│   ├── debug_backtest.py          # Debug backtesting script
│   ├── test_rthe.py               # R.T.H.E. strategy test
│   └── test_csv_merger.py         # CSV merger test
│
├── main.py                        # Main entry point
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── PROJECT_STRUCTURE.md           # This file
```

## Key Components

### Strategies
- **Base Strategy**: Abstract trading strategy class with common functionality
- **R.T.H.E. Strategy**: Advanced Renko-based strategy with trend following and hedging

### Engine
- **Backtest Engine**: Core backtesting functionality with parameter optimization
- **Results Analyzer**: Performance metrics calculation and visualization
- **Validation Suite**: Advanced testing (in-sample, walk-forward, permutation tests)
- **R.T.H.E. Analyzer**: Strategy-specific analysis and metrics

### Utils
- **Data Handler**: CSV loading, validation, and preprocessing
- **Renko Converter**: OHLC to Renko brick conversion with optimal sizing
- **CSV Merger**: Multiple file merging with duplicate removal and gap filling

### Tests
- **Debug Scripts**: Individual component testing and validation
- **Strategy Tests**: Specific strategy performance testing

## Usage

### Quick Start
```bash
# Run R.T.H.E. strategy test
python tests/test_rthe.py

# Run full validation suite
python engine/validation_suite.py

# Run debug backtest
python tests/debug_backtest.py
```

### Data Requirements
- Place CSV files in `data/` folder
- CSV should have columns: `date`, `open`, `high`, `low`, `close`
- Supports various date formats (Unix timestamp, ISO8601, etc.)

### Strategy Parameters
- `brick_size`: Renko brick size (auto-optimized)
- `tsl_offset`: Trailing stop loss offset
- `hedge_size_ratio`: Hedge position size ratio
- `min_bricks_for_trend`: Minimum bricks for trend confirmation

## Features
- ✅ Renko chart conversion with optimal brick sizing
- ✅ R.T.H.E. strategy with core/hedge position management
- ✅ Advanced validation testing suite
- ✅ Parameter optimization with multiple metrics
- ✅ Comprehensive performance analysis
- ✅ Data preprocessing and validation
- ✅ Multiple CSV file merging
- ✅ Modular, extensible architecture 