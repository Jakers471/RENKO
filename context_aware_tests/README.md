# Context-Aware Market State Testing

A comprehensive system for data-driven parameter optimization using rolling window testing to find optimal lookback periods for market state classification (trend vs. consolidation detection).

## 🎯 **Goal**

Find optimal lookback periods for detecting trend vs. consolidation with **data-driven testing** — not guessing. Uses rolling window analysis to test parameter robustness across different market conditions.

## 📁 **File Structure**

```
context_aware_tests/
├── config.py                           # Central configuration & thresholds
├── data/
│   ├── load_data.py                    # Data loading utilities
│   └── renko_converter.py              # OHLC to Renko conversion
├── indicators/
│   ├── atr.py                         # ATR calculation & volatility analysis
│   ├── adx.py                         # ADX calculation & trend strength
│   └── ma_utils.py                    # MA, slope, spread, range compression
├── classifier/
│   └── market_state_classifier.py     # Classifies bars: trend or consolidation
├── optimizer/
│   ├── grid_search_optimizer.py       # Tests parameter combinations & scores
│   └── range_lookbacks.py             # Range lookback periods [10,20,30,50]
├── chunk_runner/
│   └── rolling_window_tester.py       # Rolling window testing logic
├── visualizers/
│   └── renko_state_overlay.py         # Renko state visualization
├── results/
│   ├── optimization_summary.json      # Stores best params per chunk, scores
│   └── renko_state_viz/               # Renko visualization outputs
├── run_rolling_window_test.py         # Main runner script
├── test_renko_state_visualization.py  # Test script for visualization
└── README.md                          # This file
```

## 🔧 **How It Works**

### **1. Data Processing**
- **OHLC Data**: Load and validate traditional candlestick data
- **Renko Conversion**: Optional conversion to Renko bricks with ATR-based or fixed brick size
- **Data Validation**: Ensure complete OHLCV dataset with proper formatting

### **2. Rolling Window Testing**
- **Window Size**: 2000 bars/bricks per chunk (configurable)
- **Step Size**: 500 bars/bricks overlap between chunks (configurable)
- **Total Combinations**: 144 (4×3×3×4 parameter combinations)

### **2. Parameter Grid**
- **ATR Lookbacks**: [5, 10, 14, 21]
- **ADX Lookbacks**: [7, 14, 21]
- **MA Lengths**: [(10,20,50), (20,50,100), (20,50,200)]
- **Range Lookbacks**: [10, 20, 30, 50]

### **3. Market State Classification**
For each chunk, the system:
1. Calculates ATR, ADX, Moving Averages, and Range analysis
2. Uses **trend score system** (2 out of 5 conditions needed)
3. Classifies each bar as 'trend' or 'consolidation'
4. Evaluates classification quality using enhanced scoring

### **4. Enhanced Scoring System**
- **40% weight**: State distribution balance (30-70% ideal)
- **30% weight**: State transition frequency (not too choppy)
- **30% weight**: Trend detection success (rewards finding trends)

## 🚀 **Usage**

### **Quick Start**
```bash
cd context_aware_tests
python run_rolling_window_test.py
```

### **Config-Driven Runs**
```bash
# Use specific configuration
python run_rolling_window_test.py --config renko_fine

# Custom run ID
python run_rolling_window_test.py --config default --run-id my_test_001

# Test specific chunk
python run_rolling_window_test.py --config quick_test --chunk-index 5

# Limit combinations for faster testing
python run_rolling_window_test.py --config default --max-combinations 50
```

### **Parallel Testing**
```bash
# Run multiple configurations simultaneously
python run_parallel_tests.py --configs default renko_fine quick_test --max-parallel 3

# Test specific chunk across multiple configs
python run_parallel_tests.py --configs default renko_fine --chunk-index 0 --max-parallel 2
```

### **Available Configurations**
- **`default`**: Standard parameter optimization (144 combinations)
- **`renko_fine`**: Renko with fine-tuned parameters (18 combinations)
- **`renko_coarse`**: Renko with coarse parameters (4 combinations)
- **`quick_test`**: Fast test with limited combinations (1 combination)

### **Data Types**
1. **OHLC bars** - Traditional candlestick data
2. **Renko bricks** - Converted from OHLC with configurable brick size

### **Test Modes**
1. **Full rolling window test** - Tests all chunks
2. **Single chunk test** - Test one chunk for debugging
3. **Custom chunk test** - Test specific chunk by index

### **Renko State Visualization**
```bash
cd context_aware_tests
python test_renko_state_visualization.py
```

**Features:**
- 🟩 **Green bricks** = Trend periods
- 🟨 **Yellow bricks** = Consolidation periods
- 📊 **Timeline strip** = State transitions over time
- 💾 **Auto-save** = Plots saved to `results/renko_state_viz/`

**Parameters:**
- **ATR-based brick size** - Dynamic sizing based on volatility
- **Fixed brick size** - Consistent brick size for comparison
- **Custom classification** - Adjustable ATR, ADX, MA, and range parameters

## 🔧 **Parallel Testing Safeguards**

### **✅ Unique Output Per Run**
- Each run gets a unique run ID: `run_20250711_143022_a1b2c3d4`
- Results saved to isolated directories: `results/run_20250711_143022_a1b2c3d4/`
- No file conflicts between parallel runs

### **✅ No Shared Global State**
- Each subprocess is fully isolated
- No shared caches or global variables
- Independent parameter sets and configurations

### **✅ Config-Driven Runs**
- Predefined test configurations in `config.py`
- Easy to modify parameters without code changes
- Support for custom run IDs and data files

### **✅ Parallel Launch Script**
- `run_parallel_tests.py` manages multiple subprocesses
- Configurable max parallel processes
- Automatic result collection and summary

### **Example Output**
```
🔄 ROLLING WINDOW PARAMETER OPTIMIZATION
============================================================
Window size: 2000 bars
Step size: 500 bars
Total data: 10424 bars

🔍 TESTING CHUNK 0
   Bars: 2000 (start: 0, end: 2000)
   ✅ Best score: 0.8234
   📊 Trend: 45.2%, Consolidation: 54.8%

📊 ROLLING WINDOW OPTIMIZATION SUMMARY
============================================================
Total chunks tested: 17

🏆 Score Statistics:
  Mean: 0.7123
  Std:  0.0891
  Min:  0.6234
  Max:  0.8234
  Median: 0.7156

🎯 Most Common Parameter Set (appears 5 times):
  ATR: 14, ADX: 14, MA: (20, 50, 200), Range: 30
```

## 📊 **Results Analysis**

### **What You Get**
1. **Top parameters per chunk** - Best performing parameter set for each time period
2. **Score distribution over time** - How parameter performance varies
3. **Parameter frequency analysis** - Which parameter sets perform consistently
4. **Trend vs. consolidation statistics** - State distribution analysis

### **Output Files**
- `results/optimization_summary.json` - Complete summary with statistics
- `results/chunk_results/chunk_X.json` - Individual chunk results
- Console output with real-time progress and final summary

## ⚙️ **Configuration**

### **Key Settings in `config.py`**
```python
# Rolling Window
WINDOW_SIZE = 2000      # Bars per chunk
STEP_SIZE = 500         # Bars to slide forward

# Classification Thresholds
ADX_TREND_THRESHOLD = 15    # ADX threshold for trend
ATR_PERCENTILE = 50         # ATR percentile for volatility
MA_SPREAD_THRESHOLD = 0.01  # MA spread threshold
TREND_SCORE_MIN = 2         # Minimum trend score (out of 5)

# Scoring Weights
BALANCE_WEIGHT = 0.4        # State distribution balance
TRANSITION_WEIGHT = 0.3     # State transition frequency
TREND_DETECTION_WEIGHT = 0.3 # Trend detection success
```

## 🔍 **Debugging & Development**

### **Single Chunk Testing**
For faster iteration during development:
```python
# Test only chunk 0
python run_rolling_window_test.py
# Select mode 2 (Single chunk test)
# Enter chunk index: 0
```

### **Custom Parameter Testing**
Modify parameter grids in:
- `config.py` - Main parameter lists
- `optimizer/range_lookbacks.py` - Range lookback periods

### **Debug Output**
- First 3 combinations show detailed indicator values
- Progress tracking every 10 combinations
- Manual trend detection sanity checks

## 📈 **Advanced Features**

### **Parameter Robustness**
- Tests parameters across different market conditions
- Identifies which parameter sets work consistently
- Shows parameter frequency across chunks

### **Time-Based Analysis**
- Each chunk represents a different time period
- Shows how optimal parameters change over time
- Identifies market regime changes

### **Flexible Testing**
- Adjustable window and step sizes
- Configurable max combinations per chunk
- Support for custom time ranges

## 🎯 **Benefits**

1. **Data-Driven**: No guessing - let the data determine optimal parameters
2. **Robust**: Tests across multiple time periods and market conditions
3. **Flexible**: Easy to modify parameters, thresholds, and testing approach
4. **Comprehensive**: Full analysis with detailed results and statistics
5. **Fast Development**: Single chunk testing for quick iteration

## 🔮 **Future Enhancements**

1. **Cross-Validation**: Test on multiple datasets
2. **Parameter Sensitivity**: Analyze parameter importance
3. **Real-time Adaptation**: Dynamic parameter adjustment
4. **Advanced Scoring**: Include trading performance metrics
5. **Enhanced Visualization**: Interactive charts, heatmaps, and real-time state tracking

## 📝 **Notes**

- Results may vary based on market conditions and time periods
- Consider running on multiple datasets for robustness
- Single chunk testing is recommended for initial development
- The system uses enhanced scoring to avoid extreme parameter biases 