# Renko Optimizations Results

This directory contains results from context-aware parameter optimization tests using Renko brick data.

## 📁 Directory Structure

```
renko_optimizations/
├── optimization_summary.json     # Complete summary with statistics
├── renko_metadata.json          # Renko conversion settings and statistics
├── chunk_results/               # Individual chunk results
│   ├── chunk_0.json            # Results for chunk 0
│   ├── chunk_1.json            # Results for chunk 1
│   └── ...                     # Additional chunks
└── README.md                   # This file
```

## 📊 Results Files

### `optimization_summary.json`
Contains overall optimization results including:
- Score statistics across all chunks
- Trend vs. consolidation percentages
- Most common parameter sets
- Parameter frequency analysis

### `renko_metadata.json`
Contains Renko conversion details:
- Brick size method (ATR-based or fixed)
- ATR settings (period, multiplier)
- Conversion statistics (total bricks, up/down distribution)
- Timestamp of conversion

### `chunk_results/`
Individual results for each time chunk:
- Best parameters for that chunk
- Classification metrics
- Score and performance data

## 🔍 How to Use

1. **View Summary**: Open `optimization_summary.json` for overall results
2. **Check Renko Settings**: Review `renko_metadata.json` for conversion details
3. **Analyze Chunks**: Browse `chunk_results/` for time-specific performance
4. **Compare Results**: Use the parameter frequency data to identify robust parameter sets

## 📈 Key Metrics

- **Score Statistics**: Mean, std, min, max scores across chunks
- **Trend Detection**: Percentage of trend vs. consolidation classification
- **Parameter Robustness**: Which parameter sets work consistently across time
- **Renko Performance**: How Renko bricks affect classification accuracy 