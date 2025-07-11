# Range Lookback Periods for Context Tuning
# These periods will be used in grid search optimization to find optimal lookback periods
# for detecting consolidation vs. expansion based on price range analysis

range_lookbacks = [10, 20, 30, 50]

# Description of each lookback period:
# 10: Short-term range analysis (very responsive, may be noisy)
# 20: Medium-term range analysis (balanced approach)
# 30: Long-term range analysis (smoother, more reliable)
# 50: Very long-term range analysis (most stable, may miss short-term changes)

# Usage in optimization:
# - Lower values (10-20): More sensitive to short-term range changes
# - Higher values (30-50): More stable, better for identifying major consolidation periods
# - 20-30: Balanced approach for most market conditions 