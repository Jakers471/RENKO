"""
Central configuration for context-aware market state testing
"""

import os

# Data Configuration
DATA_FILES = [
    "../data/bitcoin_merged_data.csv",
    "../data/merged_ohlc_data.csv", 
    "../data/my_merged_data.csv"
]

# Rolling Window Configuration
WINDOW_SIZE = 2000  # Number of bars per chunk
STEP_SIZE = 500     # Bars to slide forward between chunks
MIN_CHUNK_SIZE = 1000  # Minimum bars required for a valid chunk

# Parameter Grid
ATR_LOOKBACKS = [5, 10, 14, 21]
ADX_LOOKBACKS = [7, 14, 21]
MA_LENGTHS = [(10, 20, 50), (20, 50, 100), (20, 50, 200)]
RANGE_LOOKBACKS = [10, 20, 30, 50]

# Classification Thresholds
ADX_TREND_THRESHOLD = 15  # Lowered from 25 for looser detection
ATR_PERCENTILE = 50       # Use 50th percentile for ATR comparison
MA_SPREAD_THRESHOLD = 0.01  # 1% spread for MA alignment
SLOPE_MULTIPLIER = 0.25   # Multiplier for price slope threshold
RANGE_PERCENTILE = 70     # 70th percentile for range expansion
TREND_SCORE_MIN = 2       # Minimum score out of 5 to classify as trend

# Scoring Weights
BALANCE_WEIGHT = 0.4      # State distribution balance
TRANSITION_WEIGHT = 0.3   # State transition frequency  
TREND_DETECTION_WEIGHT = 0.3  # Trend detection success

# Output Configuration
RESULTS_DIR = "results"
OPTIMIZATION_SUMMARY_FILE = "optimization_summary.json"
CHUNK_RESULTS_DIR = "chunk_results"

# Debug Configuration
DEBUG_FIRST_N_COMBINATIONS = 3  # Debug first N combinations
PROGRESS_UPDATE_INTERVAL = 10   # Show progress every N combinations

# Renko Configuration
RENKO_USE_ATR = True           # Use ATR-based brick size
RENKO_ATR_PERIOD = 14          # ATR period for brick size calculation
RENKO_ATR_MULTIPLIER = 1.0     # ATR multiplier for brick size
RENKO_FIXED_BRICK_SIZE = None  # Fixed brick size (if not using ATR)

# Results Configuration
RENKO_RESULTS_DIR = "renko_optimizations"  # Directory for Renko results

# File Paths
def get_project_root():
    """Get the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

def get_data_path(filename):
    """Get full path to data file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, filename)

def get_results_path(filename):
    """Get full path to results file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "results", filename) 

# Parallel Run Configuration
import uuid
import datetime

def generate_run_id(prefix: str = "run") -> str:
    """Generate unique run ID for parallel testing"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}_{timestamp}_{unique_id}"

def get_output_dir(run_id: str = None) -> str:
    """Get output directory for current run"""
    if run_id is None:
        run_id = generate_run_id()
    return f"results/{run_id}"

# Test Configuration Sets
TEST_CONFIGS = {
    "default": {
        "name": "Default Optimization",
        "description": "Standard parameter optimization",
        "atr_lookbacks": ATR_LOOKBACKS,
        "adx_lookbacks": ADX_LOOKBACKS,
        "ma_lengths": MA_LENGTHS,
        "range_lookbacks": RANGE_LOOKBACKS,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "use_renko": False,
        "renko_atr_period": 14,
        "renko_atr_multiplier": 1.0
    },
    "renko_fine": {
        "name": "Renko Fine Optimization",
        "description": "Renko with fine-tuned parameters",
        "atr_lookbacks": [10, 14, 21],
        "adx_lookbacks": [7, 14],
        "ma_lengths": [(20, 50, 200)],
        "range_lookbacks": [20, 30, 50],
        "window_size": 1500,
        "step_size": 500,
        "use_renko": True,
        "renko_atr_period": 14,
        "renko_atr_multiplier": 1.0
    },
    "renko_coarse": {
        "name": "Renko Coarse Optimization", 
        "description": "Renko with coarse parameters",
        "atr_lookbacks": [14, 21],
        "adx_lookbacks": [14],
        "ma_lengths": [(20, 50, 200)],
        "range_lookbacks": [30],
        "window_size": 2000,
        "step_size": 1000,
        "use_renko": True,
        "renko_atr_period": 21,
        "renko_atr_multiplier": 1.5
    },
    "quick_test": {
        "name": "Quick Test",
        "description": "Fast test with limited combinations",
        "atr_lookbacks": [14],
        "adx_lookbacks": [14],
        "ma_lengths": [(20, 50, 200)],
        "range_lookbacks": [30],
        "window_size": 1000,
        "step_size": 500,
        "use_renko": False,
        "renko_atr_period": 14,
        "renko_atr_multiplier": 1.0
    }
} 