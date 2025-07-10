import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_handler import DataHandler
from utils.renko_converter import RenkoConverter
from strategies.rthe_strategy import RTHEStrategy

# --- CONFIG ---
DATA_FILE = "data/bitcoin_merged_data.csv"
NUM_BARS = 200  # Change as desired

# --- LOAD DATA ---
data_handler = DataHandler()
ohlc_data = data_handler.load_csv(DATA_FILE)

# --- RENKO CONVERSION ---
renko_converter = RenkoConverter(brick_size=1.0)
optimal_brick_size = renko_converter.get_optimal_brick_size(ohlc_data)
renko_converter = RenkoConverter(optimal_brick_size)
renko_data = renko_converter.convert_to_renko(ohlc_data)
renko_data = renko_data.head(NUM_BARS).reset_index(drop=True)

# --- RUN STRATEGY ---
strategy = RTHEStrategy(
    brick_size=optimal_brick_size,
    tsl_offset=optimal_brick_size * 0.02,
    hedge_size_ratio=1.0,
    min_bricks_for_trend=3
)
for i in range(len(renko_data)):
    strategy.execute_trade(renko_data, i)

# --- PLOT RENKO CHART ---
fig, ax = plt.subplots(figsize=(14, 7))

# Plot Renko bricks as colored bars
for i, row in renko_data.iterrows():
    color = 'green' if row['direction'] == 1 else 'red'
    ax.bar(i, row['close'] - row['open'], bottom=row['open'], width=0.8, color=color, edgecolor='black', alpha=0.7)

# --- Overlay Trades ---
# Core trades
for trade in strategy.core_trades:
    entry_idx = renko_data[renko_data['close'] == trade.entry_price].index
    exit_idx = renko_data[renko_data['close'] == trade.exit_price].index
    if len(entry_idx) > 0:
        ax.scatter(entry_idx[0], trade.entry_price, marker='^', color='blue', s=100, label='Core Entry' if 'Core Entry' not in ax.get_legend_handles_labels()[1] else "")
    if len(exit_idx) > 0:
        ax.scatter(exit_idx[0], trade.exit_price, marker='v', color='navy', s=100, label='Core Exit' if 'Core Exit' not in ax.get_legend_handles_labels()[1] else "")

# Hedge trades
for trade in strategy.hedge_trades:
    entry_idx = renko_data[renko_data['close'] == trade.entry_price].index
    exit_idx = renko_data[renko_data['close'] == trade.exit_price].index
    if len(entry_idx) > 0:
        ax.scatter(entry_idx[0], trade.entry_price, marker='o', color='orange', s=80, label='Hedge Entry' if 'Hedge Entry' not in ax.get_legend_handles_labels()[1] else "")
    if len(exit_idx) > 0:
        ax.scatter(exit_idx[0], trade.exit_price, marker='x', color='darkorange', s=80, label='Hedge Exit' if 'Hedge Exit' not in ax.get_legend_handles_labels()[1] else "")

# --- Finalize Plot ---
ax.set_title(f"Renko Chart with R.T.H.E. Trades (First {NUM_BARS} Bricks)", fontsize=16)
ax.set_xlabel("Renko Brick Index")
ax.set_ylabel("Price")

# Custom legend (avoid duplicate labels)
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), loc='best')

plt.tight_layout()
plt.show() 