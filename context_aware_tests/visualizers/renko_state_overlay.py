"""
Renko State Overlay Visualizer
Visualizes market state classifications on Renko charts with color-coded bricks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
from typing import Optional, Tuple, Dict
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.renko_converter import RenkoConverter
from data.load_data import DataLoader
from classifier.market_state_classifier import MarketStateClassifier

class RenkoStateVisualizer:
    """
    Visualizes market state classifications on Renko charts
    """
    
    def __init__(self):
        """Initialize the Renko state visualizer"""
        self.renko_converter = RenkoConverter()
        self.data_loader = DataLoader()
        self.classifier = MarketStateClassifier()
        
        # Color scheme for states
        self.state_colors = {
            'trend': '#2E8B57',      # Sea Green
            'consolidation': '#FFD700'  # Gold
        }
        
        # Opacity settings
        self.trend_opacity = 0.8
        self.consolidation_opacity = 0.6
        
    def load_and_convert_data(self, 
                             data_file: str,
                             use_atr: bool = True,
                             atr_period: int = 14,
                             atr_multiplier: float = 1.0,
                             brick_size: Optional[float] = None) -> pd.DataFrame:
        """
        Load OHLC data and convert to Renko bricks
        
        Args:
            data_file: Path to data file
            use_atr: Whether to use ATR-based brick size
            atr_period: ATR lookback period
            atr_multiplier: Multiplier for ATR-based brick size
            brick_size: Fixed brick size (if not using ATR)
            
        Returns:
            DataFrame with Renko bricks
        """
        print(f"Loading data from: {data_file}")
        
        # Load OHLC data
        ohlc_data = self.data_loader.load_data(data_file)
        if ohlc_data.empty:
            print("Error: No data loaded")
            return pd.DataFrame()
        
        print(f"Loaded {len(ohlc_data)} OHLC bars")
        
        # Convert to Renko
        renko_data = self.renko_converter.convert_to_renko(
            ohlc_data, 
            brick_size=brick_size,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            use_atr=use_atr
        )
        
        if renko_data.empty:
            print("Error: No Renko bricks generated")
            return pd.DataFrame()
        
        print(f"Generated {len(renko_data)} Renko bricks")
        return renko_data
    
    def classify_renko_states(self, 
                            renko_data: pd.DataFrame,
                            atr_period: int = 21,
                            adx_period: int = 7,
                            ma_periods: Tuple[int, int, int] = (20, 50, 200),
                            range_lookback: int = 30,
                            debug: bool = False) -> pd.DataFrame:
        """
        Classify market states on Renko data
        
        Args:
            renko_data: DataFrame with Renko bricks
            atr_period: ATR lookback period
            adx_period: ADX lookback period
            ma_periods: Tuple of MA periods
            range_lookback: Range analysis lookback period
            debug: Whether to print debug information
            
        Returns:
            DataFrame with state classifications added
        """
        if renko_data.empty:
            return pd.DataFrame()
        
        # Prepare data for classification (need OHLC format)
        # For Renko, we'll use open/close as high/low
        classification_data = renko_data.copy()
        classification_data['high'] = np.maximum(classification_data['renko_open'], 
                                               classification_data['renko_close'])
        classification_data['low'] = np.minimum(classification_data['renko_open'], 
                                              classification_data['renko_close'])
        classification_data['open'] = classification_data['renko_open']
        classification_data['close'] = classification_data['renko_close']
        
        # Classify states
        state_series = self.classifier.classify_market_state(
            classification_data,
            atr_period=atr_period,
            adx_period=adx_period,
            ma_periods=ma_periods,
            range_lookback=range_lookback,
            debug=debug
        )
        
        # Add state to Renko data
        renko_data['market_state'] = state_series
        
        # Get classification metrics
        metrics = self.classifier.get_classification_metrics(classification_data, state_series)
        print(f"Classification metrics: {metrics}")
        
        return renko_data
    
    def plot_renko_with_states(self, 
                              renko_data: pd.DataFrame,
                              title: str = "Renko Chart with Market States",
                              show_timeline: bool = True,
                              save_plot: bool = True,
                              filename: Optional[str] = None) -> None:
        """
        Plot Renko chart with color-coded state bricks
        
        Args:
            renko_data: DataFrame with Renko bricks and state classifications
            title: Plot title
            show_timeline: Whether to show state timeline below chart
            save_plot: Whether to save the plot
            filename: Custom filename for saving
        """
        if renko_data.empty:
            print("Error: No Renko data to plot")
            return
        
        # Create figure with subplots
        if show_timeline:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        
        # Plot Renko bricks with state colors
        self._plot_renko_bricks(ax1, renko_data, title)
        
        # Plot state timeline if requested
        if show_timeline:
            self._plot_state_timeline(ax2, renko_data)
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"renko_state_viz_{timestamp}.png"
            
            save_path = os.path.join("context_aware_tests", "results", "renko_state_viz", filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def _plot_renko_bricks(self, ax, renko_data: pd.DataFrame, title: str) -> None:
        """
        Plot Renko bricks with state-based coloring
        
        Args:
            ax: Matplotlib axis
            renko_data: DataFrame with Renko bricks and states
            title: Plot title
        """
        brick_width = 0.8
        
        for i, row in renko_data.iterrows():
            # Get brick properties
            brick_open = row['renko_open']
            brick_close = row['renko_close']
            direction = row['direction']
            state = row.get('market_state', 'consolidation')
            
            # Determine brick color and opacity
            if state == 'trend':
                color = self.state_colors['trend']
                alpha = self.trend_opacity
            else:
                color = self.state_colors['consolidation']
                alpha = self.consolidation_opacity
            
            # Create brick rectangle
            if direction == 1:  # Up brick
                brick_height = brick_close - brick_open
                brick_bottom = brick_open
            else:  # Down brick
                brick_height = brick_open - brick_close
                brick_bottom = brick_close
            
            # Add brick to plot
            rect = patches.Rectangle(
                (i - brick_width/2, brick_bottom),
                brick_width,
                brick_height,
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=alpha
            )
            ax.add_patch(rect)
        
        # Customize axis
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Renko Brick Index', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            patches.Patch(color=self.state_colors['trend'], alpha=self.trend_opacity, 
                         label='Trend'),
            patches.Patch(color=self.state_colors['consolidation'], alpha=self.consolidation_opacity, 
                         label='Consolidation')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
    
    def _plot_state_timeline(self, ax, renko_data: pd.DataFrame) -> None:
        """
        Plot state timeline below the main chart
        
        Args:
            ax: Matplotlib axis
            renko_data: DataFrame with Renko bricks and states
        """
        # Create state values (1 for trend, 0 for consolidation)
        state_values = []
        for state in renko_data['market_state']:
            if state == 'trend':
                state_values.append(1)
            else:
                state_values.append(0)
        
        # Plot timeline
        x_values = range(len(state_values))
        ax.fill_between(x_values, state_values, alpha=0.7, 
                       color=self.state_colors['trend'])
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Customize axis
        ax.set_title('Market State Timeline', fontsize=12, fontweight='bold')
        ax.set_xlabel('Renko Brick Index', fontsize=10)
        ax.set_ylabel('State', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Consolidation', 'Trend'])
        ax.grid(True, alpha=0.3)
    
    def run_visualization(self, 
                         data_file: str,
                         use_atr: bool = True,
                         atr_period: int = 14,
                         atr_multiplier: float = 1.0,
                         brick_size: Optional[float] = None,
                         classification_params: Optional[Dict] = None,
                         show_timeline: bool = True,
                         save_plot: bool = True,
                         debug: bool = False) -> pd.DataFrame:
        """
        Complete pipeline: load, convert, classify, and visualize
        
        Args:
            data_file: Path to data file
            use_atr: Whether to use ATR-based brick size
            atr_period: ATR lookback period
            atr_multiplier: Multiplier for ATR-based brick size
            brick_size: Fixed brick size (if not using ATR)
            classification_params: Dictionary with classification parameters
            show_timeline: Whether to show state timeline
            save_plot: Whether to save the plot
            debug: Whether to print debug information
            
        Returns:
            DataFrame with Renko bricks and state classifications
        """
        print("=== Renko State Visualization Pipeline ===")
        
        # Load and convert data
        renko_data = self.load_and_convert_data(
            data_file, use_atr, atr_period, atr_multiplier, brick_size
        )
        
        if renko_data.empty:
            return pd.DataFrame()
        
        # Set default classification parameters
        if classification_params is None:
            classification_params = {
                'atr_period': 21,
                'adx_period': 7,
                'ma_periods': (20, 50, 200),
                'range_lookback': 30
            }
        
        # Classify states
        renko_data = self.classify_renko_states(
            renko_data, debug=debug, **classification_params
        )
        
        if renko_data.empty:
            return pd.DataFrame()
        
        # Create title
        title = f"Renko Chart with Market States\n"
        title += f"Bricks: {len(renko_data)}, "
        title += f"Trend: {len(renko_data[renko_data['market_state'] == 'trend'])} "
        title += f"({len(renko_data[renko_data['market_state'] == 'trend']) / len(renko_data) * 100:.1f}%)"
        
        # Plot visualization
        self.plot_renko_with_states(
            renko_data, title, show_timeline, save_plot
        )
        
        return renko_data

    def run_visualization_with_optimized_params(self, data_file: str, optimization_results_file: str,
                                               use_atr: bool = True,
                                               atr_period: int = 14,
                                               atr_multiplier: float = 1.0,
                                               brick_size: Optional[float] = None,
                                               show_timeline: bool = True,
                                               save_plot: bool = True,
                                               debug: bool = False) -> pd.DataFrame:
        """
        Load best parameters from optimization results and visualize Renko state overlay.
        Args:
            data_file: Path to data file
            optimization_results_file: Path to saved optimization results JSON
            use_atr, atr_period, atr_multiplier, brick_size: Renko conversion params
            show_timeline, save_plot, debug: Visualization options
        Returns:
            DataFrame with Renko bricks and state classifications
        """
        print(f"Loading optimization results from: {optimization_results_file}")
        
        # Handle both old and new file formats
        if not os.path.exists(optimization_results_file):
            # Try to find the file in results subdirectories
            results_dir = "context_aware_tests/results"
            if os.path.exists(results_dir):
                for root, dirs, files in os.walk(results_dir):
                    for file in files:
                        if file.endswith('_top10.json') and 'grid_search_results' in file:
                            potential_file = os.path.join(root, file)
                            print(f"Found potential results file: {potential_file}")
                            if input(f"Use this file? (y/n): ").lower() == 'y':
                                optimization_results_file = potential_file
                                break
                    if os.path.exists(optimization_results_file):
                        break
        
        with open(optimization_results_file, 'r') as f:
            results = json.load(f)
        
        if isinstance(results, list):
            top_result = results[0]
        elif isinstance(results, dict) and 'top_results' in results:
            top_result = results['top_results'][0]
        else:
            raise ValueError("Could not parse top result from optimization results file.")
        
        print(f"Top parameter set: {top_result}")
        
        # Extract params (handle both old and new field names)
        classification_params = {
            'atr_period': top_result.get('atr_lookback', top_result.get('ATR', 21)),
            'adx_period': top_result.get('adx_lookback', top_result.get('ADX', 7)),
            'ma_periods': tuple(top_result.get('MA', (top_result.get('ma_short', 20), 
                                                    top_result.get('ma_medium', 50), 
                                                    top_result.get('ma_long', 200)))),
            'range_lookback': top_result.get('range_lookback', top_result.get('Range', 30))
        }
        
        return self.run_visualization(
            data_file=data_file,
            use_atr=use_atr,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier,
            brick_size=brick_size,
            classification_params=classification_params,
            show_timeline=show_timeline,
            save_plot=save_plot,
            debug=debug
        )


def main():
    """Main function for testing the visualizer"""
    visualizer = RenkoStateVisualizer()
    
    # Example usage
    data_file = "data/bitcoin_merged_data.csv"  # Adjust path as needed
    
    # Run visualization
    result = visualizer.run_visualization(
        data_file=data_file,
        use_atr=True,
        atr_period=14,
        atr_multiplier=1.0,
        show_timeline=True,
        save_plot=True,
        debug=True
    )
    
    if not result.empty:
        print(f"Visualization complete. Processed {len(result)} Renko bricks.")


if __name__ == "__main__":
    main() 