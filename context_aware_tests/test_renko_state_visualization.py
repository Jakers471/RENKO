"""
Test script for Renko State Visualization
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualizers.renko_state_overlay import RenkoStateVisualizer

def test_renko_state_visualization():
    """Test the Renko state visualization"""
    print("Testing Renko State Visualization...")
    
    # Initialize visualizer
    visualizer = RenkoStateVisualizer()
    
    # Test data file path
    data_file = "../data/bitcoin_merged_data.csv"
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        print("Available data files:")
        data_dir = "../data"
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith('.csv'):
                    print(f"  - {file}")
        return
    
    # Run visualization with different parameters
    print(f"Running visualization on: {data_file}")
    
    # Test 1: ATR-based brick size
    print("\n=== Test 1: ATR-based Renko bricks ===")
    result1 = visualizer.run_visualization(
        data_file=data_file,
        use_atr=True,
        atr_period=14,
        atr_multiplier=1.0,
        classification_params={
            'atr_period': 21,
            'adx_period': 7,
            'ma_periods': (20, 50, 200),
            'range_lookback': 30
        },
        show_timeline=True,
        save_plot=True,
        debug=True
    )
    
    if not result1.empty:
        print(f"Test 1 complete: {len(result1)} Renko bricks processed")
        
        # Print some statistics
        trend_count = len(result1[result1['market_state'] == 'trend'])
        consolidation_count = len(result1[result1['market_state'] == 'consolidation'])
        
        print(f"Trend bricks: {trend_count} ({trend_count/len(result1)*100:.1f}%)")
        print(f"Consolidation bricks: {consolidation_count} ({consolidation_count/len(result1)*100:.1f}%)")
    
    # Test 2: Fixed brick size
    print("\n=== Test 2: Fixed brick size ===")
    result2 = visualizer.run_visualization(
        data_file=data_file,
        use_atr=False,
        brick_size=100.0,  # Fixed $100 brick size
        classification_params={
            'atr_period': 21,
            'adx_period': 7,
            'ma_periods': (20, 50, 200),
            'range_lookback': 30
        },
        show_timeline=True,
        save_plot=True,
        debug=False
    )
    
    if not result2.empty:
        print(f"Test 2 complete: {len(result2)} Renko bricks processed")
    
    print("\nVisualization tests complete!")

def test_renko_state_visualization_with_optimized_params():
    """Test the Renko state visualization using optimized parameters from results file"""
    print("\n=== Test 3: Visualization with optimized parameters ===")
    visualizer = RenkoStateVisualizer()
    data_file = "../data/bitcoin_merged_data.csv"
    optimization_results_file = "../results/optimization_results_20250711_top10.json"  # Update as needed
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    if not os.path.exists(optimization_results_file):
        print(f"Optimization results file not found: {optimization_results_file}")
        return
    result = visualizer.run_visualization_with_optimized_params(
        data_file=data_file,
        optimization_results_file=optimization_results_file,
        use_atr=True,
        atr_period=14,
        atr_multiplier=1.0,
        show_timeline=True,
        save_plot=True,
        debug=True
    )
    if not result.empty:
        print(f"Test 3 complete: {len(result)} Renko bricks processed with optimized parameters")

if __name__ == "__main__":
    test_renko_state_visualization()
    test_renko_state_visualization_with_optimized_params() 