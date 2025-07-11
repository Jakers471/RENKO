"""
Rolling Window Tester for parameter optimization across time chunks
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import DataLoader
from optimizer.grid_search_optimizer import GridSearchOptimizer
from config import (
    WINDOW_SIZE, STEP_SIZE, MIN_CHUNK_SIZE,
    get_results_path, RESULTS_DIR, OPTIMIZATION_SUMMARY_FILE, CHUNK_RESULTS_DIR,
    RENKO_RESULTS_DIR
)

class RollingWindowTester:
    """
    Rolling Window Tester for parameter optimization across time chunks
    """
    
    def __init__(self):
        """Initialize the rolling window tester"""
        self.data_loader = DataLoader()
        self.optimizer = GridSearchOptimizer()
        self.chunk_results = []
        self.summary_results = []
        
    def create_chunks(self, data: pd.DataFrame, 
                     window_size: int = WINDOW_SIZE, 
                     step_size: int = STEP_SIZE) -> List[pd.DataFrame]:
        """
        Create overlapping data chunks for rolling window testing
        
        Args:
            data: Full dataset
            window_size: Number of bars per chunk
            step_size: Bars to slide forward between chunks
            
        Returns:
            List of data chunks
        """
        return self.data_loader.create_data_chunks(data, window_size, step_size)
    
    def test_single_chunk(self, chunk: pd.DataFrame, 
                         chunk_id: int,
                         max_combinations: int = None) -> Dict:
        """
        Test a single chunk with full parameter grid search
        
        Args:
            chunk: Data chunk to test
            chunk_id: ID of the chunk
            max_combinations: Maximum combinations to test
            
        Returns:
            Dictionary with chunk test results
        """
        data_type = "bricks" if 'direction' in chunk.columns else "bars"
        print(f"\n🔍 TESTING CHUNK {chunk_id}")
        print(f"   {data_type.capitalize()}: {len(chunk)} (start: {chunk['chunk_start'].iloc[0]}, end: {chunk['chunk_end'].iloc[0]})")
        
        # Run grid search on this chunk
        results = self.optimizer.run_grid_search(chunk, max_combinations)
        
        if results.empty:
            print(f"   ❌ No results for chunk {chunk_id}")
            return None
        
        # Get best result
        best_result = results.iloc[0].to_dict()
        
        # Add chunk metadata
        chunk_result = {
            'chunk_id': chunk_id,
            'chunk_start': chunk['chunk_start'].iloc[0],
            'chunk_end': chunk['chunk_end'].iloc[0],
            'chunk_size': len(chunk),
            'best_score': best_result['score'],
            'best_params': {
                'atr_lookback': best_result['atr_lookback'],
                'adx_lookback': best_result['adx_lookback'],
                'ma_short': best_result['ma_short'],
                'ma_medium': best_result['ma_medium'],
                'ma_long': best_result['ma_long'],
                'range_lookback': best_result['range_lookback']
            },
            'best_metrics': best_result['metrics'],
            'total_combinations_tested': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✅ Best score: {best_result['score']:.4f}")
        print(f"   📊 Trend: {best_result['metrics']['trend_pct']:.1f}%, "
              f"Consolidation: {best_result['metrics']['consolidation_pct']:.1f}%")
        
        return chunk_result
    
    def run_rolling_window_test(self, data: pd.DataFrame,
                               window_size: int = WINDOW_SIZE,
                               step_size: int = STEP_SIZE,
                               max_combinations: int = None,
                               test_specific_chunk: Optional[int] = None,
                               use_renko: bool = False,
                               renko_config: dict = None) -> Dict:
        """
        Run rolling window optimization test
        
        Args:
            data: Full dataset
            window_size: Number of bars per chunk
            step_size: Bars to slide forward between chunks
            max_combinations: Maximum combinations to test per chunk
            test_specific_chunk: Test only specific chunk index (for debugging)
            
        Returns:
            Dictionary with overall test results
        """
        print("🔄 ROLLING WINDOW PARAMETER OPTIMIZATION")
        print("="*60)
        
        # Convert to Renko if requested
        if use_renko:
            print("📊 RENKO-BASED TESTING")
            print("-" * 30)
            
            if renko_config is None:
                renko_config = {}
            
            renko_data = self.data_loader.convert_to_renko(
                data=data,
                use_atr=renko_config.get('use_atr', True),
                atr_period=renko_config.get('atr_period', 14),
                atr_multiplier=renko_config.get('atr_multiplier', 1.0),
                fixed_brick_size=renko_config.get('fixed_brick_size', None)
            )
            
            if renko_data.empty:
                print("❌ Failed to convert to Renko data")
                return {}
            
            data = renko_data
            print(f"✅ Using Renko data: {len(data)} bricks")
            
            # Store Renko metadata
            self.renko_metadata = {
                'renko_type': 'ATR-based' if renko_config.get('use_atr', True) else 'Fixed-size',
                'atr_period': renko_config.get('atr_period', 14),
                'atr_multiplier': renko_config.get('atr_multiplier', 1.0),
                'fixed_brick_size': renko_config.get('fixed_brick_size', None),
                'total_bricks': len(renko_data),
                'up_bricks': len(renko_data[renko_data['direction'] == 1]),
                'down_bricks': len(renko_data[renko_data['direction'] == -1]),
                'conversion_timestamp': datetime.now().isoformat()
            }
        else:
            print("📊 OHLC-BASED TESTING")
            print("-" * 30)
        
        print(f"Window size: {window_size} {'bricks' if use_renko else 'bars'}")
        print(f"Step size: {step_size} {'bricks' if use_renko else 'bars'}")
        print(f"Total data: {len(data)} {'bricks' if use_renko else 'bars'}")
        
        # Create chunks
        chunks = self.create_chunks(data, window_size, step_size)
        
        if test_specific_chunk is not None:
            # Test only specific chunk
            if 0 <= test_specific_chunk < len(chunks):
                chunk = chunks[test_specific_chunk]
                chunk_result = self.test_single_chunk(chunk, test_specific_chunk, max_combinations)
                if chunk_result:
                    self.chunk_results = [chunk_result]
                return self._create_summary()
            else:
                print(f"❌ Invalid chunk index: {test_specific_chunk}")
                return {}
        
        # Test all chunks
        for i, chunk in enumerate(chunks):
            if len(chunk) < MIN_CHUNK_SIZE:
                print(f"⚠️  Skipping chunk {i}: too small ({len(chunk)} bars)")
                continue
                
            chunk_result = self.test_single_chunk(chunk, i, max_combinations)
            if chunk_result:
                self.chunk_results.append(chunk_result)
        
        return self._create_summary()
    
    def _create_summary(self) -> Dict:
        """
        Create summary of all chunk results
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.chunk_results:
            return {}
        
        # Calculate summary statistics
        scores = [r['best_score'] for r in self.chunk_results]
        trend_pcts = [r['best_metrics']['trend_pct'] for r in self.chunk_results]
        consolidation_pcts = [r['best_metrics']['consolidation_pct'] for r in self.chunk_results]
        
        # Parameter frequency analysis
        param_counts = {}
        for result in self.chunk_results:
            params_key = f"{result['best_params']['atr_lookback']}_{result['best_params']['adx_lookback']}_{result['best_params']['ma_short']}_{result['best_params']['ma_medium']}_{result['best_params']['ma_long']}_{result['best_params']['range_lookback']}"
            param_counts[params_key] = param_counts.get(params_key, 0) + 1
        
        # Find most common parameter set
        most_common_params = max(param_counts.items(), key=lambda x: x[1]) if param_counts else None
        
        summary = {
            'total_chunks_tested': len(self.chunk_results),
            'score_statistics': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores)
            },
            'trend_percentage_statistics': {
                'mean': np.mean(trend_pcts),
                'std': np.std(trend_pcts),
                'min': np.min(trend_pcts),
                'max': np.max(trend_pcts)
            },
            'consolidation_percentage_statistics': {
                'mean': np.mean(consolidation_pcts),
                'std': np.std(consolidation_pcts),
                'min': np.min(consolidation_pcts),
                'max': np.max(consolidation_pcts)
            },
            'most_common_parameter_set': most_common_params,
            'parameter_frequency': param_counts,
            'chunk_results': self.chunk_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.summary_results = summary
        return summary
    
    def save_results(self, output_dir: str = None):
        """
        Save all results to files
        
        Args:
            output_dir: Output directory (optional)
        """
        if not self.summary_results:
            print("❌ No results to save")
            return
        
        if output_dir is None:
            output_dir = get_results_path("")
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        chunk_dir = os.path.join(output_dir, CHUNK_RESULTS_DIR)
        os.makedirs(chunk_dir, exist_ok=True)
        
        # Save summary
        summary_file = os.path.join(output_dir, OPTIMIZATION_SUMMARY_FILE)
        with open(summary_file, 'w') as f:
            json.dump(self.summary_results, f, indent=2)
        print(f"📊 Summary saved to: {summary_file}")
        
        # Add Renko metadata if this was a Renko test
        if hasattr(self, 'renko_metadata'):
            renko_file = os.path.join(output_dir, "renko_metadata.json")
            with open(renko_file, 'w') as f:
                json.dump(self.renko_metadata, f, indent=2)
            print(f"📊 Renko metadata saved to: {renko_file}")
        
        # Save individual chunk results
        for chunk_result in self.chunk_results:
            chunk_file = os.path.join(chunk_dir, f"chunk_{chunk_result['chunk_id']}.json")
            with open(chunk_file, 'w') as f:
                json.dump(chunk_result, f, indent=2)
        
        print(f"📁 Chunk results saved to: {chunk_dir}")
        
        return summary_file
    
    def print_summary(self):
        """
        Print summary of results
        """
        if not self.summary_results:
            print("❌ No results to display")
            return
        
        print("\n📊 ROLLING WINDOW OPTIMIZATION SUMMARY")
        print("="*60)
        
        summary = self.summary_results
        print(f"Total chunks tested: {summary['total_chunks_tested']}")
        
        # Score statistics
        score_stats = summary['score_statistics']
        print(f"\n🏆 Score Statistics:")
        print(f"  Mean: {score_stats['mean']:.4f}")
        print(f"  Std:  {score_stats['std']:.4f}")
        print(f"  Min:  {score_stats['min']:.4f}")
        print(f"  Max:  {score_stats['max']:.4f}")
        print(f"  Median: {score_stats['median']:.4f}")
        
        # Trend statistics
        trend_stats = summary['trend_percentage_statistics']
        print(f"\n📈 Trend Percentage Statistics:")
        print(f"  Mean: {trend_stats['mean']:.1f}%")
        print(f"  Std:  {trend_stats['std']:.1f}%")
        print(f"  Min:  {trend_stats['min']:.1f}%")
        print(f"  Max:  {trend_stats['max']:.1f}%")
        
        # Most common parameters
        if summary['most_common_parameter_set']:
            params_str, count = summary['most_common_parameter_set']
            print(f"\n🎯 Most Common Parameter Set (appears {count} times):")
            atr, adx, ma_s, ma_m, ma_l, range_l = params_str.split('_')
            print(f"  ATR: {atr}, ADX: {adx}, MA: ({ma_s}, {ma_m}, {ma_l}), Range: {range_l}")
        
        # Parameter frequency table
        print(f"\n📋 Parameter Set Frequency:")
        for params_str, count in sorted(summary['parameter_frequency'].items(), 
                                       key=lambda x: x[1], reverse=True)[:5]:
            atr, adx, ma_s, ma_m, ma_l, range_l = params_str.split('_')
            print(f"  {count}x: ATR={atr}, ADX={adx}, MA=({ma_s},{ma_m},{ma_l}), Range={range_l}")
    
    def get_chunk_by_index(self, data: pd.DataFrame, 
                          chunk_index: int,
                          window_size: int = WINDOW_SIZE,
                          step_size: int = STEP_SIZE) -> Optional[pd.DataFrame]:
        """
        Get a specific chunk by index for testing
        
        Args:
            data: Full dataset
            chunk_index: Index of the chunk to retrieve
            window_size: Number of bars per chunk
            step_size: Bars to slide forward between chunks
            
        Returns:
            DataFrame chunk or None if index out of range
        """
        return self.data_loader.get_chunk_by_index(data, chunk_index, window_size, step_size)
    
    def get_chunk_by_time_range(self, data: pd.DataFrame,
                               start_date: str,
                               end_date: str) -> Optional[pd.DataFrame]:
        """
        Get a chunk by time range for testing
        
        Args:
            data: Full dataset
            start_date: Start date (string format)
            end_date: End date (string format)
            
        Returns:
            DataFrame chunk or None if no data in range
        """
        return self.data_loader.get_chunk_by_time_range(data, start_date, end_date) 