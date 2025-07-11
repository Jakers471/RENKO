"""
Grid Search Optimizer for finding optimal lookback periods
for market state classification (trend vs. consolidation)
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier.market_state_classifier import MarketStateClassifier
from config import (
    ATR_LOOKBACKS, ADX_LOOKBACKS, MA_LENGTHS, RANGE_LOOKBACKS,
    BALANCE_WEIGHT, TRANSITION_WEIGHT, TREND_DETECTION_WEIGHT,
    DEBUG_FIRST_N_COMBINATIONS, PROGRESS_UPDATE_INTERVAL,
    generate_run_id, get_output_dir
)

class GridSearchOptimizer:
    """
    Grid Search Optimizer for finding optimal lookback periods
    for market state classification (trend vs. consolidation)
    """
    
    def __init__(self, run_id: str = None):
        """Initialize the grid search optimizer"""
        self.classifier = MarketStateClassifier()
        self.results = []
        self.run_id = run_id or generate_run_id("grid_search")
        self.output_dir = get_output_dir(self.run_id)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"🆔 Run ID: {self.run_id}")
        print(f"📁 Output directory: {self.output_dir}")
        
    def evaluate_classification(self, state_series: pd.Series, ground_truth_labels: pd.Series = None) -> float:
        """
        Evaluate the quality of market state classification
        
        Args:
            state_series: Series with predicted market states
            ground_truth_labels: Series with actual market states (if available)
            
        Returns:
            Score for the classification (higher is better)
        """
        if state_series.empty:
            return 0.0
        
        total_bars = len(state_series)
        trend_bars = len(state_series[state_series == 'trend'])
        consolidation_bars = len(state_series[state_series == 'consolidation'])
        
        # Calculate state balance (avoid extreme imbalances)
        trend_ratio = trend_bars / total_bars
        consolidation_ratio = consolidation_bars / total_bars
        
        # Enhanced scoring system
        score = 0.0
        
        # 1. State distribution balance (30-70% ideal)
        if 0.3 <= trend_ratio <= 0.7:
            balance_score = 1.0 - abs(trend_ratio - 0.5) * 2
            score += balance_score * BALANCE_WEIGHT
        else:
            # Penalize extreme imbalances
            score += 0.1  # Minimal score for extreme cases
        
        # 2. State transition analysis (not too choppy)
        state_changes = (state_series != state_series.shift()).sum()
        transition_ratio = state_changes / total_bars
        
        if 0.01 <= transition_ratio <= 0.1:  # Reasonable transition frequency
            transition_score = 1.0 - abs(transition_ratio - 0.05) * 10
            score += transition_score * TRANSITION_WEIGHT
        else:
            score += 0.1  # Penalize too frequent or too rare transitions
        
        # 3. Trend detection success (reward finding trends)
        if trend_ratio > 0.1:  # At least 10% trend detection
            trend_detection_score = min(trend_ratio * 2, 1.0)  # Reward up to 50% trend
            score += trend_detection_score * TREND_DETECTION_WEIGHT
        else:
            score += 0.0  # No reward for no trend detection
        
        return score
    
    def save_result(self, atr_l: int, adx_l: int, ma_set: Tuple[int, int, int], 
                   range_l: int, score: float, metrics: Dict):
        """
        Save optimization result
        
        Args:
            atr_l: ATR lookback period
            adx_l: ADX lookback period
            ma_set: Moving average periods tuple
            range_l: Range lookback period
            score: Evaluation score
            metrics: Classification metrics
        """
        result = {
            'atr_lookback': atr_l,
            'adx_lookback': adx_l,
            'ma_short': ma_set[0],
            'ma_medium': ma_set[1],
            'ma_long': ma_set[2],
            'range_lookback': range_l,
            'score': score,
            'metrics': metrics,
            'total_combinations': len(ATR_LOOKBACKS) * len(ADX_LOOKBACKS) * len(MA_LENGTHS) * len(RANGE_LOOKBACKS)
        }
        
        self.results.append(result)
    
    def run_grid_search(self, data: pd.DataFrame, max_combinations: int = None) -> pd.DataFrame:
        """
        Run the grid search optimization
        
        Args:
            data: DataFrame with OHLC data
            max_combinations: Maximum number of combinations to test (None for all)
            
        Returns:
            DataFrame with optimization results
        """
        print("🔧 GRID SEARCH OPTIMIZATION FOR MARKET STATE CLASSIFICATION")
        print("="*70)
        
        # Calculate total combinations
        total_combinations = len(ATR_LOOKBACKS) * len(ADX_LOOKBACKS) * len(MA_LENGTHS) * len(RANGE_LOOKBACKS)
        print(f"Total parameter combinations: {total_combinations}")
        
        if max_combinations:
            print(f"Testing maximum {max_combinations} combinations")
            total_combinations = min(total_combinations, max_combinations)
        
        # Run grid search
        combination_count = 0
        
        for atr_l in ATR_LOOKBACKS:
            for adx_l in ADX_LOOKBACKS:
                for ma_set in MA_LENGTHS:
                    for range_l in RANGE_LOOKBACKS:
                        combination_count += 1
                        
                        if max_combinations and combination_count > max_combinations:
                            break
                        
                        # Calculate progress percentage
                        progress_pct = (combination_count / total_combinations) * 100
                        
                        print(f"[{progress_pct:.1f}%] Testing combination {combination_count}/{total_combinations}: "
                              f"ATR={atr_l}, ADX={adx_l}, MA={ma_set}, Range={range_l}")
                        
                        try:
                            # Classify market state with debug for first few combinations
                            debug_mode = combination_count <= DEBUG_FIRST_N_COMBINATIONS
                            state_series = self.classifier.classify_market_state(
                                data=data,
                                atr_period=atr_l,
                                adx_period=adx_l,
                                ma_periods=ma_set,
                                range_lookback=range_l,
                                debug=debug_mode
                            )
                            
                            # Evaluate classification
                            score = self.evaluate_classification(state_series)
                            
                            # Get metrics
                            metrics = self.classifier.get_classification_metrics(data, state_series)
                            
                            # Save result
                            self.save_result(atr_l, adx_l, ma_set, range_l, score, metrics)
                            
                            print(f"  Score: {score:.4f}, Trend: {metrics['trend_pct']:.1f}%, "
                                  f"Consolidation: {metrics['consolidation_pct']:.1f}%")
                            
                            # Show estimated time remaining every N combinations
                            if combination_count % PROGRESS_UPDATE_INTERVAL == 0:
                                remaining_combinations = total_combinations - combination_count
                                print(f"  ⏱️  Progress: {progress_pct:.1f}% complete, {remaining_combinations} combinations remaining")
                            
                        except Exception as e:
                            print(f"  Error testing combination: {e}")
                            continue
                    
                    if max_combinations and combination_count > max_combinations:
                        break
                
                if max_combinations and combination_count > max_combinations:
                    break
            
            if max_combinations and combination_count > max_combinations:
                break
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Sort by score (descending)
        if not results_df.empty:
            results_df = results_df.sort_values('score', ascending=False)
        
        print(f"\n✅ Grid search completed! Tested {len(results_df)} combinations.")
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_file: str = None):
        """
        Save optimization results to file with unique run ID
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"grid_search_results_{self.run_id}_{timestamp}.csv"
        
        # Save to CSV in run-specific directory
        csv_path = os.path.join(self.output_dir, output_file)
        results_df.to_csv(csv_path, index=False)
        print(f"📊 Results saved to: {csv_path}")
        
        # Save top results to JSON for easy access
        top_results = results_df.head(10).to_dict('records')
        json_file = output_file.replace('.csv', '_top10.json')
        json_path = os.path.join(self.output_dir, json_file)
        
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            else:
                return obj
        
        with open(json_path, 'w') as f:
            json.dump(convert_types(top_results), f, indent=2)
        print(f"🏆 Top 10 results saved to: {json_path}")
        
        # Save run metadata
        metadata = {
            'run_id': self.run_id,
            'timestamp': timestamp,
            'total_combinations': len(results_df),
            'best_score': float(results_df['score'].max()) if not results_df.empty else 0.0,
            'output_directory': self.output_dir
        }
        metadata_path = os.path.join(self.output_dir, f"run_metadata_{self.run_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return csv_path
    
    def print_top_results(self, results_df: pd.DataFrame, top_n: int = 10):
        """
        Print top optimization results
        
        Args:
            results_df: DataFrame with results
            top_n: Number of top results to print
        """
        if results_df.empty:
            print("No results to display")
            return
        
        print(f"\n🏆 TOP {top_n} OPTIMIZATION RESULTS:")
        print("="*80)
        
        for i, row in results_df.head(top_n).iterrows():
            print(f"{i+1:2d}. Score: {row['score']:.4f}")
            print(f"    ATR: {row['atr_lookback']}, ADX: {row['adx_lookback']}")
            print(f"    MA: ({row['ma_short']}, {row['ma_medium']}, {row['ma_long']})")
            print(f"    Range: {row['range_lookback']}")
            print(f"    Trend: {row['metrics']['trend_pct']:.1f}%, "
                  f"Consolidation: {row['metrics']['consolidation_pct']:.1f}%")
            print() 