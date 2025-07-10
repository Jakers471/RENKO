import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_handler import DataHandler
from utils.renko_converter import RenkoConverter
from strategies.rthe_strategy import RTHEStrategy
from engine.backtest_engine import BacktestEngine
from engine.results_analyzer import ResultsAnalyzer
import random

class ValidationSuite:
    def __init__(self, data_file, train_frac=0.6, test_frac=0.2, n_permutations=10, walk_forward_steps=3):
        self.data_file = data_file
        self.train_frac = train_frac
        self.test_frac = test_frac
        self.n_permutations = n_permutations
        self.walk_forward_steps = walk_forward_steps
        self.data_handler = DataHandler()
        self.ohlc_data = self.data_handler.load_csv(data_file)
        self.renko_converter = RenkoConverter(brick_size=1.0)
        self.optimal_brick_size = self.renko_converter.get_optimal_brick_size(self.ohlc_data)
        self.renko_converter = RenkoConverter(self.optimal_brick_size)
        self.renko_data = self.renko_converter.convert_to_renko(self.ohlc_data)
        self.backtest_engine = BacktestEngine(initial_capital=10000)
        self.parameter_ranges = {
            'brick_size': [self.optimal_brick_size * 0.5, self.optimal_brick_size, self.optimal_brick_size * 1.5],
            'tsl_offset': [self.optimal_brick_size * 0.01, self.optimal_brick_size * 0.02, self.optimal_brick_size * 0.03],
            'hedge_size_ratio': [0.3, 0.5, 0.7],
            'min_bricks_for_trend': [2, 3, 4]
        }
        self.analyzer = ResultsAnalyzer()

    def _split_data(self):
        n = len(self.renko_data)
        train_end = int(n * self.train_frac)
        test_end = train_end + int(n * self.test_frac)
        train = self.renko_data.iloc[:train_end].reset_index(drop=True)
        test = self.renko_data.iloc[train_end:test_end].reset_index(drop=True)
        oos = self.renko_data.iloc[test_end:].reset_index(drop=True)
        return train, test, oos

    def in_sample_excellence(self):
        print("\n=== In-Sample Excellence Test ===")
        train, _, _ = self._split_data()
        results = self.backtest_engine.run_parameter_optimization(
            RTHEStrategy, train, self.parameter_ranges, max_iterations=30)
        best = self.backtest_engine.get_best_results(results, metric='profit_factor', top_n=1)[0]
        self.analyzer.display_summary_stats([best], top_n=1)
        return best

    def in_sample_permutation(self):
        print("\n=== In-Sample Permutation Test ===")
        train, _, _ = self._split_data()
        best_results = []
        for i in range(self.n_permutations):
            permuted = train.sample(frac=1, random_state=i).reset_index(drop=True)
            results = self.backtest_engine.run_parameter_optimization(
                RTHEStrategy, permuted, self.parameter_ranges, max_iterations=10)
            best = self.backtest_engine.get_best_results(results, metric='profit_factor', top_n=1)[0]
            best_results.append(best)
        self.analyzer.display_summary_stats(best_results, top_n=3)
        return best_results

    def walk_forward(self):
        print("\n=== Walk-Forward Test ===")
        n = len(self.renko_data)
        window = int(n * self.train_frac)
        step = int(n * self.test_frac)
        results = []
        for i in range(self.walk_forward_steps):
            train_start = i * step
            train_end = train_start + window
            test_start = train_end
            test_end = test_start + step
            if test_end > n:
                break
            train = self.renko_data.iloc[train_start:train_end].reset_index(drop=True)
            test = self.renko_data.iloc[test_start:test_end].reset_index(drop=True)
            # Optimize on train
            train_results = self.backtest_engine.run_parameter_optimization(
                RTHEStrategy, train, self.parameter_ranges, max_iterations=10)
            best_params = self.backtest_engine.get_best_results(train_results, metric='profit_factor', top_n=1)[0].parameters
            # Test on test
            test_strategy = RTHEStrategy(**best_params)
            test_result = self.backtest_engine.run_single_backtest(test_strategy, test, best_params)
            results.append(test_result)
        self.analyzer.display_summary_stats(results, top_n=3)
        return results

    def walk_forward_permutation(self):
        print("\n=== Walk-Forward Permutation Test ===")
        n = len(self.renko_data)
        window = int(n * self.train_frac)
        step = int(n * self.test_frac)
        all_results = []
        for i in range(self.walk_forward_steps):
            train_start = i * step
            train_end = train_start + window
            test_start = train_end
            test_end = test_start + step
            if test_end > n:
                break
            train = self.renko_data.iloc[train_start:train_end].reset_index(drop=True)
            test = self.renko_data.iloc[test_start:test_end].reset_index(drop=True)
            for j in range(self.n_permutations):
                permuted = train.sample(frac=1, random_state=j).reset_index(drop=True)
                train_results = self.backtest_engine.run_parameter_optimization(
                    RTHEStrategy, permuted, self.parameter_ranges, max_iterations=5)
                best_params = self.backtest_engine.get_best_results(train_results, metric='profit_factor', top_n=1)[0].parameters
                test_strategy = RTHEStrategy(**best_params)
                test_result = self.backtest_engine.run_single_backtest(test_strategy, test, best_params)
                all_results.append(test_result)
        self.analyzer.display_summary_stats(all_results, top_n=5)
        return all_results

    def run_all(self):
        print("\n================ VALIDATION SUITE ================")
        print(f"Data file: {self.data_file}")
        print(f"Total Renko bricks: {len(self.renko_data)}")
        print(f"Optimal brick size: {self.optimal_brick_size:.2f}")
        print("==================================================")
        best_in_sample = self.in_sample_excellence()
        perm_results = self.in_sample_permutation()
        wf_results = self.walk_forward()
        wfp_results = self.walk_forward_permutation()
        print("\n================ SUMMARY ================")
        print(f"Best in-sample profit factor: {best_in_sample.profit_factor:.3f}")
        print(f"Permutation test best profit factor: {max([r.profit_factor for r in perm_results]):.3f}")
        print(f"Walk-forward best profit factor: {max([r.profit_factor for r in wf_results]):.3f}")
        print(f"Walk-forward permutation best profit factor: {max([r.profit_factor for r in wfp_results]):.3f}")
        print("========================================")

if __name__ == "__main__":
    suite = ValidationSuite(data_file="data/bitcoin_merged_data.csv", n_permutations=5, walk_forward_steps=3)
    suite.run_all() 