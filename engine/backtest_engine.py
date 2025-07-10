import pandas as pd
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import itertools
from tqdm import tqdm
from strategies.trading_strategy import TradingStrategy, Trade

@dataclass
class BacktestResult:
    """Results from a single backtest"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    equity_curve: pd.Series
    trades: List[Trade]
    parameters: Dict[str, Any]

class BacktestEngine:
    """Main backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run_single_backtest(self, 
                           strategy: TradingStrategy,
                           data: pd.DataFrame,
                           parameters: Dict[str, Any] = None) -> BacktestResult:
        """
        Run a single backtest with given strategy and data
        
        Args:
            strategy: Trading strategy instance
            data: Renko data DataFrame
            parameters: Strategy parameters
            
        Returns:
            BacktestResult object
        """
        # Reset strategy
        strategy.reset()
        
        # Set parameters if provided
        if parameters:
            for key, value in parameters.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
        
        # Set brick size if strategy needs it
        if hasattr(strategy, 'set_brick_size') and 'brick_size' in data.columns:
            brick_size = data['brick_size'].iloc[0]
            strategy.set_brick_size(brick_size)
        
        # Run strategy on each bar
        for i in range(len(data)):
            strategy.execute_trade(data, i)
        
        # Calculate results
        return self._calculate_results(strategy, parameters or {})
    
    def run_parameter_optimization(self,
                                 strategy_class,
                                 data: pd.DataFrame,
                                 parameter_ranges: Dict[str, List],
                                 max_iterations: int = 1000) -> List[BacktestResult]:
        """
        Run parameter optimization with multiple iterations
        
        Args:
            strategy_class: Strategy class to instantiate
            data: Renko data DataFrame
            parameter_ranges: Dict of parameter names to lists of values to test
            max_iterations: Maximum number of iterations to run
            
        Returns:
            List of BacktestResult objects
        """
        results = []
        
        # Generate parameter combinations
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        # Limit iterations if needed
        if total_combinations > max_iterations:
            # Use random sampling
            combinations = []
            for _ in range(max_iterations):
                combination = {}
                for name, values in parameter_ranges.items():
                    combination[name] = np.random.choice(values)
                combinations.append(combination)
        else:
            # Use all combinations
            combinations = []
            for combination in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combination))
                combinations.append(param_dict)
        
        # Run backtests
        print(f"Running {len(combinations)} backtest iterations...")
        for params in tqdm(combinations):
            strategy = strategy_class(**params)
            result = self.run_single_backtest(strategy, data, params)
            results.append(result)
        
        return results
    
    def _calculate_results(self, strategy: TradingStrategy, parameters: Dict[str, Any]) -> BacktestResult:
        """Calculate backtest results from strategy trades"""
        trades = strategy.trades
        
        if not trades:
            # No trades made
            return BacktestResult(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                profit_factor=0.0,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_win=0.0,
                avg_loss=0.0,
                equity_curve=pd.Series([self.initial_capital]),
                trades=trades,
                parameters=parameters
            )
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl > 0])
        losing_trades = len([t for t in trades if t.pnl < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate PnL metrics
        winning_pnls = [t.pnl for t in trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in trades if t.pnl < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0
        
        # Calculate profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0.0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate equity curve
        equity_curve = self._calculate_equity_curve(trades)
        
        # Calculate total return
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            equity_curve=equity_curve,
            trades=trades,
            parameters=parameters
        )
    
    def _calculate_equity_curve(self, trades: List[Trade]) -> pd.Series:
        """Calculate equity curve from trades"""
        if not trades:
            return pd.Series([self.initial_capital])
        
        # Create timeline of all trade dates
        all_dates = set()
        for trade in trades:
            all_dates.add(trade.entry_date)
            all_dates.add(trade.exit_date)
        
        all_dates = sorted(list(all_dates))
        
        # Calculate cumulative PnL
        equity = self.initial_capital
        equity_curve = []
        
        for date in all_dates:
            # Add PnL from trades that ended on this date
            for trade in trades:
                if trade.exit_date == date:
                    equity += trade.pnl * self.initial_capital
            
            equity_curve.append(equity)
        
        return pd.Series(equity_curve, index=all_dates)
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(equity_curve) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return abs(drawdown.min())
    
    def get_best_results(self, results: List[BacktestResult], 
                        metric: str = 'profit_factor', 
                        top_n: int = 10) -> List[BacktestResult]:
        """
        Get top N results based on specified metric
        
        Args:
            results: List of backtest results
            metric: Metric to sort by ('profit_factor', 'sharpe_ratio', 'total_return', etc.)
            top_n: Number of top results to return
            
        Returns:
            List of top N results
        """
        if not results:
            return []
        
        # Sort by metric
        sorted_results = sorted(results, key=lambda x: getattr(x, metric), reverse=True)
        
        return sorted_results[:top_n]
    
    def generate_parameter_report(self, results: List[BacktestResult]) -> pd.DataFrame:
        """
        Generate a report of parameter performance
        
        Args:
            results: List of backtest results
            
        Returns:
            DataFrame with parameter performance summary
        """
        if not results:
            return pd.DataFrame()
        
        # Extract parameters and metrics
        report_data = []
        for result in results:
            row = result.parameters.copy()
            row.update({
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades
            })
            report_data.append(row)
        
        return pd.DataFrame(report_data) 