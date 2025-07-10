import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from backtest_engine import BacktestResult

class ResultsAnalyzer:
    """Analyzes and visualizes backtest results"""
    
    def __init__(self):
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def display_summary_stats(self, results: List[BacktestResult], top_n: int = 10):
        """
        Display summary statistics for backtest results
        
        Args:
            results: List of backtest results
            top_n: Number of top results to display
        """
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS SUMMARY ({len(results)} iterations)")
        print(f"{'='*60}")
        
        # Calculate overall statistics
        profit_factors = [r.profit_factor for r in results if r.profit_factor != float('inf')]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        total_returns = [r.total_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        
        print(f"\nOverall Statistics:")
        print(f"Average Profit Factor: {np.mean(profit_factors):.3f}")
        print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.3f}")
        print(f"Average Total Return: {np.mean(total_returns)*100:.2f}%")
        print(f"Average Max Drawdown: {np.mean(max_drawdowns)*100:.2f}%")
        print(f"Average Win Rate: {np.mean(win_rates)*100:.2f}%")
        
        # Get best results by different metrics
        best_profit_factor = max(results, key=lambda x: x.profit_factor if x.profit_factor != float('inf') else 0)
        best_sharpe = max(results, key=lambda x: x.sharpe_ratio)
        best_return = max(results, key=lambda x: x.total_return)
        
        print(f"\nBest Results:")
        print(f"Best Profit Factor: {best_profit_factor.profit_factor:.3f} (Params: {best_profit_factor.parameters})")
        print(f"Best Sharpe Ratio: {best_sharpe.sharpe_ratio:.3f} (Params: {best_sharpe.parameters})")
        print(f"Best Total Return: {best_return.total_return*100:.2f}% (Params: {best_return.parameters})")
        
        # Display top N results by profit factor
        print(f"\nTop {top_n} Results by Profit Factor:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Profit Factor':<12} {'Sharpe':<8} {'Return %':<10} {'Drawdown %':<12} {'Win Rate %':<10} {'Trades':<8}")
        print("-" * 80)
        
        sorted_results = sorted(results, key=lambda x: x.profit_factor if x.profit_factor != float('inf') else 0, reverse=True)
        
        for i, result in enumerate(sorted_results[:top_n]):
            pf = result.profit_factor if result.profit_factor != float('inf') else float('inf')
            print(f"{i+1:<4} {pf:<12.3f} {result.sharpe_ratio:<8.3f} {result.total_return*100:<10.2f} "
                  f"{result.max_drawdown*100:<12.2f} {result.win_rate*100:<10.2f} {result.total_trades:<8}")
    
    def plot_equity_curves(self, results: List[BacktestResult], top_n: int = 5, save_path: str = None):
        """
        Plot equity curves for top N results
        
        Args:
            results: List of backtest results
            top_n: Number of top results to plot
            save_path: Path to save the plot
        """
        if not results:
            print("No results to plot")
            return
        
        # Get top N results by profit factor
        sorted_results = sorted(results, key=lambda x: x.profit_factor if x.profit_factor != float('inf') else 0, reverse=True)
        top_results = sorted_results[:top_n]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_results)))
        
        for i, result in enumerate(top_results):
            equity_curve = result.equity_curve
            if len(equity_curve) > 1:
                # Convert index to numeric for plotting
                if isinstance(equity_curve.index[0], str):
                    x_values = range(len(equity_curve))
                else:
                    x_values = equity_curve.index
                
                label = f"PF: {result.profit_factor:.2f}, Return: {result.total_return*100:.1f}%"
                ax.plot(x_values, equity_curve.values, label=label, color=colors[i], linewidth=2)
        
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title(f'Equity Curves - Top {top_n} Results by Profit Factor')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_parameter_heatmap(self, results: List[BacktestResult], param1: str, param2: str, 
                              metric: str = 'profit_factor', save_path: str = None):
        """
        Create a heatmap showing parameter performance
        
        Args:
            results: List of backtest results
            param1: First parameter name
            param2: Second parameter name
            metric: Metric to display
            save_path: Path to save the plot
        """
        if not results:
            print("No results to plot")
            return
        
        # Create pivot table
        data = []
        for result in results:
            if param1 in result.parameters and param2 in result.parameters:
                value = getattr(result, metric)
                if metric == 'profit_factor' and value == float('inf'):
                    value = 10  # Cap infinite values for visualization
                data.append({
                    param1: result.parameters[param1],
                    param2: result.parameters[param2],
                    metric: value
                })
        
        if not data:
            print(f"Parameters {param1} and {param2} not found in results")
            return
        
        df = pd.DataFrame(data)
        pivot_table = df.pivot_table(values=metric, index=param1, columns=param2, aggfunc='mean')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
        ax.set_title(f'{metric.replace("_", " ").title()} Heatmap: {param1} vs {param2}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_equity_curve(self, result: BacktestResult, save_path: str = None):
        """
        Create an interactive equity curve plot using Plotly
        
        Args:
            result: Single backtest result
            save_path: Path to save the HTML file
        """
        equity_curve = result.equity_curve
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Equity Curve', 'Drawdown'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index,
                y=equity_curve.values,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Calculate drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max * 100
        
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Interactive Equity Curve - Profit Factor: {result.profit_factor:.3f}',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def plot_performance_distribution(self, results: List[BacktestResult], save_path: str = None):
        """
        Plot distribution of performance metrics
        
        Args:
            results: List of backtest results
            save_path: Path to save the plot
        """
        if not results:
            print("No results to plot")
            return
        
        # Extract metrics
        profit_factors = [r.profit_factor for r in results if r.profit_factor != float('inf')]
        sharpe_ratios = [r.sharpe_ratio for r in results]
        total_returns = [r.total_return for r in results]
        max_drawdowns = [r.max_drawdown for r in results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Profit Factor distribution
        axes[0, 0].hist(profit_factors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Profit Factor Distribution')
        axes[0, 0].set_xlabel('Profit Factor')
        axes[0, 0].set_ylabel('Frequency')
        
        # Sharpe Ratio distribution
        axes[0, 1].hist(sharpe_ratios, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Sharpe Ratio Distribution')
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].set_ylabel('Frequency')
        
        # Total Return distribution
        axes[1, 0].hist([r*100 for r in total_returns], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Total Return Distribution')
        axes[1, 0].set_xlabel('Total Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Max Drawdown distribution
        axes[1, 1].hist([r*100 for r in max_drawdowns], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Max Drawdown Distribution')
        axes[1, 1].set_xlabel('Max Drawdown (%)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_detailed_report(self, result: BacktestResult) -> str:
        """
        Generate a detailed text report for a single result
        
        Args:
            result: Single backtest result
            
        Returns:
            Formatted report string
        """
        report = f"""
{'='*60}
DETAILED BACKTEST REPORT
{'='*60}

Strategy Parameters:
{'-'*30}
"""
        
        for param, value in result.parameters.items():
            report += f"{param}: {value}\n"
        
        report += f"""
Performance Metrics:
{'-'*30}
Total Return: {result.total_return*100:.2f}%
Sharpe Ratio: {result.sharpe_ratio:.3f}
Maximum Drawdown: {result.max_drawdown*100:.2f}%
Profit Factor: {result.profit_factor:.3f}
Win Rate: {result.win_rate*100:.2f}%

Trade Statistics:
{'-'*30}
Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Losing Trades: {result.losing_trades}
Average Win: {result.avg_win*100:.2f}%
Average Loss: {result.avg_loss*100:.2f}%

Risk Metrics:
{'-'*30}
        Risk-Reward Ratio: {f"{abs(result.avg_win/result.avg_loss):.3f}" if result.avg_loss != 0 else 'N/A'}
        Expected Value per Trade: {(result.avg_win * result.win_rate + result.avg_loss * (1-result.win_rate))*100:.2f}%
"""
        
        return report
    
    def save_results_to_csv(self, results: List[BacktestResult], filename: str):
        """
        Save all results to a CSV file
        
        Args:
            results: List of backtest results
            filename: Output CSV filename
        """
        if not results:
            print("No results to save")
            return
        
        # Create DataFrame
        data = []
        for result in results:
            row = result.parameters.copy()
            row.update({
                'profit_factor': result.profit_factor if result.profit_factor != float('inf') else 999,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades,
                'avg_win': result.avg_win,
                'avg_loss': result.avg_loss
            })
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}") 