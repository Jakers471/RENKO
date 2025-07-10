import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from rthe_strategy import RTHEStrategy, RTHETrade

class RTHEAnalyzer:
    """Specialized analyzer for R.T.H.E. strategy performance"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def display_rthe_metrics(self, strategy: RTHEStrategy):
        """Display comprehensive R.T.H.E. performance metrics"""
        
        metrics = strategy.get_performance_metrics()
        
        if not metrics:
            print("No R.T.H.E. trades found")
            return
        
        print("\n" + "="*80)
        print("RENKO TREND-HEDGE ENGINE (R.T.H.E.) PERFORMANCE REPORT")
        print("="*80)
        
        # Core Strategy Metrics
        print(f"\n📊 CORE STRATEGY METRICS:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Core Trades: {metrics['core_trades']}")
        print(f"  Hedge Trades: {metrics['hedge_trades']}")
        print(f"  Core Win Rate: {metrics.get('core_win_rate', 0)*100:.1f}%")
        print(f"  Core Profit Factor: {metrics.get('core_profit_factor', 0):.3f}")
        print(f"  Core Avg Win: {metrics.get('core_avg_win', 0)*100:.2f}%")
        print(f"  Core Avg Loss: {metrics.get('core_avg_loss', 0)*100:.2f}%")
        
        # Hedge Performance
        print(f"\n🛡️ HEDGE PERFORMANCE:")
        print(f"  Hedge Activations: {metrics['hedge_activations']}")
        print(f"  Hedge Activation Rate: {metrics['hedge_activation_rate']*100:.1f}%")
        print(f"  Hedge Win Rate: {metrics.get('hedge_win_rate', 0)*100:.1f}%")
        print(f"  Hedge Avg Win: {metrics.get('hedge_avg_win', 0)*100:.2f}%")
        print(f"  Hedge Avg Loss: {metrics.get('hedge_avg_loss', 0)*100:.2f}%")
        
        # Risk Management
        print(f"\n🎯 RISK MANAGEMENT:")
        print(f"  TSL Hits: {metrics['tsl_hits']}")
        print(f"  TSL Hit Rate: {metrics['tsl_hit_rate']*100:.1f}%")
        print(f"  Trend Continuations: {metrics['trend_continuations']}")
        print(f"  Trend Continuation Rate: {metrics['trend_continuation_rate']*100:.1f}%")
        
        # Strategy Effectiveness
        print(f"\n⚡ STRATEGY EFFECTIVENESS:")
        if metrics['hedge_activations'] > 0:
            hedge_effectiveness = metrics['trend_continuations'] / metrics['hedge_activations']
            print(f"  Hedge Effectiveness: {hedge_effectiveness*100:.1f}% (trends that resumed)")
        
        if metrics['core_trades'] > 0:
            tsl_effectiveness = 1 - metrics['tsl_hit_rate']
            print(f"  TSL Effectiveness: {tsl_effectiveness*100:.1f}% (trades not stopped out)")
    
    def plot_rthe_analysis(self, strategy: RTHEStrategy, save_path: str = None):
        """Create comprehensive R.T.H.E. analysis plots"""
        
        if not strategy.core_trades and not strategy.hedge_trades:
            print("No trades to analyze")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('R.T.H.E. Strategy Analysis', fontsize=16, fontweight='bold')
        
        # 1. Core vs Hedge Performance
        if strategy.core_trades and strategy.hedge_trades:
            core_pnls = [t.pnl for t in strategy.core_trades if t.trade_type == 'core_exit']
            hedge_pnls = [t.pnl for t in strategy.hedge_trades if t.trade_type == 'hedge_exit']
            
            axes[0, 0].hist([core_pnls, hedge_pnls], bins=20, alpha=0.7, 
                           label=['Core Trades', 'Hedge Trades'])
            axes[0, 0].set_title('Core vs Hedge PnL Distribution')
            axes[0, 0].set_xlabel('PnL (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
        
        # 2. TSL Hit Analysis
        if strategy.core_trades:
            tsl_hits = [t.pnl for t in strategy.core_trades if t.tsl_hit]
            non_tsl = [t.pnl for t in strategy.core_trades if not t.tsl_hit]
            
            if tsl_hits and non_tsl:
                axes[0, 1].hist([tsl_hits, non_tsl], bins=15, alpha=0.7,
                               label=['TSL Hits', 'Non-TSL Exits'])
                axes[0, 1].set_title('TSL Hit vs Non-TSL Exit Performance')
                axes[0, 1].set_xlabel('PnL (%)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].legend()
        
        # 3. Hedge Activation Timeline
        if strategy.core_trades:
            hedge_activations = [i for i, t in enumerate(strategy.core_trades) if t.hedge_activated]
            if hedge_activations:
                axes[0, 2].plot(hedge_activations, 'ro-', alpha=0.7)
                axes[0, 2].set_title('Hedge Activation Timeline')
                axes[0, 2].set_xlabel('Trade Number')
                axes[0, 2].set_ylabel('Hedge Activated')
                axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Brick Count Analysis
        if strategy.core_trades:
            brick_counts = [t.brick_count for t in strategy.core_trades if t.brick_count > 0]
            if brick_counts:
                axes[1, 0].hist(brick_counts, bins=15, alpha=0.7, color='green')
                axes[1, 0].set_title('Brick Count Distribution')
                axes[1, 0].set_xlabel('Consecutive Bricks')
                axes[1, 0].set_ylabel('Frequency')
        
        # 5. Trade Duration Analysis
        if strategy.core_trades:
            # Calculate trade durations (simplified)
            durations = []
            for i in range(1, len(strategy.core_trades)):
                if strategy.core_trades[i].trade_type == 'core_exit':
                    durations.append(i)
            
            if durations:
                axes[1, 1].hist(durations, bins=15, alpha=0.7, color='orange')
                axes[1, 1].set_title('Trade Duration Distribution')
                axes[1, 1].set_xlabel('Bricks per Trade')
                axes[1, 1].set_ylabel('Frequency')
        
        # 6. Performance Summary
        metrics = strategy.get_performance_metrics()
        if metrics:
            summary_data = {
                'Core Win Rate': metrics.get('core_win_rate', 0) * 100,
                'Hedge Win Rate': metrics.get('hedge_win_rate', 0) * 100,
                'TSL Hit Rate': metrics.get('tsl_hit_rate', 0) * 100,
                'Trend Continuation': metrics.get('trend_continuation_rate', 0) * 100
            }
            
            keys = list(summary_data.keys())
            values = list(summary_data.values())
            
            axes[1, 2].bar(keys, values, alpha=0.7, color=['blue', 'green', 'red', 'purple'])
            axes[1, 2].set_title('Key Performance Metrics')
            axes[1, 2].set_ylabel('Percentage (%)')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_trade_log(self, strategy: RTHEStrategy) -> pd.DataFrame:
        """Generate detailed trade log for R.T.H.E. strategy"""
        
        all_trades = []
        
        # Add core trades
        for trade in strategy.core_trades:
            all_trades.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': 'LONG' if trade.direction == 1 else 'SHORT',
                'pnl': trade.pnl * 100,
                'trade_type': trade.trade_type,
                'brick_count': trade.brick_count,
                'hedge_activated': trade.hedge_activated,
                'tsl_hit': trade.tsl_hit,
                'position_type': 'CORE'
            })
        
        # Add hedge trades
        for trade in strategy.hedge_trades:
            all_trades.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': 'LONG' if trade.direction == 1 else 'SHORT',
                'pnl': trade.pnl * 100,
                'trade_type': trade.trade_type,
                'brick_count': trade.brick_count,
                'hedge_activated': trade.hedge_activated,
                'tsl_hit': trade.tsl_hit,
                'position_type': 'HEDGE'
            })
        
        if all_trades:
            df = pd.DataFrame(all_trades)
            df = df.sort_values('entry_date').reset_index(drop=True)
            return df
        
        return pd.DataFrame()
    
    def save_rthe_report(self, strategy: RTHEStrategy, filename: str = "rthe_report.csv"):
        """Save comprehensive R.T.H.E. report to CSV"""
        
        # Get trade log
        trade_log = self.generate_trade_log(strategy)
        
        if not trade_log.empty:
            trade_log.to_csv(filename, index=False)
            print(f"R.T.H.E. trade log saved to: {filename}")
            
            # Also save summary metrics
            metrics = strategy.get_performance_metrics()
            if metrics:
                summary_df = pd.DataFrame([metrics])
                summary_filename = filename.replace('.csv', '_summary.csv')
                summary_df.to_csv(summary_filename, index=False)
                print(f"R.T.H.E. summary metrics saved to: {summary_filename}")
        else:
            print("No trades to save") 