import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from .trading_strategy import TradingStrategy, Trade

@dataclass
class RTHETrade:
    """Represents a R.T.H.E. trade with core and hedge positions"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    pnl: float
    trade_type: str  # 'core', 'hedge', 'core_exit', 'hedge_exit'
    brick_count: int
    hedge_activated: bool
    tsl_hit: bool
    position_size: float = 1.0  # 1.0 for core, hedge_size_ratio for hedge

class RTHEStrategy(TradingStrategy):
    """
    Renko Trend-Hedge Engine (R.T.H.E.) Strategy
    
    Philosophy:
    - Core positions follow established trends using Renko brick analysis
    - Hedge positions protect against reversals during trend continuation
    - Trailing stops manage risk while allowing trend extension
    - Position sizing adapts to market volatility
    """
    
    def __init__(self, 
                 brick_size: float = 20.0,
                 tsl_offset: float = 5.0,
                 hedge_size_ratio: float = 0.5,
                 min_bricks_for_trend: int = 2,
                 max_risk_per_trade: float = 0.02):
        """
        Initialize R.T.H.E. strategy
        
        Args:
            brick_size: Size of Renko bricks
            tsl_offset: Trailing stop loss offset from brick boundaries
            hedge_size_ratio: Ratio of hedge position size to core position
            min_bricks_for_trend: Minimum consecutive bricks for trend confirmation
            max_risk_per_trade: Maximum risk per trade as fraction of capital
        """
        super().__init__("RTHE")
        
        # Core parameters
        self.brick_size = brick_size
        self.tsl_offset = tsl_offset
        self.hedge_size_ratio = hedge_size_ratio
        self.min_bricks_for_trend = min_bricks_for_trend
        self.max_risk_per_trade = max_risk_per_trade
        
        # Position tracking
        self.core_position = 0  # 0: no position, 1: long, -1: short
        self.hedge_position = 0  # 0: no hedge, 1: long hedge, -1: short hedge
        self.core_entry_price = 0.0
        self.hedge_entry_price = 0.0
        self.core_entry_date = None
        self.hedge_entry_date = None
        
        # Risk management
        self.tsl_level = 0.0
        
        # Brick tracking
        self.consecutive_up_bricks = 0
        self.consecutive_down_bricks = 0
        self.last_brick_direction = 0
        self.last_brick_high = 0.0
        self.last_brick_low = 0.0
        
        # Performance tracking
        self.core_trades = []
        self.hedge_trades = []
        self.total_trades = 0
        self.hedge_activations = 0
        self.tsl_hits = 0
        self.trend_continuations = 0
    
    def reset(self):
        """Reset strategy state"""
        self.core_position = 0
        self.hedge_position = 0
        self.core_entry_price = 0.0
        self.hedge_entry_price = 0.0
        self.core_entry_date = None
        self.hedge_entry_date = None
        self.tsl_level = 0.0
        self.consecutive_up_bricks = 0
        self.consecutive_down_bricks = 0
        self.last_brick_direction = 0
        self.last_brick_high = 0.0
        self.last_brick_low = 0.0
        self.core_trades = []
        self.hedge_trades = []
        self.total_trades = 0
        self.hedge_activations = 0
        self.tsl_hits = 0
        self.trend_continuations = 0
    
    def execute_trade(self, data: pd.DataFrame, index: int):
        """Execute R.T.H.E. trading logic for current brick"""
        if index < 1:  # Need at least 2 bricks for analysis
            return
        
        current_brick = data.iloc[index]
        previous_brick = data.iloc[index - 1]
        
        # Update brick tracking
        self._update_brick_tracking(current_brick, previous_brick)
        
        # Check for core trend entry
        if self.core_position == 0:
            self._check_core_entry(data, index)
        
        # Check for hedge entry
        elif self.core_position != 0 and self.hedge_position == 0:
            self._check_hedge_entry(data, index)
        
        # Check for exits
        if self.core_position != 0:
            self._check_exits(data, index)
    
    def _update_brick_tracking(self, current_brick: pd.Series, previous_brick: pd.Series):
        """Update brick direction tracking"""
        current_direction = current_brick['direction']
        
        if current_direction == 1:  # Up brick
            if self.last_brick_direction == 1:
                self.consecutive_up_bricks += 1
            else:
                self.consecutive_up_bricks = 1
            self.consecutive_down_bricks = 0
        elif current_direction == -1:  # Down brick
            if self.last_brick_direction == -1:
                self.consecutive_down_bricks += 1
            else:
                self.consecutive_down_bricks = 1
            self.consecutive_up_bricks = 0
        
        self.last_brick_direction = current_direction
        self.last_brick_low = current_brick['low']
        self.last_brick_high = current_brick['high']
    
    def _check_core_entry(self, data: pd.DataFrame, index: int):
        """Check for core trend entry"""
        current_brick = data.iloc[index]
        current_price = current_brick['close']
        current_date = current_brick.get('date', index)
        
        # Long entry: Up brick breaks previous resistance
        if (current_brick['direction'] == 1 and 
            self.consecutive_up_bricks >= self.min_bricks_for_trend and
            current_price >= self.last_brick_high):
            self._enter_core_long(current_price, current_date)
        
        # Short entry: Down brick breaks previous support
        elif (current_brick['direction'] == -1 and 
              self.consecutive_down_bricks >= self.min_bricks_for_trend and
              current_price <= self.last_brick_low):
            self._enter_core_short(current_price, current_date)
    
    def _check_hedge_entry(self, data: pd.DataFrame, index: int):
        """Check for hedge entry on reversal"""
        current_brick = data.iloc[index]
        current_price = current_brick['close']
        current_date = current_brick.get('date', index)
        
        # Hedge against core long position
        if (self.core_position == 1 and 
            current_brick['direction'] == -1):
            
            self._enter_hedge_short(current_price, current_date)
        
        # Hedge against core short position
        elif (self.core_position == -1 and 
              current_brick['direction'] == 1):
            
            self._enter_hedge_long(current_price, current_date)
    
    def _check_exits(self, data: pd.DataFrame, index: int):
        """Check for various exit conditions"""
        current_brick = data.iloc[index]
        current_price = current_brick['close']
        current_date = current_brick.get('date', index)
        
        # Update trailing stop level
        if self.core_position == 1:  # Long position
            self.tsl_level = self.last_brick_low - self.tsl_offset
        elif self.core_position == -1:  # Short position
            self.tsl_level = self.last_brick_high + self.tsl_offset
        
        # Check TSL hit
        if self._check_tsl_hit(current_price):
            self._exit_core_tsl(current_price, current_date)
            if self.hedge_position != 0:
                self._exit_hedge(current_price, current_date)
            return
        
        # Check hedge resolution
        if self.hedge_position != 0:
            self._check_hedge_resolution(data, index)
    
    def _check_hedge_resolution(self, data: pd.DataFrame, index: int):
        """Check if hedge should be closed or promoted"""
        current_brick = data.iloc[index]
        current_price = current_brick['close']
        current_date = current_brick.get('date', index)
        
        # Trend resumes in core direction
        if ((self.core_position == 1 and self.hedge_position == -1 and 
             current_brick['direction'] == 1) or
            (self.core_position == -1 and self.hedge_position == 1 and 
             current_brick['direction'] == -1)):
            
            # Close hedge, keep core
            self._exit_hedge(current_price, current_date)
            self.trend_continuations += 1
        
        # Second reversal confirms new trend
        elif ((self.core_position == 1 and self.hedge_position == -1 and 
               self.consecutive_down_bricks >= 2) or
              (self.core_position == -1 and self.hedge_position == 1 and 
               self.consecutive_up_bricks >= 2)):
            
            # Close core, promote hedge to core
            self._promote_hedge_to_core(current_price, current_date)
    
    def _enter_core_long(self, price: float, date):
        """Enter core long position"""
        self.core_position = 1
        self.core_entry_price = price
        self.core_entry_date = date
        self.tsl_level = self.last_brick_low - self.tsl_offset
    
    def _enter_core_short(self, price: float, date):
        """Enter core short position"""
        self.core_position = -1
        self.core_entry_price = price
        self.core_entry_date = date
        self.tsl_level = self.last_brick_high + self.tsl_offset
    
    def _enter_hedge_short(self, price: float, date):
        """Enter hedge short position (position sizing by hedge_size_ratio)"""
        self.hedge_position = -1
        self.hedge_entry_price = price
        self.hedge_entry_date = date
        self.hedge_activations += 1
        # Position size is determined by hedge_size_ratio
    
    def _enter_hedge_long(self, price: float, date):
        """Enter hedge long position (position sizing by hedge_size_ratio)"""
        self.hedge_position = 1
        self.hedge_entry_price = price
        self.hedge_entry_date = date
        self.hedge_activations += 1
        # Position size is determined by hedge_size_ratio
    
    def _check_tsl_hit(self, current_price: float) -> bool:
        """Check if trailing stop loss is hit"""
        if self.core_position == 1:  # Long position
            return current_price <= self.tsl_level
        elif self.core_position == -1:  # Short position
            return current_price >= self.tsl_level
        return False
    
    def _exit_core_tsl(self, price: float, date):
        """Exit core position via TSL"""
        pnl = (price - self.core_entry_price) / self.core_entry_price
        if self.core_position == -1:  # Short position
            pnl = -pnl
        trade = RTHETrade(
            entry_date=str(self.core_entry_date),
            exit_date=str(date),
            entry_price=self.core_entry_price,
            exit_price=price,
            direction=self.core_position,
            pnl=pnl,
            trade_type='core_exit',
            brick_count=self.consecutive_up_bricks if self.core_position == 1 else self.consecutive_down_bricks,
            hedge_activated=self.hedge_position != 0,
            tsl_hit=True,
            position_size=1.0
        )
        self.core_trades.append(trade)
        self.tsl_hits += 1
        self.core_position = 0
        self.core_entry_price = 0.0
        self.core_entry_date = None
    
    def _exit_hedge(self, price: float, date):
        """Exit hedge position"""
        pnl = (price - self.hedge_entry_price) / self.hedge_entry_price
        if self.hedge_position == -1:  # Short hedge
            pnl = -pnl
        # Scale PnL by hedge_size_ratio
        pnl = pnl * self.hedge_size_ratio
        trade = RTHETrade(
            entry_date=str(self.hedge_entry_date),
            exit_date=str(date),
            entry_price=self.hedge_entry_price,
            exit_price=price,
            direction=self.hedge_position,
            pnl=pnl,
            trade_type='hedge_exit',
            brick_count=0,
            hedge_activated=True,
            tsl_hit=False,
            position_size=self.hedge_size_ratio
        )
        self.hedge_trades.append(trade)
        self.hedge_position = 0
        self.hedge_entry_price = 0.0
        self.hedge_entry_date = None
    
    def _promote_hedge_to_core(self, price: float, date):
        """Promote hedge to core position"""
        # Close core position
        core_pnl = (price - self.core_entry_price) / self.core_entry_price
        if self.core_position == -1:  # Short position
            core_pnl = -core_pnl
        core_trade = RTHETrade(
            entry_date=str(self.core_entry_date),
            exit_date=str(date),
            entry_price=self.core_entry_price,
            exit_price=price,
            direction=self.core_position,
            pnl=core_pnl,
            trade_type='core_exit',
            brick_count=0,
            hedge_activated=True,
            tsl_hit=False,
            position_size=1.0
        )
        self.core_trades.append(core_trade)
        
        # Promote hedge to core
        self.core_position = self.hedge_position
        self.core_entry_price = self.hedge_entry_price
        self.core_entry_date = self.hedge_entry_date
        
        # Reset hedge
        self.hedge_position = 0
        self.hedge_entry_price = 0.0
        self.hedge_entry_date = None
        
        # Update TSL
        if self.core_position == 1:
            self.tsl_level = self.last_brick_low - self.tsl_offset
        else:
            self.tsl_level = self.last_brick_high + self.tsl_offset
    
    @property
    def trades(self):
        """Return all completed trades for backtest engine compatibility"""
        return self.core_trades + self.hedge_trades
    
    def get_performance_metrics(self) -> Dict:
        """Get R.T.H.E. specific performance metrics"""
        all_trades = self.core_trades + self.hedge_trades
        
        if not all_trades:
            return {}
        
        core_trades = [t for t in self.core_trades if t.trade_type == 'core_exit']
        hedge_trades = [t for t in self.hedge_trades if t.trade_type == 'hedge_exit']
        
        metrics = {
            'total_trades': len(all_trades),
            'core_trades': len(core_trades),
            'hedge_trades': len(hedge_trades),
            'hedge_activations': self.hedge_activations,
            'tsl_hits': self.tsl_hits,
            'trend_continuations': self.trend_continuations,
            'hedge_activation_rate': self.hedge_activations / max(len(core_trades), 1),
            'tsl_hit_rate': self.tsl_hits / max(len(core_trades), 1),
            'trend_continuation_rate': self.trend_continuations / max(self.hedge_activations, 1)
        }
        
        if core_trades:
            core_pnls = [t.pnl for t in core_trades]
            metrics.update({
                'core_win_rate': len([p for p in core_pnls if p > 0]) / len(core_pnls),
                'core_avg_win': np.mean([p for p in core_pnls if p > 0]) if any(p > 0 for p in core_pnls) else 0,
                'core_avg_loss': np.mean([p for p in core_pnls if p < 0]) if any(p < 0 for p in core_pnls) else 0,
                'core_profit_factor': sum([p for p in core_pnls if p > 0]) / abs(sum([p for p in core_pnls if p < 0])) if any(p < 0 for p in core_pnls) else float('inf')
            })
        
        if hedge_trades:
            hedge_pnls = [t.pnl for t in hedge_trades]
            metrics.update({
                'hedge_win_rate': len([p for p in hedge_pnls if p > 0]) / len(hedge_pnls),
                'hedge_avg_win': np.mean([p for p in hedge_pnls if p > 0]) if any(p > 0 for p in hedge_pnls) else 0,
                'hedge_avg_loss': np.mean([p for p in hedge_pnls if p < 0]) if any(p < 0 for p in hedge_pnls) else 0
            })
        
        return metrics 