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
                 max_risk_per_trade: float = 0.02,
                 hedge_logic: str = "single_brick",
                 hedge_brick_threshold: int = 2,
                 hedge_price_confirmation: bool = True,
                 hedge_trend_strength: bool = False,
                 enable_counter_trend: bool = False,
                 ma_20_period: int = 20,
                 ma_50_period: int = 50,
                 ma_200_period: int = 200):
        """
        Initialize R.T.H.E. strategy with moving averages
        
        Args:
            brick_size: Size of Renko bricks
            tsl_offset: Trailing stop loss offset from brick boundaries
            hedge_size_ratio: Ratio of hedge position size to core position
            min_bricks_for_trend: Minimum consecutive bricks for trend confirmation
            max_risk_per_trade: Maximum risk per trade as fraction of capital
            hedge_logic: Hedge confirmation method ('single_brick', 'multiple_bricks', 'price_level', 'trend_strength', 'combined')
            hedge_brick_threshold: Number of consecutive reversal bricks required for hedge
            hedge_price_confirmation: Whether to require price level confirmation
            hedge_trend_strength: Whether to require trend strength confirmation
            enable_counter_trend: Whether to allow counter-trend entries
            ma_20_period: Period for 20-day SMA
            ma_50_period: Period for 50-day SMA
            ma_200_period: Period for 200-day SMA
        """
        super().__init__("RTHE")
        
        # Core parameters
        self.brick_size = brick_size
        self.tsl_offset = tsl_offset
        self.hedge_size_ratio = hedge_size_ratio
        self.min_bricks_for_trend = min_bricks_for_trend
        self.max_risk_per_trade = max_risk_per_trade
        
        # Hedge logic parameters
        self.hedge_logic = hedge_logic
        self.hedge_brick_threshold = hedge_brick_threshold
        self.hedge_price_confirmation = hedge_price_confirmation
        self.hedge_trend_strength = hedge_trend_strength
        
        # Moving average parameters
        self.enable_counter_trend = enable_counter_trend
        self.ma_20_period = ma_20_period
        self.ma_50_period = ma_50_period
        self.ma_200_period = ma_200_period
        
        # Moving average values (will be calculated during execution)
        self.ma_20 = 0.0
        self.ma_50 = 0.0
        self.ma_200 = 0.0
        self.ma_20_prev = 0.0
        self.ma_50_prev = 0.0
        
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
        
        # Track previous brick levels for price confirmation
        self.previous_brick_high = 0.0
        self.previous_brick_low = 0.0
        
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
        self.previous_brick_high = 0.0
        self.previous_brick_low = 0.0
        self.core_trades = []
        self.hedge_trades = []
        self.total_trades = 0
        self.hedge_activations = 0
        self.tsl_hits = 0
        self.trend_continuations = 0
        
        # Reset moving averages
        self.ma_20 = 0.0
        self.ma_50 = 0.0
        self.ma_200 = 0.0
        self.ma_20_prev = 0.0
        self.ma_50_prev = 0.0
    
    def calculate_moving_averages(self, renko_data: pd.DataFrame, index: int):
        """Calculate moving averages using Renko close series only."""
        if index < max(self.ma_200_period, self.ma_50_period, self.ma_20_period):
            return False
        
        self.ma_20_prev = self.ma_20
        self.ma_50_prev = self.ma_50
        
        close_prices = renko_data['close'].values
        if index >= self.ma_20_period:
            self.ma_20 = np.mean(close_prices[index - self.ma_20_period + 1:index + 1])
        if index >= self.ma_50_period:
            self.ma_50 = np.mean(close_prices[index - self.ma_50_period + 1:index + 1])
        if index >= self.ma_200_period:
            self.ma_200 = np.mean(close_prices[index - self.ma_200_period + 1:index + 1])
        return True
    
    def check_ma_bullish_crossover(self) -> bool:
        """Check if 20 and 50 SMA have bullish crossover"""
        if self.ma_20_prev == 0 or self.ma_50_prev == 0:
            return False
        
        # Check for bullish crossover: 20 SMA crosses above 50 SMA
        crossover = (self.ma_20_prev <= self.ma_50_prev and self.ma_20 > self.ma_50)
        
        # Check if both MAs are upsloping
        upsloping = (self.ma_20 > self.ma_20_prev and self.ma_50 > self.ma_50_prev)
        
        return crossover and upsloping
    
    def check_ma_bearish_crossover(self) -> bool:
        """Check if 20 and 50 SMA have bearish crossover"""
        if self.ma_20_prev == 0 or self.ma_50_prev == 0:
            return False
        
        # Check for bearish crossover: 20 SMA crosses below 50 SMA
        crossover = (self.ma_20_prev >= self.ma_50_prev and self.ma_20 < self.ma_50)
        
        # Check if both MAs are downsloping
        downsloping = (self.ma_20 < self.ma_20_prev and self.ma_50 < self.ma_50_prev)
        
        return crossover and downsloping
    
    def check_ma_trend_alignment(self, direction: int) -> bool:
        """Check if moving averages align with trade direction"""
        if direction == 1:  # Long
            # Price above 20 SMA, 20 SMA above 50 SMA, 50 SMA above 200 SMA
            current_price = self.last_brick_high  # Use current brick high for long entries
            return (current_price > self.ma_20 and 
                   self.ma_20 > self.ma_50 and 
                   self.ma_50 > self.ma_200)
        elif direction == -1:  # Short
            # Price below 20 SMA, 20 SMA below 50 SMA, 50 SMA below 200 SMA
            current_price = self.last_brick_low  # Use current brick low for short entries
            return (current_price < self.ma_20 and 
                   self.ma_20 < self.ma_50 and 
                   self.ma_50 < self.ma_200)
        return False
    
    def execute_trade(self, data: pd.DataFrame, index: int):
        """Execute R.T.H.E. trading logic for current brick. All MAs are calculated on Renko close series."""
        if index < 1:
            return
        
        # Calculate moving averages on Renko close
        self.calculate_moving_averages(data, index)
        
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
        
        # Store previous levels before updating
        self.previous_brick_high = self.last_brick_high
        self.previous_brick_low = self.last_brick_low
        
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
        """Check for core trend entry with MA-based rules"""
        current_brick = data.iloc[index]
        current_price = current_brick['close']
        current_date = current_brick.get('date', index)
        
        # Long entry: Up brick with bullish MA crossover and trend alignment
        if (current_brick['direction'] == 1 and 
            self.consecutive_up_bricks >= self.min_bricks_for_trend and
            current_price >= self.last_brick_high):
            
            # Check MA conditions for long entry
            ma_bullish = self.check_ma_bullish_crossover()
            ma_aligned = self.check_ma_trend_alignment(1)
            
            if ma_bullish and ma_aligned:
                self._enter_core_long(current_price, current_date)
        
        # Short entry: Down brick with bearish MA crossover and trend alignment
        elif (current_brick['direction'] == -1 and 
              self.consecutive_down_bricks >= self.min_bricks_for_trend and
              current_price <= self.last_brick_low):
            
            # Only allow short entries if counter-trend is enabled
            if self.enable_counter_trend:
                # Check MA conditions for short entry
                ma_bearish = self.check_ma_bearish_crossover()
                ma_aligned = self.check_ma_trend_alignment(-1)
                
                if ma_bearish and ma_aligned:
                    self._enter_core_short(current_price, current_date)
    
    def _check_hedge_entry(self, data: pd.DataFrame, index: int):
        """Check for hedge entry on reversal with multiple confirmation methods"""
        current_brick = data.iloc[index]
        current_price = current_brick['close']
        current_date = current_brick.get('date', index)
        
        # Check hedge conditions based on logic type
        if self.core_position == 1:  # Long core position
            if self._should_hedge_long_position(current_brick, current_price):
                self._enter_hedge_short(current_price, current_date)
        
        elif self.core_position == -1:  # Short core position
            if self._should_hedge_short_position(current_brick, current_price):
                self._enter_hedge_long(current_price, current_date)
    
    def _should_hedge_long_position(self, current_brick: pd.Series, current_price: float) -> bool:
        """Check if we should hedge a long position based on selected logic"""
        if current_brick['direction'] != -1:  # Must be down brick
            return False
        
        # Logic 1: Single brick (original behavior)
        if self.hedge_logic == "single_brick":
            return True
        
        # Logic 2: Multiple bricks confirmation
        if self.hedge_logic == "multiple_bricks":
            if self.consecutive_down_bricks < self.hedge_brick_threshold:
                return False
        
        # Logic 3: Price level confirmation
        if self.hedge_price_confirmation:
            # Price must break below previous brick low
            if current_price >= self.previous_brick_low:
                return False
        
        # Logic 4: Trend strength confirmation
        if self.hedge_trend_strength:
            # Down trend must be stronger than up trend
            if self.consecutive_down_bricks <= self.consecutive_up_bricks:
                return False
        
        # Combined logic: All enabled conditions must be met
        if self.hedge_logic == "combined":
            brick_ok = self.consecutive_down_bricks >= self.hedge_brick_threshold
            price_ok = not self.hedge_price_confirmation or current_price < self.previous_brick_low
            strength_ok = not self.hedge_trend_strength or self.consecutive_down_bricks > self.consecutive_up_bricks
            return brick_ok and price_ok and strength_ok
        
        # For other logic types, just check the specific condition
        return True
    
    def _should_hedge_short_position(self, current_brick: pd.Series, current_price: float) -> bool:
        """Check if we should hedge a short position based on selected logic"""
        if current_brick['direction'] != 1:  # Must be up brick
            return False
        
        # Logic 1: Single brick (original behavior)
        if self.hedge_logic == "single_brick":
            return True
        
        # Logic 2: Multiple bricks confirmation
        if self.hedge_logic == "multiple_bricks":
            if self.consecutive_up_bricks < self.hedge_brick_threshold:
                return False
        
        # Logic 3: Price level confirmation
        if self.hedge_price_confirmation:
            # Price must break above previous brick high
            if current_price <= self.previous_brick_high:
                return False
        
        # Logic 4: Trend strength confirmation
        if self.hedge_trend_strength:
            # Up trend must be stronger than down trend
            if self.consecutive_up_bricks <= self.consecutive_down_bricks:
                return False
        
        # Combined logic: All enabled conditions must be met
        if self.hedge_logic == "combined":
            brick_ok = self.consecutive_up_bricks >= self.hedge_brick_threshold
            price_ok = not self.hedge_price_confirmation or current_price > self.previous_brick_high
            strength_ok = not self.hedge_trend_strength or self.consecutive_up_bricks > self.consecutive_down_bricks
            return brick_ok and price_ok and strength_ok
        
        # For other logic types, just check the specific condition
        return True
    
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