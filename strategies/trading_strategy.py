import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    pnl: float
    entry_reason: str
    exit_reason: str

class TradingStrategy:
    """Base trading strategy class for Renko-based trading"""
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.current_position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.entry_date = None
        
    def reset(self):
        """Reset strategy state"""
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_date = None
    
    def should_enter_long(self, data: pd.DataFrame, index: int) -> bool:
        """Override this method to implement long entry logic"""
        return False
    
    def should_enter_short(self, data: pd.DataFrame, index: int) -> bool:
        """Override this method to implement short entry logic"""
        return False
    
    def should_exit_long(self, data: pd.DataFrame, index: int) -> bool:
        """Override this method to implement long exit logic"""
        return False
    
    def should_exit_short(self, data: pd.DataFrame, index: int) -> bool:
        """Override this method to implement short exit logic"""
        return False
    
    def execute_trade(self, data: pd.DataFrame, index: int):
        """Execute trading logic for current bar"""
        if self.current_position == 0:  # No position
            if self.should_enter_long(data, index):
                self.enter_long(data, index)
            elif self.should_enter_short(data, index):
                self.enter_short(data, index)
        
        elif self.current_position == 1:  # Long position
            if self.should_exit_long(data, index):
                self.exit_long(data, index)
        
        elif self.current_position == -1:  # Short position
            if self.should_exit_short(data, index):
                self.exit_short(data, index)
    
    def enter_long(self, data: pd.DataFrame, index: int):
        """Enter long position"""
        self.current_position = 1
        self.entry_price = data.iloc[index]['close']
        self.entry_date = data.iloc[index].get('date', index)
    
    def enter_short(self, data: pd.DataFrame, index: int):
        """Enter short position"""
        self.current_position = -1
        self.entry_price = data.iloc[index]['close']
        self.entry_date = data.iloc[index].get('date', index)
    
    def exit_long(self, data: pd.DataFrame, index: int):
        """Exit long position"""
        exit_price = data.iloc[index]['close']
        exit_date = data.iloc[index].get('date', index)
        pnl = (exit_price - self.entry_price) / self.entry_price
        
        trade = Trade(
            entry_date=str(self.entry_date),
            exit_date=str(exit_date),
            entry_price=self.entry_price,
            exit_price=exit_price,
            direction=1,
            pnl=pnl,
            entry_reason="Long Entry",
            exit_reason="Long Exit"
        )
        
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_date = None
    
    def exit_short(self, data: pd.DataFrame, index: int):
        """Exit short position"""
        exit_price = data.iloc[index]['close']
        exit_date = data.iloc[index].get('date', index)
        pnl = (self.entry_price - exit_price) / self.entry_price
        
        trade = Trade(
            entry_date=str(self.entry_date),
            exit_date=str(exit_date),
            entry_price=self.entry_price,
            exit_price=exit_price,
            direction=-1,
            pnl=pnl,
            entry_reason="Short Entry",
            exit_reason="Short Exit"
        )
        
        self.current_position = 0
        self.entry_price = 0.0
        self.entry_date = None

class RenkoBreakoutStrategy(TradingStrategy):
    """Renko breakout strategy with configurable parameters"""
    
    def __init__(self, 
                 consecutive_bricks: int = 3,
                 stop_loss_bricks: int = 2,
                 take_profit_bricks: int = 5,
                 min_bricks_for_trend: int = 2):
        super().__init__("RenkoBreakout")
        self.consecutive_bricks = consecutive_bricks
        self.stop_loss_bricks = stop_loss_bricks
        self.take_profit_bricks = take_profit_bricks
        self.min_bricks_for_trend = min_bricks_for_trend
        self.brick_size = 0.0
        
    def set_brick_size(self, brick_size: float):
        """Set the brick size for calculations"""
        self.brick_size = brick_size
    
    def should_enter_long(self, data: pd.DataFrame, index: int) -> bool:
        """Enter long on consecutive up bricks"""
        if index < self.consecutive_bricks:
            return False
        
        # Check for consecutive up bricks
        recent_bricks = data.iloc[index-self.consecutive_bricks+1:index+1]
        if len(recent_bricks) < self.consecutive_bricks:
            return False
        
        # All recent bricks should be up
        return all(recent_bricks['direction'] == 1)
    
    def should_enter_short(self, data: pd.DataFrame, index: int) -> bool:
        """Enter short on consecutive down bricks"""
        if index < self.consecutive_bricks:
            return False
        
        # Check for consecutive down bricks
        recent_bricks = data.iloc[index-self.consecutive_bricks+1:index+1]
        if len(recent_bricks) < self.consecutive_bricks:
            return False
        
        # All recent bricks should be down
        return all(recent_bricks['direction'] == -1)
    
    def should_exit_long(self, data: pd.DataFrame, index: int) -> bool:
        """Exit long on stop loss or take profit"""
        if self.current_position != 1:
            return False
        
        current_price = data.iloc[index]['close']
        price_change = current_price - self.entry_price
        
        # Stop loss
        if price_change <= -self.stop_loss_bricks * self.brick_size:
            return True
        
        # Take profit
        if price_change >= self.take_profit_bricks * self.brick_size:
            return True
        
        # Trend reversal (down brick after up trend)
        if index >= self.min_bricks_for_trend:
            recent_bricks = data.iloc[index-self.min_bricks_for_trend+1:index+1]
            if all(recent_bricks['direction'] == -1):
                return True
        
        return False
    
    def should_exit_short(self, data: pd.DataFrame, index: int) -> bool:
        """Exit short on stop loss or take profit"""
        if self.current_position != -1:
            return False
        
        current_price = data.iloc[index]['close']
        price_change = self.entry_price - current_price
        
        # Stop loss
        if price_change <= -self.stop_loss_bricks * self.brick_size:
            return True
        
        # Take profit
        if price_change >= self.take_profit_bricks * self.brick_size:
            return True
        
        # Trend reversal (up brick after down trend)
        if index >= self.min_bricks_for_trend:
            recent_bricks = data.iloc[index-self.min_bricks_for_trend+1:index+1]
            if all(recent_bricks['direction'] == 1):
                return True
        
        return False

class RenkoMeanReversionStrategy(TradingStrategy):
    """Renko mean reversion strategy"""
    
    def __init__(self, 
                 lookback_period: int = 10,
                 oversold_threshold: int = 3,
                 overbought_threshold: int = 3):
        super().__init__("RenkoMeanReversion")
        self.lookback_period = lookback_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
    
    def should_enter_long(self, data: pd.DataFrame, index: int) -> bool:
        """Enter long on oversold condition"""
        if index < self.lookback_period:
            return False
        
        recent_bricks = data.iloc[index-self.lookback_period:index]
        down_bricks = len(recent_bricks[recent_bricks['direction'] == -1])
        
        return down_bricks >= self.oversold_threshold
    
    def should_enter_short(self, data: pd.DataFrame, index: int) -> bool:
        """Enter short on overbought condition"""
        if index < self.lookback_period:
            return False
        
        recent_bricks = data.iloc[index-self.lookback_period:index]
        up_bricks = len(recent_bricks[recent_bricks['direction'] == 1])
        
        return up_bricks >= self.overbought_threshold
    
    def should_exit_long(self, data: pd.DataFrame, index: int) -> bool:
        """Exit long on reversal"""
        if self.current_position != 1:
            return False
        
        # Exit on down brick
        return data.iloc[index]['direction'] == -1
    
    def should_exit_short(self, data: pd.DataFrame, index: int) -> bool:
        """Exit short on reversal"""
        if self.current_position != -1:
            return False
        
        # Exit on up brick
        return data.iloc[index]['direction'] == 1 