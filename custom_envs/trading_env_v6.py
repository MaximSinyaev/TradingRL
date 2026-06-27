import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple
from custom_envs.trading_env_v5 import TradingEnvV5

class TradingEnvV6(TradingEnvV5):
    """Trading environment for Phase 1.4 (Production-ready futures v6).
    
    Key Features:
    - Target Portfolio Weight Action Space: Continuous actions [-1.0, 1.0] are mapped
      to a target portfolio weight (e.g., -1.0 means 100% Short, 0.0 means Cash).
    - Auto-rebalancing: Calculates the required delta to reach the target weight.
    - Deadband (Tolerance): To avoid micro-trades due to price fluctuations, trades are
      only executed if the absolute delta exceeds `weight_tolerance`.
    - No invalid action penalties.
    """

    def __init__(
        self,
        df: pd.DataFrame | list[pd.DataFrame] | dict[str, pd.DataFrame],
        initial_deposit: float = 100_000.0,
        commission: float = 0.0005,
        leverage: float = 1.0,
        base_slippage: float = 0.0001,  
        volatility_factor: float = 0.05, 
        downside_penalty: float = 3.0,   
        max_drawdown_pct: float = 0.10,  
        t_max: Optional[int] = 1440,
        domain_randomization: bool = True,
        weight_tolerance: float = 0.05, # 5% deadband
        total_assets: Optional[int] = None,
        fixed_asset_idx: Optional[int] = None,
    ):
        # We pass continuous_action=True by definition for V6
        super().__init__(
            df=df,
            initial_deposit=initial_deposit,
            commission=commission,
            leverage=leverage,
            base_slippage=base_slippage,
            volatility_factor=volatility_factor,
            downside_penalty=downside_penalty,
            max_drawdown_pct=max_drawdown_pct,
            continuous_action=True,
            t_max=t_max,
            domain_randomization=domain_randomization,
            total_assets=total_assets,
            fixed_asset_idx=fixed_asset_idx,
            invalid_action_penalty=0.0 # Unused in V6, but set to 0 just in case
        )
        
        self.weight_tolerance = weight_tolerance

    def _get_current_weight(self, price: float, portfolio_value: float) -> float:
        """Returns the current portfolio weight in range [-1.0, 1.0]"""
        if portfolio_value <= 0:
            return 0.0
            
        long_volume = sum(v for _, v in self.positions_long)
        short_volume = sum(v for _, v in self.positions_short)
        
        net_exposure_value = (long_volume - short_volume) * price
        
        # Max leverage scaling: e.g. with leverage 1.0, weight is [-1, 1]
        weight = net_exposure_value / (portfolio_value * self.leverage)
        return np.clip(weight, -1.0, 1.0)

    def step(self, action: np.ndarray) -> Tuple:
        target_weight = np.clip(float(action[0]), -1.0, 1.0)
        
        mid_price = self._get_current_price()
        current_portfolio_value = self.get_portfolio_value(mid_price)
        current_weight = self._get_current_weight(mid_price, current_portfolio_value)
        
        delta_weight = target_weight - current_weight
        
        executed = False
        trade_size_fraction = 0.0
        
        # 1. Check if we need to rebalance (exceeds tolerance)
        if abs(delta_weight) > self.weight_tolerance:
            gk_vol = self.df['gk_volatility'].iloc[self.current_step] if 'gk_volatility' in self.df.columns else 0.0
            slippage = self.base_slippage + self.volatility_factor * gk_vol
            buy_price = mid_price * (1 + slippage)
            sell_price = mid_price * (1 - slippage)
            
            # Close existing positions if we are flipping or reducing exposure
            if current_weight > 0 and target_weight < current_weight:
                # Need to reduce long
                vol_to_close = sum(v for _, v in self.positions_long) * ((current_weight - max(target_weight, 0)) / current_weight)
                if vol_to_close > 0:
                    realized_pnl = self._close_long(vol_to_close, sell_price)
                    self.total_realized_pnl += realized_pnl
                    executed = True
            elif current_weight < 0 and target_weight > current_weight:
                # Need to reduce short
                vol_to_close = sum(v for _, v in self.positions_short) * ((abs(current_weight) - abs(min(target_weight, 0))) / abs(current_weight))
                if vol_to_close > 0:
                    realized_pnl = self._close_short(vol_to_close, buy_price)
                    self.total_realized_pnl += realized_pnl
                    executed = True

            # Re-calculate portfolio after closures to know how much cash we have
            current_portfolio_value = self.get_portfolio_value(mid_price)
            current_weight = self._get_current_weight(mid_price, current_portfolio_value)
            
            # Open new positions if needed
            if target_weight > current_weight and target_weight > 0:
                # Open Long
                amount_to_spend = (target_weight - max(current_weight, 0)) * current_portfolio_value * self.leverage
                amount_to_spend = min(amount_to_spend, self.deposit)
                if amount_to_spend > 1.0:
                    amount_after_fee = amount_to_spend * (1 - self.commission)
                    volume = amount_after_fee / buy_price
                    self.positions_long.append((buy_price, volume))
                    self._update_avg_price_long()
                    self.deposit -= amount_to_spend
                    executed = True
            elif target_weight < current_weight and target_weight < 0:
                # Open Short
                amount_to_spend = (abs(target_weight) - abs(min(current_weight, 0))) * current_portfolio_value * self.leverage
                amount_to_spend = min(amount_to_spend, self.deposit)
                if amount_to_spend > 1.0:
                    amount_after_fee = amount_to_spend * (1 - self.commission)
                    volume = amount_after_fee / sell_price
                    self.positions_short.append((sell_price, volume))
                    self._update_avg_price_short()
                    self.deposit -= amount_to_spend
                    executed = True

            if executed:
                trade_size_fraction = delta_weight

        # 2. Apply Funding Fees
        if 'fundingRate' in self.df.columns:
            funding_rate = self.df['fundingRate'].iloc[self.current_step]
            long_volume = sum(v for _, v in self.positions_long)
            short_volume = sum(v for _, v in self.positions_short)
            net_position_value = (long_volume - short_volume) * mid_price
            funding_fee = funding_rate * net_position_value
            self.deposit -= funding_fee 

        self.current_step += 1
        
        # 3. Calculate Portfolio Value & Drawdown
        new_mid_price = self._get_current_price() if self.current_step < len(self.df) else mid_price
        new_portfolio_value = self.get_portfolio_value(new_mid_price)
        
        # Записываем историю для подсчета Sortino
        if hasattr(self, 'portfolio_history'):
            self.portfolio_history.append(new_portfolio_value)
        
        self.max_portfolio_value = max(self.max_portfolio_value, new_portfolio_value)
        drawdown = (new_portfolio_value - self.max_portfolio_value) / self.max_portfolio_value
        
        # 4. Reward Calculation (Sortino Proxy)
        step_return = (new_portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
        
        if step_return > 0:
            reward = step_return
        else:
            reward = step_return * self.downside_penalty
            
        # Add turnover penalty (to avoid micro-trading)
        # Using a small penalty coefficient, e.g., 0.001
        reward -= abs(delta_weight) * 0.001
            
        self.prev_portfolio_value = new_portfolio_value
        
        # In V6, no invalid_action_penalty because all actions [-1, 1] are valid weights.
        
        # 5. Check Done & Margin Call
        done = False
        margin_call = False
        if drawdown <= -self.max_drawdown_pct:
            done = True
            margin_call = True
            reward -= 0.5 # Huge penalty for margin call
        elif self.current_step - self.start_index >= self.t_max:
            done = True
        elif self.deposit <= 0.01 and not self.positions_long and not self.positions_short:
            done = True
            
        if done:
            self._finalize_pnl()
            episode_sortino = self._calculate_sortino_from_history()
        else:
            episode_sortino = 0.0

        info = {
            "pnl": (new_portfolio_value - self.initial_deposit) / self.initial_deposit * 100,
            "deposit": self.deposit,
            "long_positions": len(self.positions_long),
            "short_positions": len(self.positions_short),
            "executed": executed,
            "margin_call": margin_call,
            "drawdown": drawdown * 100,
            "real_reward": reward,
            "episode_sortino": episode_sortino,
            "target_weight": target_weight,
            "current_weight": current_weight,
            "delta_weight": delta_weight,
            "trade_size_fraction": trade_size_fraction
        }

        return self._get_observation(), reward, done, False, info
