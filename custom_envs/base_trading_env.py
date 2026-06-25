import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, List

class BaseTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, initial_deposit: float = 100_000.0, commission: float = 0.0005, leverage: float = 1.0):
        super().__init__()
        self.initial_deposit = initial_deposit
        self.commission = commission
        self.leverage = leverage
        self._reset_portfolio()

    def _reset_portfolio(self):
        self.deposit = self.initial_deposit
        self.positions_long: List[Tuple[float, float]] = []  # [(price, volume)]
        self.positions_short: List[Tuple[float, float]] = [] # [(price, volume)]
        self.pnl = 0.0
        self.trades = []
        self.total_realized_pnl = 0.0
        self.avg_price_long = 0.0
        self.avg_price_short = 0.0

    def _get_current_price(self) -> float:
        raise NotImplementedError

    def _get_observation(self) -> np.ndarray:
        raise NotImplementedError

    def get_portfolio_value(self, price: float) -> float:
        """Returns the total liquidation value of the portfolio."""
        value = self.deposit
        
        # Add liquidation value of longs
        if self.positions_long:
            long_volume = sum(v for _, v in self.positions_long)
            value += long_volume * price * (1 - self.commission)
            
        # Add liquidation value of shorts
        if self.positions_short:
            # We locked margin when opening shorts: margin = volume * entry_price / (1 - commission)
            margin_locked = sum(p * v / (1 - self.commission) for p, v in self.positions_short)
            short_volume = sum(v for _, v in self.positions_short)
            cost_basis = self.avg_price_short * short_volume
            cost_to_close = short_volume * price * (1 + self.commission)
            realized_pnl = cost_basis - cost_to_close
            value += margin_locked + realized_pnl
            
        return value

    def _execute_buy(self, size_fraction: float, price: float) -> bool:
        if self.positions_short:
            total_short_volume = sum(v for _, v in self.positions_short)
            if total_short_volume * price > 0:
                volume_to_close = total_short_volume * size_fraction
                realized_pnl = self._close_short(volume_to_close, price)
                self.total_realized_pnl += realized_pnl
                self.trades.append({"type": "close_short", "volume": volume_to_close, "pnl": realized_pnl})
                return True

        if not self._can_trade():
            return False

        available_for_trade = self.deposit * self.leverage
        amount_to_spend = min(available_for_trade * size_fraction, self.deposit)

        if amount_to_spend <= 0.01:
            return False

        amount_after_fee = amount_to_spend * (1 - self.commission)
        volume = amount_after_fee / price

        self.positions_long.append((price, volume))
        self._update_avg_price_long()
        self.deposit -= amount_to_spend
        self.trades.append({"type": "buy_long", "price": price, "volume": volume, "cost": amount_to_spend})
        return True

    def _execute_sell(self, size_fraction: float, price: float) -> bool:
        if self.positions_long:
            total_long_volume = sum(v for _, v in self.positions_long)
            if total_long_volume * price > 0:
                volume_to_close = total_long_volume * size_fraction
                realized_pnl = self._close_long(volume_to_close, price)
                self.total_realized_pnl += realized_pnl
                self.trades.append({"type": "close_long", "volume": volume_to_close, "pnl": realized_pnl})
                return True

        if not self._can_trade():
            return False

        available_for_trade = self.deposit * self.leverage
        amount_to_spend = min(available_for_trade * size_fraction, self.deposit)

        if amount_to_spend <= 0.01:
            return False

        amount_after_fee = amount_to_spend * (1 - self.commission)
        volume = amount_after_fee / price

        self.positions_short.append((price, volume))
        self._update_avg_price_short()
        self.deposit -= amount_to_spend
        self.trades.append({"type": "sell_short", "price": price, "volume": volume, "cost": amount_to_spend})
        return True

    def _close_long(self, volume_to_close: float, price: float) -> float:
        if not self.positions_long or volume_to_close <= 0:
            return 0.0

        cost_basis = self.avg_price_long * volume_to_close
        total_value_after_fee = (volume_to_close * price) * (1 - self.commission)
        realized_pnl = total_value_after_fee - cost_basis

        remaining = volume_to_close
        new_positions = []
        for pos_price, pos_volume in self.positions_long:
            if remaining <= 0:
                new_positions.append((pos_price, pos_volume))
                continue
            if pos_volume <= remaining:
                remaining -= pos_volume
            else:
                new_positions.append((pos_price, pos_volume - remaining))
                remaining = 0

        self.positions_long = [(p, v) for p, v in new_positions if v > 1e-9]
        self._update_avg_price_long()
        self.deposit += total_value_after_fee
        self.pnl += realized_pnl / self.initial_deposit * 100
        return realized_pnl

    def _close_short(self, volume_to_close: float, price: float) -> float:
        if not self.positions_short or volume_to_close <= 0:
            return 0.0

        # Calculate exact margin locked for the volume we are closing
        # Margin locked per unit of volume = pos_price / (1 - self.commission)
        margin_locked = 0.0
        remaining = volume_to_close
        new_positions = []
        
        for pos_price, pos_volume in self.positions_short:
            if remaining <= 0:
                new_positions.append((pos_price, pos_volume))
                continue
            
            close_vol = min(pos_volume, remaining)
            margin_locked += close_vol * pos_price / (1 - self.commission)
            
            if pos_volume <= remaining:
                remaining -= pos_volume
            else:
                new_positions.append((pos_price, pos_volume - remaining))
                remaining = 0

        self.positions_short = [(p, v) for p, v in new_positions if v > 1e-9]
        
        cost_basis = self.avg_price_short * volume_to_close
        total_value_with_fee = (volume_to_close * price) * (1 + self.commission)
        realized_pnl = cost_basis - total_value_with_fee

        self._update_avg_price_short()
        self.deposit += margin_locked + realized_pnl
        self.pnl += realized_pnl / self.initial_deposit * 100
        return realized_pnl

    def _update_avg_price_long(self):
        total_cost = sum(p * v for p, v in self.positions_long)
        total_volume = sum(v for _, v in self.positions_long)
        self.avg_price_long = total_cost / total_volume if total_volume > 0 else 0.0

    def _update_avg_price_short(self):
        total_cost = sum(p * v for p, v in self.positions_short)
        total_volume = sum(v for _, v in self.positions_short)
        self.avg_price_short = total_cost / total_volume if total_volume > 0 else 0.0

    def _unrealized_pnl_long(self, price: float) -> float:
        if not self.positions_long: return 0.0
        return (price - self.avg_price_long) * sum(v for _, v in self.positions_long)

    def _unrealized_pnl_short(self, price: float) -> float:
        if not self.positions_short: return 0.0
        return (self.avg_price_short - price) * sum(v for _, v in self.positions_short)

    def _unrealized_pnl_total(self, price: float) -> float:
        return (self._unrealized_pnl_long(price) + self._unrealized_pnl_short(price)) / self.initial_deposit

    def _can_trade(self) -> bool:
        return self.deposit > 0.01 * self.initial_deposit
