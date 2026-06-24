"""
TradingEnvV4 — Simple, discrete environment with position sizing and shorts.

Key features:
1. MultiDiscrete action space: [action, size]
   - action: HOLD(0), BUY(1), SELL(2)
   - size: 10%, 20%, 30%, ..., 100% of available_balance (10 levels)

2. BUY action: can open/increase long OR close/reduce short
3. SELL action: can open/increase short OR close/reduce long
4. Correct PnL calculation with weighted average price
5. Reward = realized PnL / initial_deposit (clean signal)
6. Risk penalty for large positions (>50% of balance)
7. Leverage support (structure ready, currently 1x)

This is the FOUNDATION for serious RL trading.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, List


class TradingEnvV4(gym.Env):
    """Trading environment with position sizing and shorts.

    Action space: MultiDiscrete([3, 10])
        - First dim: HOLD(0), BUY(1), SELL(2)
        - Second dim: 10%, 20%, ..., 100% of available_balance

    BUY action logic (one-way hedging):
        - If have short positions: close/reduce them first
        - If no shorts: open/increase long position

    SELL action logic (one-way hedging):
        - If have long positions: close/reduce them first
        - If no longs: open/increase short position

    Observation:
        - Technical indicators (from features)
        - Deposit state (normalized)
        - Position info (long volume, short volume, avg prices)
        - Unrealized PnL (net: long - short)
        - Action masks
    """

    metadata = {"render_modes": []}

    # Action definitions
    ACTIONS = {
        0: "HOLD",
        1: "BUY",   # Open/increase long OR close/reduce short
        2: "SELL"   # Open/increase short OR close/reduce long
    }

    # Size levels (fraction of available_balance)
    # 10 levels: 10%, 20%, ..., 100%
    SIZE_LEVELS = [i / 10 for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]

    def __init__(
        self,
        features: np.ndarray,
        real_prices: np.ndarray,
        initial_deposit: float = 100_000.0,
        commission: float = 0.0005,
        leverage: float = 1.0,  # Currently 1x, structure ready for higher
        t_max: int = 1440,
        warmup_steps: int = 0,
        invalid_action_penalty: float = -0.00005,  # Small penalty for impossible actions
    ):
        super().__init__()

        # Validation
        if features.shape[0] != real_prices.shape[0]:
            raise ValueError("features and real_prices must have same length")

        if not 1.0 <= leverage <= 10.0:
            raise ValueError("leverage must be between 1.0 and 10.0")

        self.features = features
        self.real_prices = real_prices
        self.initial_deposit = initial_deposit
        self.commission = commission
        self.leverage = leverage
        self.t_max = t_max + warmup_steps
        self.warmup_steps = warmup_steps
        self.invalid_action_penalty = invalid_action_penalty

        # Spaces
        self.num_features = features.shape[1]
        # Action: [3 actions] x [10 size levels (10%-100%)]
        self.action_space = spaces.MultiDiscrete([3, 10])

        # Observation: features + deposit + long_pos + short_pos + pnl + masks
        obs_dim = self.num_features + 10  # Expanded for long + short
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State (will be set in reset)
        self.current_step = 0
        self.start_index = 0
        self.deposit = initial_deposit
        self.positions_long: List[Tuple[float, float]] = []  # [(price, volume), ...]
        self.positions_short: List[Tuple[float, float]] = []  # [(price, volume), ...]
        self.pnl = 0.0
        self.trades = []
        self.total_realized_pnl = 0.0

        # Track average prices for PnL calculation
        self.avg_price_long = 0.0
        self.avg_price_short = 0.0

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Random start position
        if len(self.features) <= self.t_max:
            self.start_index = 0
            self.t_max = len(self.features) - 1
        else:
            self.start_index = np.random.randint(0, len(self.features) - self.t_max)

        self.current_step = self.start_index
        self.deposit = self.initial_deposit
        self.positions_long = []
        self.positions_short = []
        self.pnl = 0.0
        self.trades = []
        self.total_realized_pnl = 0.0
        self.avg_price_long = 0.0
        self.avg_price_short = 0.0

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple:
        """Execute one step.

        Args:
            action: [action_type, size_level]
                - action_type: 0=HOLD, 1=BUY, 2=SELL
                - size_level: 0-9 → 10%, 20%, ..., 100%

        Returns:
            observation, reward, done, truncated, info
        """
        action_type = int(action[0])
        size_level = int(action[1])
        size_fraction = self.SIZE_LEVELS[size_level]

        reward = 0.0
        executed = False
        price = self._get_current_price()

        # ===== EXECUTE ACTION =====
        if action_type == 1:  # BUY
            if size_fraction > 0:
                executed = self._execute_buy(size_fraction, price)

        elif action_type == 2:  # SELL
            if size_fraction > 0:
                executed = self._execute_sell(size_fraction, price)

        # Track if agent tried to trade but failed
        attempted_trade = (action_type in [1, 2]) and (size_fraction > 0)

        # Update step
        self.current_step += 1

        # Build info (always return state)
        info = {
            "pnl": self.pnl,
            "deposit": self.deposit,
            "long_positions": len(self.positions_long),
            "short_positions": len(self.positions_short),
            "executed": executed
        }

        # Skip rewards during warmup
        if self.current_step <= self.warmup_steps + self.start_index + 1:
            info["warmup"] = True
            return self._get_observation(), 0.0, False, False, info

        # ===== REWARD: Realized PnL / Initial Deposit =====
        reward = self.total_realized_pnl / self.initial_deposit

        # Risk penalty: discourage large position sizes
        if executed and size_fraction > 0.5:
            risk_penalty = -0.001 * ((size_fraction - 0.5) / 0.5)
            reward += risk_penalty

        # Invalid action penalty: discourage trying to trade when can't
        if attempted_trade and not executed:
            reward += self.invalid_action_penalty

        # Reset for next step
        self.total_realized_pnl = 0.0

        # Check termination
        done = self._check_done()

        # Final PnL calculation
        if done:
            self._finalize_pnl()

        info = {
            "pnl": self.pnl,
            "deposit": self.deposit,
            "long_positions": len(self.positions_long),
            "short_positions": len(self.positions_short),
            "executed": executed
        }

        return self._get_observation(), reward, done, False, info

    def _execute_buy(self, size_fraction: float, price: float) -> bool:
        """Execute BUY action: close short OR open long.

        For closing: size_fraction is fraction of POSITION to close (0.1-1.0)
        For opening: size_fraction is fraction of deposit to use (0.1-1.0)

        Returns True if any trade was executed.
        """
        # First: close/reduce short positions (uses position size, not deposit)
        if self.positions_short:
            total_short_volume = sum(v for _, v in self.positions_short)
            short_value = total_short_volume * price

            if short_value > 0:
                # Close size_fraction of the short position (0.1-1.0)
                volume_to_close = total_short_volume * size_fraction
                realized_pnl = self._close_short(volume_to_close, price)
                self.total_realized_pnl += realized_pnl
                self.trades.append({
                    "type": "close_short",
                    "step": self.current_step,
                    "volume": volume_to_close,
                    "pnl": realized_pnl
                })
                return True

        # Second: open/increase long position (uses deposit, requires funds)
        if not self._can_trade():
            return False

        available_for_trade = self.deposit * self.leverage
        amount_to_spend = available_for_trade * size_fraction
        amount_to_spend = min(amount_to_spend, self.deposit)

        if amount_to_spend <= 0.01:
            return False

        # Second: open/increase long position
        amount_after_fee = amount_to_spend * (1 - self.commission)
        volume = amount_after_fee / price

        self.positions_long.append((price, volume))
        self._update_avg_price_long()
        self.deposit -= amount_to_spend

        self.trades.append({
            "type": "buy_long",
            "step": self.current_step,
            "price": price,
            "volume": volume,
            "cost": amount_to_spend
        })

        return True

    def _execute_sell(self, size_fraction: float, price: float) -> bool:
        """Execute SELL action: close long OR open short.

        For closing: size_fraction is fraction of POSITION to close (0.1-1.0)
        For opening: size_fraction is fraction of deposit to use (0.1-1.0)

        Returns True if any trade was executed.
        """
        # First: close/reduce long positions (uses position size, not deposit)
        if self.positions_long:
            total_long_volume = sum(v for _, v in self.positions_long)
            long_value = total_long_volume * price

            if long_value > 0:
                # Close size_fraction of the long position (0.1-1.0)
                volume_to_close = total_long_volume * size_fraction
                realized_pnl = self._close_long(volume_to_close, price)
                self.total_realized_pnl += realized_pnl
                self.trades.append({
                    "type": "close_long",
                    "step": self.current_step,
                    "volume": volume_to_close,
                    "pnl": realized_pnl
                })
                return True

        # Second: open/increase short position (uses deposit, requires margin)
        if not self._can_trade():
            return False

        available_for_trade = self.deposit * self.leverage
        amount_to_spend = available_for_trade * size_fraction
        amount_to_spend = min(amount_to_spend, self.deposit)

        if amount_to_spend <= 0.01:
            return False

        # Second: open/increase short position
        amount_after_fee = amount_to_spend * (1 - self.commission)
        volume = amount_after_fee / price

        self.positions_short.append((price, volume))
        self._update_avg_price_short()
        self.deposit -= amount_to_spend  # Margin locked

        self.trades.append({
            "type": "sell_short",
            "step": self.current_step,
            "price": price,
            "volume": volume,
            "cost": amount_to_spend
        })

        return True

    def _close_long(self, volume_to_close: float, price: float) -> float:
        """Close long position, return realized PnL."""
        if not self.positions_long or volume_to_close <= 0:
            return 0.0

        avg_price = self.avg_price_long
        cost_basis = avg_price * volume_to_close

        total_value = volume_to_close * price
        total_value_after_fee = total_value * (1 - self.commission)

        realized_pnl = total_value_after_fee - cost_basis

        # Consume positions (FIFO)
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

        # Update PnL for display
        self.pnl += realized_pnl / self.initial_deposit * 100

        return realized_pnl

    def _close_short(self, volume_to_close: float, price: float) -> float:
        """Close short position, return realized PnL.

        Short PnL = (entry_price - exit_price) * volume
        Profit when price goes DOWN.
        """
        if not self.positions_short or volume_to_close <= 0:
            return 0.0

        avg_price = self.avg_price_short
        # For short: we sold at avg_price, buying back at price
        cost_basis = avg_price * volume_to_close  # What we got when opening

        total_value = volume_to_close * price  # Cost to close
        total_value_with_fee = total_value * (1 + self.commission)

        # Realized PnL = entry_value - exit_cost
        realized_pnl = cost_basis - total_value_with_fee

        # Consume positions (FIFO)
        remaining = volume_to_close
        new_positions = []

        for pos_price, pos_volume in self.positions_short:
            if remaining <= 0:
                new_positions.append((pos_price, pos_volume))
                continue

            if pos_volume <= remaining:
                remaining -= pos_volume
            else:
                new_positions.append((pos_price, pos_volume - remaining))
                remaining = 0

        self.positions_short = [(p, v) for p, v in new_positions if v > 1e-9]
        self._update_avg_price_short()

        # When closing short: return margin + profit/loss
        self.deposit += cost_basis - realized_pnl  # margin + PnL

        # Update PnL for display
        self.pnl += realized_pnl / self.initial_deposit * 100

        return realized_pnl

    def _update_avg_price_long(self):
        """Update weighted average price for long positions."""
        if not self.positions_long:
            self.avg_price_long = 0.0
            return

        total_cost = sum(p * v for p, v in self.positions_long)
        total_volume = sum(v for _, v in self.positions_long)
        self.avg_price_long = total_cost / total_volume if total_volume > 0 else 0.0

    def _update_avg_price_short(self):
        """Update weighted average price for short positions."""
        if not self.positions_short:
            self.avg_price_short = 0.0
            return

        total_cost = sum(p * v for p, v in self.positions_short)
        total_volume = sum(v for _, v in self.positions_short)
        self.avg_price_short = total_cost / total_volume if total_volume > 0 else 0.0

    def _unrealized_pnl_long(self, price: float) -> float:
        """Unrealized PnL for long positions."""
        if not self.positions_long:
            return 0.0

        total_volume = sum(v for _, v in self.positions_long)
        return (price - self.avg_price_long) * total_volume

    def _unrealized_pnl_short(self, price: float) -> float:
        """Unrealized PnL for short positions.

        Short profits when price goes DOWN.
        """
        if not self.positions_short:
            return 0.0

        total_volume = sum(v for _, v in self.positions_short)
        return (self.avg_price_short - price) * total_volume

    def _unrealized_pnl_total(self, price: float) -> float:
        """Total unrealized PnL as fraction of initial deposit."""
        long_pnl = self._unrealized_pnl_long(price)
        short_pnl = self._unrealized_pnl_short(price)
        return (long_pnl + short_pnl) / self.initial_deposit

    def _check_done(self) -> bool:
        """Check if episode should end."""
        if self.current_step - self.start_index >= self.t_max:
            return True
        if self.deposit <= 0.01 and not self.positions_long and not self.positions_short:
            return True
        return False

    def _finalize_pnl(self):
        """Calculate final PnL including open positions."""
        price = self._get_current_price()

        # Close all longs (already updates self.pnl)
        if self.positions_long:
            total_volume = sum(v for _, v in self.positions_long)
            self._close_long(total_volume, price)

        # Close all shorts (already updates self.pnl)
        if self.positions_short:
            total_volume = sum(v for _, v in self.positions_short)
            self._close_short(total_volume, price)

    def _get_current_price(self) -> float:
        """Get current price."""
        return self.real_prices[self.current_step]

    def _can_trade(self) -> bool:
        """Check if can trade."""
        return self.deposit > 0.01 * self.initial_deposit

    def get_possible_actions(self) -> list[list]:
        """Return list of valid [action, size] combinations.

        Logic:
        - HOLD always available
        - If deposit is high: can open new positions (BUY/SELL)
        - If deposit is low but have positions: can only CLOSE them
          - If have longs: can SELL to close
          - If have shorts: can BUY to close
        """
        actions = [[0, 0]]  # HOLD always available

        can_afford_new = self._can_trade()

        if can_afford_new:
            # High deposit: can do anything
            for size_idx in range(10):
                actions.append([1, size_idx])  # BUY (open long OR close short)
                actions.append([2, size_idx])  # SELL (open short OR close long)
        else:
            # Low deposit: only allow closing existing positions
            if self.positions_long:
                # Can close longs with SELL
                for size_idx in range(10):
                    actions.append([2, size_idx])  # SELL to close long
            if self.positions_short:
                # Can close shorts with BUY
                for size_idx in range(10):
                    actions.append([1, size_idx])  # BUY to close short

        return actions

    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        features = self.features[self.current_step]
        price = self._get_current_price()

        # Normalized state
        deposit_norm = self.deposit / self.initial_deposit

        # Long position info
        long_volume = sum(v for _, v in self.positions_long)
        long_volume_norm = long_volume * price / (self.initial_deposit * self.leverage)
        long_avg_ratio = (self.avg_price_long / price - 1.0) if self.avg_price_long > 0 else 0.0

        # Short position info
        short_volume = sum(v for _, v in self.positions_short)
        short_volume_norm = short_volume * price / (self.initial_deposit * self.leverage)
        short_avg_ratio = (self.avg_price_short / price - 1.0) if self.avg_price_short > 0 else 0.0

        # Unrealized PnL
        unrealized_pnl = self._unrealized_pnl_total(price)

        # Can trade
        can_trade = float(self._can_trade())

        obs = np.concatenate([
            features,
            [
                deposit_norm,
                long_volume_norm,
                long_avg_ratio,
                short_volume_norm,
                short_avg_ratio,
                unrealized_pnl,
                can_trade,
                self.leverage - 1.0,  # Leverage indicator
                0.0,  # Reserved
                0.0,  # Reserved
            ]
        ]).astype(np.float32)

        return obs


if __name__ == "__main__":
    # Quick test
    import numpy as np

    # Dummy data
    features = np.random.randn(1000, 15).astype(np.float32)
    prices = 50000 + np.random.randn(1000).cumsum() * 10

    env = TradingEnvV4(
        features=features,
        real_prices=prices,
        t_max=500
    )

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space.shape}")

    # Test episode
    obs, _ = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: action={action}, reward={reward:.6f}")

        if done:
            break

    print(f"Final PnL: {env.pnl:.2f}%")
