"""
TradingEnvV3 — Clean trading environment with balanced reward design.

Reward components:
1. Base reward: change in total wealth (natural signal)
2. Trade bonus: small constant for executing trades (encourages participation)
3. Realized PnL: real profit/loss on sell (scaled by initial_deposit)
4. Inactivity penalty: penalty for N steps without trades (prevents "freezing")

NO magic numbers, no unnecessary penalties.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from typing import Optional


class TradingEnvV3(gym.Env):
    """Trading environment with balanced reward design.

    Actions:
        0 - hold (no action)
        1 - buy (spend buy_fraction of deposit)
        2 - sell (sell average volume of positions)

    Observation:
        - Technical indicators (15 features)
        - Deposit info (normalized)
        - Position info (volume, avg prices)
        - Unrealized PnL
        - Action masks (can_buy, can_sell)
    """

    metadata = {"render_modes": []}

    actions_dict = {
        0: "hold",
        1: "buy",
        2: "sell"
    }

    def __init__(
        self,
        features: np.ndarray,
        real_prices: np.ndarray,
        initial_deposit: float = 100_000.0,
        buy_fraction: float = 0.1,
        commission: float = 0.0005,
        t_max: int = 1440,
        warmup_steps: int = 0,
        # Reward parameters
        trade_bonus: float = 0.01,           # Bonus за исполнение сделки
        inactivity_threshold: int = 240,      # Шагов без сделки до штрафа
        inactivity_penalty: float = -0.05,   # Штраф за бездействие
        return_pt: bool = False,
    ):
        super().__init__()

        # Валидация
        if features.shape[0] != real_prices.shape[0]:
            raise ValueError("features and real_prices must have same length")

        # Convert to torch if needed
        self.return_pt = return_pt
        if self.return_pt:
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features

        self.real_prices = real_prices
        self.initial_deposit = initial_deposit
        self.buy_fraction = buy_fraction
        self.commission = commission
        self.t_max = t_max + warmup_steps
        self.warmup_steps = warmup_steps

        # Reward parameters
        self.trade_bonus = trade_bonus
        self.inactivity_threshold = inactivity_threshold
        self.inactivity_penalty = inactivity_penalty

        # Spaces
        self.num_features = features.shape[1]
        self.action_space = spaces.Discrete(3)

        # Observation: features (15) + deposit (2) + volume (1) + prices (2) + pnl (1) + masks (2) = 23
        obs_dim = self.num_features + 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

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
        self.positions = []  # [(price, volume), ...]
        self.pnl = 0.0
        self.trades = []
        self.inactivity_steps = 0  # Steps since last trade

        return self._get_observation(), {}

    def step(self, action: int):
        reward = 0.0
        executed = False
        price = self._get_current_price()

        # Previous total wealth for base reward
        prev_wealth = self._get_total_wealth(price)

        # Execute action
        if action == 1:  # BUY
            if self._can_buy():
                amount_to_spend = min(self.initial_deposit * self.buy_fraction, self.deposit)
                amount_after_fee = amount_to_spend * (1 - self.commission)
                volume = amount_after_fee / price

                self.positions.append((price, volume))
                self.deposit -= amount_to_spend
                self.trades.append(("buy", self.current_step, amount_to_spend, volume, price))
                executed = True

        elif action == 2:  # SELL
            if self._can_sell():
                # Sell average volume
                volumes = [v for _, v in self.positions]
                avg_volume = np.mean(volumes)
                volume_to_sell = avg_volume

                # Cost basis
                cost_basis = self._consume_positions(volume_to_sell)

                # Sale value
                total_value = volume_to_sell * price
                total_value_after_fee = total_value * (1 - self.commission)

                # Realized PnL
                profit = total_value_after_fee - cost_basis
                relative_profit = profit / self.initial_deposit

                self.deposit += total_value_after_fee
                self.pnl += relative_profit * 100  # В процентах
                self.trades.append(("sell", self.current_step, cost_basis, price, volume_to_sell, profit))
                executed = True

        # ===== REWARD DESIGN =====
        self.current_step += 1

        # Skip rewards during warmup
        if self.current_step <= self.warmup_steps + self.start_index + 1:
            return self._get_observation(), 0.0, False, False, {}

        # 1. Base reward: wealth change
        curr_wealth = self._get_total_wealth(self._get_current_price())
        base_reward = (curr_wealth - prev_wealth) / self.initial_deposit

        # 2. Trade bonus (за исполнение)
        trade_reward = self.trade_bonus if executed else 0.0

        # 3. Inactivity penalty (если долго не торговали)
        if executed:
            self.inactivity_steps = 0
            inactivity_reward = 0.0
        else:
            self.inactivity_steps += 1
            if self.inactivity_steps >= self.inactivity_threshold:
                inactivity_reward = self.inactivity_penalty
            else:
                inactivity_reward = 0.0

        reward = base_reward + trade_reward + inactivity_reward

        # Check termination
        done = self._check_done()

        # Final PnL calculation
        if done:
            self._finalize_pnl()

        return self._get_observation(), reward, done, False, {}

    def _get_total_wealth(self, price: float) -> float:
        """Calculate total wealth = deposit + positions value."""
        positions_value = sum(v * price for _, v in self.positions)
        return self.deposit + positions_value

    def _check_done(self) -> bool:
        """Check if episode should end."""
        # Time limit
        if self.current_step - self.start_index >= self.t_max:
            return True
        # Bankrupt
        if self.deposit <= 0.01 and not self.positions:
            return True
        return False

    def _finalize_pnl(self):
        """Calculate final PnL including open positions."""
        if self.positions:
            price = self._get_current_price()
            total_volume = sum(v for _, v in self.positions)
            cost_basis = self._consume_positions(total_volume)
            total_value = total_volume * price * (1 - self.commission)
            profit = total_value - cost_basis
            self.pnl += profit / self.initial_deposit * 100
            self.deposit += total_value

    def _get_current_price(self) -> float:
        return self.real_prices[self.current_step]

    def _consume_positions(self, volume_to_sell: float) -> float:
        """Remove volume from positions, return cost basis."""
        remaining = volume_to_sell
        cost = 0.0
        new_positions = []

        for price, volume in self.positions:
            if remaining <= 0:
                new_positions.append((price, volume))
                continue

            if volume <= remaining:
                cost += volume * price
                remaining -= volume
            else:
                cost += remaining * price
                new_positions.append((price, volume - remaining))
                remaining = 0

        self.positions = [(p, v) for p, v in new_positions if v > 1e-9]
        return cost

    def _average_buy_price(self) -> float:
        """Volume-weighted average buy price."""
        if not self.positions:
            return 0.0
        total_cost = sum(price * volume for price, volume in self.positions)
        total_volume = sum(volume for _, volume in self.positions)
        return total_cost / total_volume if total_volume > 0 else 0.0

    def _last_buy_price(self) -> float:
        """Last buy price."""
        return self.positions[-1][0] if self.positions else 0.0

    def _unrealized_pnl(self, price: float) -> float:
        """Unrealized PnL as fraction of initial deposit."""
        if not self.positions:
            return 0.0
        avg_price = self._average_buy_price()
        volume = sum(v for _, v in self.positions)
        return ((price - avg_price) * volume) / self.initial_deposit if avg_price > 0 else 0.0

    def _can_buy(self) -> bool:
        """Can buy if have enough deposit."""
        return self.deposit > 0.01 * self.initial_deposit

    def _can_sell(self) -> bool:
        """Can sell if have positions."""
        return len(self.positions) > 0

    def get_possible_actions(self) -> list[int]:
        """Return list of valid actions."""
        actions = [0]  # hold always available
        if self._can_buy():
            actions.append(1)
        if self._can_sell():
            actions.append(2)
        return actions

    def _get_observation(self):
        """Build observation vector."""
        features = self.features[self.current_step]
        price = self._get_current_price()

        # Normalized state components
        deposit_norm = self.deposit / self.initial_deposit
        realized_deposit = 1.0 - deposit_norm
        volume = sum(v for _, v in self.positions)
        volume_norm = volume * price / self.initial_deposit
        last_buy_ratio = self._last_buy_price() / price - 1.0 if self.positions else 0.0
        avg_buy_ratio = self._average_buy_price() / price - 1.0 if self.positions else 0.0
        unrealized_pnl = self._unrealized_pnl(price)
        can_buy = float(self._can_buy())
        can_sell = float(self._can_sell())

        if self.return_pt:
            obs = torch.cat([
                features,
                torch.tensor([
                    deposit_norm,
                    realized_deposit,
                    volume_norm,
                    last_buy_ratio,
                    avg_buy_ratio,
                    unrealized_pnl,
                    can_buy,
                    can_sell
                ], dtype=torch.float32)
            ])
        else:
            obs = np.concatenate([
                features,
                [
                    deposit_norm,
                    realized_deposit,
                    volume_norm,
                    last_buy_ratio,
                    avg_buy_ratio,
                    unrealized_pnl,
                    can_buy,
                    can_sell
                ]
            ]).astype(np.float32)

        return obs
