import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple
from custom_envs.base_trading_env import BaseTradingEnv

class TradingEnvV4(BaseTradingEnv):
    """Trading environment with position sizing and shorts.

    Action space: MultiDiscrete([3, 10])
        - First dim: HOLD(0), BUY(1), SELL(2)
        - Second dim: 10%, 20%, ..., 100% of available_balance
    """

    ACTIONS = {
        0: "HOLD",
        1: "BUY",   # Open/increase long OR close/reduce short
        2: "SELL"   # Open/increase short OR close/reduce long
    }

    SIZE_LEVELS = [i / 10 for i in range(1, 11)]

    def __init__(
        self,
        features: np.ndarray,
        real_prices: np.ndarray,
        initial_deposit: float = 100_000.0,
        commission: float = 0.0005,
        leverage: float = 1.0,
        t_max: int = 1440,
        warmup_steps: int = 0,
        invalid_action_penalty: float = -0.00005,
    ):
        super().__init__(initial_deposit=initial_deposit, commission=commission, leverage=leverage)

        if features.shape[0] != real_prices.shape[0]:
            raise ValueError("features and real_prices must have same length")
        if not 1.0 <= leverage <= 10.0:
            raise ValueError("leverage must be between 1.0 and 10.0")

        self.features = features
        self.real_prices = real_prices
        self.t_max = t_max + warmup_steps
        self.warmup_steps = warmup_steps
        self.invalid_action_penalty = invalid_action_penalty

        self.num_features = features.shape[1]
        self.action_space = spaces.MultiDiscrete([3, 10])

        obs_dim = self.num_features + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.start_index = 0
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        if len(self.features) <= self.t_max:
            self.start_index = 0
            self.t_max = len(self.features) - 1
        else:
            self.start_index = np.random.randint(0, len(self.features) - self.t_max)

        self.current_step = self.start_index
        self._reset_portfolio()

        return self._get_observation(), {}

    def _get_current_price(self) -> float:
        return self.real_prices[self.current_step]

    def _check_done(self) -> bool:
        if self.current_step - self.start_index >= self.t_max:
            return True
        if self.deposit <= 0.01 and not self.positions_long and not self.positions_short:
            return True
        return False

    def _finalize_pnl(self):
        price = self._get_current_price()
        if self.positions_long:
            self._close_long(sum(v for _, v in self.positions_long), price)
        if self.positions_short:
            self._close_short(sum(v for _, v in self.positions_short), price)

    def step(self, action: np.ndarray) -> Tuple:
        action_type = int(action[0])
        size_level = int(action[1])
        size_fraction = self.SIZE_LEVELS[size_level]

        executed = False
        price = self._get_current_price()

        if action_type == 1 and size_fraction > 0:
            executed = self._execute_buy(size_fraction, price)
        elif action_type == 2 and size_fraction > 0:
            executed = self._execute_sell(size_fraction, price)

        attempted_trade = (action_type in [1, 2]) and (size_fraction > 0)
        self.current_step += 1

        info = {
            "pnl": self.pnl,
            "deposit": self.deposit,
            "long_positions": len(self.positions_long),
            "short_positions": len(self.positions_short),
            "executed": executed
        }

        if self.current_step <= self.warmup_steps + self.start_index + 1:
            info["warmup"] = True
            return self._get_observation(), 0.0, False, False, info

        reward = self.total_realized_pnl / self.initial_deposit

        if executed and size_fraction > 0.5:
            reward += -0.001 * ((size_fraction - 0.5) / 0.5)

        if attempted_trade and not executed:
            reward += self.invalid_action_penalty

        self.total_realized_pnl = 0.0

        done = self._check_done()
        if done:
            self._finalize_pnl()

        info.update({
            "pnl": self.pnl,
            "deposit": self.deposit,
            "long_positions": len(self.positions_long),
            "short_positions": len(self.positions_short),
            "executed": executed
        })

        return self._get_observation(), reward, done, False, info

    def get_possible_actions(self) -> list[list]:
        actions = [[0, 0]]
        if self._can_trade():
            for size_idx in range(10):
                actions.append([1, size_idx])
                actions.append([2, size_idx])
        else:
            if self.positions_long:
                for size_idx in range(10):
                    actions.append([2, size_idx])
            if self.positions_short:
                for size_idx in range(10):
                    actions.append([1, size_idx])
        return actions

    def _get_observation(self) -> np.ndarray:
        features = self.features[self.current_step]
        price = self._get_current_price()
        deposit_norm = self.deposit / self.initial_deposit
        
        long_volume = sum(v for _, v in self.positions_long)
        long_volume_norm = long_volume * price / (self.initial_deposit * self.leverage)
        long_avg_ratio = (self.avg_price_long / price - 1.0) if self.avg_price_long > 0 else 0.0

        short_volume = sum(v for _, v in self.positions_short)
        short_volume_norm = short_volume * price / (self.initial_deposit * self.leverage)
        short_avg_ratio = (self.avg_price_short / price - 1.0) if self.avg_price_short > 0 else 0.0

        unrealized_pnl = self._unrealized_pnl_total(price)
        can_trade = float(self._can_trade())

        return np.concatenate([
            features,
            [
                deposit_norm,
                long_volume_norm,
                long_avg_ratio,
                short_volume_norm,
                short_avg_ratio,
                unrealized_pnl,
                can_trade,
                self.leverage - 1.0,
                0.0,
                0.0,
            ]
        ]).astype(np.float32)

if __name__ == "__main__":
    features = np.random.randn(1000, 15).astype(np.float32)
    prices = 50000 + np.random.randn(1000).cumsum() * 10
    env = TradingEnvV4(features=features, real_prices=prices, t_max=500)
    obs, _ = env.reset()
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        if done: break
