import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnvV1(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        initial_deposit: float = 100.0,
        buy_fraction: float = 0.1,
        commission: float = 0.0005,
        max_inactivity_steps: int = 20, 
        inactivity_penalty: float = -0.1,
        action_window_size: int = 50,
        unused_capital_penalty: float = 0.001,
        
    ):
        super().__init__()

        self.features = features
        self.initial_deposit = initial_deposit
        self.buy_fraction = buy_fraction
        self.commission = commission
        self.max_inactivity_steps = max_inactivity_steps
        self.inactivity_penalty = inactivity_penalty
        self.action_window_size = action_window_size  # ⬅️ через сколько шагов без действия прерывать
        self.unused_capital_penalty = unused_capital_penalty  # ⬅️ штраф за неиспользуемый депозит

        self.num_features = features.shape[1]
        self.action_space = spaces.Discrete(3)  # hold, buy, sell

        # features + deposit + avg_buy_price_raw + avg_buy_price_mean
        obs_dim = self.num_features + 4
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()
        
    def get_possible_actions(self):
        actions = [0]  # hold всегда доступен
        if self._can_buy():
            actions.append(1)
        if self._can_sell():
            actions.append(2)
        return actions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.deposit = self.initial_deposit
        self.positions = []  # list of prices bought (raw)
        self.pnl = 0.0
        self.trades = []
        self.inactive_steps = 0

        return self._get_observation(), {}

    def step(self, action: int):
        """
        Выполняет шаг в среде на основе действия агента.
        Действия:
        0 - hold
        1 - buy
        2 - sell

        Args:
            action (int): _description_

        Returns:
            _type_: _description_
        """
        done = False
        reward = 0.0
        price = self._get_current_price()

        acted = False

        if action == 1 and self._can_buy():
            amount_to_spend = self.deposit * self.buy_fraction
            amount_after_fee = amount_to_spend * (1 - self.commission)
            volume = amount_after_fee / price
            self.positions.append(price)
            self.deposit -= amount_to_spend
            self.trades.append(("buy", self.current_step, price))
            acted = True

        elif action == 2 and self._can_sell():
            avg_price = self._average_buy_price()
            total_volume = len(self.positions)
            total_value = total_volume * price
            total_value_after_fee = total_value * (1 - self.commission)

            cost_basis = total_volume * avg_price
            profit = total_value_after_fee - cost_basis

            self.deposit += total_value_after_fee
            self.pnl += profit
            reward = profit
            self.trades.append(("sell", self.current_step, price))
            self.positions.clear()
            acted = True

        # ⬇️ Проверка на бездействие
        if not acted:
            self.inactive_steps += 1
            unrealized = self._unrealized_pnl(price)
            reward += 0.1 * unrealized
            # if self.inactive_steps >= self.max_inactivity_steps:
            #     reward += self.inactivity_penalty
        else:
            self.inactive_steps = 0  # сброс счётчика
            

        
        reward -= self.unused_capital_penalty * self.deposit

        self.current_step += 1
        if self.inactive_steps >= self.action_window_size:
            done = True
        elif self.current_step >= len(self.features) - 1:
            done = True
        return self._get_observation(), reward, done, False, {}
    
    def _get_current_price(self):
        # предполагаем, что feature[0] — это close price
        return self.features[self.current_step][0]

    def _average_buy_price(self):
        if not self.positions:
            return 0.0
        return sum(self.positions) / len(self.positions)

    def _average_buy_price_raw(self):
        return self.positions[-1] if self.positions else 0.0

    def _get_observation(self):
        features = self.features[self.current_step]
        avg_price_mean = self._average_buy_price()
        avg_price_raw = self._average_buy_price_raw()
        unrealized = self._unrealized_pnl(self._get_current_price())

        obs = np.concatenate([
            features,
            [self.deposit, avg_price_raw, avg_price_mean, unrealized]
        ])
        return obs.astype(np.float32)

    def _can_buy(self):
        return self.deposit >= self.deposit * self.buy_fraction

    def _can_sell(self):
        return len(self.positions) > 0
    
    def _unrealized_pnl(self, price: float) -> float:
        if not self.positions:
            return 0.0
        avg_price = self._average_buy_price()
        volume = len(self.positions)
        return (price - avg_price) * volume