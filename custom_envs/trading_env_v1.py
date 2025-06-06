import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

class TradingEnvV1(gym.Env):
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
        initial_deposit: float = 100.0,
        buy_fraction: float = 0.1,
        commission: float = 0.0005,
        # max_inactivity_steps: int = 20,
        inactivity_penalty: float = -1,
        action_window_size: int = 240,
        unused_capital_penalty: float = 0.01,
        reward_on_trades_only: bool = False, # Выдавать награду только за сделки
        t_max: int = 1440,  # Максимальное количество шагов в эпизоде
        impossible_action_penalty: float = 0.05,  # Штраф за невозможное действие
        warmup_steps = 0, # Количество шагов, которые нужно пропустить перед началом эпизода и за которые не будет начисляться награда
        sell_scaling_factor = 5.0, # Фактор масштабирования для награды за продажу
        return_pt: bool = False, # Возвращать PyTorch тензор вместо numpy массива для вектора состояний
    ):
        """
        Инициализация среды трейдинга.

        Args:
            features (np.ndarray): Массив признаков, нормализованных для подачи агенту.
            real_prices (np.ndarray): Массив реальных цен (например, close или open), используемых для сделок и PnL.
            initial_deposit (float): Начальный депозит агента.
            buy_fraction (float): Доля депозита, которая используется при покупке.
            commission (float): Комиссия брокера за сделку (в долях от суммы).
            Deprecated: max_inactivity_steps (int): Не используется, но может применяться для штрафов за бездействие.
            inactivity_penalty (float): Штраф за бездействие.
            action_window_size (int): Максимальное количество шагов бездействия до завершения эпизода.
            unused_capital_penalty (float): Штраф за неиспользуемый капитал (депозит).
            reward_on_trades_only (bool): Если True, агент получает награду только с совершённых сделок, без штрафов и бонусов за бездействие.
            t_max (int): Максимальное количество шагов в эпизоде.
        """
        super().__init__()
        self.return_pt = return_pt  # Возвращать PyTorch тензор вместо numpy массива для вектора состояний
        

        if features.shape[0] != real_prices.shape[0]:
            raise ValueError("Количество строк в features и real_prices должно совпадать.")
        if self.return_pt:
            self.features = torch.tensor(features, dtype=torch.float32)
        else:
            self.features = features
        
        self.real_prices = real_prices  # <-- реальные цены для сделок и PnL
        self.initial_deposit = initial_deposit
        self.buy_fraction = buy_fraction
        self.commission = commission
        # self.max_inactivity_steps = max_inactivity_steps
        self.inactivity_penalty = inactivity_penalty
        self.action_window_size = action_window_size  # ⬅️ через сколько шагов без действия прерывать
        self.unused_capital_penalty = unused_capital_penalty  # ⬅️ штраф за неиспользуемый депозит
        self.reward_on_trades_only = reward_on_trades_only  # Сохранение нового аргумента
        self.t_max = t_max + warmup_steps  # Сохранение нового аргумента
        self.warmup_steps = warmup_steps  # Количество шагов, которые нужно пропустить перед началом эпизода
        self.impossible_action_penalty = impossible_action_penalty  # Штраф за невозможное действие
        self.sell_scaling_factor = sell_scaling_factor # Фактор масштабирования для награды за продажу
        
        
        self.invalid_buy_streak = 0
        self.invalid_sell_streak = 0
        

        self.num_features = features.shape[1]
        self.action_space = spaces.Discrete(3)  # hold, buy, sell

        # features + deposit + avg_buy_price_raw + avg_buy_price_mean
        obs_dim = self.num_features + 8
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
        np.random.seed(seed)

        if len(self.features) <= self.t_max:
            self.start_index = 0
            self.t_max = len(self.features) - 1
        self.start_index = np.random.randint(0, len(self.features) - self.t_max)  # Новый код
        self.current_step = self.start_index  # Новый код
        self.deposit = self.initial_deposit
        self.positions = []  # <-- [(price, volume)]
        self.pnl = 0.0
        self.trades = []
        self.inactive_steps = 0

        return self._get_observation(), {}

    def step(self, action: int):
        """
        Выполняет шаг в среде на основе действия агента.

        Действия:
            0 - hold (удержание позиции)
            1 - buy (покупка на фиксированную долю от депозита)
            2 - sell (продажа всех накопленных позиций)

        Args:
            action (int): Действие агента на текущем шаге.

        Returns:
            observation (np.ndarray): Новое наблюдение за состоянием среды.
            reward (float): Вознаграждение за совершенное действие.
            done (bool): Флаг завершения эпизода.
            truncated (bool): Флаг усечения эпизода (всегда False).
            info (dict): Служебная информация (пустой словарь).
        """
        done = False
        reward = 0.0
        price = self._get_current_price()

        acted = False

        if action == 1:
            if self._can_buy():
                amount_to_spend = min(self.initial_deposit * self.buy_fraction, self.deposit)
                amount_after_fee = amount_to_spend * (1 - self.commission)
                volume = amount_after_fee / price
                self.positions.append((price, volume))  # <-- теперь сохраняем объём
                self.deposit -= amount_to_spend
                self.trades.append(("buy", self.current_step, amount_to_spend, volume, price))
                acted = True
                self.invalid_buy_streak = 0
            elif not self.reward_on_trades_only:
                # штраф за попытку купить, когда нет средств
                self.invalid_buy_streak += 1
                reward -= self.impossible_action_penalty * self.invalid_buy_streak  # штраф за попытку купить, когда нет средств

        elif action == 2:
            if self._can_sell():
                self.invalid_sell_streak = 0
                
                volumes = [v for _, v in self.positions]
                avg_volume = np.mean(volumes)  # <-- средний объём
                volume_to_sell = avg_volume

                cost_basis = self._consume_positions(volume_to_sell)  # <-- списание из self.positions
                total_value = volume_to_sell * price
                total_value_after_fee = total_value * (1 - self.commission)

                profit = total_value_after_fee - cost_basis
                relative_profit = profit / self.initial_deposit * 100
                # Old sell scaling
                if relative_profit > 0:
                    # reward += 2.5 * relative_profit
                    reward += np.clip(relative_profit, -self.sell_scaling_factor, self.sell_scaling_factor) * 2
                else:
                    reward += np.clip(relative_profit, -self.sell_scaling_factor, self.sell_scaling_factor) * 1.
                
                self.deposit += total_value_after_fee
                self.pnl += relative_profit
                self.trades.append(("sell", self.current_step, cost_basis, price, volume_to_sell, profit, relative_profit))
                acted = True
            elif not self.reward_on_trades_only:
                # штраф за попытку продать, когда нет позиций
                self.invalid_sell_streak += 1
                reward -= self.impossible_action_penalty * self.invalid_sell_streak  # штраф за попытку продать, когда нет позиций
                
                
        if self.current_step < self.warmup_steps:
            return self._get_observation(), reward, done, False, {}

        # ⬇️ Проверка на бездействие
        if not acted:
            self.inactive_steps += 1
            # unrealized = self._unrealized_pnl(price)
            # if not self.reward_on_trades_only:  # Новая проверка
            #     reward += 0.1 * unrealized
                # if self.inactive_steps >= self.max_inactivity_steps:
                #     reward += self.inactivity_penalty
        else:
            self.inactive_steps = 0  # сброс счётчика
            if self.positions and not self.reward_on_trades_only:
                reward += 0.2 * self._unrealized_pnl(price)
                
            
        if not self.reward_on_trades_only:
            reward -= self.unused_capital_penalty * (self.deposit / self.initial_deposit)  # штраф за неиспользуемый капитал

        self.current_step += 1
        if not self.reward_on_trades_only:  # Новая проверка
            if self.inactive_steps >= self.action_window_size:
                done = True
                reward += self.inactivity_penalty
                
        if self.current_step - self.start_index >= self.t_max:  # Изменённая проверка
            done = True
        elif self.deposit <= 0 and not self.positions:
            done = True
        # Если эпизод завершён, считаем PnL по всем позициям по текущей цене
        if done and self.positions:
            current_price = self._get_current_price()
            total_volume = sum(v for _, v in self.positions)
            cost_basis = self._consume_positions(total_volume)
            total_value = total_volume * current_price * (1 - self.commission)
            profit = total_value - cost_basis
            self.deposit += total_value
            self.pnl += profit / self.initial_deposit * 100
        return self._get_observation(), reward, done, False, {}
    
    def _get_current_price(self):
        # предполагаем, что feature[3] — это close price
        return self.real_prices[self.current_step]
    
    def _consume_positions(self, volume_to_sell: float) -> float:
        # ⬇️ Новая функция: списывает объём из self.positions и возвращает его стоимость (cost basis)
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

    def _average_buy_price(self):
        if not self.positions:
            return 0.0
        total_cost = sum(price * volume for price, volume in self.positions)
        total_volume = sum(volume for _, volume in self.positions)
        return total_cost / total_volume if total_volume > 0 else 0.0

    def _average_buy_price_raw(self):
        return self.positions[-1][0] if self.positions else 0.0

    def _get_observation(self):
        features = self.features[self.current_step]
        current_price = self._get_current_price()
        deposit_norm = self.deposit / self.initial_deposit
        realized_deposit = 1 - deposit_norm
        volume = sum(v for _, v in self.positions)
        volume_norm = volume * current_price / self.initial_deposit
        avg_price_mean = self._average_buy_price()
        avg_price_raw = self._average_buy_price_raw()
        unrealized_pnl = self._unrealized_pnl(self._get_current_price())
        can_buy = float(self._can_buy())
        can_sell = float(self._can_sell())

        if self.return_pt:
            obs = torch.cat([
                features,
                torch.tensor([
                    deposit_norm,
                    realized_deposit,
                    volume_norm,
                    avg_price_raw / current_price - 1.0,
                    avg_price_mean / current_price - 1.0,
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
                    avg_price_raw / current_price - 1.0,
                    avg_price_mean / current_price - 1.0,
                    unrealized_pnl,
                    can_buy,
                    can_sell
                ]
            ]).astype(np.float32)
        return obs

    def _can_buy(self):
        return self.deposit > 0.01 * self.initial_deposit

    def _can_sell(self):
        return len(self.positions) > 0
    
    def _unrealized_pnl(self, price: float) -> float:
        if not self.positions:
            return 0.0
        avg_price = self._average_buy_price()
        volume = sum(v for _, v in self.positions)
        return ((price - avg_price) * volume) / self.initial_deposit if avg_price > 0 else 0.0