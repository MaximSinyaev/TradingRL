# Trading RL — Project Overview

## Цель проекта

Deep Reinforcement Learning для крипто-трейдинга. Обучение агентов торговле BTC/ETH на minutely данных с Binance.

## Архитектура

```
Data (Binance) → Feature Engineering → TradingEnv (Gymnasium) → RL Agent → Actions → PnL
```

### Компоненты

#### 1. Данные (`utils/binance_loader.py`)
- Источник: Binance REST API
- Символы: BTCUSDT, ETHUSDT
- Интервал: 1m
- Кэш: `binance_data_cache/*.parquet.gz`

#### 2. Фичи (`utils/feature_generator.py`)
Технические индикаторы для каждого timestamp:
- SMA/EMA (5, 20, 50)
- RSI (14)
- MACD
- Bollinger Bands
- Volume indicators

#### 3. Окружения (`custom_envs/`)

| Версия | Файл | Ключевые особенности |
|--------|------|----------------------|
| V1 | `trading_env_v1.py` | Базовое: 3 действия (hold, buy, sell), простые награды |
| V2 | `trading_env_v2.py` | Улучшенные награды, штраф за холдинг |
| V3 | `trading_env_v3.py` | Балансированные награды (wealth change + trade bonus + inactivity penalty) |
| V4 | `trading_env_v4.py` | **Актуальная версия**: MultiDiscrete action space [action, size], shorts, position sizing (10-100%) |

**TradingEnvV4 details:**
- Action space: `MultiDiscrete([3, 10])` → [action∈{0,1,2}, size∈{0..9}]
  - action: HOLD(0), BUY(1), SELL(2)
  - size: 10%, 20%, ..., 100% of available_balance
- BUY: может открыть long ИЛИ закрыть short
- SELL: может открыть short ИЛИ закрыть long
- Reward: `realized_pnl / initial_deposit` (чистый сигнал)
- Risk penalty для позиций >50% balance

#### 4. Агенты (`agents/`)

| Агент | Файл | Описание |
|-------|------|----------|
| DQN | `dqn_agent.py` | Deep Q-Network с PER |
| DoubleDQN | `double_dqn_agent.py` | DDQN (reduces overestimation) |
| Random | `random_agent.py` | Baseline |

**Buffers:**
- `prioritized_replay_buffer.py` — Prioritized Experience Replay (TD-error sampling)

#### 5. Обучение (`utils/train_dqn_agent.py`)
- Custom training loop (не стандартный Gymnasium API)
- Поддержка device/batch конфигурации
- Валидация на отложенной выборке

### Структура обучения

```
1. Загрузка данных (binance_loader)
2. Генерация фичей (feature_generator)
3. Создание окружения (TradingEnvV4)
4. Инициализация агента (DQN/DoubleDQN + PER)
5. Training loop:
   - Select action (ε-greedy)
   - Environment step
   - Store transition (PER)
   - Update network (sampled from PER)
   - Decay ε
6. Валидация
7. Визуализация результатов
```

## Ноутбуки (эксперименты)

| Ноутбук | Env | Статус |
|---------|-----|--------|
| `1. base_rl_test.ipynb` | V1 | Исторический |
| `2. base_rl_tes_v2.ipynb` | V2 | Исторический |
| `1. base_rl_test_env_v4.ipynb` | V4 | ✅ Актуальный (полный) |
| `5. test_v4_simple.ipynb` | V4 | ✅ Актуальный (простой) |

## Текущее состояние

**Что готово:**
- ✅ TradingEnv V4 (position sizing, shorts)
- ✅ PrioritizedReplayBuffer
- ✅ DQN + DoubleDQN агенты
- ✅ Infrastructure: uv, pyproject.toml, tests, scripts

**Что в планах:**
- 🔄 Улучшение агентов (target network, proper decay)
- 🔄 New features (sentiment, on-chain data)
- 🔄 Production-ready inference

## Конфигурация

- Python: 3.10+ (см. `.python-version`)
- Package manager: uv (см. `pyproject.toml`, `uv.lock`)
- Tests: pytest (`tests/`)
- Scripts: `scripts/`

## Для агентов (AI context)

Если вы агент и работаете с этим проектом:

1. **Актуальное окружение**: `custom_envs/trading_env_v4.py`
2. **Актуальные агенты**: `agents/dqn_agent.py`, `agents/double_dqn_agent.py`
3. **Эксперименты**: ноутбуки с `v4` в названии
4. **Тесты**: `pytest tests/` перед изменениями
5. **Conventions**: Follow patterns from existing agents/envs

## Ключевые паттерны кода

### Создание окружения
```python
from custom_envs.trading_env_v4 import TradingEnvV4
env = TradingEnvV4(features=X, real_prices=prices, commission=0.001, initial_deposit=10000)
```

### Создание агента
```python
from agents.dqn_agent import DQNAgent
agent = DQNAgent(state_dim=15, action_dim=30, device='cuda')
```

### Обучение
```python
from utils.train_dqn_agent import train_agent
metrics = train_agent(env, agent, train_episodes=1000)
```
