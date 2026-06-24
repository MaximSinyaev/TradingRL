# Trading RL

Deep Reinforcement Learning для крипто-трейдинга (BTC, ETH).

## 🚀 Quick Start

### Установка через uv (рекомендуется)

```bash
# Установка зависимостей
uv sync --all-extras

# Загрузка данных
uv run python scripts/download_data.py

# Тесты
uv run pytest tests/ -v
```

### Установка через pip

```bash
pip install -e .
python scripts/download_data.py
```

## 📦 Структура проекта

```
trading_rl/
├── agents/               # RL агенты
│   ├── dqn_agent.py              # DQN с Prioritized Experience Replay
│   ├── double_dqn_agent.py       # Double DQN
│   ├── random_agent.py            # Baseline
│   └── prioritized_replay_buffer.py  # PER buffer
├── custom_envs/         # Trading среды (Gymnasium)
│   ├── trading_env_v1.py          # V1: Базовая среда
│   ├── trading_env_v2.py          # V2: Улучшенные награды
│   ├── trading_env_v3.py          # V3: Балансированные награды
│   ├── trading_env_v4.py          # V4: Position sizing + shorts
│   └── trading_env_frame_stack.py # Frame stacking для LSTM
├── utils/               # Утилиты
│   ├── binance_loader.py          # Загрузка данных с Binance
│   ├── feature_generator.py       # Генерация фичей
│   ├── train_dqn_agent.py         # Обучение агента
│   ├── visualizer_candles.py      # Визуализация свечей
│   └── utils.py                    # Общие утилиты
├── scripts/             # Скрипты
│   ├── download_data.py           # Загрузка данных
│   ├── benchmark_agent.py          # Бенчмарк агентов
│   └── benchmark_device_batch.py  # Бенчмарк device/batch
├── tests/               # Тесты
│   ├── test_trading_env_v4_edge_cases.py
│   └── unit/test_trading_env_v1.py
└── *.ipynb              # Ноутбуки для экспериментов
```

## 🏔️ Версии окружений

| Версия | Описание |
|--------|----------|
| **V1** | Базовая среда с простыми наградами |
| **V2** | Улучшенные награды, штраф за холдинг |
| **V3** | Балансированные награды, без magic numbers |
| **V4** | Position sizing (10-100%), shorts, MultiDiscrete action space |

## 📊 Данные

Данные скачиваются с Binance и кэшируются в `binance_data_cache/`:

- **Символы**: BTCUSDT, ETHUSDT
- **Интервал**: 1 минута
- **Формат**: parquet.gz

## 🤖 Агенты

- **DQN**: Deep Q-Network с Prioritized Experience Replay
- **Double DQN**: Разделяет выбор и оценку действия (reduces overestimation)
- **Random**: Baseline для сравнения

## 🧪 Эксперименты

Ноутбуки для обучения и тестирования:

| Ноутбук | Env | Описание |
|---------|-----|----------|
| `1. base_rl_test.ipynb` | V1 | Исторические эксперименты |
| `2. base_rl_tes_v2.ipynb` | V2 | Исторические эксперименты |
| `1. base_rl_test_env_v4.ipynb` | V4 | Полный цикл обучения v4 |
| `5. test_v4_simple.ipynb` | V4 | Простой тест v4 |

## 🧪 Тесты

```bash
uv run pytest tests/ -v
```

## 📈 Текущее состояние

- ✅ TradingEnv V4 готов (position sizing, shorts)
- ✅ PrioritizedReplayBuffer реализован
- ✅ Проектная структура (uv, tests, scripts)
- 🔄 В разработке: улучшенные агенты, новые фичи

## 🛠️ Технологии

- Python 3.10+
- PyTorch
- Gymnasium
- pandas, numpy
- uv (package manager)
