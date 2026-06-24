# Trading RL

Deep Reinforcement Learning для крипто-трейдинга (BTC, ETH).

## 🚀 Quick Start

### Установка через uv (рекомендуется)

```bash
# Установка зависимостей
uv sync --all-extras

# Запуск скриптов
uv run python scripts/download_data.py
uv run python -m pytest tests/
```

### Установка через pip

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

## 📦 Структура проекта

```
trading_rl/
├── agents/          # RL агенты (DQN, Double DQN)
├── custom_envs/     # Trading среды (Gymnasium)
├── utils/           # Утилиты (загрузка данных, фичи, обучение)
├── scripts/         # Скрипты для загрузки данных
├── tests/           # Unit тесты
└── notebooks/       # Jupyter ноутбуки для экспериментов
```

## 📊 Данные

Данные скачиваются с Binance и кэшируются в `binance_data_cache/`:

- **Символы**: BTCUSDT, ETHUSDT
- **Интервал**: 1 минута
- **Формат**: parquet.gz

### Загрузка данных

```bash
# Тест (3 дня)
uv run python scripts/download_data_test.py

# Полная загрузка (6 месяцев)
uv run python scripts/download_data.py
```

## 🤖 Агенты

- **DQN**: Basic Deep Q-Network с Prioritized Experience Replay
- **Double DQN**: Разделяет выбор и оценку действия
- **Random**: Baseline для сравнения

## 🏋️ Обучение

Смотри ноутбуки:
- `1. base_rl_test.ipynb` — эксперименты с V1 средой
- `2. base_rl_tes_v2.ipynb` — эксперименты с V2 средой

## 🧪 Тесты

```bash
uv run pytest tests/ -v
```
