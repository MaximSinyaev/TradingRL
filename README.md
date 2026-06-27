# Trading RL

Deep Reinforcement Learning исследовательская платформа для крипто-трейдинга (бессрочные фьючерсы BTC, ETH, BNB).

Проект использует SOTA подходы к алгоритмическому трейдингу с помощью RL, включая Purged Walk-Forward Cross Validation, детектирование режимов рынка через HMM и Байесовскую оптимизацию гиперпараметров.

## 🚀 Quick Start

### Установка (используется uv)

```bash
# Установка зависимостей
uv sync --all-extras
```

### 1. Подготовка данных и HMM
Загрузка исторических данных, генерация признаков (Funding Rate, Open Interest, Fractional Diff) и обучение HMM-модели для детекции режимов рынка (Bull, Bear, Flat):
```bash
uv run python scripts/train_hmm.py
```

### 2. Оптимизация Гиперпараметров (HPO)
Перед финальным обучением **настоятельно рекомендуется** найти оптимальные гиперпараметры с помощью Optuna (Tree-structured Parzen Estimator). Скрипт перебирает архитектуру (CNN, GRU, Transformer), LR, entropy coefficient, n_steps и batch_size, используя Median Pruner для ранней остановки неудачных агентов:
```bash
uv run python scripts/tune_ppo.py --n_trials 50
```
*Рекомендуется запускать на отдельной машине/сервере на длительное время.*

### 3. Тренировка Агента
После нахождения лучших параметров, запускаем полное обучение агента (PPO). Агент обучается на непрерывных отрезках данных, из которых вырезаны (purged) валидационные куски + эмбарго (чтобы избежать заглядывания в будущее):
```bash
uv run python scripts/train_ppo_purged.py --exp_name "ppo_v6_final" --extractor transformer
```
Мониторинг процесса обучения через TensorBoard:
```bash
uv run tensorboard --logdir ./tensorboard_logs/
```

## 📦 Ключевые компоненты

* **Среда (TradingEnvV6):** Непрерывное пространство действий `[-1.0, 1.0]`, интерпретируемое как *Target Portfolio Weight*. Среда учитывает комиссии (0.05%), динамическое проскальзывание (на базе Garman-Klass Volatility) и штрафует за излишний оборот (Turnover Penalty).
* **Модели (Extractors):** `GatedTransformerExtractor`, `GatedGruExtractor` и `GatedCnnExtractor`. Используют Feature-wise Linear Modulation (FiLM) для кондиционирования слоев сети на текущий режим рынка (вероятности HMM).
* **Алгоритм:** Stable Baselines 3 `PPO` с кастомными коллбеками для Out-Of-Sample (OOS) валидации во время обучения.

## 🛠️ Технологии

- Python 3.12+
- PyTorch (с поддержкой MPS/CUDA)
- Stable Baselines 3
- Optuna (Bayesian HPO)
- Gymnasium
- Weights & Biases / TensorBoard
