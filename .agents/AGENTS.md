# Workspace Rules: Trading RL

## Language Policy
- **Code, Comments, and Documentation:** All code comments, docstrings, and entries in `.agents/SYNC_LOG.md` MUST be written in English. This ensures token efficiency and aligns with standard software engineering practices.
- **Communication:** Communication with the user in the chat MUST be in Russian.

## Experiment Management
In this repository, all trained models, normalizers, and logs are strictly structured by experiments. This is a mandatory rule for all new training scripts and notebooks.

### 1. Storage Structure
All training artifacts must be saved in the `models/experiments/<exp_name>/` directory.
Inside this folder, there should be:
- `model.zip` (the model itself)
- `vec_normalize.pkl` (environment/normalizer state)

Tensorboard logs are saved in `tensorboard_logs/<exp_name>/`.

### 2. Using `experiment_manager`
Always use `core.experiment_manager` to set up paths before training starts:
```python
from core.experiment_manager import get_experiment_paths

exp_name = "my_experiment_name"
model_path, norm_path, tb_log_dir = get_experiment_paths(exp_name)

# Pass tb_log_dir to the agent (TensorBoard is connected by default)
model = create_ppo_agent(..., tensorboard_log=tb_log_dir)

# Training...

# Saving using the correct paths
model.save(model_path)
vec_env.save(norm_path)
```

### 3. Tracking (W&B and TensorBoard)
- **W&B (Weights & Biases)**: Must be connected **to all** training runs to log metrics and hyperparameters.
- **TensorBoard**: Connected by default to all agents (via `tensorboard_log` in the SB3 constructor).
- **Running TensorBoard**: It is **FORBIDDEN** to run TensorBoard inside Jupyter Notebook cells (via `%tensorboard` or `!tensorboard`). TensorBoard is always run as a separate process in the terminal.

## Synchronization and State Management
Multiple AI assistants may be working on the project simultaneously. To avoid breaking each other's code, use the `.agents/SYNC_LOG.md` file.
- **Always read `.agents/SYNC_LOG.md`** at the beginning of a session or before modifying key components (`data_loader`, `feature_generator`, `trading_env`).
- **Always write to `.agents/SYNC_LOG.md`** after completing a logical block of code or research.
- If you change the output data format (e.g., added a new column to a DataFrame), you **must** indicate this in the sync log under the *"Attention for other assistants"* section.

## Architecture and Separation of Concerns
The project has a strict modular structure. Mixing logic is forbidden!
- `core/data/data_loader.py` and related files: Only downloading data and caching. No feature math.
- `core/features/feature_generator.py`: Only indicator math, normalization, and forming the `state_vector`.
- `custom_envs/`: Gymnasium environments (interaction with the environment, commission calculation, slippage, reward).
- `agents/`: Only neural network logic, replay buffers, extractors, and algorithms (DQN, PPO).

## Trading RL Specifics (Our Edge)
We are building a system aimed at the real market:
- **Instrument**: Perpetual Futures. Take into account the possibility of shorting and Funding Rate mechanics.
- **Timeframe**: The main timeframe is 4h (to eliminate microstructural noise).
- **Environment > Algorithm**: The environment must be as harsh as possible (commissions, dynamic slippage depending on volatility, risk management).
- **Reward Function**: DO NOT use simple PnL. The reward must be risk-adjusted (Sortino Ratio, penalties for drawdown).

## Dependency Management (uv)
**uv** is used for managing the environment and dependencies (not pip or conda).
- **Installing packages**: Use `uv add <package>` or `uv pip install <package>`.
- **Running scripts**: Run via `uv run python script.py` (or Jupyter Notebook).

## Roadmap
Before proposing large-scale architectural changes, consult `assistant_docs/roadmap.md`. This is the approved project development plan.
