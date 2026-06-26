# Workspace Rules: Trading RL

## Language Policy
- **Code, Comments, and Documentation:** All code comments, docstrings, and entries in `.agents/SYNC_LOG.md` MUST be written in English. This ensures token efficiency and aligns with standard software engineering practices.
- **Communication:** Communication with the user in the chat MUST be in Russian.

## Core Principle: Transparency and Consistency
The user values control and understanding of what is happening in the codebase.
- **Do not work silently.** Before rewriting half the project, tell the user: *"I plan to do X, Y, and Z. Shall we proceed?"*
- **Iterative approach.** Break complex tasks into phases. Deliver one logical part → report → move on.
- **Justification.** If you make an architectural decision (e.g., replacing PPO with SAC), justify it based on research and market logic.

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

## Multi-Agent Synchronization: BLACKBOARD.md vs SYNC_LOG.md
To manage parallel work across multiple AI assistants safely, we use two distinct tools. **Do NOT confuse their purposes.**

### 1. Real-time Control: `BLACKBOARD.md`
This is a strict "control panel" and announcement board for active parallel execution. It is **NOT A FORUM**.
- **State Broadcasting Only (No Chatter):** Exists strictly for stating facts and passing state. Only direct statements are allowed: *"Locked file X for refactoring"* or *"Changed interface of function Y, use Z"*. Any forms of dialogue, "thanks", or discussions are strictly forbidden. If there's a conflict or architectural doubt, go to the user in chat.
- **Actionable Information Only:** Only write things that *directly affect* the very next step of another assistant. Do not write progress reports here (use SYNC_LOG for that). Write warnings: *"Warning: DataFrame X structure changed, added column Y"*.
- **Mandatory Garbage Collection:** As soon as an assistant reads a message addressed to them, incorporates it, or when a file lock is no longer active, **it is their direct responsibility to delete that entry from the file**. The Blackboard must always strive to be empty.
- **Hard Size Limit:** `.agents/BLACKBOARD.md` **must NEVER exceed 30 lines**. If it grows larger, the assistant opening it must compress it, aggregate data, or delete stale garbage before adding a new entry.

**Format for Blackboard entries:**
`[TIMESTAMP] [AGENT_NAME]: [TARGET/GLOBAL] - [ACTIONABLE_MESSAGE]`

### 2. Asynchronous Project Log: `SYNC_LOG.md`
- Used for recording completed logical blocks, research findings, and architectural decisions.
- Read at the beginning of a session to gain context.
- Written to *only after* successful completion of a phase/task (e.g., "Implemented metric X, logic is in file Y").
- If you change the output data format, you **must** indicate this here under an *"Attention for other assistants"* section, AND broadcast it immediately on the Blackboard.


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
