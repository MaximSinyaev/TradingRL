# Trading RL - Instructions for Claude

## CRITICAL: First Step in Every Session

**IMMEDIATELY upon starting a new session or reset, you MUST:**

```bash
# 1. Read the project laws (AGENTS.md contains ALL rules)
Read .agents/AGENTS.md

# 2. Understand current context and recent changes
Read .agents/SYNC_LOG.md

# 3. Review the approved development plan
Read assistant_docs/roadmap.md
```

**These files are your BIBLE for this project. Follow them strictly.**

---

## Quick Summary (All details in AGENTS.md)

- **Code/Comments:** English | **Communication:** Russian
- **Storage:** `models/experiments/<exp_name>/` → use `core.experiment_manager.get_experiment_paths()`
- **W&B:** Mandatory for all training | TensorBoard in notebooks: FORBIDDEN
- **Dependencies:** Use `uv` (not pip/conda)
- **Architecture:** `core/data/` → `core/features/` → `custom_envs/` → `agents/`

---

## Project State (from SYNC_LOG.md)

- **Status:** Post-refactoring, stable
- **Environment:** `TradingEnvV6` (Target Portfolio Weight paradigm)
- **Data:** Bybit Futures | 4h timeframe | Multi-asset (BTC, ETH, BNB)
- **Cache:** `smart_data_cache/` with ready-to-use parquet files
- **DELETED:** `utils/` folder — do NOT recreate, use `core/` instead

---

## Memory Location

Project-specific memories are stored in:
```
.memory/
```

Use the Write tool to create memories there for important context that should persist within this session.

---

## This File

This file (`CLAUDE.md`) is read by the Claude Code harness when it initializes in this project directory.
