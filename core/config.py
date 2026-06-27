"""
Global configuration and constants for the Trading RL project.
"""

# Validation slices used for Purged Cross-Validation and OOS Evaluation.
# These specific periods are "punched out" from the training data to ensure
# the agent never sees them during training, preventing data leakage.
VAL_SLICES = {
    "bull_1": ("2021-02-01", "2021-03-31"),
    "bull_2": ("2021-10-01", "2021-11-15"),
    "bear_1": ("2022-05-01", "2022-06-15"),
    "bear_2": ("2022-11-01", "2022-12-15"),
    "flat_1": ("2021-07-15", "2021-08-31"),
    "flat_2": ("2023-02-01", "2023-03-15"),
    "bull_3": ("2024-02-01", "2024-03-15"),  # ETF Bull run
    "flat_3": ("2024-05-01", "2024-06-15"),  # Post-halving consolidation
}

# Number of frames to stack for the PPO agent's observation space
N_STACK = 18
