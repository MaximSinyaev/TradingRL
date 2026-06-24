import pandas as pd
import numpy as np


def transform_int_actions(actions):
    """
    Transform actions to a more readable format.

    Supports:
    - Discrete: [0, 1, 2] → ['hold', 'buy', 'sell']
    - MultiDiscrete: [[0, 3], [1, 5], ...] → ['hold_30%', 'buy_50%', ...]

    Args:
        actions: List or array of actions (int or [action_type, size_level])

    Returns:
        pd.Series with string labels
    """
    actions = pd.Series(actions)

    # Check if MultiDiscrete (list/array elements)
    if len(actions) > 0 and isinstance(actions.iloc[0], (list, np.ndarray)):
        # MultiDiscrete format: [action_type, size_level]
        action_names = {0: 'hold', 1: 'buy', 2: 'sell'}
        size_levels = [f'{i*10}%' for i in range(1, 11)]  # ['10%', ..., '100%']

        def format_multi_action(action):
            if isinstance(action, (list, np.ndarray)):
                action_type, size_idx = int(action[0]), int(action[1])
                if action_type == 0:
                    return 'hold'  # HOLD ignores size
                return f'{action_names[action_type]}_{size_levels[size_idx]}'
            return str(action)

        return actions.apply(format_multi_action)

    # Discrete format: [0, 1, 2]
    return actions.replace({0: 'hold', 1: 'buy', 2: 'sell'})