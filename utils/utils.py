import pandas as pd

def transform_int_actions(actions):
    """
    Transform the actions to a more readable format.
    """
    actions = pd.Series(actions)
    return actions.replace({0: 'hold', 1: 'buy', 2: 'sell'})