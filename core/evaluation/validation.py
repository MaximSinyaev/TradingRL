import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from custom_envs.trading_env_v5 import TradingEnvV5
from core.visualization.visualizer_candles import CandleChartVisualizer

def get_base_env(vec_env):
    while hasattr(vec_env, 'venv'):
        vec_env = vec_env.venv
    return vec_env.envs[0]

def run_validation(data_features, model_path, norm_path, use_frame_stack=True, n_stack=10, test_size=360, random_start=True):
    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # If random_start is True, we pick a random 360-candle slice from the data
    if random_start and len(data_features) > test_size:
        start_idx = np.random.randint(0, len(data_features) - test_size)
        test_df = data_features.iloc[start_idx:start_idx + test_size].reset_index(drop=True)
        print(f"Random validation slice selected: from index {start_idx} to {start_idx + test_size}.")
    else:
        test_df = data_features.iloc[-test_size:].reset_index(drop=True)
        print("Using the last test_size candles for validation.")

    test_env = TradingEnvV5(df=test_df, continuous_action=True, t_max=len(test_df), initial_deposit=100_000.0)
    vec_env = DummyVecEnv([lambda: test_env])

    if use_frame_stack:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    try:
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    except Exception as e:
        print(f"Error loading normalizer: {e}")
        return

    obs = vec_env.reset()
    done = [False]

    actions_list = []
    portfolio_values = []

    print("Running simulation...")
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = vec_env.step(action)
        
        act_val = action[0][0] if isinstance(action, np.ndarray) else action
        size_str = f"{abs(act_val)*100:.0f}%"
        if act_val > 0.05:
            actions_list.append(f'buy_{size_str}')
        elif act_val < -0.05:
            actions_list.append(f'sell_{size_str}')
        else:
            actions_list.append('hold')
            
        if done[0]:
            # SB3 automatically resets the environment upon done=True.
            # To get the true final PnL before reset, we read it from info[0].
            pnl = info[0].get("pnl", 0.0)
            final_port_val = 100_000.0 * (1 + pnl / 100.0)
            portfolio_values.append(final_port_val)
        else:
            real_env = get_base_env(vec_env)
            cur_price = real_env._get_current_price()
            portfolio_values.append(real_env.get_portfolio_value(cur_price))

    real_env = get_base_env(vec_env)
    
    print(f"\n{'='*30}")
    print(f"📊 VALIDATION REPORT")
    print(f"{'='*30}")
    print(f"Initial balance: $100,000.00")
    print(f"Final balance:   ${portfolio_values[-1]:,.2f}")
    roi = (portfolio_values[-1] - 100_000) / 100_000 * 100
    print(f"Net Profit:      {roi:.2f}%")

    active_steps = len([a for a in actions_list if a != 'hold'])
    print(f"Candles in pos:  {active_steps} out of {len(actions_list)}")

    if len(actions_list) >= len(test_df) - 1:
        print("Status: 🏁 Successfully reached end of slice")
    else:
        print("Status: 💀 Liquidated (Margin Call / Max Drawdown)")

    actions_series = pd.Series(actions_list)
    port_series = pd.Series(portfolio_values)
    
    # Trim test_df if agent died early
    test_df_trimmed = test_df.iloc[:len(actions_list)].copy()

    visualizer = CandleChartVisualizer(use_volume_width=False)
    visualizer.plot_candlestick(
        data=test_df_trimmed, 
        title=f"Validation Backtest ({roi:.2f}% PnL)", 
        actions=actions_series,
        portfolio_values=port_series
    )

def plot_boruta_importance(importance_scores):
    import plotly.graph_objects as go
    import numpy as np
    
    # Сортируем фичи по медианному падению PnL
    medians = {f: np.median(scores) for f, scores in importance_scores.items()}
    sorted_features = sorted(medians.keys(), key=lambda x: medians[x])

    fig = go.Figure()

    for feature in sorted_features:
        scores = importance_scores[feature]
        
        # Цвет: зеленый если медиана > 0, красный если медиана < 0
        box_color = 'mediumseagreen' if medians[feature] > 0 else 'indianred'
        
        fig.add_trace(go.Box(
            x=scores,
            y=[feature] * len(scores),
            name=feature,
            orientation='h',
            marker_color=box_color,
            boxmean=True # Показывать среднее пунктирной линией
        ))

    # Добавляем вертикальную линию на нуле
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")

    fig.update_layout(
        title="Permutation Feature Importance (Boruta-style 100 Iterations)",
        xaxis_title="Падение PnL без фичи (Положительное = фича полезна)",
        yaxis_title="Фича",
        height=800,
        showlegend=False,
        template="plotly_dark",
        margin=dict(l=150)
    )

    fig.show()

    print("\nИНСТРУКЦИЯ:")
    print("1. Весь ЯЩИК ПРАВЕЕ нуля: Фича ПОЛЕЗНАЯ. Удалять нельзя!")
    print("2. Ящик ПЕРЕСЕКАЕТ ноль: Фича — ШУМ. Модель не уверена в её пользе.")
    print("3. Весь ЯЩИК ЛЕВЕЕ нуля: Фича ВРЕДНАЯ. Удаление этой колонки увеличит прибыль!")

def evaluate_model(model, vec_normalize_path, df_to_test, use_frame_stack=True, n_stack=10, num_seeds=10):
    pnls = []
    for seed in range(num_seeds):
        env = TradingEnvV5(df=df_to_test, continuous_action=True, t_max=len(df_to_test), initial_deposit=100_000.0, domain_randomization=False)
        vec_env = DummyVecEnv([lambda: env])
        
        if use_frame_stack:
            vec_env = VecFrameStack(vec_env, n_stack=n_stack)
            
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        
        vec_env.seed(seed)
        obs = vec_env.reset()
        done = [False]
        
        final_port_val = 100_000.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = vec_env.step(action)
            if done[0]:
                pnl = info[0].get("pnl", 0.0)
                final_port_val = 100_000.0 * (1 + pnl / 100.0)
                break
                
        pnls.append(final_port_val)
        
    return np.mean(pnls)

def calculate_permutation_importance(model, vec_normalize_path, test_df, final_features, base_pnl, n_iterations=10, use_frame_stack=True, n_stack=10, num_seeds=10):
    from tqdm.auto import tqdm
    importance_scores = {feature: [] for feature in final_features}
    
    print(f"\\nНачинаем перемешивание фичей ({n_iterations} итераций на каждую, {num_seeds} seed-прогонов внутри)...\\n")
    rng = np.random.default_rng()
    
    for feature in tqdm(final_features, desc="Features"):
        for i in range(n_iterations):
            shuffled_df = test_df.copy()
            
            # Перемешиваем только одну колонку
            shuffled_df[feature] = rng.permutation(shuffled_df[feature].values)
            
            # Пересобираем state_vector ТОЛЬКО из нужных фичей, чтобы шейп совпадал с моделью
            feature_arrays = []
            for c in final_features:
                feature_arrays.append(shuffled_df[c].values)
            
            shuffled_state_matrix = np.column_stack(feature_arrays)
            shuffled_df['state_vector'] = list(shuffled_state_matrix)
            
            # Тестируем
            pnl_with_noise = evaluate_model(model, vec_normalize_path, shuffled_df, use_frame_stack, n_stack, num_seeds)
            
            # Падение баланса
            performance_drop = base_pnl - pnl_with_noise
            importance_scores[feature].append(performance_drop)
            
        mean_drop = np.mean(importance_scores[feature])
        std_drop = np.std(importance_scores[feature])
        print(f"Фича {feature}: среднее падение = ${mean_drop:,.2f} ± ${std_drop:,.2f}")
        
    return importance_scores
