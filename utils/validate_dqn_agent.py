import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tqdm
from utils.visualizer_candles import CandleChartVisualizer
from utils.utils import transform_int_actions

candle_visualizer = CandleChartVisualizer(use_volume_width=False)

def validate_agent(env, agent, test_features, episodes: int = 10, t_max: int = None, render: bool = False, verbose: bool = True, base_env=None):
    """
    Валидирует агента в среде, используя только Q-network без обучения.
    
    Args:
        env: Торговая среда (должна реализовывать .reset() и .step(action)).
        agent: Объект агента с .q_network (torch.nn.Module).
        episodes: Количество эпизодов для оценки.
        t_max: Максимальная длина одного эпизода (в шагах). Если None — ограничение только от среды.
        render: Если True, отображает график действий каждые 100 шагов.
        verbose: Если True, печатает итоговую статистику по каждому эпизоду.
        base_env: Альтернативная среда для получения статистики (по умолчанию None).
    
    Returns:
        Список наград за каждый эпизод.
    """
    rewards = []
    end_pnls = []
    if base_env is None:
        base_env = env
    for ep in tqdm.tqdm(range(episodes), desc="Validation Progress", total=episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        prices = []
        actions = []
        steps = []

        t = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = agent.q_network(state_tensor)
                action = int(torch.argmax(q_values).item())

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            state = next_state
            
            actions.append(action)

            if render:
                price = base_env._get_current_price()  # предполагается, что первый фичер — это цена
                prices.append(price)
                
                steps.append(base_env.current_step)

                if base_env.current_step % 1000 == 0:
                    clear_output(wait=True)
                    env_start_idx = base_env.start_index
                    t_max = base_env.t_max
                    if getattr(env, 'stack_size', None) is not None:
                        t_max += env.stack_size
                    candle_visualizer.plot_candlestick(test_features.iloc[env_start_idx:env_start_idx+len(actions)].reset_index(),
                                                       actions=transform_int_actions(actions),
                    )
                    # _plot_prices_and_actions(prices, actions, steps, ep, env.current_step)

            t += 1
            if t_max is not None and t >= t_max:
                done = True

        rewards.append(total_reward)
        end_pnl = base_env.pnl
        end_pnls.append(end_pnl)
        
        
        if verbose:
            print(f"Episode {ep+1}/{episodes} — Total Reward: {total_reward:.2f}")

    avg_reward = np.mean(rewards)
    avg_pnl = np.mean(end_pnls)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}, Average PNL: {avg_pnl:.2f}")
    return end_pnls, rewards, actions

def _plot_prices_and_actions(prices, actions, steps, episode_num, step_num):
    """Вспомогательная функция для построения графика."""
    clear_output(wait=True)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, prices, label="Price", color='gray', alpha=0.7)

    buy_steps = [s for s, a in zip(steps, actions) if a == 1]
    buy_prices = [p for p, a in zip(prices, actions) if a == 1]
    sell_steps = [s for s, a in zip(steps, actions) if a == 2]
    sell_prices = [p for p, a in zip(prices, actions) if a == 2]

    plt.scatter(buy_steps, buy_prices, marker='^', color='green', label='Buy')
    plt.scatter(sell_steps, sell_prices, marker='v', color='red', label='Sell')

    plt.title(f"Episode {episode_num + 1}, Step {step_num}")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()