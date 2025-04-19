import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from utils.utils import transform_int_actions
from IPython.display import clear_output
import numpy as np

def adaptive_epsilon_schedule(episode, total_episodes, epsilon_min=0.05, epsilon_max=1.0):
    # Функция, убывающая сначала и растущая к концу
    decay = np.exp(-5 * episode / total_episodes)
    recovery = (episode / total_episodes) ** 3
    if episode < total_episodes // 2:
        return np.clip(epsilon_min + (epsilon_max - epsilon_min) * (decay + recovery), epsilon_min, epsilon_max)
    else:
        return np.clip(epsilon_min + (epsilon_max - epsilon_min) * (decay + recovery), epsilon_min, epsilon_max - 0.3)


def train_dqn_agent(env, agent, 
                    num_episodes=100, target_update_freq=10, 
                    t_max=100_000, render=False, base_env=None,
                    adaptive_epsilon=False, epsilon_min=0.05, epsilon_max=1.0
                    ):
    if base_env is None:
        base_env = env
    actions_per_episode = defaultdict(list)
    rewards_per_episode = []
    pnl_per_episode = []

    for episode in tqdm.trange(num_episodes):
        if adaptive_epsilon:
            agent.epsilon = adaptive_epsilon_schedule(episode, num_episodes, epsilon_min=epsilon_min, epsilon_max=epsilon_max)
        state, _ = env.reset()
        total_reward = 0
        done = False

        for i in range(t_max):
            action = agent.act(state)
            actions_per_episode[episode].append(action)

            state_next, reward, done, _, _ = env.step(action)
            agent.store(state, action, reward, state_next, done)
            agent.train_step()
            state = state_next
            total_reward += reward
            if done:
                break

        if target_update_freq is not None and episode % target_update_freq == 0:
            agent.update_target_network()

        rewards_per_episode.append(total_reward)
        pnl_per_episode.append(base_env.pnl)

        if episode % 10 == 0:
            try:
                clear_output(wait=True)
            except NameError:
                pass

            print(f"For episodes {episode} to {episode - 10}: mean reward {np.mean(rewards_per_episode[-10:]):.2f}, mean pnl {np.mean(pnl_per_episode[-10:]):.2f}")
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, PNL: {base_env.pnl:.2f}")
            print(f"Actions: {transform_int_actions(actions_per_episode[episode]).value_counts(normalize=True).to_dict()}")
            if render:
                plt.figure(figsize=(10, 4))
                plt.plot(pnl_per_episode, label="PNL")
                plt.xlabel("Episode")
                plt.ylabel("PNL")
                plt.title("PNL over episodes")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.show()

    return rewards_per_episode, pnl_per_episode, actions_per_episode