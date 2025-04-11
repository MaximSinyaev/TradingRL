import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

class FrameStackTradingEnvV1(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env: gym.Env,
        stack_size: int = 4
    ):
        super().__init__()

        self.base_env = base_env
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

        low = np.repeat(self.base_env.observation_space.low, stack_size, axis=0)
        high = np.repeat(self.base_env.observation_space.high, stack_size, axis=0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = self.base_env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.frames.clear()
        self.frames.append(obs)
        # Накопим стек через шаги действия hold (0)
        for _ in range(1, self.stack_size):
            obs, _, done, _, _ = self.base_env.step(0)
            self.frames.append(obs)
            if done:
                break
        return self._get_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.base_env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()
