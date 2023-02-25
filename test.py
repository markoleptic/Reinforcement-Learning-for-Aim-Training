import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
import ML_Env

# print(Box(0, 50, shape=(2,), dtype=int))
# observation_space = Box(0, 3, shape=(17-1, 9-1), dtype=int)

# print(observation_space.sample())
from gymnasium.utils.env_checker import check_env

env = gym.make('ML_Env/ML_RL_Env-v0',render_mode="human")
env.reset()
for _ in range(1000):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        observation, info = env.reset()