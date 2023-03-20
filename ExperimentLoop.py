from Algos import Algos
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env

def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)

env = gym.make("ML_Env/ML_RL_Env-v0", numRows=11, numCols=5, timeStep=1, episodeLength=300)
numEpisodes = 10000

# -----------------------
# --- experiment loop ---
# -----------------------

#------------------#
#------Random------#
#------------------#
algos = Algos(numRows=11, numCols=5, alpha=1, gamma=0.9, epsilon=0.9)
# Choose A from S using epsilon-greedy policy
action = (np.random.randint(0, 11), np.random.randint(0, 5))
totalCumRewards = np.array([])
episodeCumRewards = np.array([])
for index in range(numEpisodes):
    env.reset()
    terminated = False
    episodeRewards = np.array([])
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(action)
        episodeRewards = np.append(episodeRewards, float(reward))
        # Randomly choose a location
        action = (np.random.randint(0, 11), np.random.randint(0, 5))
        if terminated or truncated:
            episodeCumRewards = np.append(episodeCumRewards, (episodeRewards.sum(0)))
            break
RandomTotalRewards = episodeCumRewards

#------------------#
#-------Sarsa------#
#------------------#
algos = Algos(numRows=11, numCols=5, alpha=1, gamma=0.9, epsilon=0.9)
# initialize S
state = (5,3)
# Choose A from S using epsilon-greedy policy
action = algos.getNextAction(state, algos.epsilon)
totalCumRewards = np.array([])
episodeCumRewards = np.array([])
for index in range(numEpisodes):
    env.reset()
    terminated = False
    episodeRewards = np.array([])
    algos.alpha = 1 - (float(index)/numEpisodes)
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(action)
        episodeRewards = np.append(episodeRewards, float(reward))
        state = tuple(observation.get("prevPos"))
        state_2 = tuple(observation.get("position"))
        # Choose A from S using epsilon-greedy policy
        action_2 = algos.getNextAction(state_2, algos.epsilon)
        algos.updateQTable_Sarsa(state, action, state_2, action_2, reward, algos.epsilon)
        action = action_2
        state = state_2
        if terminated or truncated:
            algos.q_table_sum_SarsaLearning = algos.updateQTableSum(algos.q_table_sum_SarsaLearning)
            episodeCumRewards = np.append(episodeCumRewards, (episodeRewards.sum(0)))
            break
SarsaTotalRewards = episodeCumRewards
SarsaQTableSum = algos.q_table_sum_SarsaLearning

# Sarsa epsilon = 0.5

algos = Algos(numRows=11, numCols=5, alpha=1, gamma=0.9, epsilon=0.5)
# initialize S
state = (5,3)
# Choose A from S using epsilon-greedy policy
action = algos.getNextAction(state, algos.epsilon)
totalCumRewards = np.array([])
episodeCumRewards = np.array([])
for index in range(numEpisodes):
    env.reset()
    terminated = False
    episodeRewards = np.array([])
    algos.alpha = 1 - (float(index)/numEpisodes)
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(action)
        episodeRewards = np.append(episodeRewards, float(reward))
        state = tuple(observation.get("prevPos"))
        state_2 = tuple(observation.get("position"))
        # Choose A from S using epsilon-greedy policy
        action_2 = algos.getNextAction(state_2, algos.epsilon)
        algos.updateQTable_Sarsa(state, action, state_2, action_2, reward, algos.epsilon)
        action = action_2
        state = state_2
        if terminated or truncated:
            algos.q_table_sum_SarsaLearning = algos.updateQTableSum(algos.q_table_sum_SarsaLearning)
            episodeCumRewards = np.append(episodeCumRewards, (episodeRewards.sum(0)))
            break
SarsaEpsilonHalfTotalRewards = episodeCumRewards

#------------------#
#----Q-Learning----#
#------------------#
algos = Algos(numRows=11, numCols=5, alpha=1, gamma=0.9, epsilon=0.9)
# initialize S
state = (5,3)
# Choose A from S using epsilon-greedy policy
action = algos.getNextAction(state, algos.epsilon)
totalCumRewards = np.array([])
episodeCumRewards = np.array([])
for index in range(numEpisodes):
    env.reset()
    terminated = False
    episodeRewards = np.array([])
    algos.alpha = 1 - (float(index)/numEpisodes)
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(action)
        episodeRewards = np.append(episodeRewards, float(reward))
        state = tuple(observation.get("prevPos"))
        state_2 = tuple(observation.get("position"))
        # Choose A from S using epsilon-greedy policy
        action_2 = algos.getNextAction(state_2, algos.epsilon)
        algos.updateQTable_QLearning(state, action, state_2, action_2, reward, algos.epsilon)
        action = action_2
        state = state_2
        if terminated or truncated:
            algos.q_table_sum_QLearning = algos.updateQTableSum(algos.q_table_sum_QLearning)
            episodeCumRewards = np.append(episodeCumRewards, (episodeRewards.sum(0)))
            break
QLearningTotalRewards = episodeCumRewards
QLearningQTableSum = algos.q_table_sum_QLearning

# Q-Learning epsilon = 0.5

algos = Algos(numRows=11, numCols=5, alpha=1, gamma=0.9, epsilon=0.5)
# initialize S
state = (5,3)
# Choose A from S using epsilon-greedy policy
action = algos.getNextAction(state, algos.epsilon)
totalCumRewards = np.array([])
episodeCumRewards = np.array([])
for index in range(numEpisodes):
    env.reset()
    terminated = False
    episodeRewards = np.array([])
    algos.alpha = 1 - (float(index)/numEpisodes)
    while not terminated:
        observation, reward, terminated, truncated, info = env.step(action)
        episodeRewards = np.append(episodeRewards, float(reward))
        state = tuple(observation.get("prevPos"))
        state_2 = tuple(observation.get("position"))
        # Choose A from S using epsilon-greedy policy
        action_2 = algos.getNextAction(state_2, algos.epsilon)
        algos.updateQTable_QLearning(state, action, state_2, action_2, reward, algos.epsilon)
        action = action_2
        state = state_2
        if terminated or truncated:
            algos.q_table_sum_QLearning = algos.updateQTableSum(algos.q_table_sum_QLearning)
            episodeCumRewards = np.append(episodeCumRewards, (episodeRewards.sum(0)))
            break
QLearningEpsilonHalfTotalRewards = episodeCumRewards

def SMA(data, step):
    w=np.repeat(1,step)/step
    result = np.convolve(w, data, mode="valid")
    return result

print("RandomTotalRewards",np.sum(RandomTotalRewards,0))
print("SarsaTotalRewards",np.sum(SarsaTotalRewards,0))
print("SarsaEpsilonHalfTotalRewards",np.sum(SarsaEpsilonHalfTotalRewards,0))
print("QLearningTotalRewards",np.sum(QLearningTotalRewards,0))
print("QLearningEpsilonHalfTotalRewards",np.sum(QLearningEpsilonHalfTotalRewards,0))
#plt.plot(RandomTotalRewards, label="Reward Received - Random")
#plt.plot(SMA(RandomTotalRewards, 100), label="SMA of Rewards - Random")
#plt.plot(SarsaTotalRewards, label="Reward Received - Sarsa")
#plt.plot(QLearningTotalRewards, label="Reward Received - QLearning")
plt.plot(SMA(SarsaTotalRewards, 100), label="SMA of Rewards - Sarsa \u03B5 = 0.9")
plt.plot(SMA(SarsaEpsilonHalfTotalRewards, 100), label="SMA of Rewards - Sarsa \u03B5 = 0.5")
plt.plot(SMA(QLearningTotalRewards, 100), label="SMA of Rewards - QLearning \u03B5 = 0.9")
plt.plot(SMA(QLearningEpsilonHalfTotalRewards, 100), label="SMA of Rewards - QLearning \u03B5 = 0.5")
font = {"weight": "bold"}
plt.rc("font", **font)
plt.xlabel("Episode")
plt.ylabel("Sum of Rewards for Each Episode")
plt.legend()
plt.show()

qTableDivide = np.full((algos.rows, algos.cols), algos.rows * algos.cols * numEpisodes)
algos.Average_And_Visualize_QTable(SarsaQTableSum, qTableDivide, "QTable of Sarsa-Learning")
algos.Average_And_Visualize_QTable(QLearningQTableSum, qTableDivide, "QTable of Q-Learning")