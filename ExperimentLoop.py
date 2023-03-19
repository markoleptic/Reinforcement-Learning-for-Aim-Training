from Algos import Algos
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env


def clamp(num, min_value, max_value):
    return max(min(num, max_value), min_value)

algos = Algos(numRows=11, numCols=5, alpha=1, gamma=0.9, epsilon=0.9)
env = gym.make("ML_Env/ML_RL_Env-v0", numRows=algos.rows, numCols=algos.cols, timeStep=1, episodeLength=300)

# -----------------------
# --- experiment loop ---
# -----------------------

# Sarsa

# initialize S
state = (5,3)

# Choose A from S using epsilon-greedy policy
action = algos.getNextAction(state, algos.epsilon)

totalCumRewards = np.array([])
numEpisodes = 2000
episodeCumRewards = np.array([])
for index in range(numEpisodes):
    env.reset()
    terminated = False
    episodeRewards = np.array([])
    algos.alpha = clamp(1 - (float(index)/(numEpisodes - float(numEpisodes)/5)), 0, 1)
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
algos.plotRewards(episodeCumRewards, False, algos.epsilon)
qTableDivide = np.full((algos.rows, algos.cols), algos.rows * algos.cols * numEpisodes)
algos.Average_And_Visualize_QTable(algos.q_table_sum_SarsaLearning, qTableDivide, "QTable of Sarsa-Learning")


# Q-Learning
algos = Algos(numRows=11, numCols=5, alpha=1, gamma=0.9, epsilon=0.9)

# initialize S
state = (5,3)

# Choose A from S using epsilon-greedy policy
action = algos.getNextAction(state, algos.epsilon)

totalCumRewards = np.array([])
numEpisodes = 2000
episodeCumRewards = np.array([])
for index in range(numEpisodes):
    env.reset()
    terminated = False
    episodeRewards = np.array([])
    algos.alpha = clamp(1 - (float(index)/(numEpisodes - float(numEpisodes)/5)), 0, 1)
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
algos.plotRewards(episodeCumRewards, False, algos.epsilon)
qTableDivide = np.full((algos.rows, algos.cols), algos.rows * algos.cols * numEpisodes)
algos.Average_And_Visualize_QTable(algos.q_table_sum_QLearning, qTableDivide, "QTable of Q-Learning")



# algos.plotRewards(episodeRewards.cumsum(0), False, epsilon)
# totalCumRewards = (np.cumsum(episodeCumRewards.mean(0), 0) if (totalCumRewards.size == 0) else np.vstack([totalCumRewards, np.cumsum(episodeCumRewards.mean(0), 0)]))
# episodeCumRewards = (episodeRewards if (episodeCumRewards.size == 0) else (np.vstack([episodeCumRewards, episodeRewards])))

# print(np.around(np.mean(np.mean(qTable.transpose(),2),2), 2))
# Visualize_QTable(qTable,"Q-Learning Q-Table")
# qTable[0,0,0,0] = 9
# print(qTable[0,0])
# print(qTable[0,0,0])
# qTable = np.zeros((rows,cols,rows,cols))
# qTable[0,0,0] = 18
# print(qTable[0,0])
# print(qTable[0,0,0])
# print(qTable.transpose())