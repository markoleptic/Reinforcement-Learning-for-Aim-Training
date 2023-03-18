from Algos import Algos
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env

algos = Algos(numRows=10, numCols=5, alpha=0.1, gamma=0.9, epsilon=1)
env = gym.make("ML_Env/ML_RL_Env-v0", numRows=algos.rows, numCols=algos.cols, timeStep=1, episodeLength=100)

# -----------------------
# --- experiment loop ---
# -----------------------

# Sarsa

totalCumRewards = np.array([])
epsilon = 1
action = np.array([0, 0])
NumberOfIterations = 100
while epsilon >= 0:
    epsilonCumRewards = np.array([])
    for index in range(NumberOfIterations):
        env.reset()
        terminated = False
        episodeRewards = np.array([])
        while not terminated:
            observation, reward, terminated, truncated, info = env.step(action)
            action = algos.updateQTable_Sarsa(observation.get("prevPos"), observation.get("position"), reward, epsilon)
            episodeRewards = np.append(episodeRewards, float(reward))
            if terminated or truncated:
                algos.q_table_sum_SarsaLearning = algos.updateQTableSum(algos.q_table_sum_SarsaLearning)
                epsilonCumRewards = (episodeRewards if (epsilonCumRewards.size == 0) else (np.vstack([epsilonCumRewards, episodeRewards])))
                break
    totalCumRewards = (np.cumsum(epsilonCumRewards.mean(0), 0) if (totalCumRewards.size == 0) else np.vstack([totalCumRewards, np.cumsum(epsilonCumRewards.mean(0), 0)]))
    epsilon -= 0.2
algos.plotRewards(totalCumRewards)

# Q-Learning

epsilon = 1
totalCumRewards = np.array([])
action = np.array([0, 0])
while epsilon >= 0:
    epsilonCumRewards = np.array([])
    for index in range(NumberOfIterations):
        env.reset()
        terminated = False
        episodeRewards = np.array([])
        qTable = np.zeros((algos.rows, algos.cols, algos.rows, algos.cols))
        while not terminated:
            observation, reward, terminated, truncated, info = env.step(action)
            action = algos.updateQTableQLearning(observation.get("prevPos"), observation.get("position"), reward, epsilon)
            episodeRewards = np.append(episodeRewards, float(reward))
            if terminated or truncated:
                algos.q_table_sum_QLearning = algos.updateQTableSum(algos.q_table_sum_QLearning)
                epsilonCumRewards = (episodeRewards if (epsilonCumRewards.size == 0) else (np.vstack([epsilonCumRewards, episodeRewards])))
                break
    totalCumRewards = (np.cumsum(epsilonCumRewards.mean(0), 0) if (totalCumRewards.size == 0) else np.vstack([totalCumRewards, np.cumsum(epsilonCumRewards.mean(0), 0)]))
    epsilon -= 0.2
algos.plotRewards(totalCumRewards)

qTableDivide = np.full((algos.rows, algos.cols), algos.rows * algos.cols * NumberOfIterations)
algos.Average_And_Visualize_QTable(algos.q_table_sum_QLearning, qTableDivide, "QTable of Q-Learning")
algos.Average_And_Visualize_QTable(algos.q_table_sum_SarsaLearning, qTableDivide, "QTable of Sarsa-Learning")

# print("Higher q-values represent greater rewards from that position")
# # taking the mean values of the inner arrays
# print(np.around(np.mean(np.mean(qTable.transpose(),2),2), 2))
# Visualize_QTable(qTable,"Q-Learning Q-Table")

"""
If you're curious what 4-d arrays looks like, uncomment this section
"""
# qTable[0,0,0,0] = 9
# print(qTable[0,0])
# print(qTable[0,0,0])
# qTable = np.zeros((rows,cols,rows,cols))
# qTable[0,0,0] = 18
# print(qTable[0,0])
# print(qTable[0,0,0])
# print(qTable.transpose())