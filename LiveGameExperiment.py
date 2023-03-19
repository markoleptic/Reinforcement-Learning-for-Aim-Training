from Algos import Algos
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env
from FileModHandler import FileModified

# Sarsa

sarsa = Algos("QTable_Sarsa.npy", numRows=11, numCols=5, alpha=0.8, gamma=0.9, epsilon=0.8)
# initialize S
state = (5,3)

# Choose A from S using epsilon-greedy policy
action = sarsa.getNextAction(state, sarsa.epsilon)

# Update the QTable after detected a file change, which means action was taken
def file_modified():
    # Oberve R, S'
    results = np.loadtxt("Accuracy.csv", delimiter=',')
    checkEndEpisode(results, sarsa)

    # results include the previous state, in case ordering was changed in game
    state = (int(results[0]), int(results[1]))
    state_2 = (int(results[2]), int(results[3]))
    action = state_2
    reward = int(results[4])
    sarsa.episodeRewards = np.append(sarsa.episodeRewards, float(reward))

    # Choose A from S using epsilon-greedy policy
    action_2 = sarsa.getNextAction(state_2, sarsa.epsilon)

    # update QTable
    sarsa.updateQTable_Sarsa(state, action, state_2, action_2, sarsa.epsilon)

    # Push next spawn location to game
    printNextSpawnLocation("SpawnLocation.txt", action_2)

    action = action_2
    state = state_2
    sarsa.saveQTable("QTable_Sarsa.npy")
    return False

def checkEndEpisode(results, AlgoClass):
    if (results.__len__() == 0):
        qTableDivide = np.full((AlgoClass.rows, AlgoClass.cols), AlgoClass.rows * AlgoClass.cols)
        if AlgoClass == sarsa:
            AlgoClass.q_table_sum_SarsaLearning = AlgoClass.updateQTableSum(AlgoClass.q_table_sum_SarsaLearning)
            AlgoClass.Average_And_Visualize_QTable(AlgoClass.q_table_sum_SarsaLearning, qTableDivide, "QTable of Sarsa-Learning")
        else:
            AlgoClass.q_table_sum_QLearning = AlgoClass.updateQTableSum(AlgoClass.q_table_sum_QLearning)
            AlgoClass.Average_And_Visualize_QTable(AlgoClass.q_table_sum_QLearning, qTableDivide, "QTable of Q-Learning")
        AlgoClass.plotRewards(AlgoClass.episodeRewards.cumsum(0), False, AlgoClass.epsilon)
        exit()

# This function saves a new spawn location to "SpawnLocation.txt" after updating the QTable in response to a file change
def printNextSpawnLocation(path, location):
    np.savetxt(path, location, delimiter=",")
    print("Next Spawn Location written to SpawnLocation.txt: ", location)

# Start monitoring for file changes
fileModifiedHandler = FileModified(r"Accuracy.csv", file_modified)
fileModifiedHandler.start()


# Q-Learning

qlearning = Algos("QTable_QLearning.npy", numRows=11, numCols=5, alpha=0.8, gamma=0.9, epsilon=0.8)
# initialize S
state = (5,3)

# Choose A from S using epsilon-greedy policy
action = qlearning.getNextAction(state, qlearning.epsilon)

# Update the QTable after detected a file change, which means action was taken
def file_modified():
    # Oberve R, S'
    results = np.loadtxt("Accuracy.csv", delimiter=',')
    checkEndEpisode(results, qlearning)

    # results include the previous state, in case ordering was changed in game
    state = (int(results[0]), int(results[1]))
    state_2 = (int(results[2]), int(results[3]))
    action = state_2
    reward = int(results[4])
    qlearning.episodeRewards = np.append(qlearning.episodeRewards, float(reward))

    # Choose A from S using epsilon-greedy policy
    action_2 = qlearning.getNextAction(state_2, qlearning.epsilon)

    # update QTable
    qlearning.updateQTable_QLearning(state, action, state_2, action_2, qlearning.epsilon)

    # Push next spawn location to game
    printNextSpawnLocation("SpawnLocation.txt", action_2)
    state = state_2
    qlearning.saveQTable("QTable_QLearning.npy")
    return False