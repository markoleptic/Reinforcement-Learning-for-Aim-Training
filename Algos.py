import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
import numpy as np
import ML_Env

np.set_printoptions(threshold=np.inf, linewidth=140)

class Algos:
    """
    Class that updates qTable values, implements Sarsa and QLearning algorithms
    """
    def __init__(self, qTableFileLoc = "", numRows=11, numCols=5, alpha=0.1, gamma=0.9, epsilon=1):
        # size of grid
        self.rows = numRows
        self.cols = numCols
        # learning rate
        self.alpha = alpha
        # discount/long-term reward factor
        self.gamma = gamma
        self.epsilon = epsilon
        # the action to pass to the environment to take (position)
        self.action = np.array([0, 0])
        self.episodeRewards = np.array([])

        # qTable: an array of the size of the grid (outer) where each element also has size of the grid (inner) (11x5)x(11x5)
        # each element in the inner contains the q-value for starting at outer state and moving to inner state
        if (qTableFileLoc == ""):
            self.qTable = np.zeros((self.rows, self.cols, self.rows, self.cols))
        else:
            self.qTable = np.load(qTableFileLoc)
        
        self.q_table_sum_QLearning = np.zeros((self.rows, self.cols))
        self.q_table_sum_SarsaLearning = np.zeros((self.rows, self.cols))

    def Average_And_Visualize_QTable(self, QTableSum, QTableDivide, title):
        avg_qTable = np.divide(QTableSum, QTableDivide)
        cmap = plt.colormaps.get_cmap("Greens")
        plt.title(title)
        plt.imshow(avg_qTable.transpose(), cmap=cmap)
        plt.colorbar(shrink=0.5)
        plt.show()

    def SMA(self, data,step):
        w=np.repeat(1,step)/step
        result = np.convolve(w, data, mode="valid")
        return result

    def plotRewards(self, cumulativeRewardArray, bVaryEpsilon = True, epsilon = 1):
        if bVaryEpsilon == False:
            plt.plot(cumulativeRewardArray, label="Reward Received per Episode")
            plt.plot(self.SMA(cumulativeRewardArray, 50), label="Simple Moving Average")
        else:
            for item in cumulativeRewardArray:
                plt.plot(item, label="\u03B5 = " + str(round(epsilon, 1)))
                epsilon -= 0.2
        font = {"weight": "bold"}
        plt.rc("font", **font)
        plt.xlabel("Episode")
        plt.ylabel("Sum of Rewards for Each Episode")
        plt.legend()
        plt.show()

    def updateQTableSum(self, QTableSum):
        for i in range(self.rows):
            for j in range(self.cols):
                QTableSum = np.add(QTableSum, self.qTable[i][j])
        return QTableSum

    def saveQTable(self, path):
        np.save(path, self.qTable)

    def loadQTable(self, path):
        self.qTable = np.load(path, encoding="utf8")
        return self.qTable

    def getMaxActionIndex(self, startPosition):
        """
        Returns the indices of next action (position) which has the highest q value
        Args:
            startPosition: a tuple of the starting state/position
        """
        return np.unravel_index(np.argmax(self.qTable[startPosition]), self.qTable[startPosition].shape)

    def getNextAction(self, prevPos, epsilon):
        """
        Returns the next action (position) to take based on the policy implemented (epsilon).
        Either uses "epsilon-greedy approach" (exploitation) or random location (exploration),
        based on the value of epsilon.
        Args:
            prevPos: a tuple of the previous agent position
            epsilon: epsilon value to use
        """
        # create random float between 0 and 1, then compare with epsilon
        actionToTake = np.random.uniform(0, 1)
        if actionToTake > epsilon:
            # exploitation
            return self.getMaxActionIndex(prevPos)
        # exploration
        return (np.random.randint(0, self.rows), np.random.randint(0, self.cols))

    def updateQTable_Sarsa(self, state, action, state_2, action_2, reward, epsilon):
        """
        Updates the qTable using the q update function for sarsa
        Args:
            state: a tuple of the previous agent position
            action: a tuple representing the action used to get to state 2
            state_2: a tuple of the current agent position
            action_2: the next action to take
            reward: The reward recieved from taking action (position) from state
            epsilon: epsilon value to use
        """
        # action and state_2 are the same since we are choosing locations anywhere on a grid
        # previous prediction
        predict = self.qTable[*state, *action]
        target = reward + self.gamma * self.qTable[*state_2, *action_2]
        self.qTable[*state, *action] = self.qTable[*state, *action] + self.alpha * (target - predict)

    def updateQTable_QLearning(self, state, action, state_2, action_2, reward, epsilon):
        """
        Updates the qTable using the q update function for Q-Learning
        Args:
            state: a tuple of the previous agent position
            action: a tuple representing the action used to get to state 2
            state_2: a tuple of the current agent position
            action_2: the next action to take
            reward: The reward recieved from taking action (position) from state
            epsilon: epsilon value to use
        """
        # action and state_2 are the same since we are choosing locations anywhere on a grid
        # previous prediction
        predict = self.qTable[*state, *action]
        # optimal Qvalue of the next state, always greedy
        q_Max = self.qTable[*state_2, *self.getMaxActionIndex(action_2)]
        target = reward + self.gamma * q_Max
        self.qTable[*state, *action] = self.qTable[*state, *action] + self.alpha * (target - predict)