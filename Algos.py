import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env

np.set_printoptions(threshold=np.inf, linewidth=140)

class Algos:
    """
    Class that updates qTable values, implements Sarsa and QLearning algorithms
    """
    def __init__(self, numRows=10, numCols=5, alpha=0.1, gamma=0.9, epsilon=1):
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

        # qTable: an array of the size of the grid (outer) where each element also has size of the grid (inner) (17x9)x(17x9)
        # each element in the inner contains the q-value for starting at outer state and moving to inner state
        self.qTable = np.zeros((self.rows, self.cols, self.rows, self.cols))
        self.q_table_sum_QLearning = np.zeros((self.rows, self.cols))
        self.q_table_sum_SarsaLearning = np.zeros((self.rows, self.cols))

    def Average_And_Visualize_QTable(self, QTableSum, QTableDivide, title):
        avg_qTable = np.divide(QTableSum, QTableDivide)
        cmap = plt.colormaps.get_cmap("Greens")
        plt.title(title)
        plt.imshow(avg_qTable.transpose(), cmap=cmap)
        plt.colorbar(shrink=0.5)
        plt.show()

    def plotRewards(self, cumulativeRewardArray):
        epsilon = 1
        font = {"weight": "bold"}
        plt.rc("font", **font)
        for item in cumulativeRewardArray:
            plt.plot(item, label="\u03B5 = " + str(round(epsilon, 1)))
            epsilon -= 0.2
        plt.xlabel("Time Step in Episode")
        plt.ylabel("Cumulative Avg Reward across All Eps")
        plt.legend()
        plt.show()

    def updateQTableSum(self, QTableSum):
        for i in range(self.rows):
            for j in range(self.cols):
                QTableSum = np.add(QTableSum, self.qTable[i][j])
        return QTableSum

    def saveQTable(self, path = "QTable.csv"):
        np.savetxt(path, self.qTable, delimiter=",")

    def loadQTable(self, path = "QTable.csv"):
        self.qTable = np.loadtxt(path)
        return np.loadtxt(path)

    def getMaxActionIndex(self, startPosition):
        """
        Returns the full (4) indices of next action (position) to take based on the policy implemented (epsilon).
        Args:
            Q: the 4-dimensional q-table
            startPosition: the starting state/position
        """
        return np.unravel_index(np.argmax(self.qTable[startPosition]), self.qTable.shape)

    def getNextAction(self, prevPos, epsilon):
        """
        Returns the next action (position) to take based on the policy implemented (epsilon).
        Either uses "epsilon-greedy approach" (exploitation) or random location (exploration),
        based on the value of epsilon.
        Args:
            prevPos: the previous agent position
        """
        # create random float between 0 and 1, then compare with epsilon
        actionToTake = np.random.uniform(0, 1)
        if actionToTake > epsilon:
            # exploitation
            x1, y1, x2, y2 = self.getMaxActionIndex(prevPos)
            return np.array([x2, y2])
        # exploration
        x1 = np.random.randint(0, self.rows)
        y1 = np.random.randint(0, self.cols)
        return np.array([x1, y1])

    def updateQTable_Sarsa(self, prevPos, position, reward, epsilon):
        """
        Updates the qTable by using the qfunction
        Args:
            Q: the state action table
            prevPos: the previous agent position
            position: the current agent position
            reward: The reward recieved from taking action (position) from prevPos
        """
        x1, y1 = prevPos
        x2, y2 = position
        x3, y3 = self.getNextAction(position, epsilon)
        next = self.qTable[x2, y2, x3, y3]
        # newqValue(prevPos -> action -> position) = oldqValue + alpha(gamma * nextqValue(position -> nextPosition) - oldqValue)
        self.qTable[x1, y1, x2, y2] = self.qTable[x1, y1, x2, y2] + self.alpha * (
            reward + self.gamma * next - self.qTable[x1, y1, x2, y2]
        )
        return self.getNextAction(position, epsilon)

    def updateQTableQLearning(self, prevPos, position, reward, epsilon):
        """
        Args:
            Q: the state action table
            prevPos: the previous agent position
            position: the current agent position
            reward: The reward recieved from taking action (position) from prevPos
        """
        x1, y1 = prevPos
        x2, y2 = position
        # optimal qvalue of the next state
        optimalNext = self.qTable[self.getMaxActionIndex((x2, y2))]
        # newqValue(prevPos -> action -> position) = oldqValue + alpha(gamma * nextqValue(position -> nextPosition) - oldqValue)
        self.qTable[x1, y1, x2, y2] = self.qTable[x1, y1, x2, y2] + self.alpha * (
            reward + self.gamma * optimalNext - self.qTable[x1, y1, x2, y2]
        )
        return self.getNextAction(position, epsilon)
