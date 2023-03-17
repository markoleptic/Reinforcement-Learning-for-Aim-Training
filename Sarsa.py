import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env
np.set_printoptions(threshold=np.inf, linewidth=140)

# size of grid
rows = 10
cols = 5
# learning rate
alpha = 0.1
# discount/long-term reward factor
gamma = 0.9
epsilon = 1
# the action to pass to the environment to take (position)
action = np.array([0,0])
# the number of iterations of the env to perform
NumberOfIterations = 100

# qTable: an array of the size of the grid (outer) where each element also has size of the grid (inner) (17x9)x(17x9)
# each element in the inner contains the q-value for starting at outer state and moving to inner state
qTable = np.zeros((rows,cols,rows,cols))
q_table_sum_QLearning = np.zeros((rows, cols))
q_table_sum_SarsaLearning = np.zeros((rows, cols))

def Average_And_Visualize_QTable(QTableSum,QTableDivide,title):
    avg_qTable = np.divide(QTableSum, QTableDivide)
    cmap = plt.colormaps.get_cmap('Greens')
    plt.title(title)
    plt.imshow(avg_qTable.transpose(), cmap=cmap)
    plt.colorbar(shrink = 0.5)
    plt.show()

def plotRewards(cumulativeRewardArray):
    epsilon = 1
    font = {'weight' : 'bold'}
    plt.rc('font', **font)
    for item in cumulativeRewardArray:
        plt.plot(item, label = "\u03B5 = " + str(round(epsilon, 1)))
        epsilon-=0.2
    plt.xlabel("Time Step in Episode")
    plt.ylabel("Cumulative Avg Reward across All Eps")
    plt.legend()
    plt.show()

def QTableSum(QTable, QTableSum):
    for i in range(rows):
        for j in range(cols):
            QTableSum = np.add(QTableSum,QTable[i][j])
    return QTableSum

def saveQTable(QTable):
    np.savetxt('data.csv', QTable, delimiter=',')

def loadQTable():
    return np.loadtxt('data.csv')

def getMaxActionIndex(Q, startPosition):
    """
    Returns the full (4) indices of next action (position) to take based on the policy implemented (epsilon).
    Args:
        Q: the 4-dimensional q-table
        startPosition: the starting state/position
    """
    return(np.unravel_index(np.argmax(Q[startPosition]), Q.shape))

def getNextAction(prevPos):
    """
    Returns the next action (position) to take based on the policy implemented (epsilon).
    Either uses "epsilon-greedy approach" (exploitation) or random location (exploration),
    based on the value of epsilon.
    Args:
        prevPos: the previous agent position
    """
    # create random float between 0 and 1, then compare with epsilon
    actionToTake = np.random.uniform(0, 1)
    if (actionToTake > epsilon):
        # exploitation
        x1,y1,x2,y2 = getMaxActionIndex(qTable, prevPos)
        return np.array([x2,y2])
    # exploration
    x1 = np.random.randint(0, rows)
    y1 = np.random.randint(0, cols)
    return np.array([x1,y1])

def updateQTable_Sarsa(Q, prevPos, position, reward):
    """
    Updates the qTable by using the qfunction
    Args:
        Q: the state action table
        prevPos: the previous agent position
        position: the current agent position
        reward: The reward recieved from taking action (position) from prevPos
    """
    x1,y1 = prevPos 
    x2,y2 = position
    x3,y3 = getNextAction(position)
    next = qTable[x2,y2,x3,y3]
    # newqValue(prevPos -> action -> position) = oldqValue + alpha(gamma * nextqValue(position -> nextPosition) - oldqValue)
    qTable[x1,y1,x2,y2] = qTable[x1,y1,x2,y2] + alpha * (reward + gamma * next - qTable[x1,y1,x2,y2])
    return getNextAction(position)

def updateQTableQLearning(Q, prevPos, position, reward):
    """
    Args:
        Q: the state action table
        prevPos: the previous agent position
        position: the current agent position
        reward: The reward recieved from taking action (position) from prevPos
    """
    x1,y1 = prevPos 
    x2,y2 = position
    # optimal qvalue of the next state
    optimalNext = qTable[getMaxActionIndex(qTable,(x2,y2))]
    # newqValue(prevPos -> action -> position) = oldqValue + alpha(gamma * nextqValue(position -> nextPosition) - oldqValue)
    qTable[x1,y1,x2,y2] = qTable[x1,y1,x2,y2] + alpha * (reward + gamma * optimalNext - qTable[x1,y1,x2,y2])
    return getNextAction(position)

# -----------------------
# --- experiment loop ---
# -----------------------

# create ML_RL_Env environment, everything besides first parameter is optional
# set render_mode to "human" to view the game
env = gym.make('ML_Env/ML_RL_Env-v0', numRows=rows, numCols=cols, timeStep = 1, episodeLength = 100)
totalCumRewards = np.array([])
while epsilon >= 0:
    epsilonCumRewards = np.array([])
    for index in range(NumberOfIterations):
        env.reset()
        terminated = False
        episodeRewards = np.array([])
        qTable = np.zeros((rows,cols,rows,cols))
        while not terminated:
            # get the results from taking an action
            observation, reward, terminated, truncated, info = env.step(action)
            # update qTable using results from taking action
            action = updateQTable_Sarsa(qTable, observation.get('prevPos'), observation.get('position'), reward)
            episodeRewards = np.append(episodeRewards, float(reward))
            if terminated or truncated:
                q_table_sum_QLearning = QTableSum(qTable, q_table_sum_QLearning)
                epsilonCumRewards = episodeRewards if (epsilonCumRewards.size == 0) else (np.vstack([epsilonCumRewards, episodeRewards]))
                break
    totalCumRewards = np.cumsum(epsilonCumRewards.mean(0),0) if (totalCumRewards.size == 0) else np.vstack([totalCumRewards, np.cumsum(epsilonCumRewards.mean(0),0)])
    epsilon -=0.2
plotRewards(totalCumRewards)

epsilon = 1
totalCumRewards = np.array([])
while epsilon >= 0:
    epsilonCumRewards = np.array([])
    for index in range(NumberOfIterations):
        env.reset()
        terminated = False
        episodeRewards = np.array([])
        qTable = np.zeros((rows,cols,rows,cols))
        while not terminated:
            # get the results from taking an action
            observation, reward, terminated, truncated, info = env.step(action)
            # update qTable using results from taking action
            action = updateQTableQLearning(qTable, observation.get('prevPos'), observation.get('position'), reward)
            episodeRewards = np.append(episodeRewards, float(reward))
            if terminated or truncated:
                q_table_sum_SarsaLearning = QTableSum(qTable, q_table_sum_SarsaLearning)
                epsilonCumRewards = episodeRewards if (epsilonCumRewards.size == 0) else (np.vstack([epsilonCumRewards, episodeRewards]))
                break
    totalCumRewards = np.cumsum(epsilonCumRewards.mean(0),0) if (totalCumRewards.size == 0) else np.vstack([totalCumRewards, np.cumsum(epsilonCumRewards.mean(0),0)])
    epsilon -=0.2
plotRewards(totalCumRewards)
qTableDivide = np.full((rows,cols), rows*cols*NumberOfIterations)
Average_And_Visualize_QTable(q_table_sum_QLearning,qTableDivide,"QTable of Q-Learning")
Average_And_Visualize_QTable(q_table_sum_SarsaLearning,qTableDivide,"QTable of Sarsa-Learning")

# print("Higher q-values represent greater rewards from that position")
# # taking the mean values of the inner arrays
# print(np.around(np.mean(np.mean(qTable.transpose(),2),2), 2))
#Visualize_QTable(qTable,"Q-Learning Q-Table")

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