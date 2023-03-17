import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env
np.set_printoptions(threshold=np.inf, linewidth=140)
rows = 10
cols = 5
# learning rate
alpha = 0.1
# discount/long-term reward factor
gamma = 0.9
epsilon = 0.9
# the action to pass to the environment to take (position)
action = np.array([0,0])

NumberOfIterations = 100

# qTable: an array of the size of the grid (outer) where each element also has size of the grid (inner) (17x9)x(17x9)
# each element in the inner contains the q-value for starting at outer state and moving to inner state
qTable = np.zeros((rows,cols,rows,cols))

q_table_sum_QLearning = np.zeros((rows, cols))
q_table_sum_SarsaLearning = np.zeros((rows, cols))
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

def Average_And_Visualize_QTable(QTableSum,QTableDivide,title):
    avg_qTable = np.divide(QTableSum, QTableDivide)
    cmap = plt.cm.get_cmap('Greens')
    plt.title(title)
    plt.imshow(avg_qTable.transpose(), cmap=cmap)
    plt.colorbar(shrink = 0.5)
    plt.show()

def QTableSum(QTable, QTableSum):
    for i in range(rows):
        for j in range(cols):
            QTableSum = np.add(QTableSum,QTable[i][j])
    return QTableSum

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

def updateQTable(Q, prevPos, position, reward):
    """
    Q-Learning or sarsa, not 100% sure if correct or which one.
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
    Q-Learning or sarsa, not 100% sure if correct or which one.
    Updates the qTable by using the qfunction
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
env = gym.make('ML_Env/ML_RL_Env-v0', numRows=rows, numCols=cols, timeStep = 1, episodeLength = 10000)
env.reset()
rewards = np.array([])
rewardcumulative = np.array([])
cumulative = np.array([])
while epsilon >= 0:
    for index in range(NumberOfIterations):
        terminated = False
        while not terminated:
            # get the results from taking an action
            observation, reward, terminated, truncated, info = env.step(action)
            # update qTable using results from taking action
            rewards = np.append(rewards,float(reward))
            action = updateQTable(qTable, observation.get('prevPos'), observation.get('position'), reward)
            if terminated or truncated:
                # print("Higher q-values represent greater rewards from that position")
                # # taking the mean values of the inner arrays
                # print(np.around(np.mean(np.mean(qTable.transpose(),2),2), 2))
                #Visualize_QTable(qTable,"Q-Learning Q-Table")
                q_table_sum_QLearning = QTableSum(qTable, q_table_sum_QLearning)
                qTable = np.zeros((rows,cols,rows,cols))
                if rewardcumulative.size == 0:
                    rewardcumulative = rewards
                else:
                    rewardcumulative = np.vstack([rewardcumulative, rewards])
                rewards = np.array([])
                env.reset()
                break
    if cumulative.size == 0:
        cumulative = np.cumsum(rewardcumulative.mean(0),0)
    else:
        cumulative = np.vstack([cumulative, np.cumsum(rewardcumulative.mean(0),0)])
    rewardcumulative = np.array([])
    epsilon -=0.2

font = {'weight' : 'bold'}
plt.rc('font', **font)
for item in cumulative:
    plt.plot(item, label = "\u03B5 = " + str(round(epsilon, 1)))
    epsilon-=0.2
plt.xlabel("Time Step in Episode")
plt.ylabel("Cumulative Avg Reward across All Eps")
plt.legend()
plt.show()

epsilon = 1
env.reset()
rewards = np.array([])
rewardcumulative = np.array([])
cumulative = np.array([])
while epsilon >= 0:
    for index in range(NumberOfIterations):
        terminated = False
        while not terminated:
            # get the results from taking an action
            observation, reward, terminated, truncated, info = env.step(action)
            # update qTable using results from taking action
            rewards = np.append(rewards,float(reward))
            action = updateQTableQLearning(qTable, observation.get('prevPos'), observation.get('position'), reward)
            if terminated or truncated:
                # print("Higher q-values represent greater rewards from that position")
                # # taking the mean values of the inner arrays
                # print(np.around(np.mean(np.mean(qTable.transpose(),2),2), 2))
                q_table_sum_SarsaLearning = QTableSum(qTable, q_table_sum_SarsaLearning)
                qTable = np.zeros((rows,cols,rows,cols))
                if rewardcumulative.size == 0:
                    rewardcumulative = rewards
                else:
                    rewardcumulative = np.vstack([rewardcumulative, rewards])
                rewards = np.array([])
                env.reset()
                break
    if cumulative.size == 0:
        cumulative = np.cumsum(rewardcumulative.mean(0),0)
    else:
        cumulative = np.vstack([cumulative, np.cumsum(rewardcumulative.mean(0),0)])
    print(cumulative)
    rewardcumulative = np.array([])
    epsilon -=0.2

epsilon = 1.0
font = {'weight' : 'bold'}
plt.rc('font', **font)
for item in cumulative:
    plt.plot(item, label = "\u03B5 = " + str(round(epsilon, 1)))
    epsilon-=0.2
plt.xlabel("Time Step in Episode")
plt.ylabel("Cumulative Avg Reward across All Eps")
plt.legend()
plt.show()
qTableDivide = np.full((rows,cols), rows*cols*NumberOfIterations)
Average_And_Visualize_QTable(q_table_sum_QLearning,qTableDivide,"QTable of Q-Learning")
Average_And_Visualize_QTable(q_table_sum_SarsaLearning,qTableDivide,"QTable of Sarsa-Learning")