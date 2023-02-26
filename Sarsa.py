import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import numpy as np
import ML_Env
np.set_printoptions(threshold=np.inf, linewidth=140)
rows = 17
cols = 9
# learning rate
alpha = 0.1
# discount/long-term reward factor
gamma = 0.9
epsilon = 1

# the action to pass to the environment to take (position)
action = np.array([0,0])

# qTable: an array of the size of the grid (outer) where each element also has size of the grid (inner) (17x9)x(17x9)
# each element in the inner contains the q-value for starting at outer state and moving to inner state
qTable = np.zeros((rows,cols,rows,cols))

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

def getMaxActionIndex(Q, startPosition):
    """
    Returns the full (4) indices of next action (position) to take based on the policy implemented (epsilon).
    Args:
        Q: the 4-dimensional q-table
        startPosition: the starting state/position
    """
    return(np.unravel_index(np.argmax(Q[startPosition]), Q.shape))

# TODO: properly implement epsilon
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
    # optimal qvalue of the next state
    optimalNext = qTable[getMaxActionIndex(qTable,(x2,y2))]
    # newqValue(prevPos -> action -> position) = oldqValue + alpha(gamma * nextqValue(position -> bestAction -> nextPosition) - oldqValue)
    qTable[x1,y1,x2,y2] = qTable[x1,y1,x2,y2] + alpha * (reward + gamma * optimalNext - qTable[x1,y1,x2,y2])

# -----------------------
# --- experiment loop ---
# -----------------------

# create ML_RL_Env environment, everything besides first parameter is optional
# set render_mode to "human" to view the game
env = gym.make('ML_Env/ML_RL_Env-v0', render_mode="rgb_array", numRows=17, numCols=9, timeStep = 0.35, episodeLength = 350000)

env.reset()
terminated = False
while not terminated:
    # get the results from taking an action
    observation, reward, terminated, truncated, info = env.step(action)
    # update qTable using results from taking action
    updateQTable(qTable, observation.get('prevPos'), observation.get('position'), reward)
    action = getNextAction(observation.get('position'))
    if terminated or truncated:
        print("Higher q-values represent greater rewards from that position")
        # only using transpose to make it more readable
        # print(np.around(qTable[0,0].transpose(), 2))
        # print('\n')
        # print(np.around(qTable[8,4].transpose(), 2))
        # print('\n')
        # print(np.around(qTable[16,8].transpose(), 2))
        # taking the mean values of the inner arrays
        print(np.around(np.mean(np.mean(qTable.transpose(),2),2), 2))
        break
        observation, info = env.reset()