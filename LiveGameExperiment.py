from Algos import Algos
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env
from FileModHandler import FileModified

# TODO This function should update the QTable after detected a file change, and call printNextSpawnLocation
def file_modified():
    array = np.loadtxt("Accuracy.csv", delimiter=',')
    print("Location: (", array[0], array[1], ") Hit: ", array[2])
    printNextSpawnLocation("SpawnLocation.txt", np.array([0,0]))
    return False

# TODO This function should save a new spawn location to "SpawnLocation.txt" after reading updating the QTable in response to a file change
def printNextSpawnLocation(path, location):
    np.savetxt(path, location, delimiter=",")
    print("Next Spawn Location written to SpawnLocation.txt: ", location)

algos = Algos(numRows=11, numCols=6, alpha=0.1, gamma=0.9, epsilon=1)

# TODO: Setup 

# Start monitoring for file changes
fileModifiedHandler = FileModified(r"Accuracy.csv", file_modified)
fileModifiedHandler.start()