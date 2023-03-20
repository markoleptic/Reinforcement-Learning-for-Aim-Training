from Algos import Algos
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import traceback

class FileModified():

    def __init__(self, file_path, algo_name):
        self.file_path = file_path
        self.callback = self.file_modified
        self.modifiedOn = os.path.getmtime(file_path)
        self.algo_name = algo_name
        if self.algo_name == "sarsa":
            self.algo = Algos("QTable_Sarsa_Live.npy", numRows=11, numCols=5, alpha=0.8, gamma=0.9, epsilon=0.5)
        elif self.algo_name == "qlearning":
            self.algo = Algos("QTable_QLearning.npy", numRows=11, numCols=5, alpha=0.8, gamma=0.9, epsilon=0.5)
        # initialize S
        self.state = (5,3)
        # Choose A from S using epsilon-greedy policy
        self.action = self.algo.getNextAction(self.state, self.algo.epsilon)

    def start(self):
            try:
                while (True):
                    time.sleep(0.1)
                    modified = os.path.getmtime(self.file_path)
                    if modified != self.modifiedOn:
                        self.modifiedOn = modified
                        if self.callback():
                            break
            except Exception as e:
                print(traceback.format_exc())
                time.sleep(0.1)
                self.start()
    
    # Update the QTable after detected a file change, which means action was taken
    def file_modified(self):
        # Oberve R, S'
        results = np.loadtxt("Accuracy.csv", delimiter=',')
        self.checkEndEpisode(results)
        # results include the previous state, in case ordering was changed in game
        self.state = (int(results[0]), int(results[1]))
        state_2 = (int(results[2]), int(results[3]))
        self.action = state_2
        reward = int(results[4])
        if (reward == 0):
            print("Results read from Accuracy.csv: ", self.state, state_2, " Hit")
            reward = -1
        else:
            print("Results read from Accuracy.csv: ", self.state, state_2, " Miss")
            reward = 0
        self.algo.episodeRewards = np.append(self.algo.episodeRewards, float(reward))
        # Choose A from S using epsilon-greedy policy
        action_2 = self.algo.getNextAction(state_2, self.algo.epsilon)
        # update QTable
        if self.algo_name == "sarsa":
            self.algo.updateQTable_Sarsa(self.state, self.action, state_2, action_2, reward, self.algo.epsilon)
        elif self.algo_name == "qlearning":
            self.algo.updateQTable_QLearning(self.state, self.action, state_2, action_2, reward, self.algo.epsilon)
        # Push next spawn location to game
        self.printNextSpawnLocation("SpawnLocation.txt", action_2)
        self.action = action_2
        self.state = state_2
        if self.algo_name == "sarsa":
            self.algo.saveQTable("QTable_Sarsa_Live.npy")
        elif self.algo_name == "qlearning":
            self.algo.saveQTable("QTable_QLearning_Live.npy")
        return False

    def checkEndEpisode(self, results):
        if (results.__len__() == 0):
            qTableDivide = np.full((self.algo.rows, self.algo.cols), self.algo.rows * self.algo.cols)
            if self.algo_name == "sarsa":
                self.algo.q_table_sum_SarsaLearning = self.algo.updateQTableSum(self.algo.q_table_sum_SarsaLearning)
                self.algo.Average_And_Visualize_QTable(self.algo.q_table_sum_SarsaLearning, qTableDivide, "QTable of Sarsa-Learning")
            elif self.algo_name == "qlearning":
                self.algo.q_table_sum_QLearning = self.algo.updateQTableSum(self.algo.q_table_sum_QLearning)
                self.algo.Average_And_Visualize_QTable(self.algo.q_table_sum_QLearning, qTableDivide, "QTable of Q-Learning")
            self.algo.plotRewards(self.algo.episodeRewards.cumsum(0), False, self.algo.epsilon)
            exit()
            
    # This function saves a new spawn location to "SpawnLocation.txt" after updating the QTable in response to a file change
    def printNextSpawnLocation(self, path, location):
        np.savetxt(path, location, delimiter=",")
        print("Next Spawn Location written to SpawnLocation.txt: ", location)
    
# Start monitoring for file changes
fileModifiedHandler = FileModified(r"Accuracy.csv", algo_name="sarsa")
fileModifiedHandler.start()