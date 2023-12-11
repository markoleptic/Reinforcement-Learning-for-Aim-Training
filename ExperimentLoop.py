from Algos import Algos
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import matplotlib.pyplot as plt
import numpy as np
import ML_Env


class ExperimentLoop:
    def __init__(self, num_episodes = 1000, num_rows = 11, num_cols = 5, time_step = 1, episode_length = 300):
        self.num_episodes = num_episodes
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.env = gym.make(
            "ML_Env/ML_RL_Env-v0",
            render_mode=None,
            numRows=num_rows,
            numCols=num_cols,
            timeStep=time_step,
            episodeLength=episode_length)
        
    def clamp(self, num, min_value, max_value):
        return max(min(num, max_value), min_value)
    
    def SMA(self, data, step):
        w = np.repeat(1, step) / step
        result = np.convolve(w, data, mode="valid")
        return result
    
    def perform_random_simulation(self):
        # Choose A from S using epsilon-greedy policy
        action = (np.random.randint(0, self.num_rows), np.random.randint(0, self.num_cols))
        totalCumRewards = np.array([])
        episodeCumRewards = np.array([])
        for index in range(self.num_episodes):
            self.env.reset()
            terminated = False
            episodeRewards = np.array([])
            while not terminated:
                observation, reward, terminated, truncated, info = self.env.step(action)
                episodeRewards = np.append(episodeRewards, float(reward))
                # Randomly choose a location
                action = (np.random.randint(0, self.num_rows), np.random.randint(0, self.num_cols))
                if terminated or truncated:
                    episodeCumRewards = np.append(episodeCumRewards, (episodeRewards.sum(0)))
                    break
        return episodeCumRewards
    
    def perform_sarsa_simulation(self, alpha=1, gamma=0.9, epsilon=0.9):
        algos = Algos(numRows=self.num_rows, numCols=self.num_cols, alpha=alpha, gamma=gamma, epsilon=epsilon)
        # initialize S
        state = (5, 3)
        # Choose A from S using epsilon-greedy policy
        action = algos.getNextAction(state, algos.epsilon)
        totalCumRewards = np.array([])
        episodeCumRewards = np.array([])
        for index in range(self.num_episodes):
            self.env.reset()
            terminated = False
            episodeRewards = np.array([])
            algos.alpha = 1 - (float(index) / self.num_episodes)
            while not terminated:
                observation, reward, terminated, truncated, info = self.env.step(action)
                episodeRewards = np.append(episodeRewards, float(reward))
                state = tuple(observation.get("prevPos"))
                state_2 = tuple(observation.get("position"))
                # Choose A from S using epsilon-greedy policy
                action_2 = algos.getNextAction(state_2, algos.epsilon)
                algos.updateQTable_Sarsa(
                    state, action, state_2, action_2, reward, algos.epsilon
                )
                action = action_2
                state = state_2
                if terminated or truncated:
                    algos.q_table_sum_SarsaLearning = algos.updateQTableSum(
                        algos.q_table_sum_SarsaLearning
                    )
                    episodeCumRewards = np.append(episodeCumRewards, (episodeRewards.sum(0)))
                    break
        return episodeCumRewards, algos.q_table_sum_SarsaLearning
    

    def perform_qlearning_simulation(self, alpha=1, gamma=0.9, epsilon=0.9):
        algos = Algos(numRows=self.num_rows, numCols=self.num_cols, alpha=alpha, gamma=gamma, epsilon=epsilon)
        # initialize S
        state = (5, 3)
        # Choose A from S using epsilon-greedy policy
        action = algos.getNextAction(state, algos.epsilon)
        totalCumRewards = np.array([])
        episodeCumRewards = np.array([])
        for index in range(self.num_episodes):
            self.env.reset()
            terminated = False
            episodeRewards = np.array([])
            algos.alpha = 1 - (float(index) / self.num_episodes)
            while not terminated:
                observation, reward, terminated, truncated, info = self.env.step(action)
                episodeRewards = np.append(episodeRewards, float(reward))
                state = tuple(observation.get("prevPos"))
                state_2 = tuple(observation.get("position"))
                # Choose A from S using epsilon-greedy policy
                action_2 = algos.getNextAction(state_2, algos.epsilon)
                algos.updateQTable_QLearning(
                    state, action, state_2, action_2, reward, algos.epsilon
                )
                action = action_2
                state = state_2
                if terminated or truncated:
                    algos.q_table_sum_QLearning = algos.updateQTableSum(
                        algos.q_table_sum_QLearning
                    )
                    episodeCumRewards = np.append(episodeCumRewards, (episodeRewards.sum(0)))
                    break
        return episodeCumRewards, algos.q_table_sum_QLearning

if __name__ == "__main__":
    num_episodes = 5000
    loop = ExperimentLoop(num_episodes, 11, 5, 1, 300)
    random_total_rewards = loop.perform_random_simulation()
    sarsa_total_rewards_09, sarsa_qtable_sum = loop.perform_sarsa_simulation(1, 0.9, 0.9)
    sarsa_total_rewards_07, _ = loop.perform_sarsa_simulation(1, 0.9, 0.7)
    sarsa_total_rewards_05, _ = loop.perform_sarsa_simulation(1, 0.9, 0.5)
    qlearning_total_rewards_09, qlearning_qtable_sum = loop.perform_qlearning_simulation(1, 0.9, 0.9)
    qlearning_total_rewards_07, _ = loop.perform_qlearning_simulation(1, 0.9, 0.7)
    qlearning_total_rewards_05, _ = loop.perform_qlearning_simulation(1, 0.9, 0.5)

    print("RandomTotalRewards", np.sum(random_total_rewards, 0))
    print("SarsaTotalRewards \u03B5 = 0.9", np.sum(sarsa_total_rewards_09, 0))
    print("SarsaTotalRewards \u03B5 = 0.5", np.sum(sarsa_total_rewards_05, 0))
    print("QLearningTotalRewards \u03B5 = 0.9", np.sum(qlearning_total_rewards_09, 0))
    print("QLearningTotalRewards \u03B5 = 0.5", np.sum(qlearning_total_rewards_05, 0))

    plt.plot(loop.SMA(random_total_rewards, 100), label="SMA of Rewards - Random")
    plt.plot(loop.SMA(sarsa_total_rewards_09, 100), label="SMA of Rewards - Sarsa")
    plt.plot(loop.SMA(qlearning_total_rewards_09, 100), label="SMA of Rewards - QLearning")
    font = {"weight": "bold"}
    plt.rc("font", **font)
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards for Each Episode")
    plt.title("SMA of Sum of Rewards for Each Episode")
    plt.legend()
    plt.show()

    plt.plot(loop.SMA(random_total_rewards, 100), label="SMA of Rewards - Random")
    plt.plot(loop.SMA(sarsa_total_rewards_09, 100), label="SMA of Rewards - Sarsa \u03B5 = 0.9")
    plt.plot(loop.SMA(sarsa_total_rewards_07, 100), label="SMA of Rewards - Sarsa \u03B5 = 0.7")
    plt.plot(loop.SMA(sarsa_total_rewards_05, 100), label="SMA of Rewards - Sarsa \u03B5 = 0.5")
    plt.plot(loop.SMA(qlearning_total_rewards_09, 100), label="SMA of Rewards - QLearning \u03B5 = 0.9")
    plt.plot(loop.SMA(qlearning_total_rewards_07, 100), label="SMA of Rewards - QLearning \u03B5 = 0.7")
    plt.plot(loop.SMA(qlearning_total_rewards_05, 100),label="SMA of Rewards - QLearning \u03B5 = 0.5")
    font = {"weight": "bold"}
    plt.rc("font", **font)
    plt.xlabel("Episode")
    plt.ylabel("Sum of Rewards for Each Episode")
    plt.title("SMA of Sum of Rewards for Each Episode with Varying \u03B5")
    plt.legend()
    plt.show()

    algos = Algos(numRows=11, numCols=5, alpha=1, gamma=0.9, epsilon=0.9)
    qTableDivide = np.full((algos.rows, algos.cols), algos.rows * algos.cols * num_episodes)
    algos.Average_And_Visualize_QTable(sarsa_qtable_sum, qTableDivide, "QTable of Sarsa-Learning")
    algos.Average_And_Visualize_QTable(qlearning_qtable_sum, qTableDivide, "QTable of Q-Learning")
