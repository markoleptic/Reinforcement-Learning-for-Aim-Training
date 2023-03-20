import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_env
import pygame
import numpy as np

Profile1Matrix = [[0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6, 0.7, 0.7],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.7, 0.8, 0.7],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.6, 0.7, 0.6],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.7, 0.6, 0.6, 0.6, 0.6],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.5, 0.5, 0.5, 0.5],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.7, 0.5, 0.4, 0.4, 0.4],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.6, 0.5, 0.4, 0.0, 0.3],
                  [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.5, 0.4, 0.3, 0.3]]

Profile2Matrix = [[ 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6],
                  [ 0.8, 0.8, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.5, 0.5, 0.6],
                  [ 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.7, 0.6, 0.4, 0.4, 0.5],
                  [ 0.8, 0.8, 0.8, 0.9, 1.0, 0.8, 0.7, 0.4, 0.0, 0.3, 0.3],
                  [ 0.8, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 0.4, 0.3, 0.3, 0.4]]

class ML_RL_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, numRows=17, numCols=9, timeStep = 1, episodeLength = 35000):
        self.numRows = numRows
        self.numCols = numCols
        self.timeStep = timeStep
        self.currentTimeStep = 0
        self.episodeLength = episodeLength
        self.observation_space = Dict(
            {
                "prevPos": Box(0, np.array([numRows-1, numRows-1]), shape=(2,), dtype=int),
                "position": Box(0, np.array([numRows-1, numCols-1]), shape=(2,), dtype=int),
            }
        )
        self.action_space = Box(0, np.array([numRows-1, numCols-1]), shape=(2,), dtype=int)
        self.numTargetSpawns = np.array([numRows-1, numCols-1])

        """
        Create a profile for a person with given accuracy in every state. This accuracy number
        will then be used in incrementReward(). The agent's chance to recieve the reward is
        (100 - (accuracy * 100))% chance. 
        Could take this further and slowly increment the accuracy by 0.01 for each time they encounter
        the state and successfully shoot it (agent does not recieve reward).
        """
        self.accuracyMatrix = np.zeros((self.numRows,self.numCols))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"position": self.position,
                "prevPos": self.prevPos,
                }

    # not really sure what getobs vs getinfo is supposed to do
    def _get_info(self):
        return {"position": self.position,
                "prevPos": self.prevPos,
                "timestep": self.currentTimeStep,
                }
    
    # return a random location in the state space
    def chooseRandomLocation(self):
        return np.array([np.random.randint(0, self.numRows), np.random.randint(0, self.numCols)])

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's start location at random
        self.position = self.chooseRandomLocation()
        self.prevPos = self.position
        self.numTargetSpawns = np.zeros((self.numCols, self.numRows)).transpose()
        self.currentTimeStep = 0
        self.accuracyMatrix = np.array(Profile2Matrix).transpose()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    # get reward from traveling to a position
    def getReward(self,position):
        x,y = position
        threshold = self.accuracyMatrix[x][y]
        return -threshold
        # if np.random.uniform(0, 1) > threshold:
        #     return 0
        # return -1

    # updates environment based on the action
    def step(self, action):
        # increment time step
        self.currentTimeStep+=self.timeStep

        # update position
        self.prevPos = self.position
        self.position = action

        # get reward from performing action
        reward = self.getReward(self.position)

        #increment the number of target spawns (debug purposes)
        x,y = action
        self.numTargetSpawns[x,y]+=1

        # terminate condition
        terminated = (self.currentTimeStep >= self.episodeLength / self.timeStep)
        # if terminated:
        #     print("Number of target spawns per location:")
        #     print(self.numTargetSpawns.transpose())
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(( 50*self.numRows, 50*self.numCols))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(( 50*self.numRows, 50*self.numCols))
        canvas.fill((255, 255, 255))
        pix_row = self.window.get_rect().width / self.numRows
        pix_col = self.window.get_rect().height / self.numCols
        # current position
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self.prevPos*(pix_row,pix_col),
                (pix_row, pix_col),
            ),
        )
        # prev position
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                self.position*(pix_row,pix_col),
                (pix_row, pix_col),
            ),
        )
        # horizontal lines
        for x in range(int(pix_row) + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_row * x),
                (50*self.numRows, pix_row * x),
                width=3,
            )
        # vertical lines
        for y in range(int(pix_col) + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_col * y, 0),
                (pix_col * y, 50*self.numCols),
                width=3,
            )
        number_font = pygame.font.SysFont( None, 16 )
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            for x in range(self.numRows):
                for y in range(self.numCols):
                    self.window.blit(number_font.render(str(int(self.numTargetSpawns[x, y])), True, (255,0,0)), ((50*x)+5,(50*y)+5))
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
      if self.window is not None:
        pygame.display.quit()
        pygame.quit()