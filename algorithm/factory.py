import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from map.Nona2Grid import Nona2Grid

class FactoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, start:list, goal:list, render_mode=None):
        self.start = start
        self.goal = goal
        self.render_mode = render_mode
        self.continusous = False
        self.tau = 0.02  # seconds between state updates
        map_dir = "/home/zhicun/code/atr_rl/Code/python_tools/map/config/nona_description.json"
        path_dir = "/home/zhicun/code/atr_rl/Code/python_tools/map/config/nodes_tuve.json"
        self.grid = Nona2Grid(map_dir, path_dir, grid_resolution=0.2) # ndarrary 40 * 65, upside down
        self.height_bound = self.grid.grid_map.shape[0] # 40
        self.width_bound = self.grid.grid_map.shape[1] # 65
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0,0]), high=np.array([self.height_bound-1, self.width_bound-1]), shape=(2,), dtype=int),
                "targer": spaces.Box(low=np.array([0,0]), high=np.array([self.height_bound-1, self.width_bound-1]), shape=(2,), dtype=int),
            }
        )


        if self.continusous:
            self.action_space = spaces.Box()
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1,0]),
            1: np.array([0,1]),
            2: np.array([-1,0]),
            3: np.array([0,-1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
    
    
    def _get_obs(self):
        """
        Compute the observation from states. May be used in `reset()` and `step()`.
        Will return a dictionary, contains the agent's and the target's location.

        ### Return
        - a dictionary contains the agent's and the target's location.

        ### Examples
        >>> agent_location = self._get_obs()["agent"] 
        (1,2)
        >>> target_location = self._get_obs()["target"]
        (3,4)
        """
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self) -> dict:
        """
        Helper function, return a dictionary, contains the manhattan distance between agent and target.

        ### Return
        - a dictionary contains the manhattan distance between agent and target.
        ### Examples
        >>> self._get_info()["distance"]
        {"distance": 4}
        """
        return {"distance": np.linalg.norm(self._agent_location - self._target_location)}
    
    def reset(self, seed=None, options=None)->tuple:
        super().reset(seed=seed)
        # Choose the agent's location uniformaly at random
        x_grid, y_grid = self.grid.point_to_grid_coordinates(self.start[0], self.start[1])
        self._agent_location = np.array([x_grid, y_grid])
        x_grid, y_grid = self.grid.point_to_grid_coordinates(self.goal[0], self.goal[1])
        self._target_location = np.array([x_grid, y_grid])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def check_collision(self, location, grid_coord=True)->bool:
        """
        #### Parameters
        - `location`: a list, contains the location of the agent in real coordinate frame
        """
        ifcolli = self.grid.check_collision(location[0], location[1], grid_coord) == 1
        return ifcolli

    def step(self, action)->tuple:
        """
        `step` method contains most of the logic of the environment. 
        Maps the action (element of {0,1,2,3}) to the direction we walk in the gridworld

        ### Parameters
        - action: an integer in {0,1,2,3}, corresponding to the action to be taken.

        ### Return
        - a tuple, (observation: dict, reward: float, done: bool, info: dict)

        ### Examples
        >>> self.step(0) # agent should move right on grid
        """
        direction = self._action_to_direction[action]
        self._agent_location = self._agent_location + direction

        min_values = np.array([0,0])
        max_values = np.array([self.height_bound, self.width_bound])
        outofbound_terminated = not (np.all((self._agent_location >= min_values) & (self._agent_location <= max_values)))
        collsion_terminated = self.check_collision(self._agent_location)
        terminated = outofbound_terminated or collsion_terminated
        reward = 1 if terminated else 0
        observation = self._get_obs()

        return observation, reward, terminated, False, info

    def render(self):
        self.grid.render()
        plt.show()

if __name__ == "__main__":
    env = FactoryEnv(start=[0.66, 5.09], goal=[10.31, 3.68])
    observation, info = env.reset()
    env.render()
    
    env.step(3)
    