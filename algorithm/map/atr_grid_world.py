import numpy as np
import gymnasium as gym
import pygame
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        """
        - render_mode: None, "human", "rgb_array"
        - size: size of the grid world
        """
        self.size = size # size of the square grid world
        self.window_size = 512 # size of the Pygame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., 'size'}^2, i.e. MultiDiscrete([size, size]).
        # MultiDiscrete is a class in gym, https://gymnasium.farama.org/api/spaces/fundamental/#multidiscrete
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
                "targer": spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(4) # 4 actions: up, down, left, right

        """
        The following dictionary maps abstract actions from `self.action_space` to the direction we will walk 
        in if that action is taken.
        I.e. 0 corresponding moving right, 1 to `up`
        """
        self._action_to_direction = {
            0: np.array([1,0]),
            1: np.array([0,1]),
            2: np.array([-1,0]),
            3: np.array([0,-1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference to the window that we draw to.
        `self.clocl` will be a cloack that is used to ensure the environment is rendered at the correct framerate in 
        human-mode. They will remain `None` until human-mode is used for the first time.
        """
        self.window = None
        self.clock = None
    
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
        """
        `Reset` method will be called to initialize a new episode. `Step` may not be called before `reset` has been called. 
        Moreover, `reset` should be called whenever a `done` signal has been issued.
        This method should return a tuple of the initial observation and some auxilliary information.

        ### Parameters
        - seed: seed for the random number generator. Recommended to use `self.np_random`
        ### return
        - return a tuple, (observation: dict, info: dict)

        ### Examples
        >>> observation, info = self.reset()
        >>> observation
        {"agent": (1,2), "target": (3,4)}
        >>> info
        {"distance": 4}
        """

        # Need this lilne to sed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformaly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location

        # Sample the target's location randomly until it does not conincide with the agent's location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

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
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        observaton = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observaton, reward, terminated, False, info