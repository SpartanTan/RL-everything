import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

import gymnasium as gym
from gymnasium import spaces

from env_path import Path
from RK4 import ATR_RK4

class SimFactoryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, dt=0.1):
        self.path:Path = Path(trajectory_point_interval=0.1,
                No=8, Nw=8, Lp=6, mu_r=0.25, sigma_d=0.8, shift_distance=1)
        # take the first waypoint and the first yaw angle as the atr initial state
        init_state = np.array([self.path.waypoints[0][0], self.path.waypoints[0][1], self.path.yaw_angles[0]])
        self.atr = ATR_RK4(init_state=init_state, dt=dt)       
        
        pass
    
    def step(self, action):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        self.path.render(True)
        plt.show()
    
    def render(self):
        pass
    
    def reset(self):
        pass
