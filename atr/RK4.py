import numpy as np
import matplotlib.pyplot as plt

class ATR_RK4():
    
    def __init__(self,
                 wheel_radius=0.125,
                 track_width=0.48,
                 init_state = np.array([0, 0, 0], dtype=float),
                 dt=0.1):
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        self.state = init_state
        self.dt = dt
        
        
    def robot_pose_derivative(self, state:np.ndarray, wheel_speeds:list):
        x, y, theta = state
        left_wheel_speed, right_wheel_speed = wheel_speeds

        # Compute the linear and angular velocities
        linear_velocity = self.wheel_radius * (left_wheel_speed + right_wheel_speed) / 2
        angular_velocity = self.wheel_radius * \
            (right_wheel_speed - left_wheel_speed) / self.track_width

        # Calculate the change in x, y, and theta
        dx = linear_velocity * np.cos(theta)
        dy = linear_velocity * np.sin(theta)
        dtheta = angular_velocity

        return np.array([dx, dy, dtheta])

    def runge_kutta_4_step(self, wheel_speeds:list, method:str="RK4"):
        """
        simulate the robot's movement
        
        ### Parameters
        - `wheel_speeds`: [wr, wl]. Note that the sequnece of input is [right, left], however the algorithm
        used here is [left, right], so have to switch the input first. 
        
        ### Returns
        - `state`: [x, y, theta]
        - `linear_velocity`: linear velocity m/s
        - `angular_velocity`: angular velocity rad/s
        """
        
        # switch wheel speed sequence
        wheel_speeds = [wheel_speeds[1], wheel_speeds[0]] # now wheel_speed is [wl, wr]
        left_wheel_speed, right_wheel_speed = wheel_speeds
        linear_velocity = self.wheel_radius * (left_wheel_speed + right_wheel_speed) / 2
        angular_velocity = self.wheel_radius * \
            (right_wheel_speed - left_wheel_speed) / self.track_width
        if method == "simple":
            k1 = self.robot_pose_derivative(self.state, wheel_speeds)
            self.state = self.state + self.dt * k1
            return np.copy(self.state), linear_velocity, angular_velocity
        else:  
            k1 = self.robot_pose_derivative(self.state, wheel_speeds)
            k2 = self.robot_pose_derivative(self.state + self.dt / 2 * k1, wheel_speeds)
            k3 = self.robot_pose_derivative(self.state + self.dt / 2 * k2, wheel_speeds)
            k4 = self.robot_pose_derivative(self.state + self.dt * k3, wheel_speeds)

            self.state = self.state + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x_normalized = np.arctan2(np.sin(self.state[2]), np.cos(self.state[2]))
            self.state[2] = x_normalized
            return np.copy(self.state), linear_velocity, angular_velocity
    
    def reset(self, position:np.ndarray):
        self.state = position