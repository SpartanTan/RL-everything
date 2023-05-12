import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# define constants for the potential field
K_GOAL = 1.0  # attraction constant
K_OBSTACLE = 100.0  # repulsion constant
OBSTACLE_THRESHOLD = 1.0  # only consider obstacles closer than this
DT = 0.01  # time step
SPEED = 0.1  # robot speed
Kp = 1.0  # proportional gain


def calculate_potential_field(goal, obstacles):
    # create a grid of points
    x = np.arange(-10, 10, 0.1)
    y = np.arange(-10, 10, 0.1)
    X, Y = np.meshgrid(x, y)
    
    # calculate the attraction field
    dx_goal = goal[0] - X
    dy_goal = goal[1] - Y
    distance_goal = np.hypot(dx_goal, dy_goal)
    U_goal = 0.5 * K_GOAL * distance_goal**2

    # calculate the repulsion field for each obstacle
    U_obstacle = 0
    for ox, oy in obstacles:
        dx_obstacle = ox - X
        dy_obstacle = oy - Y
        distance_obstacle = np.hypot(dx_obstacle, dy_obstacle)
        repulsion = np.where(distance_obstacle < OBSTACLE_THRESHOLD,
                             0.5 * K_OBSTACLE * (1/distance_obstacle - 1/OBSTACLE_THRESHOLD)**2, 
                             0)
        U_obstacle += repulsion

    # total potential field is the sum of attraction and repulsion fields
    U = U_goal + U_obstacle
    return U


def gradient_descent(potential_field, start):
    # take the negative gradient of the potential field to get a vector field
    V = -np.array(np.gradient(potential_field))

    # interpolate the vector field at the robot's current position
    X, Y = np.mgrid[0:potential_field.shape[0], 0:potential_field.shape[1]]
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    vx = griddata(positions, V[0].ravel(), (start[1], start[0]), method='cubic')
    vy = griddata(positions, V[1].ravel(), (start[1], start[0]), method='cubic')

    return vx, vy

class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

    def move(self, speed, omega):
        self.x += speed * np.cos(self.theta) * DT
        self.y += speed * np.sin(self.theta) * DT
        self.theta += omega * DT

def robot_control(goal, obstacles):
    # calculate the potential field
    U = calculate_potential_field(goal, obstacles)

    # start at some position
    robot = Robot()

    # record the trajectory of the robot
    robot_trajectory = []

    # robot control loop
    for _ in range(10000):  # limit the number of iterations to avoid infinite loop
        # record the current position
        robot_trajectory.append([robot.x, robot.y])

        # calculate the desired velocity
        vx, vy = gradient_descent(U, [robot.y, robot.x])

        # calculate desired orientation
        theta_desired = np.arctan2(vy, vx)

        # calculate error in orientation
        e_theta = theta_desired - robot.theta

        # wrap angle error to [-pi, pi]
        e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))

        # calculate angular velocity command using proportional control
        omega = Kp * e_theta

        # move the robot
        robot.move(SPEED, omega)

        # stop if the goal is reached
        if np.linalg.norm([robot.x - goal[0], robot.y - goal[1]]) <= 0.1:
            break

    # convert to numpy array for easier manipulation
    robot_trajectory = np.array(robot_trajectory)

    # plot the reference trajectory (goal)
    plt.plot(goal[0], goal[1], 'r*', label='Goal')

    # plot the obstacles
    for obstacle in obstacles:
        plt.plot(obstacle[0], obstacle[1], 'ko', label='Obstacles')

    # plot the robot's trajectory
    plt.plot(robot_trajectory[:, 0], robot_trajectory[:, 1], 'b', label='Robot trajectory')

    # configure the plot
    plt.legend()
    plt.grid(True)
    plt.show()


# define goal and obstacles
goal = np.array([5.0, 5.0])
obstacles = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

# run the control loop
robot_control(goal, obstacles)
