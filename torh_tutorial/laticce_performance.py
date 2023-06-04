from env_path import Path
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString
from IPython.display import display, clear_output
from RK4 import ATR_RK4
import math
import time
import os
import cProfile

np.set_printoptions(precision=3, suppress=True)

# max speed of ATR, 1m/s
# 8 rad/s 

# Parameters
# initial_state = [0.0, 0.0, 0.0]  # [x, y, theta]

num_of_speeds = 5
v_left_set = np.linspace(0.1, 1.0, 5)  # Range of left wheel speeds rad/s
v_right_set = np.linspace(0.1, 1.0, 5)  # Range of right wheel speeds

v_left_set = np.concatenate(([-0.2, 0], v_left_set))
v_right_set = np.concatenate(([-0.2, 0], v_right_set))

scale = 12
v_left_set *= scale
v_right_set *= scale

dt = 0.1  # Time step
N = int(100/scale)  # Number of steps
WHEEL_RADIUS = 0.125  # Wheel radius
TRACK_WIDTH = 0.48 # track width

# Cost factors
class C:
    K_OFFSET = 20 # 3.5
    K_COLLISION = 1000.0
    K_walked_distance = -10
    K_distance_to_goal = 4 # 4
    K_distance_to_obs = 1.0 # 0.0

# Define a path class for convenience
class LPath:
    def __init__(self, N=100):
        self.x = []
        self.y = []
        self.theta = []
        self.v = 0.0
        self.w = 0.0
        self.v_left = 0.0
        self.v_right = 0.0
        self.end_distance = 0.0
        self.walked_distance = 0.0
        self.cost = 0
        self.collision = False

def is_path_collision(path, obstacles):
    for i in range(len(path.x)):
        for obs in obstacles:
            if np.hypot(path.x[i] - obs[0], path.y[i] - obs[1]) <= (obs[2]):
                return True
    return False

def distance_to_obstacles(point, obstacles):
    distances = []
    for obs in obstacles:
        distances.append(np.hypot(point[0] - obs[0], point[1] - obs[1]))
    return np.min(distances)


def sample_paths_diff_drive_robot(v_left_set, v_right_set, initial_state, dt, N, ref_path, obstacles, goal, polygon, atr:ATR_RK4):
    PATHS = []
    approach_goal = False
    for v_left in v_left_set:
        for v_right in v_right_set:
            # if v_left == 0 and v_right == 0:
            #     continue
            path = LPath(N)
            atr.reset(initial_state)
            x, y, theta = initial_state
            v_right = v_right
            v_left = v_left
            # Calculate linear and angular velocities from wheel speeds

            v = (v_right*WHEEL_RADIUS + v_left*WHEEL_RADIUS) / 2
            w = WHEEL_RADIUS * (v_right - v_left) / TRACK_WIDTH
            
            
            # vel = atr.robot_pose_derivative(atr.state, [v_right, v_left])
            
            for i in range(N):
                # atr.runge_kutta_4_step([v_right, v_left], "simple")
                dx = v * np.cos(theta)
                dy = v * np.sin(theta)
                x += dx * dt
                y += dy * dt
                theta += w * dt
                # x = atr.state[0]
                # y = atr.state[1]
                # theta = atr.state[2]
                
                path.x.append(x)
                path.y.append(y)
                path.theta.append(theta)
            path.walked_distance = np.hypot(x - initial_state[0], y - initial_state[1])
            path.v_left = v_left
            path.v_right = v_right
            # Calculate the cost of the path
            last_point = np.array([path.x[-1], path.y[-1]])
            _, distance_on_line = find_closest_point_on_line(last_point, ref_path)
            obs_collision = is_path_collision(path, obstacles)
            wall_collision = not polygon.contains(Point(last_point))
            if obs_collision or wall_collision:
                continue
            path.collision = 0

            # a list stores the distances to the goal
            distances = []
            for i in range(len(path.x)):
                distance = math.sqrt((path.x[i] - goal[0]) ** 2 + (path.y[i] - goal[1]) ** 2)
                distances.append(distance)
            distance_to_goal = min(distances)
            if distance_to_goal <= 0.2:
                distance_to_goal == 0.2
                approach_goal = True
            if approach_goal:
                mul_scale = np.abs([ int(0.8*scale/ (v_left+0.001)), int(0.8*scale / (v_right+0.001))])
                mul_scale = np.min(mul_scale)
                if mul_scale > 1:
                    path.v_left = v_left * mul_scale
                    path.v_right = v_right * mul_scale
            distance_to_goal = distance_to_goal / 10.0
            # distance_to_goal = np.hypot(x - goal[0], y - goal[1])
            distance_to_obs = distance_to_obstacles(last_point, obstacles)
            if  path.walked_distance == 0:
                not_walking_penalty = 20
            else:
                not_walking_penalty = 0
            path.cost = C.K_OFFSET * distance_on_line + \
                        C.K_COLLISION * path.collision + \
                        C.K_walked_distance * path.walked_distance + \
                        C.K_distance_to_goal * distance_to_goal + \
                        C.K_distance_to_obs * 1/distance_to_obs + not_walking_penalty
            # if approach_goal:
            #     path.cost = C.K_walked_distance * path.walked_distance + C.K_COLLISION * collision

            path.end_distance = distance_on_line
            path.v = v
            path.w = w
            PATHS.append(path)

    return PATHS

def find_closest_point_on_line(point, ref_path):
    # Create LineString object from reference path
    line = LineString(ref_path)
    
    # Create Point object from input point
    p = Point(point)
    
    # Find the closest point on the line
    closest_point_on_line = line.interpolate(line.project(p))
    
    # Calculate the distance to the closest point
    distance = p.distance(closest_point_on_line)

    return closest_point_on_line, distance

STUCK_COUNT = 0

def select_optimal_path(paths):
    global STUCK_COUNT
    min_cost = float('inf')
    optimal_path = None
    optimal_index = 0

    assert len(paths) > 0, "No paths provided"
    for idx, path in enumerate(paths):
        if path.cost < min_cost:
            min_cost = path.cost
            optimal_path = path
            optimal_index = idx

    # print(f"optimal path vel: {optimal_path.v}")
    if optimal_path.v / scale <= 0.02:
        print("select random path")
        optimal_index = np.random.randint(len(paths))
        optimal_path = paths[optimal_index]
        STUCK_COUNT += 1
        C.K_OFFSET -= 0.2
        if C.K_OFFSET < 0:
            C.K_OFFSET = 0    
    else:
        if STUCK_COUNT > 0:
            STUCK_COUNT -= 1
        if STUCK_COUNT < 5:
            C.K_OFFSET = 20
    
    return optimal_path, optimal_index    



def run_loop():
    testPath = Path(trajectory_point_interval=0.1,
                No=8, Nw=8, Lp=6, mu_r=0.25, sigma_d=0.8, shift_distance=1)
    testPath.reset()
    testPath.render()

    state = [testPath.waypoints[0][0], testPath.waypoints[0][1], testPath.yaw_angles[0]]
    atr = ATR_RK4(init_state=state, dt=dt)
    path_obs = [(round(float(item[0][0]),3), round(float(item[0][1]),3), round(item[1], 3)) for item in testPath.obstacles]
    goal_index = 1
    # goal = [testPath.waypoints[1][0], testPath.waypoints[1][1]]
    goal = [testPath.waypoints[-1][0], testPath.waypoints[-1][1]]

    paths = sample_paths_diff_drive_robot(v_left_set, v_right_set, state, dt, N, testPath.even_trajectory, path_obs, goal, testPath.bounding_box_polygon, atr)
    for path in paths:
        plt.plot(path.x, path.y, color='gray', alpha=0.1)
    testPath.render()
    trajectory = [state]

    start_time = time.time()
    while True:
        loop_time = time.time()
        paths = sample_paths_diff_drive_robot(v_left_set, v_right_set, state, dt, N, testPath.even_trajectory, path_obs, goal, testPath.bounding_box_polygon, atr)
        optimal_path, optimal_path_index = select_optimal_path(paths)
        v_left_optimal = optimal_path.v_left / scale
        v_right_optimal = optimal_path.v_right / scale
        state,_,_ = atr.runge_kutta_4_step([v_right_optimal, v_left_optimal], "simple")
        state = [state[0], state[1], state[2]]
        # state = [optimal_path.x[1], optimal_path.y[1], optimal_path.theta[1]]
        trajectory.append(state)
        plt.cla()
        testPath.render()
        plt.scatter(state[0], state[1], color='green', s=10)
        for path in paths:
            plt.plot(path.x, path.y, color='gray', alpha=0.1)

        plt.plot(optimal_path.x, optimal_path.y, color='blue', linewidth=2)
        elapsed_time = time.time() - start_time
        computation_time = time.time() - loop_time
        plt.title(f"v :{str(optimal_path.v/scale)[0:4]} m/s, v_left: {round(v_left_optimal,2)}, v_right: {round(v_right_optimal,2)}\n \
                  elapsed time: {round(elapsed_time,2)} s, comp time: {round(computation_time,2)} s")

        if np.hypot(state[0] - goal[0], state[1] - goal[1]) < 0.2:
            goal_index += 1
            # if goal_index >= len(testPath.waypoints):
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', linewidth=2)
            plt.pause(5)
            plt.show()
            break
            # goal = [testPath.waypoints[goal_index][0], testPath.waypoints[goal_index][1]]
        # print(f"loop time: {computation_time}")
        pause_time = dt - computation_time
        if pause_time < 0:
            pause_time = 0
        plt.pause(computation_time)

if __name__ == "__main__":
    run_loop()