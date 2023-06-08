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
import pickle

np.set_printoptions(precision=3, suppress=True)

# max speed of ATR, 1m/s
# 8 rad/s 

# Parameters
# initial_state = [0.0, 0.0, 0.0]  # [x, y, theta]

num_of_speeds = 5
v_left_set = np.linspace(0.1, 0.8, 5)  # Range of left wheel speeds rad/s
v_right_set = np.linspace(0.1, 0.8, 5)  # Range of right wheel speeds

v_left_set = np.concatenate(([-0.2, 0], v_left_set))
v_right_set = np.concatenate(([-0.2, 0], v_right_set))

scale = 12
v_left_set *= scale
v_right_set *= scale

dt = 0.1  # Time step
N = int(100/scale)  # Number of steps
WHEEL_RADIUS = 0.125  # Wheel radius
TRACK_WIDTH = 0.48 # track width

w_max = WHEEL_RADIUS * 1.6 / TRACK_WIDTH


def draw_arrow(x, y, angle, length=0.5, color='b'):
    dx = length * np.cos(angle)
    dy = length * np.sin(angle)
    plt.arrow(x, y, dx, dy, head_width=0.05, head_length=0.1, fc=color, ec='black')

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
        self.lock_goal = False

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
                path.lock_goal = True
            if approach_goal:
                mul_scale = np.abs([0.8*scale/ (v_left+0.001), 0.8*scale / (v_right+0.001)])
                mul_scale = np.min(mul_scale)
                if mul_scale >= 1:
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
    if not path.lock_goal and optimal_path.v / scale <= 0.02:
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

k_rou = 0.8
k_alpha = 10
k_beta = 0.0

def cal_v_w(state, goal):
    rou = np.hypot(goal[0] - state[0], goal[1] - state[1])
    alpha = np.arctan2(goal[1] - state[1], goal[0] - state[0]) - state[2]
    beta = goal[2] - (state[2] + alpha)
    v = k_rou * rou
    if v > 0.1:
        v = 0.1
    w = k_alpha * alpha + k_beta * beta
    # w = np.remainder(w + np.pi, 2 * np.pi) - np.pi
    
    if abs(w) > w_max:
        w = w_max * np.sign(w)
    return v, w

def cal_wheelspeed(v, w):
    iJ = np.array([[1/WHEEL_RADIUS, TRACK_WIDTH/(2*WHEEL_RADIUS)],
                   [1/WHEEL_RADIUS, -TRACK_WIDTH/(2*WHEEL_RADIUS)]])
    W = np.matmul(iJ, np.array([v, w])) # [wr, wl]
    W = [W[0], W[1]]
    max_speed = np.max(np.abs(W))
    scale_factor = 0.8 / max_speed
    W = np.array(W) * scale_factor
    W = [W[0], W[1]]
    return W
    
def run_loop(oldPath=False):
    testPath = Path(trajectory_point_interval=0.1,
                No=8, Nw=8, Lp=6, mu_r=0.25, sigma_d=0.8, shift_distance=1)
    if oldPath:
        with open("lattice_path.pkl", "rb") as file:
            testPath = pickle.load(file)
    else:
        testPath.reset()
        with open("lattice_path.pkl", "wb") as file:
            pickle.dump(testPath, file)
    testPath.render()

    state = [testPath.waypoints[0][0], testPath.waypoints[0][1], testPath.yaw_angles[0]]
    state_ref = [testPath.waypoints[0][0], testPath.waypoints[0][1], testPath.yaw_angles[0]]
    
    atr_ref = ATR_RK4(init_state=state, dt=dt)
    atr = ATR_RK4(init_state=state, dt=dt)
    
    path_obs = [(round(float(item[0][0]),3), round(float(item[0][1]),3), round(item[1], 3)) for item in testPath.obstacles]
    goal_index = 1
    # goal = [testPath.waypoints[1][0], testPath.waypoints[1][1]]
    goal = [testPath.waypoints[-1][0], testPath.waypoints[-1][1]]

    # paths = sample_paths_diff_drive_robot(v_left_set, v_right_set, state, dt, N, testPath.even_trajectory, path_obs, goal, testPath.bounding_box_polygon, atr)
    # for path in paths:
    #     plt.plot(path.x, path.y, color='gray', alpha=0.1)
    testPath.render()
    trajectory = [state]
    trajectory_ref = [state_ref]
    
    start_time = time.time()
    look_ahead_index = 0
    ref_stop = False
    sim_stop = False
    while True:
        loop_time = time.time()
        if not ref_stop:
            min_distance, index = testPath.trejectory_kd_tree.query(np.array([state_ref[0], state_ref[1]]))
            if index + 1 >= look_ahead_index:
                look_ahead_index = index + 1 # the target point is 0.1 meter away from the closest point
            target_point = testPath.even_trajectory[look_ahead_index]
            target_point = [target_point[0], target_point[1], testPath.yaw_angles[look_ahead_index]]
            v, w = cal_v_w(state_ref, target_point)
            W = cal_wheelspeed(v, w)
            state_ref, v_act, w_act = atr_ref.runge_kutta_4_step(W, "simple")
            state_ref = [state_ref[0], state_ref[1], state_ref[2]]
            trajectory_ref.append(state_ref)
            if np.hypot(state_ref[0] - goal[0], state_ref[1] - goal[1]) < 0.2:
                ref_stop = True
                trajectory_ref = np.array(trajectory_ref)
        # plt.cla()
        # testPath.render()
        # plt.scatter(state[0], state[1], color='green', s=10)
        # plt.scatter(target_point[0], target_point[1], color='red', s=10)
        # computation_time = time.time() - loop_time
        # elapsed_time = time.time() - start_time   
        
        # plt.title(f"v :{v} m/s, v_left: {round(W[1],2)}, v_right: {round(W[0],2)}\n \
        #     elapsed time: {round(elapsed_time,2)} s, comp time: {round(computation_time,2)} s")
        # if np.hypot(state[0] - goal[0], state[1] - goal[1]) < 0.2:
        #     goal_index += 1
        #     # if goal_index >= len(testPath.waypoints):
        #     trajectory = np.array(trajectory)
        #     plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', linewidth=2)
        #     plt.pause(5)
        #     plt.show()
        #     break
        # # goal = [testPath.waypoints[goal_index][0], testPath.waypoints[goal_index][1]]
        # pause_time = dt - computation_time
        # if pause_time < 0:
        #     pause_time = 0
        # plt.pause(0.005)
        if not sim_stop:
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
        # plt.scatter(state[0], state[1], color='green', s=10)
        draw_arrow(state[0], state[1], state[2])
        draw_arrow(state_ref[0], state_ref[1], state_ref[2])
        
        # plt.scatter(state_ref[0], state_ref[1], color='green', s=10)
        plt.scatter(target_point[0], target_point[1], color='red', s=10)
        for path in paths:
            plt.plot(path.x, path.y, color='gray', alpha=0.1)

        plt.plot(optimal_path.x, optimal_path.y, color='blue', linewidth=2)
        elapsed_time = time.time() - start_time
        computation_time = time.time() - loop_time
        plt.title(f"v :{str(optimal_path.v/scale)[0:4]} m/s, v_left: {round(v_left_optimal,2)}, v_right: {round(v_right_optimal,2)}\n \
                    v_ref:{round(v_act, 3)}, w_ref: {round(w_act, 3)}, v_left:{round(W[1],2)}, v_right:{round(W[0],2)}\n \
                  elapsed time: {round(elapsed_time,2)} s, comp time: {round(computation_time,2)} s")

        if np.hypot(state[0] - goal[0], state[1] - goal[1]) < 0.2:
            sim_stop = True
            goal_index += 1        
            trajectory = np.array(trajectory)
            
        if sim_stop and ref_stop:
            plt.plot(trajectory[:, 0], trajectory[:, 1], color='red', linewidth=2)
            plt.plot(trajectory_ref[:, 0], trajectory_ref[:, 1], color='green', linewidth=2)
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
    run_loop(oldPath=False)