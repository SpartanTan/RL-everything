from RK4 import ATR_RK4
import numpy as np
import matplotlib.pyplot as plt
import time
from nona_polygon import NONA_POLY

state = np.array([0, 5.0, 0], dtype=float)

atr = ATR_RK4(init_state=state, dt=0.1)

simulation_duration = 20  # in seconds
time_step = 0.1  # in seconds
speed = 0.1
radius = 0.125
wheel_speed = speed / radius
differential_rate = 1.5

wheel_speeds = (wheel_speed*differential_rate, wheel_speed)  # in rad/s (right wheel, left wheel)
trajectory = [state]
trajectory_simple = [state]

def render(polygons, ax):
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'k-') # change 'k-' to any other color, linestyle
        ax.fill(x, y, alpha=0.3) # change alpha to control the transparency

testNONA = NONA_POLY()

n_steps = int(simulation_duration / time_step)
for step in range(n_steps):
    state, _, _ = atr.runge_kutta_4_step(wheel_speeds, "RK4")
    trajectory.append(state)
trajectory = np.array(trajectory)

for step in range(n_steps):
    state, _, _ = atr.runge_kutta_4_step(wheel_speeds, "simple")
    trajectory_simple.append(state)
trajectory_simple = np.array(trajectory)

fig, ax = plt.subplots()
# render(testNONA.polygons, ax)
plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o',
            markersize=0.2, linestyle='-', linewidth=1)
plt.plot(trajectory_simple[:, 0], trajectory_simple[:, 1], marker='o',
            markersize=0.2, linestyle='-', linewidth=1)
plt.grid(True)
plt.axis('equal')
plt.show()

# atr.reset(np.array([0, 5.0, 0], dtype=float))
# start = time.time()
# for step in range(n_steps):
#     stime = time.time()
#     plt.cla()
#     render(testNONA.polygons, ax)
#     plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o',
#             markersize=0.2, linestyle='-', linewidth=1)
#     state, linear_velocity, angular_velocity = atr.runge_kutta_4_step(wheel_speeds)
#     plt.scatter(state[0], state[1], color='red', s=5)
#     etime = time.time()
#     computation_time = etime - stime
#     pause_time = time_step - computation_time
#     plt.title(f"v: {linear_velocity}  time: {round(etime-start)} s")
#     plt.pause(pause_time)