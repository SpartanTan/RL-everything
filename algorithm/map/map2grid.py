import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path


def point_in_polygon(poly, x, y):
    poly_2d = [(point[0], point[1]) for point in poly]  # Extract the (x, y) coordinates from the 3D points
    path = Path(poly_2d)
    return path.contains_points([(x, y)])

def point_to_grid_coordinates(x, y, grid_resolution):
    i = int(y // grid_resolution)
    j = int(x // grid_resolution)
    return i, j


# Read the JSON file
with open("./config/nona_description.json", "r") as file:
    # Load the JSON content into a Python dictionary
    data = json.load(file)

polygons = data["polygons"]

with open("./config/nodes_tuve.json", "r") as file:
    # Load the JSON content into a Python dictionary
    data = json.load(file)

paths = data["Paths"]

node_positions = {node['id']: np.array(node['position'][:2]) for node in data['Nodes']}

seq = 0
for path in paths:
    
    path_id = path['id']
    node_ids = path['graph']
    path_positions = [node_positions[node_id] for node_id in node_ids]
    for wp in path_positions:
        seq += 1
        plt.text(wp[0], wp[1], f'{seq}, {wp}', fontsize=8)
    path_positions = np.array(path_positions)
    plt.plot(path_positions[:, 0], path_positions[:, 1], marker='o', markersize=5, linestyle='-', linewidth=1, label=f"Path {path_id}")
    # for pos in path_positions:
    #     plt.text()
    break

# Grid map parameters
grid_resolution = 0.2  # The size of each grid cell
grid_width = 13  # The width of the grid map
grid_height = 8  # The height of the grid map

# Create an empty grid map
grid_map = np.zeros((int(grid_height / grid_resolution), int(grid_width / grid_resolution)))

# Iterate through each cell in the grid map
for i in range(grid_map.shape[0]):
    for j in range(grid_map.shape[1]):
        # Calculate the cell center coordinates
        x = j * grid_resolution + grid_resolution / 2
        y = i * grid_resolution + grid_resolution / 2

        # Check if the cell center is inside any of the polygons
        for polygon in polygons:
            if point_in_polygon(polygon, x, y):
                grid_map[i, j] = 1
                break
robot_x, robot_y = 0.66, 5.09
robot_coord_x, robot_coord_y = point_to_grid_coordinates(robot_x, robot_y, grid_resolution)



plt.scatter(robot_x, robot_y, color='red')
plt.text(robot_x, robot_y+0.2, f'{robot_coord_y, robot_coord_x}', fontsize=12)
# grid_map[robot_coord_y, robot_coord_x]
# Display the grid map
plt.imshow(grid_map, cmap='binary', origin='lower', extent=[0, grid_width, 0, grid_height])
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Grid Map total resolution: {grid_width / grid_resolution, grid_height / grid_resolution, }')
plt.show()
