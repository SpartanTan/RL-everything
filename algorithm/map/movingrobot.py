import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation

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

# Prepare the plot
fig, ax = plt.subplots()
ax.imshow(grid_map, cmap='binary', origin='lower', extent=[0, grid_width, 0, grid_height])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Grid Map with Moving Point')

# Initial point coordinates
x_point = 0.66
y_point = 5.09

# Create a scatter plot with the initial point
point_plot, = ax.plot(x_point, y_point, marker='o', markersize=5, color='red')

# Animation update function
def update(frame):
    global x_point
    x_point += 0.2
    point_plot.set_data(x_point, y_point)
    return point_plot,

# Create the animation
ani = FuncAnimation(fig, update, frames=range(int(grid_width)), interval=1000, blit=True)

# Show the plot with animation
plt.show()
