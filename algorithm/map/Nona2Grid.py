import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
import matplotlib.ticker as ticker

class Nona2Grid():
    def __init__(self,
                 map_dir: str = "./config/nona_description.json",
                 path_dir: str = "./config/nodes_tuve.json",
                 grid_resolution: float = 0.2,
                 grid_width: int = 13,
                 grid_height: int = 8,
                 if_render: bool = True
                 ):
        # Create an empty grid map
        self.map_dir = map_dir
        self.path_dir = path_dir
        self.grid_resolution = grid_resolution
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.if_render = if_render

        self.grid_map = np.zeros(
            (int(self.grid_height / self.grid_resolution), int(self.grid_width / self.grid_resolution)))

        self.readJson()
        self.construct_grid()
        self.construct_path()

    def readJson(self):
        # Read the JSON file
        with open(self.map_dir, "r") as file:
            # Load the JSON content into a Python dictionary
            data = json.load(file)

        self.polygons = data["polygons"]

        with open(self.path_dir, "r") as file:
            # Load the JSON content into a Python dictionary
            data = json.load(file)

        self.paths = data["Paths"]
        self.nodes = data["Nodes"]

    def construct_grid(self):
        # Iterate through each cell in the grid map
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                # Calculate the cell center coordinates
                x = j * self.grid_resolution + self.grid_resolution / 2
                y = i * self.grid_resolution + self.grid_resolution / 2

                # Check if the cell center is inside any of the polygons
                for polygon in self.polygons:
                    if self.point_in_polygon(polygon, x, y):
                        self.grid_map[i, j] = 1
                        break

    def construct_path(self):
        node_positions = {node['id']: np.array(
            node['position'][:2]) for node in self.nodes}

        for path in self.paths:
            seq = 0
            path_id = path['id']
            node_ids = path['graph']
            path_positions = [node_positions[node_id] for node_id in node_ids]
            if(self.if_render):
                for wp in path_positions:
                    seq += 1
                    plt.text(wp[0], wp[1], f'{seq}, {wp}', fontsize=8)
            path_positions = np.array(path_positions)
            if(self.if_render):
                plt.plot(path_positions[:, 0], path_positions[:, 1], marker='o',
                     markersize=5, linestyle='-', linewidth=1, label=f"Path {path_id}")
            break

    def point_in_polygon(self, poly, x, y):
        """
        Check if a point is inside a polygon
        Return tru if it is."""
        poly_2d = [(point[0], point[1])
                   for point in poly]  # Extract the (x, y) coordinates from the 3D points
        path = Path(poly_2d)
        return path.contains_points([(x, y)])

    def point_to_grid_coordinates(self, x, y):
        i = int(y // self.grid_resolution)
        j = int(x // self.grid_resolution)
        return i, j
    
    def check_collision(self, x, y, grid_coord=True):
        if not grid_coord:
            x, y = self.point_to_grid_coordinates(x, y)
        return self.grid_map[x, y] == 1

    def render(self):
        plt.imshow(self.grid_map, cmap='binary', origin='lower', extent=[0, self.grid_width, 0, self.grid_height])
        plt.xlabel('X')
        plt.ylabel('Y')

        ax = plt.gca()
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
        plt.grid(True, which='both', linestyle='--', alpha=0.5)

        # plt.grid(True)
        plt.title(f'Grid Map total resolution: {self.grid_resolution}, {self.grid_width / self.grid_resolution, self.grid_height / self.grid_resolution, }')
        # plt.show()

if __name__ == "__main__":
    grid_resolution = 0.2  # The size of each grid cell
    grid_width = 13  # The width of the grid map
    grid_height = 8  # The height of the grid map
    grid = Nona2Grid(grid_resolution=grid_resolution, grid_width=grid_width, grid_height=grid_height)
    grid.render()
    
    robot_x, robot_y = 0.66, 5.09
    robot_coord_x, robot_coord_y = grid.point_to_grid_coordinates(robot_x, robot_y)
    if grid.check_collision(robot_x, robot_y):
        print(f'robot in collection')
    plt.scatter(robot_x, robot_y, color='red')
    plt.text(robot_x, robot_y+0.2, f'{robot_coord_y, robot_coord_x}', fontsize=12)
    # grid_map[robot_coord_y, robot_coord_x]
    plt.show()
