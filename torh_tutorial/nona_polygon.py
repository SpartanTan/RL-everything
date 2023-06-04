import json
from shapely.geometry import Point, Polygon, MultiLineString
from shapely.ops import nearest_points

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon


class NONA_POLY:
    def __init__(self, map_dir:str="testnona.json"):
        self.map_dir = map_dir
        with open(map_dir, "r") as file:
                    # Load the JSON content into a Python dictionary
                    data = json.load(file)

        self.polygons = [Polygon(coords) for coords in data["polygons"]]

    def is_point_within_polygons(self, point):
        for polygon in self.polygons:
            if point.within(polygon):
                return True
        return False

    def find_nearest_boundary_point(self, query_point):
        # Combine the exterior boundaries of all polygons into a MultiLineString
        boundary = MultiLineString([polygon.exterior for polygon in self.polygons])
        
        # Find the nearest points on the boundary to the query_point
        nearest_point = nearest_points(query_point, boundary)[1]
        distance = nearest_point.distance(query_point)
        return nearest_point, distance

def render(polygons, ax):
    for polygon in polygons:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'k-') # change 'k-' to any other color, linestyle
        ax.fill(x, y, alpha=0.3) # change alpha to control the transparency

if __name__ == "__main__":   
    # Replace the x and y coordinates with the ones you want to check
    testNONA = NONA_POLY()
    x = 10.5
    y = 3
    point = Point(x, y)
    nearest_boundary_point, distance = testNONA.find_nearest_boundary_point(point)
    fig, ax = plt.subplots()
    render(testNONA.polygons, ax)
        
    plt.scatter(point.x, point.y, color='red', marker='o', s=50)
    plt.scatter(nearest_boundary_point.x, nearest_boundary_point.y, color='blue', marker='x', s=50)
    print(nearest_boundary_point)
    print(distance)
    if testNONA.is_point_within_polygons(point):
        print("The point is within one of the polygons.")
    else:
        print("The point is not within any of the polygons.")
    plt.show()