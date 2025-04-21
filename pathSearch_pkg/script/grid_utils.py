import math
import numpy as np
from sklearn.neighbors import KDTree

class GridRunner:
    def __init__(
            self, x_length: float, y_length: float, z_length: float, resolution: float
        ):
        self.x_size: int = math.ceil(x_length / resolution)
        self.y_size: int = math.ceil(y_length / resolution)
        self.z_size: int = math.ceil(z_length / resolution)

    def obstacle_point_uniform(self, obstacle_points: np.ndarray, obstacle_radius: float):
        xs = np.linspace(0, self.x_size, self.x_size)
        ys = np.linspace(0, self.y_size, self.y_size)
        zs = np.linspace(0, self.z_size, self.z_size)

        xs, ys, zs = np.meshgrid(xs, ys, zs)
        grid_points = np.concatenate([
            xs[..., np.newaxis], 
            ys[..., np.newaxis], 
            zs[..., np.newaxis]
        ], axis=-1)
        grid_points = grid_points.reshape(-1, 3)

        tree = KDTree(obstacle_points)
        indices_list = tree.query_radius(grid_points, r=obstacle_radius, return_distance=False)
        
        uniform_obstacle_points = []
        for indices in indices_list:
            if len(indices) > 0:
                continue
            
            uniform_obstacle_points.append(grid_points[indices])

        return np.array(uniform_obstacle_points)

    def distance_cell_to_closest_obstacle(self, uniform_obstacle_points: np.ndarray):
        xs = np.linspace(0, self.x_size, self.x_size)
        ys = np.linspace(0, self.y_size, self.y_size)
        zs = np.linspace(0, self.z_size, self.z_size)

        xs, ys, zs = np.meshgrid(xs, ys, zs)
        grid_points = np.concatenate([
            xs[..., np.newaxis], 
            ys[..., np.newaxis], 
            zs[..., np.newaxis]
        ], axis=-1)
        grid_points = grid_points.reshape(-1, 3)

        tree = KDTree(uniform_obstacle_points)
        distances_list, indices_list = tree.query(grid_points, k=1, return_distance=True)
        distances = np.concatenate(distances_list, axis=-1)
        
        assert distances.shape == (grid_points.shape[0],)
        
        return {'grid_points': grid_points, 'distances': distances}

    