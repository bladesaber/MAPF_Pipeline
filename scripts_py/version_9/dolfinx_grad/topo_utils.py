import numpy as np
import pyvista
import scipy
from sklearn.neighbors import KDTree
from tqdm import tqdm


class DistanceBandFilter(object):
    def __init__(
            self,
            grid: pyvista.UnstructuredGrid,
            filter_radius: float,
            is_cell_field=True,
            is_center_important=True
    ):
        self.grid = grid
        self.is_cell_field = is_cell_field
        self.filter_radius = filter_radius
        self.is_center_important = is_center_important
        self.dist_mat: scipy.sparse.coo_matrix = None
        self.distSum_mat: np.matrix = None
        self.generate_filter(grid)

    def generate_filter(self, grid: pyvista.UnstructuredGrid):
        if self.is_cell_field:
            centers = grid.cell_centers().points
            num = centers.shape[0]
        else:
            centers = grid.points
            num = centers.shape[0]

        row_idxs = []
        center_tree = KDTree(centers, metric='minkowski')
        neighbour_idxs, dists = center_tree.query_radius(centers, r=self.filter_radius, return_distance=True)
        for row_id, neg_idxs in tqdm(enumerate(neighbour_idxs)):
            row_idxs.append(np.full(shape=(neg_idxs.shape[0],), fill_value=row_id))
        row_idxs = np.concatenate(row_idxs, axis=-1).reshape(-1)
        col_idxs = np.concatenate(neighbour_idxs, axis=-1).reshape(-1)
        dists = np.concatenate(dists, axis=-1).reshape(-1)
        if self.is_center_important:
            dists = self.filter_radius - dists
        self.dist_mat = scipy.sparse.coo_matrix((dists, (row_idxs, col_idxs)), shape=(num, num))
        self.distSum_mat = self.dist_mat.sum(axis=1).reshape((-1, 1))

    def run(self, value: np.ndarray):
        return np.divide(self.dist_mat.dot(value), self.distSum_mat + 1e-8)


class DensityBandFilter(object):
    def __init__(self, grid: pyvista.UnstructuredGrid, radius_filter: float):
        self.grid = grid
        self.radius_filter = radius_filter
        self.matrix_num: int = None
        self.row_idxs: np.ndarray = None
        self.col_idxs: np.ndarray = None
        self.generate_filter(grid)

    def generate_filter(self, grid: pyvista.UnstructuredGrid):
        cell_centers = grid.cell_centers().points
        self.matrix_num = cell_centers.shape[0]
        row_idxs = []
        center_tree = KDTree(cell_centers, metric='minkowski')
        neighbour_idxs = center_tree.query_radius(cell_centers, r=self.radius_filter, return_distance=False)
        for row_id, neg_idxs in tqdm(enumerate(neighbour_idxs)):
            row_idxs.append(np.full(shape=(neg_idxs.shape[0],), fill_value=row_id))
        self.row_idxs = np.concatenate(row_idxs, axis=-1).reshape(-1)
        self.col_idxs = np.concatenate(neighbour_idxs, axis=-1).reshape(-1)

    def run(self, data: np.ndarray, density: np.ndarray):
        values = density[self.col_idxs].reshape(-1)
        value_mat = scipy.sparse.coo_matrix(
            (values, (self.row_idxs, self.col_idxs)), shape=(self.matrix_num, self.matrix_num)
        )
        value_sum_mat = value_mat.sum(axis=1).reshape((-1, 1))
        return np.divide(value_mat.dot(data), value_sum_mat + 1e-8)
