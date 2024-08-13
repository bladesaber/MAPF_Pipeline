import numpy as np
import scipy.sparse
from sklearn.neighbors import KDTree
from scipy import sparse
import pyvista
from tqdm import tqdm


class DistanceBandFilter(object):
    def __init__(self, grid: pyvista.UnstructuredGrid, radius_filter: float, is_center=True):
        self.grid = grid
        self.radius_filter = radius_filter
        self.is_center = is_center
        self.dist_mat: scipy.sparse.coo_matrix = None
        self.distSum_mat: np.matrix = None
        self.generate_filter()

    def generate_filter(self):
        cell_centers = self.grid.cell_centers().points
        cells_num = cell_centers.shape[0]
        row_idxs = []
        center_tree = KDTree(cell_centers, metric='minkowski')
        neighbour_idxs, dists = center_tree.query_radius(cell_centers, r=self.radius_filter, return_distance=True)
        for row_id, neg_idxs in tqdm(enumerate(neighbour_idxs)):
            row_idxs.append(np.full(shape=(neg_idxs.shape[0],), fill_value=row_id))
        row_idxs = np.concatenate(row_idxs, axis=-1).reshape(-1)
        col_idxs = np.concatenate(neighbour_idxs, axis=-1).reshape(-1)
        if self.is_center:
            dists = self.radius_filter - np.concatenate(dists, axis=-1).reshape(-1)
        self.dist_mat = sparse.coo_matrix((dists, (row_idxs, col_idxs)), shape=(cells_num, cells_num))
        self.distSum_mat = self.dist_mat.sum(axis=1).reshape((-1, 1))

    def run(self, data: np.ndarray):
        return np.divide(self.dist_mat.dot(data), self.distSum_mat + 1e-8)


class DensityBandFilter(object):
    def __init__(self, grid: pyvista.UnstructuredGrid, radius_filter: float):
        self.grid = grid
        self.radius_filter = radius_filter
        self.matrix_num: int = None
        self.row_idxs: np.ndarray = None
        self.col_idxs: np.ndarray = None
        self.generate_filter()

    def generate_filter(self):
        cell_centers = self.grid.cell_centers().points
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
        value_mat = sparse.coo_matrix(
            (values, (self.row_idxs, self.col_idxs)), shape=(self.matrix_num, self.matrix_num)
        )
        value_sum_mat = value_mat.sum(axis=1).reshape((-1, 1))
        return np.divide(value_mat.dot(data), value_sum_mat + 1e-8)
