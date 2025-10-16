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
            with_weight=True
    ):
        '''
        Args:
            grid: pyvista.UnstructuredGrid, mesh
            filter_radius: float, gaussian filter radius
            is_cell_field: bool, if True, use cell centers as centers, otherwise use points as centers
            with_weight: bool, if True, use distance as weight, otherwise use 1 as weight
        '''
        self.grid = grid
        self.is_cell_field = is_cell_field
        self.filter_radius = filter_radius
        self.with_weight = with_weight
        self.dist_mat: scipy.sparse.coo_matrix
        self.distSum_mat: np.matrix
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
        for row_id, neg_idxs in tqdm(enumerate(neighbour_idxs), desc='Generate topology filter'):
            row_idxs.append(np.full(shape=(neg_idxs.shape[0],), fill_value=row_id))
        
        row_idxs = np.concatenate(row_idxs, axis=-1).reshape(-1)
        col_idxs = np.concatenate(neighbour_idxs, axis=-1).reshape(-1)
        dists = np.concatenate(dists, axis=-1).reshape(-1)
        assert row_idxs.shape[0] == col_idxs.shape[0] == dists.shape[0]
        
        if self.with_weight:
            dists = self.filter_radius - dists
        else:
            dists = np.ones_like(dists)
        
        self.dist_mat = scipy.sparse.coo_matrix((dists, (row_idxs, col_idxs)), shape=(num, num))
        self.distSum_mat = self.dist_mat.sum(axis=1).reshape((-1, 1))

    def run(self, value: np.ndarray):
        return np.divide(self.dist_mat.dot(value), self.distSum_mat + 1e-8)


