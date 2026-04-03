from typing import Literal

import numpy as np

from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj


class ShapeUtils(object):
    @staticmethod
    def inflate_convex_hull(mesh: SeqMeshObj, cell_weights: np.ndarray):
        """inflate cell weights to convex hull shape """
        new_cell_weights = cell_weights.copy()
        for cell_idx, weight in enumerate(cell_weights):
            cell_neighbors = mesh.cells_neighbors[cell_idx]
            if mesh.tdim == 2:
                if cell_weights[cell_neighbors].sum() >= 2 and weight == 0.0:
                        new_cell_weights[cell_idx] = 1.0
            else:
                if cell_weights[cell_neighbors].sum() >= 3 and weight == 0.0:
                    new_cell_weights[cell_idx] = 1.0
        return new_cell_weights

    @staticmethod
    def corrosion_convex_hull(mesh: SeqMeshObj, cell_weights: np.ndarray):
        """corrode cell weights to convex hull shape """
        new_cell_weights = cell_weights.copy()
        for cell_idx, weight in enumerate(cell_weights):
            cell_neighbors = mesh.cells_neighbors[cell_idx]        
            if mesh.tdim == 2:
                if cell_weights[cell_neighbors].sum() <= 1 and weight == 1.0:
                        new_cell_weights[cell_idx] = 0.0
            else:
                if cell_weights[cell_neighbors].sum() <= 1 and weight == 1.0:
                    new_cell_weights[cell_idx] = 0.0
        return new_cell_weights

    @staticmethod
    def convert_weight_to_ConvexHull(
        mesh: SeqMeshObj, cell_weights: np.ndarray, 
        inflate_times: int = 3, corrosion_times: int = 3
    ):
        for _ in range(inflate_times):
            new_cell_weights = ShapeUtils.inflate_convex_hull(mesh, cell_weights)
            if new_cell_weights.sum() == cell_weights.sum():
                break
            cell_weights = new_cell_weights
        
        for _ in range(corrosion_times):
            new_cell_weights = ShapeUtils.corrosion_convex_hull(mesh, cell_weights)
            if new_cell_weights.sum() == cell_weights.sum():
                break
            cell_weights = new_cell_weights

        return cell_weights

    @staticmethod
    def convert_weight_to_standard_grid(
        standard_grid_to_cells: np.ndarray,
        cell_weights: np.ndarray,
        mode: Literal['inflate', 'corrosion'] = 'inflate'
    ) -> tuple[np.ndarray, np.ndarray]:
        grid_cell_weights = cell_weights[standard_grid_to_cells]
        if mode == 'corrosion':
            pos_bool = np.sum(grid_cell_weights, axis=1) == standard_grid_to_cells.shape[1]
        elif mode == 'inflate':
            pos_bool = np.sum(grid_cell_weights, axis=1) > 0.0
        else:
            raise ValueError(f"Invalid mode: {mode}")

        pos_idxs = np.unique(standard_grid_to_cells[pos_bool].reshape(-1))
        new_cell_weights = np.zeros(cell_weights.shape[0])
        new_cell_weights[pos_idxs] = 1.0
        
        grid_pos_weight = np.zeros(shape=(grid_cell_weights.shape[0], ))
        grid_pos_weight[pos_bool] = 1.0
        
        return new_cell_weights, grid_pos_weight

    @staticmethod
    def inflate_standard_grid_weight(
        mesh: SeqMeshObj,
        grid_pos_weight: np.ndarray,
        standard_grid_neighbors: list[np.ndarray],
        standard_grid_to_cells: np.ndarray,
    ):        
        new_grid_pos_weight = np.zeros(shape=(grid_pos_weight.shape[0], ))
        for c_idx, pos_tag in enumerate(grid_pos_weight):
            if pos_tag == 1.0:
                new_grid_pos_weight[c_idx] = 1.0
            else:
                neighbour_idxs = standard_grid_neighbors[c_idx]
                if mesh.tdim == 3:
                    if grid_pos_weight[neighbour_idxs].sum() >= 5:
                        new_grid_pos_weight[c_idx] = 1.0
                else:
                    if grid_pos_weight[neighbour_idxs].sum() >= 3:
                        new_grid_pos_weight[c_idx] = 1.0
        
        pos_idxs = np.unique(standard_grid_to_cells[new_grid_pos_weight == 1.0].reshape(-1))
        new_cell_weights = np.zeros(mesh.num_cells)
        new_cell_weights[pos_idxs] = 1.0
        
        return new_cell_weights
    
