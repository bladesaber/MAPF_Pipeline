import dolfinx

import numpy as np
import pandas as pd
import networkx as nx
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils


class ReMesher(object):
    def __init__(
        self, mesh: MeshObj, surface_marker: list[int]
    ):
        self.mesh = mesh
        self.surface_marker = surface_marker
        self.surface_indices, self.internal_indices = self.extract_surface_indices()
        self.cell_connectivity: np.ndarray = MeshUtils.cell_connectivity(self.mesh.grid, dim=self.mesh.tdim)
        self.line_connectivity = MeshUtils.line_connectivity(self.mesh.grid)
    
        self.internal_line_connectivity: dict[int, np.ndarray] = {}  # key: neighbor length, value: line indices
        self.compute_internal_line_connectivity()
        
        self.surface_line_connectivity: dict[int, np.ndarray] = {}  # key: neighbor length, value: line indices
        self.compute_surface_line_connectivity()
    
    def extract_surface_indices(self) -> tuple[np.ndarray, np.ndarray]:
        indices_list = []
        for marker in self.surface_marker:
            indices_list.append(self.mesh.extract_surface_indices(marker))
        surface_indices = np.unique(np.concatenate(indices_list, axis=0))
        internal_indices = np.array(list(set(np.arange(0, self.mesh.num_points, 1)) - set(surface_indices)))
        return surface_indices, internal_indices

    def compute_internal_line_connectivity(self):
        graph = nx.Graph()
        graph.add_edges_from(self.line_connectivity)
        internal_line_dict: dict[int, list[list[int]]] = {}
        for inner_idx in self.internal_indices:
            neighbor_idxs = list(graph.neighbors(inner_idx))
            neighbor_len = len(neighbor_idxs)
            if neighbor_len not in internal_line_dict.keys():
                internal_line_dict[neighbor_len] = []
            internal_line_dict[neighbor_len].append([inner_idx] + neighbor_idxs)
        for neg_len, combin_pack in internal_line_dict.items():
            self.internal_line_connectivity[neg_len] = np.array(combin_pack)

    def compute_surface_line_connectivity(self):
        graph = nx.Graph()
        graph.add_edges_from(self.line_connectivity)
        surface_line_dict: dict[int, list[list[int]]] = {}
        for surface_idx in self.surface_indices:
            neighbor_idxs = list(graph.neighbors(surface_idx))
            neighbor_len = len(neighbor_idxs)
            if neighbor_len not in surface_line_dict.keys():
                surface_line_dict[neighbor_len] = []
            surface_line_dict[neighbor_len].append([surface_idx] + neighbor_idxs)
        for neg_len, combin_pack in surface_line_dict.items():
            self.surface_line_connectivity[neg_len] = np.array(combin_pack)

    def check_mesh_volume(self, new_coord: np.ndarray, min_volume: float = 0.0, max_volume: float = np.inf) -> dict:
        if self.mesh.tdim == 2:
            points_a = new_coord[self.cell_connectivity[:, 0], :self.mesh.tdim]
            points_b = new_coord[self.cell_connectivity[:, 1], :self.mesh.tdim]
            points_c = new_coord[self.cell_connectivity[:, 2], :self.mesh.tdim]
            areas = 0.5 * np.abs(
                points_a[:, 0] * (points_b[:, 1] - points_c[:, 1])
                + points_b[:, 0] * (points_c[:, 1] - points_a[:, 1])
                + points_c[:, 0] * (points_a[:, 1] - points_b[:, 1])
            )
            return {
                "is_valid": np.all(areas > min_volume) and np.all(areas < max_volume),
                "areas": areas
            }
        else:
            points_a = new_coord[self.cell_connectivity[:, 0], :self.mesh.tdim]
            points_b = new_coord[self.cell_connectivity[:, 1], :self.mesh.tdim]
            points_c = new_coord[self.cell_connectivity[:, 2], :self.mesh.tdim]
            points_d = new_coord[self.cell_connectivity[:, 3], :self.mesh.tdim]
            vec_ab = points_b - points_a
            vec_ac = points_c - points_a
            vec_ad = points_d - points_a
            volume = 1. / 6. * np.sum(vec_ab * np.cross(vec_ac, vec_ad, axisa=1), axis=1)
            return {
                "is_valid": np.all(volume > min_volume) and np.all(volume < max_volume),
                "volume": volume
            }

    def inner_laplacian_smoothing(self, max_iteration=1000, tol=1e-2) -> dict:
        '''
        laplacian smoothing for mesh deformation: compute mean position of neighbor points as new position. Only update internal points.
        '''
        coord_cur = self.mesh.grid.points.copy()[:, :self.mesh.tdim]
        iteration = 0
        while True:
            inner_idxs, coord_new = [], []
            for _, lin_conn_pack in self.internal_line_connectivity.items():
                new_pos = np.mean(coord_cur[lin_conn_pack[:, 1:]], axis=1)
                inner_idxs.append(lin_conn_pack[:, 0])
                coord_new.append(new_pos)
            inner_idxs = np.concatenate(inner_idxs, axis=-1)
            coord_new = np.concatenate(coord_new, axis=0)

            residual = np.max(np.linalg.norm(coord_cur[inner_idxs, :] - coord_new, ord=2, axis=1))
            coord_cur[inner_idxs, :] = coord_new

            iteration += 1
            if iteration > max_iteration:
                return {
                    'is_success': False,
                    'info': f'[Info]: laplacian smoothing reach max iteration, laplacian smoothing re-mesh error:{residual}.',
                    'coord': coord_cur
                }

            if residual <= tol:
                return {
                    'is_success': True,
                    'info': f'[Info]: laplacian smoothing [Info]: converge, laplacian smoothing re-mesh error:{residual}.',
                    'coord': coord_cur
                }
    
    def reset_mesh(self, coord: np.ndarray):
        assert coord.shape[0] == self.mesh.num_points, f'[Error]: reset mesh error, coord shape:{coord.shape}, mesh num_points:{self.mesh.num_points}.'
        assert coord.shape[1] in [2, 3], f'[Error]: reset mesh error, coord shape:{coord.shape}.'
        if coord.shape[1] == 2:
            coord = np.concatenate((coord, np.zeros((coord.shape[0], 1))), axis=1)
        self.mesh.domain.geometry.x[:] = coord
    
    def draw_boundary(self):
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.mesh.grid, color='black', style='wireframe')
        
        bc_points = self.mesh.points[self.surface_indices, :]
        if bc_points.shape[1] == 2:
            bc_points = np.concatenate((bc_points, np.zeros((bc_points.shape[0], 1))), axis=1)
        plotter.add_mesh(pyvista.PolyData(bc_points), color='green', style='points')
        
        inner_points = self.mesh.points[self.internal_indices, :]
        if inner_points.shape[1] == 2:
            inner_points = np.concatenate((inner_points, np.zeros((inner_points.shape[0], 1))), axis=1)
        plotter.add_mesh(pyvista.PolyData(inner_points), color='blue', style='points')
        
        plotter.show()
        