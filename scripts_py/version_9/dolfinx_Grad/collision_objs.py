import dolfinx
import numpy as np
from typing import List, Dict
from sklearn.neighbors import KDTree
import pyvista

from .dolfinx_utils import MeshUtils


class MeshCollisionObj(object):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            facet_tags: dolfinx.mesh.MeshTags,
            cell_tags: dolfinx.mesh.MeshTags,
            bry_markers: List[int]
    ):
        self.domain = domain
        self.facet_tags = facet_tags
        self.cell_tags = cell_tags
        self.tdim = domain.topology.dim
        self.fdim = self.tdim - 1
        self.V = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))

        bry_dofs = []
        for marker in bry_markers:
            bry_dofs.append(
                MeshUtils.extract_entity_dofs(
                    self.V, self.fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
                ))
        self.bry_idxs = np.concatenate(bry_dofs, axis=-1)

        self.bry_coords: np.ndarray = None
        self.tree: KDTree = None

    def create_tree(self):
        self.bry_coords = self.domain.geometry.x[self.bry_idxs, :self.tdim]
        self.tree = KDTree(self.bry_coords)

    def release_tree(self):
        del self.tree
        self.tree = None
        self.bry_coords = None

    def query_radius_neighbors_from_xyz(
            self, coords: np.ndarray, radius,
            return_dist=False, count_only=False, sort_results=False
    ):
        res = self.tree.query_radius(
            coords, r=radius,
            return_distance=return_dist, count_only=count_only, sort_results=sort_results
        )
        return res

    def query_k_neighbors_from_xyz(self, coords: np.ndarray, k, return_dist=False):
        res = self.tree.query(coords, k=k, return_distance=return_dist)
        return res

    def get_bry_coords(self):
        return self.domain.geometry.x[self.bry_idxs, :self.tdim]

    def find_conflict_bry_nodes(self, coords: np.ndarray, radius):
        idxs, dists = self.query_radius_neighbors_from_xyz(coords=coords, radius=radius, return_dist=True)
        idxs = np.concatenate(idxs, axis=-1)
        idxs = np.unique(idxs)
        idxs = self.bry_idxs[idxs]

        dists = np.concatenate(dists, axis=-1)
        if dists.shape[0] > 0:
            dist_min = np.min(dists)
        else:
            dist_min = np.inf

        return idxs, dist_min

    def find_node_neighbors(self, coords: np.ndarray, radius):
        relate_idxs = self.query_radius_neighbors_from_xyz(coords, radius)
        relate_idxs = np.concatenate(relate_idxs)
        relate_idxs = np.unique(relate_idxs)
        relate_idxs = self.bry_idxs[relate_idxs]
        return relate_idxs


class ObstacleCollisionObj(object):
    def __init__(self, coords: np.ndarray):
        self.coords: np.ndarray = coords

    def get_coords(self):
        return self.coords

    def save_vtu(self, file: str):
        if self.coords.shape[1] == 2:
            coords = np.zeros((self.coords.shape[0], 3))
            coords[:, :2] = self.coords
        else:
            coords = self.coords

        mesh = pyvista.PolyData(coords)
        pyvista.save_meshio(file, mesh)
