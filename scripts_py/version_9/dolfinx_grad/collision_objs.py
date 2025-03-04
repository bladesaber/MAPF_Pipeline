import dolfinx
import numpy as np
from typing import List, Dict
from sklearn.neighbors import KDTree
import pyvista

from .dolfinx_utils import MeshUtils
from .surface_fields import TopoLogyField


class MeshCollisionObj(object):
    def __init__(
            self,
            name, domain: dolfinx.mesh.Mesh, facet_tags: dolfinx.mesh.MeshTags, cell_tags: dolfinx.mesh.MeshTags,
            bry_markers: List[int], point_radius
    ):
        self.name = name
        self.domain = domain
        self.facet_tags = facet_tags
        self.cell_tags = cell_tags
        self.tdim = domain.topology.dim
        self.fdim = self.tdim - 1
        self.point_radius = point_radius

        bry_dofs = []
        for marker in bry_markers:
            bry_dofs.append(
                MeshUtils.extract_entity_dofs(
                    dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1)), self.fdim,
                    MeshUtils.extract_facet_entities(domain, facet_tags, marker)
                ))
        self.bry_idxs = np.concatenate(bry_dofs, axis=-1)
        self.n_points = self.bry_idxs.shape[0]

        self.bry_coords: np.ndarray = None
        self.tree: KDTree = None

    def update_tree(self):
        if self.tree is not None:
            del self.tree
            self.tree = None
            self.bry_coords = None

        self.bry_coords = self.domain.geometry.x[self.bry_idxs, :self.tdim]
        self.tree = KDTree(self.bry_coords)

    def query_radius_neighbors_from_xyz(
            self, coords: np.ndarray, radius, return_dist=False, count_only=False, sort_results=False
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

    def get_coords(self):
        return self.domain.geometry.x[:, :self.tdim]

    def find_conflict_bry_nodes(self, obs_coords: np.ndarray, radius, with_dist=False):
        """
        Find the mesh boundary nodes which conflicted by obstacle
        """
        if with_dist:
            idxs, dists = self.query_radius_neighbors_from_xyz(obs_coords, radius, return_dist=with_dist)
        else:
            idxs = self.query_radius_neighbors_from_xyz(obs_coords, radius, return_dist=with_dist)
        idxs = np.concatenate(idxs, axis=-1)

        dist_min = np.inf
        if idxs.shape[0] > 0:
            idxs = np.unique(idxs)
            idxs = self.bry_idxs[idxs]
            if with_dist:
                dists = np.concatenate(dists, axis=-1)
                dist_min = np.min(dists)

        if with_dist:
            return idxs, dist_min
        else:
            return idxs

    def approximate_extract(self, bbox_w, part_pcd_idxs: np.ndarray = None):
        if part_pcd_idxs is None:
            part_pcd_idxs = np.arange(0, self.n_points, 1)

        labels = TopoLogyField.semi_octree_fit(self.get_coords()[part_pcd_idxs, :], bbox_w)
        label_idxs_dict = {}
        for label in np.unique(labels):
            label_idxs_dict[label] = part_pcd_idxs[labels == label]
        return label_idxs_dict

    def find_conflict_bbox_infos(self, coords: np.ndarray, radius, bbox_w):
        obs_idxs = self.find_conflict_bry_nodes(coords, radius, with_dist=False)
        if obs_idxs.shape[0] == 0:
            return {}

        label_idxs_dict = self.approximate_extract(bbox_w, obs_idxs)
        bbox_infos = {}
        for label_idx in label_idxs_dict.keys():
            bbox_infos[f"{self.name}_{label_idx}"] = {
                'points': self.get_coords()[label_idxs_dict[label_idx], :],
                'obs_radius': self.point_radius,
                'bbox_w': bbox_w
            }
        return bbox_infos


class ObstacleCollisionObj(object):
    def __init__(self, name: str, coords: np.ndarray, point_radius, with_tree=False):
        self.name = name
        self.coords: np.ndarray = coords
        self.ndim = 3 if self.coords.shape[1] == 3 else 2
        self.n_points = self.coords.shape[0]
        self.point_radius = point_radius

        self.with_tree = with_tree
        if with_tree:
            self.tree = KDTree(self.coords)

    def approximate_extract(self, bbox_w, part_pcd_idxs: np.ndarray = None):
        if part_pcd_idxs is None:
            part_pcd_idxs = np.arange(0, self.n_points, 1)

        labels = TopoLogyField.semi_octree_fit(self.coords[part_pcd_idxs, :], bbox_w)
        label_idxs_dict = {}
        for label in np.unique(labels):
            label_idxs_dict[label] = part_pcd_idxs[labels == label]
        return label_idxs_dict

    def find_conflict_nodes(self, mesh_coords: np.ndarray, radius, with_dist=False):
        if with_dist:
            idxs, dists = self.tree.query_radius(mesh_coords, radius, return_distance=with_dist)
        else:
            idxs = self.tree.query_radius(mesh_coords, radius, return_distance=with_dist)
        idxs = np.concatenate(idxs, axis=-1)

        dist_min = np.inf
        if idxs.shape[0] > 0:
            idxs = np.unique(idxs)
            if with_dist:
                dists = np.concatenate(dists, axis=-1)
                dist_min = np.min(dists)

        if with_dist:
            return idxs, dist_min
        else:
            return idxs

    def find_conflict_bbox_infos(self, mesh_coords: np.ndarray, radius, bbox_w):
        obs_idxs = self.find_conflict_nodes(mesh_coords, radius, with_dist=False)
        if obs_idxs.shape[0] == 0:
            return {}

        label_idxs_dict = self.approximate_extract(bbox_w, obs_idxs)
        bbox_infos = {}
        for label_idx in label_idxs_dict.keys():
            bbox_infos[f"{self.name}_{label_idx}"] = {
                'points': self.coords[label_idxs_dict[label_idx], :],
                'obs_radius': self.point_radius,
                'bbox_w': bbox_w
            }
        return bbox_infos

    def _labels_plot(self, label_idxs_dict: dict):
        color_map = np.random.random((len(label_idxs_dict), 3))
        colors = np.zeros((self.n_points, 3))
        for idx, label_idx in enumerate(label_idxs_dict.keys()):
            colors[label_idxs_dict[label_idx], :] = color_map[idx]

        if self.ndim == 2:
            mesh = pyvista.PointSet(np.concatenate([self.coords, np.zeros((self.n_points, 1))], axis=1))
        else:
            mesh = pyvista.PointSet(self.coords)
        mesh.point_data['color'] = colors

        plt = pyvista.Plotter()
        plt.add_mesh(mesh)
        plt.show()

    @staticmethod
    def save_vtu(file: str, coords: np.ndarray):
        if coords.shape[1] == 2:
            coords = np.zeros((coords.shape[0], 3))
            coords[:, :2] = coords
        else:
            coords = coords
        mesh = pyvista.PolyData(coords)
        pyvista.save_meshio(file, mesh)

    @staticmethod
    def load(name, point_radius, dim, file: str):
        format = file.split('.')[-1]
        if format in ['vtu', 'stl']:
            mesh = pyvista.read(file)
            coords = np.array(mesh.points[:, :dim]).astype(float)
        else:
            raise ValueError("[ERROR]: Non-Support Format")

        return ObstacleCollisionObj(name, coords, point_radius, with_tree=True)

    @staticmethod
    def remove_intersection_points(coords, mesh_objs: List[MeshCollisionObj], radius):
        for mesh_obj in mesh_objs:
            shell_radius = radius + mesh_obj.point_radius
            idxs_list = mesh_obj.query_radius_neighbors_from_xyz(coords, shell_radius)
            accept_idxs = np.ones(coords.shape[0], dtype=np.bool)
            for i, idxs in enumerate(idxs_list):
                if len(idxs) > 0:
                    accept_idxs[i] = False
            coords = coords[accept_idxs]
        return coords
