import numpy as np
import pandas as pd
import pyvista
import math
from sklearn.neighbors import KDTree
from typing import Dict, Union
from tqdm import tqdm

from build import mapf_pipeline
from scripts_py.version_9.mapf_pkg.shape_utils import ShapeUtils, AxisTransform
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils


class SegmentSurfaceCell(object):
    def __init__(
            self, path: np.ndarray, radius: float,
            left_direction: np.ndarray = None, right_direction: np.ndarray = None
    ):
        self.path = path
        self.radius = radius
        self.left_direction = left_direction
        self.right_direction = right_direction

        self.pcd_data: np.ndarray = None
        self.left_clamp_pcd: np.ndarray = None
        self.right_clamp_pcd: np.ndarray = None

    @staticmethod
    def pcd2line_dist(pcd: np.ndarray, line_0: np.ndarray, line_1: np.ndarray):
        line02pcd = pcd - line_0
        line12pcd = pcd - line_1
        line02line1 = line_1 - line_0

        line02pcd_dist = np.linalg.norm(line02pcd, ord=2, axis=1)
        line12pcd_dist = np.linalg.norm(line12pcd, ord=2, axis=1)
        line02line1_dist = np.linalg.norm(line02line1, ord=2)

        cos_dist = np.sum(line02pcd * line02line1, axis=1) / line02line1_dist

        res_dist = np.zeros(shape=(pcd.shape[0],))

        left_clamp_bool = cos_dist < 0
        res_dist[left_clamp_bool] = line02pcd_dist[left_clamp_bool]

        right_clamp_bool = cos_dist > line02line1_dist
        res_dist[right_clamp_bool] = line12pcd_dist[right_clamp_bool]

        inner_bool = np.bitwise_and(cos_dist >= 0., cos_dist <= line02line1_dist)
        res_dist[inner_bool] = np.sqrt(np.power(line02pcd_dist[inner_bool], 2.0) - np.power(cos_dist[inner_bool], 2.0))

        return res_dist

    @staticmethod
    def remove_inner_pcd(pcd_data: np.ndarray, path: np.ndarray, radius, relax_factor: float):
        pcd_tree = KDTree(pcd_data)
        valid_bool = np.full(shape=(pcd_data.shape[0],), fill_value=True, dtype=bool)
        for i in range(path.shape[0] - 1):
            l0, l1 = path[i, :], path[i + 1, :]
            line_length = np.linalg.norm(l1 - l0, ord=2)
            search_radius = np.sqrt(np.power(line_length * 0.5, 2.0) + np.power(radius, 2.0)) + 1e-6
            search_line_points = np.concatenate([l0.reshape((1, -1)), l1.reshape((1, -1))], axis=0)
            idxs_list, dist_list = pcd_tree.query_radius(search_line_points, r=search_radius, return_distance=True)

            for pcd_idxs, dists in zip(idxs_list, dist_list):
                if len(pcd_idxs) == 0:
                    continue

                real_dist = SegmentSurfaceCell.pcd2line_dist(pcd_data[pcd_idxs, :], l0, l1)
                fail_pcd_idxs = pcd_idxs[real_dist < radius - relax_factor]
                valid_bool[fail_pcd_idxs] = False

        pcd_data = pcd_data[valid_bool]
        return pcd_data

    def generate_pcd_by_cylinder(
            self, reso: float, with_left_clamp: bool, with_right_clamp: bool, relax_factor,
            deplicate_format: int = 4
    ):
        pcd_data = []
        for i in range(self.path.shape[0] - 1):
            last_xyz = self.path[i, :]
            cur_xyz = self.path[i + 1, :]

            xyz_dif = cur_xyz[:3] - last_xyz[:3]
            height = np.linalg.norm(xyz_dif, ord=2)
            norm_vec = xyz_dif / height
            rot_mat = AxisTransform.rot_mat_from_vectors(norm_vec)
            center = (cur_xyz[:3] + last_xyz[:3]) * 0.5
            pcd = ShapeUtils.create_cylinder_pcd(
                center, self.radius, height, None, reso, is_solid=False,
                left_camp='wireframe', right_clamp='wireframe',
                rot_mat=rot_mat
            )
            pcd_data.append(pcd)

        pcd_data = np.concatenate(pcd_data, axis=0)

        pcd_data = np.round(pcd_data, decimals=deplicate_format)
        pcd_data = pd.DataFrame(data=pcd_data, columns=['x', 'y', 'z'])
        pcd_data.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
        pcd_data = pcd_data[['x', 'y', 'z']].values

        pcd_data = self.remove_inner_pcd(pcd_data, self.path, self.radius, relax_factor)
        if pcd_data.shape[0] > 20000:
            pcd_idxs = np.random.choice(np.arange(0, pcd_data.shape[0], 1), size=20000)
            pcd_data = pcd_data[pcd_idxs, :]

        if with_left_clamp:
            left_clamp_pcd_data = self.generate_left_clamp(reso)
            if left_clamp_pcd_data.shape[0] > 3000:
                pcd_idxs = np.random.choice(np.arange(0, left_clamp_pcd_data.shape[0], 1), size=3000)
                left_clamp_pcd_data = left_clamp_pcd_data[pcd_idxs, :]
            else:
                left_clamp_pcd_data = left_clamp_pcd_data
            self.left_clamp_pcd = left_clamp_pcd_data

        if with_right_clamp:
            right_clamp_pcd_data = self.generate_right_clamp(reso)
            if right_clamp_pcd_data.shape[0] > 3000:
                pcd_idxs = np.random.choice(np.arange(0, right_clamp_pcd_data.shape[0], 1), size=3000)
                right_clamp_pcd_data = right_clamp_pcd_data[pcd_idxs, :]
            else:
                right_clamp_pcd_data = right_clamp_pcd_data
            self.right_clamp_pcd = right_clamp_pcd_data

        self.pcd_data = pcd_data

    def generate_pcd_by_sphere(
            self, length_reso: float, sphere_reso: float, relax_factor: float, deplicate_format: int = 4
    ):
        pcd_data = []
        for i in range(self.path.shape[0] - 1):
            last_xyz = self.path[i, :]
            cur_xyz = self.path[i + 1, :]

            xyz_dif = cur_xyz[:3] - last_xyz[:3]
            length = np.linalg.norm(xyz_dif, ord=2)
            norm_vec = xyz_dif / length
            step = math.ceil(length / length_reso)
            step_length = length / step

            for j in range(step + 1):
                center = last_xyz + j * norm_vec * step_length
                pcd = ShapeUtils.create_sphere_pcd(center, norm_vec, self.radius, sphere_reso)
                pcd_data.append(pcd)

        pcd_data = np.concatenate(pcd_data, axis=0)

        pcd_data = np.round(pcd_data, decimals=deplicate_format)
        pcd_data = pd.DataFrame(data=pcd_data, columns=['x', 'y', 'z'])
        pcd_data.drop_duplicates(subset=['x', 'y', 'z'], inplace=True)
        pcd_data = pcd_data[['x', 'y', 'z']].values

        pcd_data = self.remove_inner_pcd(pcd_data, self.path, self.radius, relax_factor)

        self.pcd_data = pcd_data

    @staticmethod
    def generate_clamp(radius: float, direction: np.ndarray, center: np.ndarray, reso: float):
        uvs = []
        u_steps = math.ceil(radius / reso)
        for cell_radius in np.linspace(0, radius, u_steps):
            length = 2 * cell_radius * np.pi
            num = max(math.ceil(length / reso), 1)
            rads = np.deg2rad(np.linspace(0, 360.0, num))
            uv = np.zeros(shape=(num, 2))
            uv[:, 0] = np.cos(rads) * cell_radius
            uv[:, 1] = np.sin(rads) * cell_radius
            uvs.append(uv)
        uvs = np.concatenate(uvs, axis=0)
        pcd = np.concatenate([np.zeros((uvs.shape[0], 1)), uvs[:, 0:1], uvs[:, 1:2]], axis=1)

        norm_vec = direction / np.linalg.norm(direction)
        rot_mat = AxisTransform.rot_mat_from_vectors(norm_vec)
        pcd = AxisTransform.pcd_transform(pcd, rot_mat)
        pcd = pcd + center
        return pcd

    def generate_left_clamp(self, reso):
        center = self.path[0, :]
        if self.left_direction is None:
            direction = self.path[1, :] - self.path[0, :]
        else:
            direction = self.left_direction
        return self.generate_clamp(self.radius, direction, center, reso)

    def generate_right_clamp(self, reso):
        center = self.path[-1, :]
        if self.right_direction is None:
            direction = self.path[-1, :] - self.path[-2, :]
        else:
            direction = self.right_direction
        return self.generate_clamp(self.radius, direction, center, reso)

    def draw(self, with_path=True, with_pcd=True, vis: VisUtils = None):
        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        if with_path:
            line_set = VisUtils.create_line_set(np.arange(0, self.path.shape[0], 1))
            line_mesh = VisUtils.create_line(self.path, line_set)
            vis.plot(line_mesh)

        if with_pcd:
            pcd_mesh = VisUtils.create_point_cloud(self.pcd_data)
            vis.plot(pcd_mesh, color=(0.5, 0.5, 0.5))

            if self.left_clamp_pcd is not None:
                left_mesh = VisUtils.create_point_cloud(self.left_clamp_pcd)
                vis.plot(left_mesh, color=(1.0, 0.0, 0.0))

            if self.right_clamp_pcd is not None:
                right_mesh = VisUtils.create_point_cloud(self.right_clamp_pcd)
                vis.plot(right_mesh, color=(0.0, 1.0, 0.0))

        if show_plot:
            vis.show()


class Pcd2MeshConverter(object):
    def __init__(self):
        self.segment_cell: Dict[str, Dict[str, Union[SegmentSurfaceCell, bool]]] = {}

    def add_segment(
            self, name: str, path: np.ndarray, radius: float, reso_info: Dict,
            left_direction: np.ndarray = None, right_direction: np.ndarray = None,
            with_left_clamp=False, with_right_clamp=False
    ):
        self.segment_cell[name] = {
            'surface_cell': SegmentSurfaceCell(path, radius, left_direction, right_direction),
            'with_left_clamp': with_left_clamp,
            'with_right_clamp': with_right_clamp,
            'reso_info': reso_info
        }

    def generate_pcd_data(self):
        for name, info in tqdm(self.segment_cell.items()):
            # info['surface_cell'].generate_pcd_by_cylinder(
            #     reso=info['reso_info']['reso'],
            #     with_left_clamp=info['with_left_clamp'],
            #     with_right_clamp=info['with_right_clamp'],
            #     relax_factor=info['reso_info']['relax_factor']
            # )
            info['surface_cell'].generate_pcd_by_sphere(
                length_reso=info['reso_info']['length_reso'],
                sphere_reso=info['reso_info']['sphere_reso'],
                relax_factor=info['reso_info']['relax_factor']
            )

    def remove_inner_pcd(self):
        for _, info_i in tqdm(self.segment_cell.items()):
            seg_cell_i = info_i['surface_cell']
            for _, info_j in self.segment_cell.items():
                seg_cell_j = info_j['surface_cell']

                seg_cell_i.pcd_data = SegmentSurfaceCell.remove_inner_pcd(
                    seg_cell_i.pcd_data, seg_cell_j.path, seg_cell_j.radius,
                    relax_factor=info_j['reso_info']['relax_factor']
                )

    def get_pcd_data(self):
        pcds = []
        for name, info in self.segment_cell.items():
            cell = info['surface_cell']
            pcds.append(cell.pcd_data)
            if cell.left_clamp_pcd is not None:
                pcds.append(cell.left_clamp_pcd)
            if cell.right_clamp_pcd is not None:
                pcds.append(cell.right_clamp_pcd)
        pcds = np.concatenate(pcds, axis=0)
        return pcds

    @staticmethod
    def pcd2mesh(pcd: np.ndarray, alpha=0.0, tol=0.001, offset=2.5) -> pyvista.PolyData:
        cloud = pyvista.PolyData(pcd)
        return cloud.delaunay_3d(
            alpha=alpha,
            tol=tol,
            offset=offset,
            progress_bar=True
        ).extract_geometry()

    @staticmethod
    def save_ply(mesh: pyvista.PolyData, file: str):
        assert file.endswith('.ply')
        mesh.save(file)
