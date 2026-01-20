"""
想法转为:
1.为防止仿真的失真,所有映射均由gmsh建模形成而不再使用Topology Weight (要测试一下gmsh的效能)
2.由于不再使用Topology Weight,这意味着整个Mesh表面都是标准化的,并且将Mesh内部划分为标准的Cell(例如Box),Cell采用内部多点的均值计算
3.由SIMP方法转向BESO/PTO方法,将优化出的形状映射为由多个小型Box构成的
"""

import os
import shutil
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import pyvista
import math
import json


class GmshUtils(object):
    @staticmethod
    def define_gmsh_kernel():
        return 'SetFactory("OpenCASCADE");'
    
    @staticmethod
    def define_gmsh_curve_physical_group(group_name: str, group_id: int, tags: list[int]) -> str:
        assert group_id > 0, "gmsh only supports positive group id"
        return f'Physical Curve(\"{group_name}\", {group_id}) = {{{", ".join([str(tag) for tag in tags])}}};'
    
    @staticmethod
    def define_gmsh_surface_physical_group(group_name: str, group_id: int, tags: list[int]) -> str:
        assert group_id > 0, "gmsh only supports positive group id"
        return f'Physical Surface(\"{group_name}\", {group_id}) = {{{", ".join([str(tag) for tag in tags])}}};'

    @staticmethod
    def define_gmsh_volume_physical_group(group_name: str, group_id: int, tags: list[int]) -> str:
        assert group_id > 0, "gmsh only supports positive group id"
        return f'Physical Volume(\"{group_name}\", {group_id}) = {{{", ".join([str(tag) for tag in tags])}}};'
  
    @staticmethod
    def define_gmsh_point(point_id: int, x: float, y: float, z: float, size: float = 0.1):
        assert point_id > 0, "gmsh only supports positive point id"
        return f'Point({point_id}) = {{{x}, {y}, {z}, {size}}};'
    
    @staticmethod
    def define_gmsh_line(line_id: int, point_id_1: int, point_id_2: int):
        assert line_id > 0, "gmsh only supports positive line id"
        return f'Line({line_id}) = {{{point_id_1}, {point_id_2}}};'
    
    @staticmethod
    def define_gmsh_surface(surface_id: int, line_ids: list[int]):
        assert surface_id > 0, "gmsh only supports positive surface id"
        content = f'Curve Loop({surface_id}) = {{{", ".join([str(line_id) for line_id in line_ids])}}};\n'
        content += f'Plane Surface({surface_id}) = {{{surface_id}}};'
        return content
    
    @staticmethod
    def define_gmsh_volume(volume_id: int, surface_ids: list[int]):
        assert volume_id > 0, "gmsh only supports positive volume id"
        content = f'Surface Loop({volume_id}) = {{{", ".join([str(surface_id) for surface_id in surface_ids])}}};\n'
        content += f'Volume({volume_id}) = {{{volume_id}}};'
        return content

    @staticmethod
    def define_mesh_size(min_size: float, max_size: float):
        return f'Mesh.MeshSizeMin = {min_size};\nMesh.MeshSizeMax = {max_size};\n'


class GridMesher(object):
    def __init__(
        self, 
        dim: int,
        left_bottom_corner: np.ndarray,
        right_top_corner: np.ndarray,
        resolution: float,
    ):
        self.resolution = resolution
        self.left_bottom_corner = left_bottom_corner
        self.right_top_corner = right_top_corner
        self.dim = dim
        
        self.points: np.ndarray
        self.line2points: np.ndarray
        self.surface2lines: np.ndarray
        self.volume2surfaces: np.ndarray
        self.cell2corners: np.ndarray
        self.cell2neighbors: list[np.ndarray]
        
        self.generate_init_topology()
        self.fixed_boundaries = {}
       
    def generate_init_topology(self):
        if self.dim == 2:
            x_step = math.ceil((self.right_top_corner[0] - self.left_bottom_corner[0]) / self.resolution)
            y_step = math.ceil((self.right_top_corner[1] - self.left_bottom_corner[1]) / self.resolution)
            xs = np.linspace(self.left_bottom_corner[0], self.right_top_corner[0], x_step)
            ys = np.linspace(self.left_bottom_corner[1], self.right_top_corner[1], y_step)
            xs, ys = np.meshgrid(xs, ys)
            xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)
            
            xys_shape = xys.shape[:-1]
            points = xys.reshape((-1, 2))
            xys_idxs = np.arange(points.shape[0])
            xys_idxs = xys_idxs.reshape(xys_shape)
            
            line_x = np.concatenate([xys_idxs[:-1, :, np.newaxis], xys_idxs[1:, :, np.newaxis]], axis=-1)
            line_x_shape = line_x.shape[:-1]
            line_x = line_x.reshape((-1, 2))
            line_x_idxs = np.arange(line_x.shape[0])
            line_x_idxs = line_x_idxs.reshape(line_x_shape)
            
            line_y = np.concatenate([xys_idxs[:, :-1, np.newaxis], xys_idxs[:, 1:, np.newaxis]], axis=-1)
            line_y_shape = line_y.shape[:-1]
            line_y = line_y.reshape((-1, 2))
            line_y_idxs = np.arange(line_x.shape[0], line_x.shape[0] + line_y.shape[0], 1)
            line_y_idxs = line_y_idxs.reshape(line_y_shape)
            
            surface2lines = np.concatenate([
                line_x_idxs[:, :-1, np.newaxis],  # positive direction
                line_y_idxs[1:, :, np.newaxis],   # positive direction
                line_x_idxs[:, 1:, np.newaxis],   # negative direction
                line_y_idxs[:-1, :, np.newaxis],  # negative direction
            ], axis=-1)
            surface2lines = surface2lines.reshape((-1, 4))
            
            surface2corners = np.concatenate([xys_idxs[:-1, :-1, np.newaxis], xys_idxs[1:, 1:, np.newaxis]], axis=-1)
            surface2corners = surface2corners.reshape((-1, 2))
            
            surface_centers = np.mean(points[surface2corners], axis=1)
            tree = KDTree(surface_centers)
            surface2neighbors = tree.query_radius(surface_centers, r=self.resolution * 1.25, return_distance=False)
            
            self.points = points
            self.line2points = np.concatenate([line_x, line_y], axis=0)
            self.surface2lines = surface2lines
            self.cell2corners = surface2corners
            self.cell2neighbors = surface2neighbors
            
        elif self.dim == 3:
            x_step = math.ceil((self.right_top_corner[0] - self.left_bottom_corner[0]) / self.resolution)
            y_step = math.ceil((self.right_top_corner[1] - self.left_bottom_corner[1]) / self.resolution)
            z_step = math.ceil((self.right_top_corner[2] - self.left_bottom_corner[2]) / self.resolution)
            
            xs = np.linspace(self.left_bottom_corner[0], self.right_top_corner[0], x_step)
            ys = np.linspace(self.left_bottom_corner[1], self.right_top_corner[1], y_step)
            zs = np.linspace(self.left_bottom_corner[2], self.right_top_corner[2], z_step)
            xs, ys, zs = np.meshgrid(xs, ys, zs)
            xyzs = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]], axis=-1)
            xyzs_shape = xyzs.shape[:-1]
            xyzs = xyzs.reshape((-1, 3))
            xyzs_idxs = np.arange(xyzs.shape[0])
            xyzs_idxs = xyzs_idxs.reshape(xyzs_shape)
            
            line_x = np.concatenate([xyzs_idxs[:-1, :, :, np.newaxis], xyzs_idxs[1:, :, :, np.newaxis]], axis=-1)
            line_x_shape = line_x.shape[:-1]
            line_x = line_x.reshape((-1, 2))
            line_x_idxs = np.arange(line_x.shape[0])
            line_x_idxs = line_x_idxs.reshape(line_x_shape)
            
            line_y = np.concatenate([xyzs_idxs[:, :-1, :, np.newaxis], xyzs_idxs[:, 1:, :, np.newaxis]], axis=-1)
            line_y_shape = line_y.shape[:-1]
            line_y = line_y.reshape((-1, 2))
            line_y_idxs = np.arange(line_x.shape[0], line_x.shape[0] + line_y.shape[0], 1)
            line_y_idxs = line_y_idxs.reshape(line_y_shape)
            
            line_z = np.concatenate([xyzs_idxs[:, :, :-1, np.newaxis], xyzs_idxs[:, :, 1:, np.newaxis]], axis=-1)
            line_z_shape = line_z.shape[:-1]
            line_z = line_z.reshape((-1, 2))
            line_z_idxs = np.arange(line_x.shape[0] + line_y.shape[0], line_x.shape[0] + line_y.shape[0] + line_z.shape[0], 1)
            line_z_idxs = line_z_idxs.reshape(line_z_shape)
            
            surface_count = 0
            surface_xy = np.concatenate([
                line_x_idxs[:,  :-1, :, np.newaxis],  # positive direction
                line_y_idxs[1:, :,   :, np.newaxis],   # positive direction
                line_x_idxs[:,  1:,  :, np.newaxis],   # negative direction
                line_y_idxs[:-1, :,  :, np.newaxis],  # negative direction
            ], axis=-1)
            surface_xy_shape = surface_xy.shape[:-1]
            surface_xy = surface_xy.reshape((-1, 4))
            surface_xy_idxs = np.arange(surface_count, surface_count + surface_xy.shape[0], 1)
            surface_xy_idxs = surface_xy_idxs.reshape(surface_xy_shape)
            surface_count += surface_xy.shape[0]
            
            surface_xz = np.concatenate([
                line_x_idxs[:,   :, :-1, np.newaxis],  # positive direction
                line_z_idxs[1:,  :, :, np.newaxis],    # positive direction
                line_x_idxs[:,   :, 1:, np.newaxis],   # negative direction
                line_z_idxs[:-1, :, :, np.newaxis],    # negative direction
            ], axis=-1)
            surface_xz_shape = surface_xz.shape[:-1]
            surface_xz = surface_xz.reshape((-1, 4))
            surface_xz_idxs = np.arange(surface_count, surface_count + surface_xz.shape[0], 1)
            surface_xz_idxs = surface_xz_idxs.reshape(surface_xz_shape)
            surface_count += surface_xz.shape[0]
            
            surface_yz = np.concatenate([
                line_y_idxs[:,  :,   :-1, np.newaxis],  # positive direction
                line_z_idxs[:, 1:,   :, np.newaxis],    # positive direction
                line_y_idxs[:,  :,  1:, np.newaxis],   # negative direction
                line_z_idxs[:,  :-1, :, np.newaxis],    # negative direction
            ], axis=-1)
            surface_yz_shape = surface_yz.shape[:-1]
            surface_yz = surface_yz.reshape((-1, 4))
            surface_yz_idxs = np.arange(surface_count, surface_count + surface_yz.shape[0], 1)
            surface_yz_idxs = surface_yz_idxs.reshape(surface_yz_shape)
            
            surface2lines = np.concatenate([surface_xy, surface_xz, surface_yz], axis=0)

            box2surfaces = np.concatenate([
                surface_xy_idxs[:,   :,   :-1, np.newaxis],
                surface_xy_idxs[:,   :,  1:, np.newaxis],
                surface_xz_idxs[:,   :-1, :, np.newaxis],
                surface_xz_idxs[:,  1:,   :, np.newaxis],
                surface_yz_idxs[:-1, :,   :, np.newaxis],
                surface_yz_idxs[1:,  :,   :, np.newaxis],
            ], axis=-1)
            box2surfaces = box2surfaces.reshape((-1, 6))
            
            box2corners = np.concatenate([xyzs_idxs[:-1, :-1, :-1, np.newaxis], xyzs_idxs[1:, 1:, 1:, np.newaxis]], axis=-1)
            box2corners = box2corners.reshape((-1, 2))
            
            box_centers = np.mean(xyzs[box2corners], axis=1)
            tree = KDTree(box_centers)
            box2neighbors = tree.query_radius(box_centers, r=self.resolution * 1.25, return_distance=False)
            
            self.points = xyzs
            self.line2points = np.concatenate([line_x, line_y, line_z], axis=0)
            self.surface2lines = surface2lines
            self.volume2surfaces = box2surfaces
            self.cell2corners = box2corners
            self.cell2neighbors = box2neighbors
            
        else:
            raise ValueError(f"[Error]: non valid dim {self.dim}.")
    
    @property
    def cell_centers(self):
        return np.mean(self.points[self.cell2corners], axis=1)

    @property
    def surface_centers(self):
        return np.mean(self.points[self.line2points[self.surface2lines]], axis=(1, 2))

    @property
    def cell_num(self):
        return self.cell2corners.shape[0]

    @property
    def line_num(self):
        return self.line2points.shape[0]
    
    @property
    def surface_num(self):
        return self.surface2lines.shape[0]
    
    @property
    def volume_num(self):
        return self.volume2surfaces.shape[0]

    @property
    def point_num(self):
        return self.points.shape[0]

    # ------ boundary operation ------
    def add_group_boundary(
        self,
        face_center: np.ndarray, face_radius: float, face_normal: np.ndarray,
        group_name: str, group_id: int, boundary_type: Literal['velocity', 'pressure', 'wall'],
    ):
        assert boundary_type != 'wall', "Wall boundary should not be fixed"
        assert group_name not in self.fixed_boundaries, f"Group {group_name} already exists"
        assert group_id > 0, "Group id must be greater than 0"
        assert face_radius > 0, "Face radius must be greater than 0"
        assert face_radius * 2.0 >= self.resolution, "Face radius must be greater than resolution * 0.5"
        
        face_normal = face_normal / np.linalg.norm(face_normal)
        self.fixed_boundaries[group_name] = {
            'group_id': group_id,
            'face_center': face_center,
            'face_radius': face_radius,
            'face_normal': face_normal,
            'boundary_type': boundary_type,
        }
    
    # ------ field operation ------
    def extract_field_cell_idxs(self, field_value: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        field_bool = field_value > threshold
        cells_count = np.zeros(shape=self.cell_num)
        for neighbor_cell_idxs in self.cell2neighbors[field_bool]:  # type: ignore
            cells_count[neighbor_cell_idxs] += 1
        face_num = 4 if self.dim == 2 else 6
        inner_bool = cells_count != face_num
        outer_bool = cells_count == face_num
        cell_idxs = np.arange(self.cell_num)
        inner_idxs = cell_idxs[inner_bool]
        outer_idxs = cell_idxs[outer_bool]
        return inner_idxs, outer_idxs

    def extract_field_boundary_idxs(self, field_value: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        if field_value is a 3D field, retrun indices of the boundary surfaces
        if field_value is a 2D field, retrun indices of the boundary lines
        """
        field_bool = field_value >= threshold
        if self.dim == 2:
            boundary_idxs = np.arange(self.line_num)
            boundary_count = np.zeros(shape=self.line_num)
            for line_idxs in self.surface2lines[field_bool]:
                boundary_count[line_idxs] += 1
        else:
            boundary_idxs = np.arange(self.surface_num)
            boundary_count = np.zeros(shape=self.surface_num)
            for surface_idxs in self.volume2surfaces[field_bool]:
                boundary_count[surface_idxs] += 1
        return boundary_idxs[boundary_count == 1]

    # ------ gmsh operation ------
    @staticmethod
    def sort_line_segments(line_segments: np.ndarray, line_idxs: np.ndarray):        
        unique_ps = np.unique(line_segments)
        path_dict = {p: {} for p in unique_ps}
        p_exist_dict = {p: False for p in unique_ps}
        
        for line_idx, (p1, p2) in enumerate(line_segments):
            path_dict[p1][p2] = {'direction': 1, 'line_idx': line_idx}
            path_dict[p2][p1] = {'direction': -1, 'line_idx': line_idx}
        
        sorted_line_idxs = [0]
        sorted_directions = [1]
        start_p = line_segments[sorted_line_idxs[-1]][0]
        p_exist_dict[start_p] = True
        while True:
            if sorted_directions[-1] == 1:
                p2 = line_segments[sorted_line_idxs[-1]][1]
            else:
                p2 = line_segments[sorted_line_idxs[-1]][0]
            if p_exist_dict[p2]:
                break
            
            p_exist_dict[p2] = True
            for other_p in path_dict[p2].keys():
                if not p_exist_dict[other_p]:
                    info = path_dict[p2][other_p]
                    sorted_line_idxs.append(info['line_idx'])
                    sorted_directions.append(info['direction'])
                    break
        
        for other_p in path_dict[p2].keys():
            if other_p == start_p:
                sorted_line_idxs.append(path_dict[p2][other_p]['line_idx'])
                sorted_directions.append(path_dict[p2][other_p]['direction'])
                break
        
        sorted_line_idxs = np.array(sorted_line_idxs)
        sorted_line_idxs = line_idxs[sorted_line_idxs]
        sorted_directions = np.array(sorted_directions)
        assert len(sorted_line_idxs) == line_segments.shape[0], "Sorted line idxs length mismatch"
        
        # sorted_line_segments = line_segments[sorted_line_idxs]
        # b = sorted_line_segments[sorted_directions == -1]
        # sorted_line_segments[sorted_directions == -1, 0] = b[:, 1]
        # sorted_line_segments[sorted_directions == -1, 1] = b[:, 0]
        # print(sorted_line_segments)
                
        return (sorted_line_idxs * sorted_directions).astype(int).tolist()

    def extract_2d_field(self, field_value: np.ndarray, threshold: float, output_type: Literal['geo', 'vtk'], **kwargs):
        line_idxs = self.extract_field_boundary_idxs(field_value, threshold)
        point_idxs = np.unique(self.line2points[line_idxs])
        
        vertices = self.points[point_idxs]
        vertice_idxs = np.arange(vertices.shape[0])
        p_idxs_map = np.zeros(shape=self.point_num, dtype=int)
        p_idxs_map[point_idxs] = vertice_idxs            
        new_line2points = p_idxs_map[self.line2points[line_idxs]]
        
        max_group_id = -1
        for _, bry_info in self.fixed_boundaries.items():
            max_group_id = max(max_group_id, bry_info['group_id'])
        max_group_id += 1
        
        group_idx_to_name = {}
        point_group_idxs = np.ones(shape=vertices.shape[0], dtype=int) * max_group_id
        line_group_idxs = np.ones(shape=new_line2points.shape[0], dtype=int) * max_group_id
        for group_name, group_info in self.fixed_boundaries.items():
            face_center = group_info['face_center']
            face_radius = group_info['face_radius']
            face_normal = group_info['face_normal']
            group_id = group_info['group_id']
            group_idx_to_name[group_id] = group_name
            
            inside_points_bool = np.bitwise_and(
                np.linalg.norm(vertices - face_center, axis=1) < face_radius,
                np.abs(np.dot(vertices - face_center, face_normal)) < self.resolution * 0.5
            )
            inside_lines_bool = np.sum(inside_points_bool[new_line2points], axis=1) > 0
            assert np.sum(inside_lines_bool) > 0, f"No lines found inside group {group_name}"
            
            point_group_idxs[inside_points_bool] = int(group_id)
            line_group_idxs[inside_lines_bool] = int(group_id)
        
        if output_type == 'vtk':
            if vertices.shape[1] == 2:
                vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1))], axis=1)
            mesh = pyvista.UnstructuredGrid({pyvista.CellType.LINE: new_line2points}, vertices)
            mesh.point_data['group_id'] = point_group_idxs
            return mesh
        
        elif output_type == 'geo':
            file_path = kwargs['file_path']
            assert file_path.endswith('geo'), "file_path must be a .geo file"
            
            point_idxs = np.arange(vertices.shape[0]) + 1
            line_idxs = np.arange(new_line2points.shape[0]) + 1
            new_line2points = new_line2points + 1
            surface_idx = 1
            
            with open(file_path, 'w') as f:
                f.write(f'{GmshUtils.define_gmsh_kernel()}\n')
                f.write('\n')
                
                for p_idx, vertex in zip(point_idxs, vertices):
                    f.write(f'{GmshUtils.define_gmsh_point(p_idx, vertex[0], vertex[1], 0.0)}\n')
                f.write('\n')
                              
                for line_idx, line_conn in zip(line_idxs, new_line2points):
                    f.write(f'{GmshUtils.define_gmsh_line(line_idx, line_conn[0], line_conn[1])}\n')
                f.write('\n')
                
                # ------ sort line2points to ensure the line is always in the same direction ------                
                f.write(f'{GmshUtils.define_gmsh_surface(
                    surface_idx, line_ids=self.sort_line_segments(new_line2points, line_idxs)
                )}\n')
                f.write('\n')
                
                for group_id in group_idx_to_name.keys():
                    group_name = group_idx_to_name[group_id]
                    if group_id == max_group_id:
                        continue
                    f.write(f'{GmshUtils.define_gmsh_curve_physical_group(
                        group_name, group_id, [line_idx for line_idx in line_idxs[line_group_idxs == group_id]]
                    )}\n')
                f.write(f'{GmshUtils.define_gmsh_curve_physical_group(
                    'wall', max_group_id, [line_idx for line_idx in line_idxs[line_group_idxs == max_group_id]]
                )}\n')
                f.write('\n')
                
                f.write(f'{GmshUtils.define_gmsh_surface_physical_group(group_name='surface', group_id=1, tags=[surface_idx])}\n')
                f.write('\n')
            
                if 'min_size' in kwargs and 'max_size' in kwargs:
                    f.write(f'{GmshUtils.define_mesh_size(kwargs['min_size'], kwargs['max_size'])}\n')
                f.write('\n')
        
        else:
            raise ValueError(f"Invalid output type: {output_type}")
    
    def extract_3d_field(self, field_value: np.ndarray, threshold: float, output_type: Literal['geo', 'vtk'], **kwargs):
        surface_idxs = self.extract_field_boundary_idxs(field_value, threshold)
        line_idxs = np.unique(self.surface2lines[surface_idxs])
        point_idxs = np.unique(self.line2points[line_idxs])
        
        vertices = self.points[point_idxs]
        vertice_idxs = np.arange(vertices.shape[0])
        p_idxs_map = np.zeros(shape=self.point_num, dtype=int)
        p_idxs_map[point_idxs] = vertice_idxs            
        new_line2points = p_idxs_map[self.line2points[line_idxs]]
        
        new_line_idxs = np.arange(line_idxs.shape[0])
        line_idxs_map = np.zeros(shape=self.line_num, dtype=int)
        line_idxs_map[line_idxs] = new_line_idxs
        new_surface2lines = line_idxs_map[self.surface2lines[surface_idxs]]
        
        max_group_id = -1
        for _, bry_info in self.fixed_boundaries.items():
            max_group_id = max(max_group_id, bry_info['group_id'])
        max_group_id += 1
        
        group_idx_to_name = {}
        point_group_idxs = np.ones(shape=vertices.shape[0], dtype=int) * max_group_id
        surface_group_idxs = np.ones(shape=new_surface2lines.shape[0], dtype=int) * max_group_id
        for group_name, group_info in self.fixed_boundaries.items():
            face_center = group_info['face_center']
            face_radius = group_info['face_radius']
            face_normal = group_info['face_normal']
            group_id = group_info['group_id']
            group_idx_to_name[group_id] = group_name
                        
            inside_points_bool = np.bitwise_and(
                np.linalg.norm(vertices - face_center, axis=1) < face_radius,
                np.abs(np.dot(vertices - face_center, face_normal)) < self.resolution * 1e-2
            )
            inside_lines_bool = np.sum(inside_points_bool[new_line2points], axis=1) > 0
            assert np.sum(inside_lines_bool) > 0, f"No lines found inside group {group_name}"
            inside_surfaces_bool = np.sum(inside_lines_bool[new_surface2lines], axis=1) > 0
            assert np.sum(inside_surfaces_bool) > 0, f"No surfaces found inside group {group_name}"
            
            point_group_idxs[inside_points_bool] = int(group_id)
            surface_group_idxs[inside_surfaces_bool] = int(group_id)
        
        if output_type == 'vtk':
            surface2points = new_line2points[new_surface2lines]
            surface2points = np.concatenate([
                surface2points[:, 0, 0, np.newaxis],
                surface2points[:, 1, 0, np.newaxis],
                surface2points[:, 2, 1, np.newaxis],
                surface2points[:, 3, 1, np.newaxis],
            ], axis=1)

            if vertices.shape[1] == 2:
                vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1))], axis=1)
            mesh = pyvista.UnstructuredGrid({pyvista.CellType.QUAD: surface2points}, vertices)
            mesh.point_data['group_id'] = point_group_idxs
            return mesh

        elif output_type == 'geo':
            file_path = kwargs['file_path']
            assert file_path.endswith('geo'), "file_path must be a .geo file"
            
            # TODO: please check, gmsh can not support id 0
            point_idxs = np.arange(vertices.shape[0]) + 1
            line_idxs = np.arange(new_line2points.shape[0]) + 1
            surface_idxs = np.arange(new_surface2lines.shape[0]) + 1
            new_line2points += 1
            new_surface2lines += 1
            
            volume_idx = 1
            
            with open(file_path, 'w') as f:
                f.write(f'{GmshUtils.define_gmsh_kernel()}\n')
                f.write('\n')
                
                for p_idx, vertex in zip(point_idxs, vertices):
                    f.write(f'{GmshUtils.define_gmsh_point(p_idx, vertex[0], vertex[1], vertex[2])}\n')
                f.write('\n')
                
                for line_idx, line_conn in zip(line_idxs, new_line2points):
                    f.write(f'{GmshUtils.define_gmsh_line(line_idx, line_conn[0], line_conn[1])}\n')
                f.write('\n')
    
                for surface_idx, surface_conn in zip(surface_idxs, new_surface2lines):
                    f.write(f'{GmshUtils.define_gmsh_surface(surface_idx, surface_conn)}\n')
                f.write('\n')
                
                f.write(f'{GmshUtils.define_gmsh_volume(volume_idx, surface_ids=surface_idxs)}\n')
                f.write('\n')
                
                for group_id in group_idx_to_name.keys():
                    group_name = group_idx_to_name[group_id]
                    if group_id == max_group_id:
                        continue
                    f.write(f'{GmshUtils.define_gmsh_surface_physical_group(
                        group_name, group_id, [surface_idx for surface_idx in surface_idxs[surface_group_idxs == group_id]]
                    )}\n')
                f.write(f'{GmshUtils.define_gmsh_surface_physical_group(
                    'wall', max_group_id, [surface_idx for surface_idx in surface_idxs[surface_group_idxs == max_group_id]]
                )}\n')
                f.write('\n')
                
                f.write(f'{GmshUtils.define_gmsh_volume_physical_group(group_name='volume', group_id=1, tags=[volume_idx])}\n')
                f.write('\n')
                
                if 'min_size' in kwargs and 'max_size' in kwargs:
                    f.write(f'{GmshUtils.define_mesh_size(kwargs['min_size'], kwargs['max_size'])}\n')
        
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    def extract_field(self, field_value: np.ndarray, threshold: float, output_type: Literal['geo', 'vtk'], **kwargs):
        if self.dim == 2:
            return self.extract_2d_field(field_value, threshold, output_type, **kwargs)
        elif self.dim == 3:
            return self.extract_3d_field(field_value, threshold, output_type, **kwargs)
        else:
            raise ValueError(f"Invalid dimension: {self.dim}")

    # ------ save & load ------
    def save(self, save_dir: str):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        
        points_file = os.path.join(save_dir, 'points.npy')
        np.save(points_file, self.points)
        
        line2points_file = os.path.join(save_dir, 'line2points.npy')
        np.save(line2points_file, self.line2points)
        
        surface2lines_file = os.path.join(save_dir, 'surface2lines.npy')
        np.save(surface2lines_file, self.surface2lines)
        
        if self.dim == 3:
            volume2surfaces_file = os.path.join(save_dir, 'volume2surfaces.npy')
            np.save(volume2surfaces_file, self.volume2surfaces)
        else:
            volume2surfaces_file = None
        
        cell2corners_file = os.path.join(save_dir, 'cell2corners.npy')
        np.save(cell2corners_file, self.cell2corners)
        
        cell2neighbors_file = os.path.join(save_dir, 'cell2neighbors.npy')
        np.save(cell2neighbors_file, self.cell2neighbors)
        
        config_file = os.path.join(save_dir, 'config.json')
        with open(config_file, 'w') as f:
            config = {
                'dim': self.dim,
                'left_bottom_corner': self.left_bottom_corner.tolist(),
                'right_top_corner': self.right_top_corner.tolist(),
                'resolution': self.resolution,
                'fixed_boundaries': {},
                'points_file': points_file,
                'line2points_file': line2points_file,
                'surface2lines_file': surface2lines_file,
                'volume2surfaces_file': volume2surfaces_file,
                'cell2corners_file': cell2corners_file,
                'cell2neighbors_file': cell2neighbors_file,
            }
            for group_name, group_info in self.fixed_boundaries.items():
                config['fixed_boundaries'][group_name] = {
                    'group_id': group_info['group_id'],
                    'face_center': list(group_info['face_center']),
                    'face_radius': group_info['face_radius'],
                    'face_normal': list(group_info['face_normal']),
                    'boundary_type': group_info['boundary_type'],
                }        
            json.dump(config, f, indent=4)
    
    def load(self, load_config_file: str):
        with open(load_config_file, 'r') as f:
            config = json.load(f)
        self.dim = config['dim']
        self.left_bottom_corner = np.array(config['left_bottom_corner'])
        self.right_top_corner = np.array(config['right_top_corner'])
        self.resolution = config['resolution']
        for group_name, group_info in config['fixed_boundaries'].items():
            self.add_group_boundary(
                face_center=np.array(group_info['face_center']),
                face_radius=group_info['face_radius'],
                face_normal=np.array(group_info['face_normal']),
                group_name=group_name,
                group_id=group_info['group_id'],
                boundary_type=group_info['boundary_type'],
            )
        self.points = np.load(config['points_file'], allow_pickle=True)
        self.line2points = np.load(config['line2points_file'], allow_pickle=True)
        self.surface2lines = np.load(config['surface2lines_file'], allow_pickle=True)
        if self.dim == 3:
            self.volume2surfaces = np.load(config['volume2surfaces_file'], allow_pickle=True)
        self.cell2corners = np.load(config['cell2corners_file'], allow_pickle=True)
        self.cell2neighbors = np.load(config['cell2neighbors_file'], allow_pickle=True)
        

def debug():
    mesher = GridMesher(
        left_bottom_corner=np.array([0.0, 0.0, 0.0]),
        right_top_corner=np.array([1.0, 1.0, 1.0]),
        resolution=0.04,
        dim=3,
    )
    mesher.add_group_boundary(
        face_center=np.array([0.0, 0.2, 0.2]),
        face_radius=0.1,
        face_normal=np.array([1.0, 0.0, 0.0]),
        group_name='group_1',
        group_id=1,
        boundary_type='velocity',
    )
    mesher.add_group_boundary(
        face_center=np.array([0.8, 1.0, 0.8]),
        face_radius=0.1,
        face_normal=np.array([0.0, 1.0, 0.0]),
        group_name='group_2',
        group_id=2,
        boundary_type='pressure',
    )

    path = np.array([
        [0.0, 0.2, 0.2],
        [0.5, 0.2, 0.2],
        [0.5, 0.6, 0.2],
        [0.5, 0.6, 0.6],
        [0.8, 0.8, 0.8],
        [0.8, 1.0, 0.8],
    ])
    detail_step = 0.025
    detail_path = []
    for i in range(path.shape[0] - 1):
        step = int(np.linalg.norm(path[i + 1] - path[i]) / detail_step)
        if i == 0:
            detail_path.append(np.linspace(path[i], path[i + 1], step))
        else:
            detail_path.append(np.linspace(path[i], path[i + 1], step)[1:])
    detail_path = np.concatenate(detail_path, axis=0)

    tree = KDTree(detail_path)
    dists_list, idxs_list = tree.query(mesher.cell_centers, k=1, return_distance=True)
    dists_list = np.concatenate(dists_list, axis=-1)
    inside_bool = dists_list < 0.1
    field_value = np.zeros(shape=mesher.cell2corners.shape[0])
    field_value[inside_bool] = 1.0

    # vtk_mesh = mesher.extract_3d_field(field_value, threshold=0.5, output_type='vtk')
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(vtk_mesh, style='wireframe', show_edges=True)
    # plotter.add_mesh(pyvista.PointSet(vtk_mesh.points[vtk_mesh.point_data['group_id'] == 1]), color='blue')
    # plotter.add_mesh(pyvista.PointSet(vtk_mesh.points[vtk_mesh.point_data['group_id'] == 2]), color='red')
    # plotter.show()

    # vtk_mesh = mesher.extract_3d_field(
    #     field_value, threshold=0.5, output_type='geo', 
    #     file_path='/home/admin123456/Desktop/work/gmsh_test/t44/test.geo',
    #     min_size=0.04,
    #     max_size=0.04,
    # )

if __name__ == '__main__':
    # debug()
    pass