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
import subprocess

from scripts_py.mesh_utils import MeshUtils
from scripts_py.gmsh_utils import GmshUtils
from scripts_py.simulator.dolfin_simulator import FluidBoundary


class GridMesher(object):
    def __init__(
        self, 
        dim: int,
        left_bottom_corner: np.ndarray,
        right_top_corner: np.ndarray,
        resolution: float,
        boundaries: list[FluidBoundary],
        inout_pipe_length: float = 0.05
    ):
        self.dim = dim
        self.resolution = resolution
        self.left_bottom_corner = left_bottom_corner
        self.right_top_corner = right_top_corner
        self.inout_pipe_length = inout_pipe_length
        
        self.points: np.ndarray
        self.line2points: np.ndarray
        self.surface2lines: np.ndarray
        self.volume2surfaces: np.ndarray
        self.cell2corners: np.ndarray
        self.cell2neighbors: list[np.ndarray]
        
        self.generate_init_topology()
        self.boundaries = boundaries
       
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
        """
        line_segments: each contains two point indexes
        line_idxs: line index of the line_segments
        """
        
        def find_one_segment(line_segments: np.ndarray, line_idxs: np.ndarray):
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
            
            contain_line = np.zeros(shape=line_segments.shape[0], dtype=bool)
            contain_line[sorted_line_idxs] = True
            
            sorted_line_idxs = line_idxs[sorted_line_idxs]
            sorted_directions = np.array(sorted_directions)
            
            # sorted_line_segments = line_segments[sorted_line_idxs]
            # b = sorted_line_segments[sorted_directions == -1]
            # sorted_line_segments[sorted_directions == -1, 0] = b[:, 1]
            # sorted_line_segments[sorted_directions == -1, 1] = b[:, 0]
            # print(sorted_line_segments)
            
            return sorted_line_idxs, sorted_directions, contain_line
            
        # assert len(sorted_line_idxs) == line_segments.shape[0], "Sorted line idxs length mismatch"
        
        cur_line_segments = line_segments.copy()
        cur_line_idxs = line_idxs.copy()
        sorted_res = []
        while True:
            sorted_line_idxs, sorted_directions, contain_line = find_one_segment(cur_line_segments, cur_line_idxs)
            sorted_res.append((sorted_line_idxs * sorted_directions).astype(int).tolist())
            
            rest_line_segments = cur_line_segments[~contain_line]
            rest_line_idxs = cur_line_idxs[~contain_line]
            if rest_line_segments.shape[0] == 0:
                break
            cur_line_segments = rest_line_segments
            cur_line_idxs = rest_line_idxs
        
        return sorted_res

    @staticmethod
    def find_surface_parts(surface_segments: np.ndarray, surface_idxs: np.ndarray):
        """
        surface_segments: each contains four line indexes
        surface_idxs: surface index of the surface_segments
        """
        
        def find_one_part(surface_segments: np.ndarray):
            line2surface_map = {}
            for map_i, line_idxs in enumerate(surface_segments):
                for line_idx in line_idxs:
                    if line_idx not in line2surface_map:
                        line2surface_map[line_idx] = []
                    line2surface_map[line_idx].append(map_i)
            
            contain_surface = np.zeros(shape=surface_segments.shape[0], dtype=bool)
            contain_surface[0] = True
            related_surface_map_idxs = [0]
            while True:
                new_related_line_idxs = []
                for map_idx in related_surface_map_idxs:
                    new_related_line_idxs.extend(surface_segments[map_idx].tolist())
                new_related_line_idxs = np.unique(np.array(new_related_line_idxs))
                
                new_related_surface_map_idxs = []
                for line_idx in new_related_line_idxs:
                    for new_surface_map_idx in line2surface_map[line_idx]:
                        if not contain_surface[new_surface_map_idx]:
                            new_related_surface_map_idxs.append(new_surface_map_idx)
                            contain_surface[new_surface_map_idx] = True
                
                new_related_surface_map_idxs = np.array(new_related_surface_map_idxs)
                if new_related_surface_map_idxs.shape[0] == 0:
                    break
                related_surface_map_idxs = np.unique(new_related_surface_map_idxs)
            
            return contain_surface
        
        cur_surface_segments = surface_segments.copy()
        cur_surface_idxs = surface_idxs.copy()
        surface_parts = []
        while True:
            contain_surface = find_one_part(cur_surface_segments)
            surface_parts.append(cur_surface_idxs[contain_surface])
            
            cur_surface_segments = cur_surface_segments[~contain_surface]
            cur_surface_idxs = cur_surface_idxs[~contain_surface]
            if cur_surface_segments.shape[0] == 0:
                break
        
        return surface_parts

    def extract_2d_field(self, field_value: np.ndarray, threshold: float, output_type: Literal['geo', 'vtk'], **kwargs):
        line_idxs = self.extract_field_boundary_idxs(field_value, threshold)
        point_idxs = np.unique(self.line2points[line_idxs])
        
        vertices = self.points[point_idxs]
        vertice_idxs = np.arange(vertices.shape[0])
        p_idxs_map = np.zeros(shape=self.point_num, dtype=int)
        p_idxs_map[point_idxs] = vertice_idxs            
        new_line2points = p_idxs_map[self.line2points[line_idxs]]
        
        if output_type == 'vtk':
            if vertices.shape[1] == 2:
                vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1))], axis=1)
            mesh = pyvista.UnstructuredGrid({pyvista.CellType.LINE: new_line2points}, vertices)
            return mesh
        
        elif output_type == 'geo':
            geo_file_path = kwargs['geo_file_path']
            assert geo_file_path.endswith('geo'), "file_path must be a .geo file"
            
            # ------ generate corase group idxs ------
            wall_marker = 9999999
            for bry_cfg in self.boundaries:
                if bry_cfg.boundary_type == 'wall':
                    wall_marker = bry_cfg.marker
                    break
                
            inout_marker = []
            line_marker_idxs = np.ones(shape=new_line2points.shape[0], dtype=int) * wall_marker
            for bry_cfg in self.boundaries:
                if bry_cfg.boundary_type == 'wall':
                    continue
                
                assert self.resolution < bry_cfg.face_radius * 2.0, f"Resolution is too large for boundary {bry_cfg.name}, it should be less than {bry_cfg.face_radius * 2.0}"
                inout_marker.append(bry_cfg.marker)
                inside_points_bool = np.bitwise_and(
                    np.linalg.norm(vertices - bry_cfg.face_center, axis=1) < bry_cfg.face_radius,
                    np.abs(np.dot(vertices - bry_cfg.face_center, bry_cfg.face_norm_vector)) < self.resolution * 0.5
                )
                
                inside_lines_bool = np.sum(inside_points_bool[new_line2points], axis=1) > 0
                assert np.sum(inside_lines_bool) > 0, f"No lines found inside group {bry_cfg.name}"
                line_marker_idxs[inside_lines_bool] = int(bry_cfg.marker)
            
            # ------ prepare gmsh file ------
            # NOTE: Be careful, gmsh can not support id 0
            point_idxs = np.arange(vertices.shape[0]) + 1
            line_idxs = np.arange(new_line2points.shape[0]) + 1
            new_line2points = new_line2points + 1
            surface_idx = 1
            
            with open(geo_file_path, 'w') as f:
                f.write(f'{GmshUtils.define_gmsh_kernel()}\n')
                f.write('\n')
                for p_idx, vertex in zip(point_idxs, vertices):
                    f.write(f'{GmshUtils.define_gmsh_point(p_idx, vertex[0], vertex[1], 0.0)}\n')
                f.write('\n')   
                for line_idx, line_conn in zip(line_idxs, new_line2points):
                    f.write(f'{GmshUtils.define_gmsh_line(line_idx, line_conn[0], line_conn[1])}\n')
                f.write('\n')
                
                # ------ sort line2points to ensure the line is always in the same direction ------
                curve_loops = self.sort_line_segments(new_line2points, line_idxs)
                curve_loop_idxs = (np.arange(len(curve_loops)) + 1).astype(int).tolist()
                for curve_loop_idx, curve_loop in zip(curve_loop_idxs, curve_loops):
                    f.write(f'{GmshUtils.define_gmsh_curve_loop(curve_loop_idx, curve_loop)}\n')
                f.write(f'{GmshUtils.define_gmsh_surface_from_curves(surface_idx, curve_loop_idxs)}\n')                
                f.write('\n')
                
                f.write(f'{GmshUtils.define_gmsh_all_lines_physical_group(group_name='lines', group_id=1)}\n')                
                f.write(f'{GmshUtils.define_gmsh_surface_physical_group(group_name='surface', group_id=1, tags=[surface_idx])}\n')
                f.write('\n')
                
                inout_curves = []
                for marker in inout_marker:
                    inout_curves.extend(line_idxs[line_marker_idxs == marker].tolist())
                f.write(f'{GmshUtils.define_field_mesh_size_to_curve(
                    distance_field_id=1,
                    threshold_field_id=2,
                    curve_tags=inout_curves,
                    sampling=300,
                    size_min=kwargs['min_size'],
                    size_max=kwargs['max_size'],
                    dist_min=kwargs['dist_min'],
                    dist_max=kwargs['dist_max'],
                )}\n')
                f.write(f'{GmshUtils.define_minimum_field(aggregate_field_id=3, field_ids=[2])}\n')
                # ------ generate fine based on coarse group idxs ------
                
                compress_mesh = kwargs.get('compress_mesh_extend_from_boundary', False)
                if compress_mesh:
                    f.write(f'{GmshUtils.compress_extend_from_boundary()}\n')  # TODO: please check whether it work
            
            msh_file = kwargs['msh_file_path']
            base_dir = os.path.dirname(msh_file)
            original_msh_file = os.path.join(base_dir, 'original.msh')
            log_file = os.path.join(base_dir, 'gmsh.log')
            cmd_str = f'gmsh {geo_file_path} -2 -o {original_msh_file} > {log_file}'
            subprocess.run(cmd_str, shell=True)
            assert os.path.exists(original_msh_file), f"Failed to generate msh file: {original_msh_file}"
            
            MeshUtils.refine_msh_from_boundaries(
                msh_file=original_msh_file,
                dim=self.dim,
                boundaries=self.boundaries,
                new_msh_file=msh_file
            )

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
            return mesh

        elif output_type == 'geo':
            geo_file_path = kwargs['geo_file_path']
            assert geo_file_path.endswith('geo'), "file_path must be a .geo file"

            if self.inout_pipe_length <= 0.0:
                wall_marker = 9999999
                for bry_cfg in self.boundaries:
                    if bry_cfg.boundary_type == 'wall':
                        wall_marker = bry_cfg.marker
                        break
                
                inout_marker = []
                surface_marker_idxs = np.ones(shape=new_surface2lines.shape[0], dtype=int) * wall_marker
                for bry_cfg in self.boundaries:
                    if bry_cfg.boundary_type == 'wall':
                        continue
                    
                    assert self.resolution < bry_cfg.face_radius * 2.0, f"Resolution is too large for boundary {bry_cfg.name}, it should be less than {bry_cfg.face_radius * 2.0}"
                    inside_points_bool = np.bitwise_and(
                        np.linalg.norm(vertices - bry_cfg.face_center, axis=1) < bry_cfg.face_radius,
                        np.abs(np.dot(vertices - bry_cfg.face_center, bry_cfg.face_norm_vector)) < self.resolution * 0.5
                    )
                    assert np.sum(inside_points_bool) > 0, f"No points found inside group {bry_cfg.name}"
                    inside_lines_bool = np.sum(inside_points_bool[new_line2points], axis=1) > 0
                    inside_surfaces_bool = np.sum(inside_lines_bool[new_surface2lines], axis=1) > 0
                    
                    surface_marker_idxs[inside_surfaces_bool] = int(bry_cfg.marker)
                    inout_marker.append(bry_cfg.marker)
                
                # NOTE: Be careful, gmsh can not support id 0
                point_idxs = np.arange(vertices.shape[0]) + 1
                line_idxs = np.arange(new_line2points.shape[0]) + 1
                surface_idxs = np.arange(new_surface2lines.shape[0]) + 1
                new_line2points += 1
                new_surface2lines += 1
                volume_idx = 1
                with open(geo_file_path, 'w') as f:
                    f.write(f'{GmshUtils.define_gmsh_kernel()}\n')
                    f.write('\n')
                    for p_idx, vertex in zip(point_idxs, vertices):
                        f.write(f'{GmshUtils.define_gmsh_point(p_idx, vertex[0], vertex[1], vertex[2])}\n')
                    f.write('\n')
                    for line_idx, line_conn in zip(line_idxs, new_line2points):
                        f.write(f'{GmshUtils.define_gmsh_line(line_idx, line_conn[0], line_conn[1])}\n')
                    f.write('\n')
                    for surface_idx, surface_conn in zip(surface_idxs, new_surface2lines):
                        f.write(f'{GmshUtils.define_gmsh_surface_from_lines(surface_idx, surface_conn)}\n')
                        
                    surface_loops = self.find_surface_parts(new_surface2lines, surface_idxs)
                    surface_loop_idxs = (np.arange(len(surface_loops)) + 1).astype(int).tolist()
                    for surface_loop_idx, surface_loop in zip(surface_loop_idxs, surface_loops):
                        f.write(f'{GmshUtils.define_gmsh_surface_loop(surface_loop_idx, surface_loop)}\n')
                    f.write(f'{GmshUtils.define_gmsh_volume_from_surface_loops(volume_idx, surface_loop_idxs)}\n')
                    f.write('\n')
                    
                    f.write(f'{GmshUtils.define_gmsh_all_surfaces_physical_group(group_name='surface', group_id=1)}\n')
                    f.write(f'{GmshUtils.define_gmsh_volume_physical_group(group_name='volume', group_id=1, tags=[1])}\n')
                    
                    inout_surfaces = []
                    for marker in inout_marker:
                        inout_surfaces.extend(surface_idxs[surface_marker_idxs == marker].tolist())
                    f.write(f'{GmshUtils.define_field_mesh_size_to_surface(
                        distance_field_id=1,
                        threshold_field_id=2,
                        surface_tags=inout_surfaces,
                        sampling=200,
                        size_min=kwargs['min_size'],
                        size_max=kwargs['max_size'],
                        dist_min=kwargs['dist_min'],
                        dist_max=kwargs['dist_max'],
                    )}\n')
                    f.write(f'{GmshUtils.define_minimum_field(aggregate_field_id=3, field_ids=[2])}\n')
                    
                    compress_mesh = kwargs.get('compress_mesh_extend_from_boundary', False)
                    if compress_mesh:
                        f.write(f'{GmshUtils.compress_extend_from_boundary()}\n')  # TODO: please check whether it work
                    
                msh_file = kwargs['msh_file_path']
                base_dir = os.path.dirname(msh_file)
                original_msh_file = os.path.join(base_dir, 'original.msh')
                log_file = os.path.join(base_dir, 'gmsh.log')
                cmd_str = f'gmsh {geo_file_path} -3 -o {original_msh_file} > {log_file}'
                subprocess.run(cmd_str, shell=True)
                assert os.path.exists(original_msh_file), f"Failed to generate msh file: {original_msh_file}"
                    
                MeshUtils.refine_msh_from_boundaries(
                    msh_file=original_msh_file,
                    dim=self.dim,
                    boundaries=self.boundaries,
                    new_msh_file=msh_file
                )
            
            else:
                # NOTE: Be careful, gmsh can not support id 0
                point_idxs = np.arange(vertices.shape[0]) + 1
                line_idxs = np.arange(new_line2points.shape[0]) + 1
                surface_idxs = np.arange(new_surface2lines.shape[0]) + 1
                new_line2points += 1
                new_surface2lines += 1
                volume_idx = 1
            
                with open(geo_file_path, 'w') as f:
                    f.write(f'{GmshUtils.define_gmsh_kernel()}\n')
                    f.write('\n')
                    for p_idx, vertex in zip(point_idxs, vertices):
                        f.write(f'{GmshUtils.define_gmsh_point(p_idx, vertex[0], vertex[1], vertex[2])}\n')
                    f.write('\n')
                    for line_idx, line_conn in zip(line_idxs, new_line2points):
                        f.write(f'{GmshUtils.define_gmsh_line(line_idx, line_conn[0], line_conn[1])}\n')
                    f.write('\n')
                    for surface_idx, surface_conn in zip(surface_idxs, new_surface2lines):
                        f.write(f'{GmshUtils.define_gmsh_surface_from_lines(surface_idx, surface_conn)}\n')
                    f.write('\n')
                    
                    surface_loops = self.find_surface_parts(new_surface2lines, surface_idxs)
                    surface_loop_idxs = (np.arange(len(surface_loops)) + 1).astype(int).tolist()
                    for surface_loop_idx, surface_loop in zip(surface_loop_idxs, surface_loops):
                        f.write(f'{GmshUtils.define_gmsh_surface_loop(surface_loop_idx, surface_loop)}\n')
                    f.write(f'{GmshUtils.define_gmsh_volume_from_surface_loops(volume_idx, surface_loop_idxs)}\n')
                    f.write('\n')
                    
                    source_volume_idx = volume_idx
                    union_volume_idxs = []
                    for bry_cfg in self.boundaries:
                        if bry_cfg.boundary_type == 'wall':
                            continue
                        volume_idx += 1
                        
                        f.write(f'{GmshUtils.define_gmsh_cylinder(
                            volume_id=volume_idx,
                            center=bry_cfg.face_center,
                            vector=bry_cfg.face_norm_vector * self.inout_pipe_length * (1.0 if bry_cfg.boundary_type == 'velocity' else -1.0),
                            radius=bry_cfg.face_radius,
                        )}\n')
                        union_volume_idxs.append(volume_idx)
                    f.write(f'{GmshUtils.union_volume(source_volume_id=source_volume_idx, union_volume_ids=union_volume_idxs)}\n')
                    f.write('\n')
                    
                dim_cfgs = GmshUtils.read_geometry_from_geo_file(geo_file_path, dims=[1, 2])
                fine_surface_tags, valid_boundary_count = {}, 0
                wall_marker = 9999999
                for bry_cfg in self.boundaries:
                    if bry_cfg.boundary_type == 'wall':
                        wall_marker = bry_cfg.marker
                        continue
                    
                    valid_boundary_count += 1
                    for face_tag, face_cfg in dim_cfgs[2].items():
                        if len(face_cfg['down_tags']) > 1:
                            continue
                        line_tag_cfg = dim_cfgs[1][face_cfg['down_tags'][0]]
                        if line_tag_cfg['type'] != 'Circle':
                            continue
                        if np.isclose(np.array(face_cfg['center']), bry_cfg.face_center).all():
                            fine_surface_tags[face_tag] = {
                                'name': bry_cfg.name,
                                'marker': bry_cfg.marker,
                            }
                
                assert valid_boundary_count == len(fine_surface_tags), f"Valid boundary count {valid_boundary_count} does not match fine surface tags count {len(fine_surface_tags)}"
                
                wall_tags = list(set(list(dim_cfgs[2].keys())) - set(list(fine_surface_tags.keys())))
                with open(geo_file_path, 'a') as f:
                    for tag, tag_cfg in fine_surface_tags.items():
                        f.write(f'{GmshUtils.define_gmsh_surface_physical_group(tag_cfg['name'], tag_cfg['marker'], tags=[tag])}\n')
                    
                    f.write(f'{GmshUtils.define_gmsh_surface_physical_group('wall', wall_marker, wall_tags)}\n')
                        
                    f.write(f'{GmshUtils.define_gmsh_volume_physical_group(group_name='volume', group_id=1, tags=[1])}\n')
                    f.write('\n')
                    
                    f.write(f'{GmshUtils.define_field_mesh_size_to_surface(
                        distance_field_id=1,
                        threshold_field_id=2,
                        surface_tags=list(fine_surface_tags.keys()),
                        sampling=100,
                        size_min=kwargs['min_size'],
                        size_max=kwargs['max_size'],
                        dist_min=kwargs['dist_min'],
                        dist_max=kwargs['dist_max'],
                    )}\n')
                    f.write(f'{GmshUtils.define_minimum_field(aggregate_field_id=3, field_ids=[2])}\n')
                    f.write('\n')
                    
                    compress_mesh = kwargs.get('compress_mesh_extend_from_boundary', False)
                    if compress_mesh:
                        f.write(f'{GmshUtils.compress_extend_from_boundary()}\n')  # TODO: please check whether it work
                
                msh_file = kwargs['msh_file_path']
                base_dir = os.path.dirname(msh_file)
                log_file = os.path.join(base_dir, 'gmsh.log')
                cmd_str = f'gmsh {geo_file_path} -3 -o {msh_file} > {log_file}'
                subprocess.run(cmd_str, shell=True)
                assert os.path.exists(msh_file), f"Failed to generate msh file: {msh_file}"
        
        else:
            raise ValueError(f"Invalid output type: {output_type}")
        
    def extract_field(self, field_value: np.ndarray, threshold: float, output_type: Literal['geo', 'vtk'], **kwargs):
        if self.dim == 2:
            return self.extract_2d_field(field_value, threshold, output_type, **kwargs)
        elif self.dim == 3:
            return self.extract_3d_field(field_value, threshold, output_type, **kwargs)
        else:
            raise ValueError(f"Invalid dimension: {self.dim}")


if __name__ == '__main__':
    pass