"""
想法转为:
1.为防止仿真的失真,所有映射均由gmsh建模形成而不再使用Topology Weight (要测试一下gmsh的效能)
2.由于不再使用Topology Weight,这意味着整个Mesh表面都是标准化的,并且将Mesh内部划分为标准的Cell(例如Box),Cell采用内部多点的均值计算
3.由SIMP方法转向BESO/PTO方法,将优化出的形状映射为由多个小型Box构成的
"""

from typing import Literal
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import pyvista
import math


class GmshUtils(object):
    @staticmethod
    def define_gmsh_kernel():
        return 'SetFactory("OpenCASCADE");'
    
    @staticmethod
    def define_gmsh_physical_group(group_name: str, group_id: int, tags: list[int], dim: int) -> str:
        """return the tag of the physical group"""
        if dim == 2:
            obj_tag = 'Surface'
        elif dim == 3:
            obj_tag = 'Volume'
        else:
            raise ValueError(f"[Error]: non valid dim {dim}.")
        content = f'Physical {obj_tag}(\"{group_name}\", {group_id}) = {{{tags}}};'
        return content

    @staticmethod
    def define_gmsh_point(point_id: int, x: float, y: float, z: float, size: float = 0.1):
        content = f'Point({point_id}) = {{{x}, {y}, {z}, {size}}};'
        return content
    
    @staticmethod
    def define_gmsh_line(line_id: int, point_id_1: int, point_id_2: int):
        content = f'Line({line_id}) = {{{point_id_1}, {point_id_2}}};'
        return content
    
    @staticmethod
    def define_gmsh_surface(surface_id: int, line_ids: list[int]):
        content = f'Curve Loop({surface_id}) = {{{", ".join([str(line_id) for line_id in line_ids])}}};\n'
        content += f'Plane Surface({surface_id}) = {{{surface_id}}};'
        return content
    
    @staticmethod
    def define_gmsh_volume(volume_id: int, surface_ids: list[int]):
        content = f'Surface Loop({volume_id}) = {{{", ".join([str(surface_id) for surface_id in surface_ids])}}};\n'
        content += f'Volume({volume_id}) = {{{volume_id}}};'
        return content
    
    @staticmethod
    def convert_wireframe_to_gmsh_code(file_path: str, wireframe: pyvista.UnstructuredGrid, dim: int):
        with open(file_path, 'w') as f:
            f.write(f'{GmshUtils.define_gmsh_kernel()}\n')
            f.write('\n')
            
            for p_idx, vertex in enumerate(wireframe.points):
                f.write(f'{GmshUtils.define_gmsh_point(p_idx, vertex[0], vertex[1], vertex[2])}\n')
            f.write('\n')
            
            # TODO: how to extract line
            if dim == 2:
                pass
            else:
                for loop_p_idxs in wireframe.cells_dict[pyvista.CellType.QUAD]:  # type: ignore
                    pass


class GridMesher(object):
    def __init__(
        self, 
        left_bottom_corner: np.ndarray,
        right_top_corner: np.ndarray,
        resolution: float,
    ):
        self.resolution = resolution
        self.left_bottom_corner = left_bottom_corner
        self.right_top_corner = right_top_corner
        self.dim = left_bottom_corner.shape[0]
        
        self.points: np.ndarray
        self.line2points: np.ndarray
        self.surface2lines: np.ndarray
        self.volume2surfaces: np.ndarray
        self.cell2corners: np.ndarray
        self.cell2neighbors: list[np.ndarray]
        self.generate_init_topology()
       
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

    def lines_to_wireframe_vtk(self, line_idxs: np.ndarray) -> pyvista.UnstructuredGrid:
        point_idxs = np.unique(self.line2points[line_idxs])    
        vertices = self.points[point_idxs]
        if vertices.shape[1] == 2:
            vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1))], axis=1)
        vertice_idxs = np.arange(vertices.shape[0])  
        p_idxs_map = np.zeros(shape=self.point_num, dtype=int)
        p_idxs_map[point_idxs] = vertice_idxs            
        new_line_conn = p_idxs_map[self.line2points[line_idxs]]
        return pyvista.UnstructuredGrid({pyvista.CellType.LINE: new_line_conn}, vertices)

    def surfaces_to_wireframe_vtk(self, surface_idxs: np.ndarray) -> pyvista.UnstructuredGrid:        
        surface2points = self.line2points[self.surface2lines[surface_idxs]]  # n x 4 x 2
        
        # ------ will destroy the order of the points
        # surface2points = surface2points.reshape((-1, 8))
        # surface2points = np.unique(surface2points, axis=1)
        # ------ will destroy the order of the points
        
        surface2points = surface2points = np.concatenate([
            surface2points[:, 0, 0, np.newaxis],
            surface2points[:, 1, 0, np.newaxis],
            surface2points[:, 2, 1, np.newaxis],
            surface2points[:, 3, 1, np.newaxis],
        ], axis=1)

        point_idxs = np.unique(surface2points)    
        vertices = self.points[point_idxs]
        if vertices.shape[1] == 2:
            vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1))], axis=1)
        vertice_idxs = np.arange(vertices.shape[0])  
        p_idxs_map = np.zeros(shape=self.point_num, dtype=int)
        p_idxs_map[point_idxs] = vertice_idxs            
        new_conn = p_idxs_map[surface2points]
        return pyvista.UnstructuredGrid({pyvista.CellType.QUAD: new_conn}, vertices)
    
    def cells_to_wireframe_vtk(self, cell_idxs: np.ndarray) -> pyvista.UnstructuredGrid:
        if self.dim == 2:
            return self.surfaces_to_wireframe_vtk(cell_idxs)
        else:
            surface_idxs = np.unique(self.volume2surfaces[cell_idxs])
            return self.surfaces_to_wireframe_vtk(surface_idxs)

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
        field_bool = field_value > threshold
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
        boundary_bool = boundary_count == 1
        return boundary_idxs[boundary_bool]

    def extract_field_wireframe_vtk(self, field_value: np.ndarray, threshold: float = 0.5) -> pyvista.UnstructuredGrid:
        boundary_idxs = self.extract_field_boundary_idxs(field_value, threshold)
        if self.dim == 2:
            return self.lines_to_wireframe_vtk(boundary_idxs)
        else:
            return self.surfaces_to_wireframe_vtk(boundary_idxs)


mesher = GridMesher(
    left_bottom_corner=np.array([0.0, 0.0, 0.0]),
    right_top_corner=np.array([1.0, 1.0, 1.0]),
    resolution=0.025,
)
path = np.array([
    [0.0, 0.2, 0.2],
    [0.5, 0.2, 0.2],
    [0.5, 0.6, 0.2],
    [0.5, 0.6, 0.6],
    [1.0, 1.0, 1.0]
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

vtk_mesh = mesher.extract_field_wireframe_vtk(field_value, threshold=0.5)
print(vtk_mesh.cells_dict)

# plotter = pyvista.Plotter()
# plotter.add_mesh(vtk_mesh, style='surface', show_edges=True, color='lightblue')
# plotter.show()

