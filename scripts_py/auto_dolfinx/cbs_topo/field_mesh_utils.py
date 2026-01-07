from typing import Any, Literal

import dolfinx
from mpi4py import MPI

import math
import numpy as np
import pyvista

import mcubes
from skimage import measure

from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.mesh_utils import MeshObj, MeshUtils

# TODO: 我无法解决如何从凹凸不平的精细平面构造出光滑的平面
# 应该由TransformedFieldObj吸收一个MeshObj,然后产出新的MeshObj

class TransformedFieldObj(object):
    @staticmethod
    def marching_cubes_from_3D_field(
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        zmin: float, zmax: float,
        resolution: int,
        field: np.ndarray,
        level: float = 0.1
    ) -> dict:
        assert field.ndim == 3, "field must be 3D"
        vertices, triangles = mcubes.marching_cubes(field, level)  # should larger than min mesh size
        vertices[:, 0] = xmin + vertices[:, 0] / (resolution - 1) * (xmax - xmin)
        vertices[:, 1] = ymin + vertices[:, 1] / (resolution - 1) * (ymax - ymin)
        vertices[:, 2] = zmin + vertices[:, 2] / (resolution - 1) * (zmax - zmin)
        return {
            "vertices": vertices,
            "connectivity": triangles
        }
    
    @staticmethod
    def marching_cubes_from_2D_field(
        field: np.ndarray,
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        level: float = 0.1
    ) -> list[dict]:
        assert field.ndim == 2, "field must be 2D"
        raw_contours = measure.find_contours(field, level=level)
        
        height, width = field.shape
        boundary_dicts = []
        for contour in raw_contours:
            if len(contour) < 3:
                raise ValueError("Contour is not a valid contour")
            
            if np.allclose(contour[0], contour[-1]):                
                vertices = np.array(contour[:-1])
                edges = np.hstack([
                    np.arange(vertices.shape[0]).reshape(-1, 1),
                    np.array(list(range(1, vertices.shape[0])) + [0]).reshape(-1, 1)
                ])
                
            else:
                vertices = np.array(contour)
                edges = np.hstack([
                    np.arange(vertices.shape[0] - 1).reshape(-1, 1),
                    np.arange(1, vertices.shape[0]).reshape(-1, 1)
                ])
            
            # ------ transform vertices from pixel coordinates to original bounds ------
            vertices[:, 1] = xmin + (vertices[:, 1] / (width - 1)) * (xmax - xmin)
            vertices[:, 0] = ymax - (vertices[:, 0] / (height - 1)) * (ymax - ymin)
            new_vertices = np.zeros_like(vertices)
            new_vertices[:, 0] = vertices[:, 1]
            new_vertices[:, 1] = vertices[:, 0]
            # ------ transform vertices from pixel coordinates to original bounds ------
            
            boundary_dicts.append({
                'vertices': new_vertices,
                'edges': edges,
            })
        
        return boundary_dicts
     
    def __init__(
        self, 
        left_bottom_corner: np.ndarray, 
        right_top_corner: np.ndarray,
        resolution: float
    ):
        """left_bottom_corner and right_top_corner should larger than the domain bounds and contain the whole domain bounds
        1. 在由Field生成的新的domain中,我们无法保证边界的精确程度,边界可能不再平整
        """
        self.tdim = left_bottom_corner.shape[0]
        self.left_bottom_corner = left_bottom_corner
        self.right_top_corner = right_top_corner
        self.boundary_cfgs: dict[str, Any] = {}
        
        if self.tdim == 2:
            xs = np.linspace(
                self.left_bottom_corner[0], self.right_top_corner[0], 
                math.ceil((self.right_top_corner[0] - self.left_bottom_corner[0]) / resolution)
            )
            ys = np.linspace(
                self.left_bottom_corner[1], self.right_top_corner[1], 
                math.ceil((self.right_top_corner[1] - self.left_bottom_corner[1]) / resolution)
            )
            xs, ys = np.meshgrid(xs, ys)
            xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)

            self.field_shape = xys.shape[:-1]
            self.field_points = xys.reshape((-1, 2))
        
        elif self.tdim == 3:
            raise NotImplementedError("3D is not implemented")
        else:
            raise ValueError(f"Invalid tdim: {self.tdim}")
        
    def add_boundary(self, name: str, style: Literal['circle'], **kwargs):
        bry_dict = {'name': name, 'style': style}
        if style == 'circle':
            bry_dict.update({
                'face_center': kwargs['face_center'],
                'face_norm_vector': kwargs['face_norm_vector'],
                'face_radius': kwargs['face_radius'],
                'marker': kwargs['marker'],
            })
        else:
            raise ValueError(f"Invalid boundary style: {style}")
        self.boundary_cfgs[name] = bry_dict
    
    @staticmethod
    def define_circle_surface_indices(
        positions: np.ndarray,
        face_center: np.ndarray,
        face_norm_vector: np.ndarray,
        face_radius: float,
        with_dists: bool = True
    ):
        face_norm_vector = face_norm_vector / np.linalg.norm(face_norm_vector)
  
        dist2center = np.linalg.norm(positions - face_center, ord=2, axis=1)
        dist2plane = np.sum((positions - face_center) * face_norm_vector, axis=1)
        indices = np.arange(0, positions.shape[0], 1)
        
        # wrong here, we can't expect precise boundary constraints
        # indices = indices[(dist2center <= face_radius) & (np.abs(dist2plane) < 1e-8)]
        
        if with_dists:
            return indices, dist2center[indices]
        else:
            return indices

    def interpolate_field(
        self, 
        mesh: MeshObj, 
        field_func: dolfinx.fem.Function, 
        points: np.ndarray, 
        threshold: float,
    ):
        assert threshold > 0.0 and threshold < 1.0, "threshold must be between 0.0 and 1.0"
        
        field_values = np.zeros(shape=points.shape[0])
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
        cell_idxs: np.ndarray = MeshUtils.cell_indices_of_points(mesh.grid, points)  # type: ignore        
        inside_bool = cell_idxs >= 0
        
        # # ------ visual inside points ------
        # plotter = pyvista.Plotter()
        # plotter.add_mesh(pyvista.PointSet(points[inside_bool]), color='red', style='points')
        # plotter.add_mesh(mesh.grid, style='wireframe')
        # plotter.show()
        # # ------ visual inside points ------
        
        field_values[inside_bool] = DolfinUtils.function_eval(
            field_func, 
            points[inside_bool], 
            cell_idxs[inside_bool].tolist()
        ).reshape(-1)
        # self.vis_point_field(points, field_values, mesh)
        
        field_values[field_values >= threshold] = 1.0
        field_values[field_values < threshold] = 0.0
        
        # TODO: 这里是错的,哪怕是边界也应该不定型,不定位置
        for name, bry_cfg in self.boundary_cfgs.items():
            if bry_cfg['style'] == 'circle':
                indices, dists = self.define_circle_surface_indices(
                    positions=points[:, :mesh.tdim],
                    face_center=bry_cfg['face_center'],
                    face_norm_vector=bry_cfg['face_norm_vector'],
                    face_radius=bry_cfg['face_radius'],
                    with_dists=True,
                )
                field_values[indices] = 1.0  # must be 1.0
            else:
                raise ValueError(f"Invalid boundary style: {bry_cfg['style']}")
            
        # self.vis_point_field(points, field_values, mesh)
        return field_values

    def generate_2d_surface(self, mesh: MeshObj, field_func: dolfinx.fem.Function, threshold: float, resolution: float):
        xs = np.linspace(
            self.left_bottom_corner[0], self.right_top_corner[0], 
            math.ceil((self.right_top_corner[0] - self.left_bottom_corner[0]) / resolution)
        )
        ys = np.linspace(
            self.left_bottom_corner[1], self.right_top_corner[1], 
            math.ceil((self.right_top_corner[1] - self.left_bottom_corner[1]) / resolution)
        )
        xs, ys = np.meshgrid(xs, ys)
        xys = np.concatenate([xs[..., np.newaxis], ys[..., np.newaxis]], axis=-1)

        xys_shape = xys.shape[:-1]
        square_points = xys.reshape((-1, 2))
        
        field_value = self.interpolate_field(mesh, field_func, square_points, threshold)
        field_value = field_value.reshape(xys_shape)
        
        boundary_dicts = self.marching_cubes_from_2D_field(
            field=field_value, 
            xmin=self.left_bottom_corner[0], xmax=self.right_top_corner[0], 
            ymin=self.left_bottom_corner[1], ymax=self.right_top_corner[1], 
            level=0.5
        )
        
        # ------ reconstruct boundary connectivity ------
        for boundary_dict in boundary_dicts:
            print(boundary_dict['edges'])
        
        # for name, bry_cfg in self.boundary_cfgs.items():
        #     if bry_cfg['style'] == 'circle':
        #         pass
        #     else:
        #         raise ValueError(f"Invalid boundary style: {bry_cfg['style']}")
        # ------ reconstruct boundary connectivity ------
        
        # self.vis_2d_surface(topo_dict['vertices'], topo_dict['connectivity'], mesh)
        
        

    @staticmethod
    def vis_point_field(points: np.ndarray, field_values: np.ndarray, mesh: MeshObj):
        grid = pyvista.PointSet(points)
        grid.point_data['field_values'] = field_values
        grid.warp_by_scalar('field_values')
        plotter = pyvista.Plotter()
        plotter.add_mesh(grid, style='surface')
        plotter.add_mesh(mesh.grid, style='wireframe', color='black')
        plotter.show()

    @staticmethod
    def vis_2d_surface(boundary_dict: dict, mesh: MeshObj):
        plotter = pyvista.Plotter()
        plotter.add_mesh(mesh.grid, style='wireframe', color='blue')
        for boundary in boundary_dict:
            vertices = boundary['vertices']
            if vertices.shape[1] == 2:
                vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1))], axis=1)
            grid = pyvista.UnstructuredGrid({pyvista.CellType.LINE: boundary['edges']}, vertices)
            plotter.add_mesh(grid, style='wireframe', color='red')
        plotter.show()


# msh_file = '/home/admin123456/Desktop/work/gmsh_test/t42/t42_model.msh'
# domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
# mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)

# res_file = '/home/admin123456/Desktop/work/gmsh_test/t42/topo_opt_test/report_00044.vtu'
# res_grid: pyvista.UnstructuredGrid = pyvista.read(res_file)  # type: ignore

# cell_scalar_space = DolfinUtils.cell_scalar_function_space(mesh.domain)
# topo_weight_fun = dolfinx.fem.Function(cell_scalar_space)
# topo_weight_fun.x.array[:] = res_grid.cell_data['topo_weight'].reshape(-1)  # type: ignore

# xmin = mesh.grid.bounds.x_min
# xmax = mesh.grid.bounds.x_max
# ymin = mesh.grid.bounds.y_min
# ymax = mesh.grid.bounds.y_max

# transform_obj = TransformedFieldObj(
#     left_bottom_corner=np.array([xmin, ymin]), 
#     right_top_corner=np.array([xmax, ymax])
# )
# transform_obj.add_boundary(
#     name='inlet',
#     style='circle',
#     face_center=np.array([-0.05, 0.5]),
#     face_norm_vector=np.array([1.0, 0.0]),
#     face_radius=0.1,
#     marker=1,
# )
# transform_obj.add_boundary(
#     name='outlet',
#     style='circle',
#     face_center=np.array([1.05, 0.5]),
#     face_norm_vector=np.array([1.0, 0.0]),
#     face_radius=0.1,
#     marker=2,
# )


# transform_obj.generate_2d_surface(mesh, topo_weight_fun, threshold=0.5, resolution=0.1)