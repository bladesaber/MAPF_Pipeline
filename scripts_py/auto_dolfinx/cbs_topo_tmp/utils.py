from typing import Literal

import dolfinx
from mpi4py import MPI

import numpy as np
import pyvista
from functools import partial
from sklearn.neighbors import KDTree

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidBoundary


def dist_pcd2center(pcd: np.ndarray, center_point: np.ndarray):
    return np.linalg.norm(pcd - center_point, axis=1)


def location_func(
    x: np.ndarray,
    face_center: np.ndarray,
    face_norm_vector: np.ndarray,
    face_radius: float,
    dim: int,
):
    if x.shape[0] in [2, 3]:
        position = x.T
    else:
        position = x
    pcd2center_dist = dist_pcd2center(position[:, :dim], face_center)
    is_on_surface = pcd2center_dist <= face_radius
    return is_on_surface


def generate_box_mesh(
    boundary_list: list[AutoFluidBoundary],
    left_bottom_corner: np.ndarray,
    right_top_corner: np.ndarray,
    num_cells: list[int],
    xdmf_file: str,
    cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.tetrahedron,
    comm=MPI.COMM_WORLD,
) -> MeshObj:
    domain = dolfinx.mesh.create_box(
        comm, [left_bottom_corner, right_top_corner], num_cells, cell_type=cell_type
    )

    mesh_tag_indices = []
    mesh_tag_values = []
    wall_marker = None
    wall_count = 0
    for boundary in boundary_list:
        if boundary.boundary_type in ["velocity", "pressure"]:
            mesh_tag = MeshUtils.define_meshTag(
                domain=domain,
                marker=boundary.marker,
                dim=domain.topology.dim - 1,
                location_func=partial(
                    location_func,
                    face_center=boundary.face_center,  # type: ignore
                    face_norm_vector=boundary.face_norm_vector,  # type: ignore
                    face_radius=boundary.face_radius,  # type: ignore
                    dim=domain.topology.dim,
                ),
            )
            mesh_tag_indices.append(mesh_tag.indices)
            mesh_tag_values.append(mesh_tag.values)

        elif boundary.boundary_type == "wall":
            wall_marker = boundary.marker
            wall_count += 1

        else:
            raise ValueError(
                f"[Error]: Invalid Boundary Type: {boundary.boundary_type}."
            )

    assert wall_count == 1, f"[Error]: Only one wall boundary is allowed."

    inout_tag_indices = np.concatenate(mesh_tag_indices)
    inout_tag_values = np.concatenate(mesh_tag_values)

    wall_indices = MeshUtils.extract_entity_indices(domain, style="facet", marker=None)
    wall_indices = np.array(
        list(set(wall_indices.tolist()) - set(inout_tag_indices.tolist()))
    ).astype(np.int32)

    mesh_tag_indices = np.concatenate([inout_tag_indices, wall_indices]).astype(np.int32)
    mesh_tag_values = np.concatenate(
        [inout_tag_values, np.full_like(wall_indices, wall_marker)]
    ).astype(np.int32)

    grid = MeshUtils.mesh_to_vtk(domain, domain.topology.dim)
    facet_tags = dolfinx.mesh.meshtags(
        domain, domain.topology.dim - 1, mesh_tag_indices, mesh_tag_values
    )
    cell_tags = dolfinx.mesh.meshtags(
        domain,
        domain.topology.dim,
        np.arange(0, grid.n_cells, 1).astype(np.int32),
        np.full(shape=(grid.n_cells,), fill_value=9).astype(np.int32),
    )

    MeshUtils.save_XDMF(
        xdmf_file=xdmf_file, domain=domain, facet_tags=facet_tags, cell_tags=cell_tags
    )

    return MeshObj(xdmf_file=xdmf_file)


def generate_rectangle_mesh(
    boundary_list: list[AutoFluidBoundary],
    left_bottom_corner: np.ndarray,
    right_top_corner: np.ndarray,
    num_cells: list[int],
    xdmf_file: str,
    cell_type: dolfinx.mesh.CellType = dolfinx.mesh.CellType.triangle,
    comm=MPI.COMM_WORLD,
):
    domain = dolfinx.mesh.create_rectangle(
        comm, [left_bottom_corner, right_top_corner], num_cells, 
        # cell_type=cell_type
    )
    
    mesh_tag_indices = []
    mesh_tag_values = []
    wall_marker = None
    wall_count = 0
    for boundary in boundary_list:
        if boundary.boundary_type in ["velocity", "pressure"]:
            mesh_tag = MeshUtils.define_meshTag(
                domain=domain,
                marker=boundary.marker,
                dim=domain.topology.dim - 1,
                location_func=partial(
                    location_func,
                    face_center=boundary.face_center,  # type: ignore
                    face_norm_vector=boundary.face_norm_vector,  # type: ignore
                    face_radius=boundary.face_radius,  # type: ignore
                    dim=domain.topology.dim,
                ),
            )
            mesh_tag_indices.append(mesh_tag.indices)
            mesh_tag_values.append(mesh_tag.values)
        elif boundary.boundary_type == "wall":
            wall_marker = boundary.marker
            wall_count += 1
        else:
            raise ValueError(
                f"[Error]: Invalid Boundary Type: {boundary.boundary_type}."
            )
    
    assert wall_count == 1, f"[Error]: Only one wall boundary is allowed."

    inout_tag_indices = np.concatenate(mesh_tag_indices)
    inout_tag_values = np.concatenate(mesh_tag_values)

    wall_indices = MeshUtils.extract_entity_indices(domain, style="facet", marker=None)
    wall_indices = np.array(
        list(set(wall_indices.tolist()) - set(inout_tag_indices.tolist()))
    ).astype(np.int32)
    mesh_tag_indices = np.concatenate([inout_tag_indices, wall_indices]).astype(np.int32)
    mesh_tag_values = np.concatenate(
        [inout_tag_values, np.full_like(wall_indices, wall_marker)]
    ).astype(np.int32)

    grid = MeshUtils.mesh_to_vtk(domain, domain.topology.dim)
    facet_tags = dolfinx.mesh.meshtags(
        domain, domain.topology.dim - 1, mesh_tag_indices, mesh_tag_values
    )
    cell_tags = dolfinx.mesh.meshtags(
        domain,
        domain.topology.dim,
        np.arange(0, grid.n_cells, 1).astype(np.int32),
        np.full(shape=(grid.n_cells,), fill_value=9).astype(np.int32),
    )

    MeshUtils.save_XDMF(
        xdmf_file=xdmf_file, domain=domain, facet_tags=facet_tags, cell_tags=cell_tags
    )
    
    return MeshObj(xdmf_file=xdmf_file)


def extract_grid_info_from_standard_mesh(mesh: MeshObj) -> tuple[np.ndarray, float]:
    mesh_points = mesh.points.copy()[:, :mesh.tdim]
    
    tree = KDTree(mesh_points)
    dists_list, idxs_list = tree.query(mesh_points, k=2, sort_results=True, return_distance=True)
    dists_list = np.concatenate(dists_list, axis=-1)
    select_idxs = np.arange(1, idxs_list.shape[0], 2)
    dists = dists_list[select_idxs]
    assert np.isclose(np.max(dists) - np.min(dists), 0.0, atol=1e-8), "grid is not standard"
    
    grid_size = np.mean(dists)
    bound_min = np.min(mesh_points, axis=0)
    bound_max = np.max(mesh_points, axis=0)
    
    left_bottom_corner = mesh_points - np.array([grid_size] * mesh.tdim)
    left_bottom_corner_bool = np.sum(left_bottom_corner >= bound_min.reshape((1, -1)), axis=1) == mesh.tdim
    left_bottom_corner = left_bottom_corner[left_bottom_corner_bool]
    right_top_corner = left_bottom_corner + np.array([grid_size] * mesh.tdim)
    
    # # ----- visualize grid info -----
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(mesh.grid, color='red', style='wireframe')
    # plotter.add_mesh(pyvista.PointSet(left_bottom_corner), color='blue', style='points')
    # # plotter.add_mesh(pyvista.PointSet(right_top_corner), color='green', style='points')
    # plotter.show()
    # # ----- visualize grid info -----
    
    grid_info = np.concatenate([
        left_bottom_corner[:, np.newaxis, :], 
        right_top_corner[:, np.newaxis, :]
    ], axis=1)
    
    return grid_info, float(grid_size)


def debug_generate_box():
    debug_boundary_list = [
        AutoFluidBoundary(
            name='inlet_1',
            boundary_type='velocity',
            value=[1.0, 0.0, 0.0],
            marker=1,
            face_center=np.array([0.0, 0.2, 0.2]),
            face_norm_vector=np.array([1.0, 0.0, 0.0]),
            face_radius=0.15,
        ),
        AutoFluidBoundary(
            name='outlet_1',
            boundary_type='pressure',
            value=0.0,
            marker=2,
            face_center=np.array([0.8, 1.0, 0.8]),
            face_norm_vector=np.array([0.0, 1.0, 0.0]),
            face_radius=0.15,
        ),
        AutoFluidBoundary(
            name='inlet_2',
            boundary_type='velocity',
            value=[0.0, 1.0, 0.0],
            marker=3,
            face_center=np.array([0.5, 0.0, 0.2]),
            face_norm_vector=np.array([0.0, 1.0, 0.0]),
            face_radius=0.15,
        ),
        AutoFluidBoundary(
            name='outlet_2',
            boundary_type='pressure',
            value=0.0,
            marker=4,
            face_center=np.array([0.8, 0.5, 1.0]),
            face_norm_vector=np.array([0.0, 0.0, 1.0]),
            face_radius=0.15,
        ),
        AutoFluidBoundary(
            name='wall',
            boundary_type='wall',
            value=0.0,
            marker=5,
        )
    ]
    
    mesh = generate_box_mesh(
        boundary_list=debug_boundary_list,
        left_bottom_corner=np.array([0.0, 0.0, 0.0]),
        right_top_corner=np.array([1.0, 1.0, 1.0]),
        num_cells=[20, 20, 20],
        xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf',
        cell_type=dolfinx.mesh.CellType.tetrahedron,
        comm=MPI.COMM_WORLD,
    )
    # print(np.unique(mesh.facet_tags.values))


def debug_generate_rectangle():
    debug_boundary_list = [
        AutoFluidBoundary(
            name='inlet',
            boundary_type='velocity',
            value=[1.0, 0.0],
            marker=1,
            face_center=np.array([0.0, 0.2]),
            face_norm_vector=np.array([1.0, 0.0]),
            face_radius=0.1,
        ),
        AutoFluidBoundary(
            name='outlet',
            boundary_type='pressure',
            value=0.0,
            marker=2,
            face_center=np.array([1.0, 0.8]),
            face_norm_vector=np.array([1.0, 0.0]),
            face_radius=0.1,
        ),
        AutoFluidBoundary(
            name='wall',
            boundary_type='wall',
            value=0.0,
            marker=3,
        ),
    ]
    mesh = generate_rectangle_mesh(
        boundary_list=debug_boundary_list,
        left_bottom_corner=np.array([0.0, 0.0]),
        right_top_corner=np.array([1.0, 1.0]),
        num_cells=[50, 50],
        xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t36/model.xdmf',
        cell_type=dolfinx.mesh.CellType.tetrahedron,
        comm=MPI.COMM_WORLD,
    )
    
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh.grid, style='wireframe')
    for boundary in debug_boundary_list:
        idxs = mesh.extract_surface_indices(boundary.marker)
        if boundary.boundary_type == 'velocity':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='red', style='points')
        elif boundary.boundary_type == 'pressure':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='blue', style='points')
        elif boundary.boundary_type == 'wall':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='green', style='points')
    plotter.show()


def debug_extract_grid_info_from_standard_mesh():
    mesh = MeshObj(xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t25/model.xdmf')
    extract_grid_info_from_standard_mesh(mesh)


if __name__ == '__main__':
    # debug_generate_box()
    # debug_generate_rectangle()
    debug_extract_grid_info_from_standard_mesh()
    pass