from typing import Literal

import dolfinx
import ufl
from mpi4py import MPI

import numpy as np
from functools import partial

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import TopoFluidOptimizer
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)


def dist_pcd2center(pcd: np.ndarray, center_point: np.ndarray):
    return np.linalg.norm(pcd - center_point, axis=1)


def location_func(
    x: np.ndarray,
    face_center: np.ndarray,
    face_norm_vector: np.ndarray,
    face_radius: float,
):
    if x.shape[0] in [2, 3]:
        position = x.T
    else:
        position = x
    pcd2center_dist = dist_pcd2center(position, face_center)
    is_on_surface = pcd2center_dist <= face_radius
    return is_on_surface


def generate_mesh(
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

    mesh_tag_indices = np.concatenate([inout_tag_indices, wall_indices]).astype(
        np.int32
    )
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


class TopoObstacleFluidOptimizer(TopoFluidOptimizer):
    def __init__(
        self,
        name: str,
        mesh: MeshObj,
        fluid_type: Literal["stoke", "naiver_stoke", "rans"],
        nut_init: float,
        cost_item: AutoFluidCost,
        regularization_item: AutoFluidRegularization,
        topo_weight_init: float,
        weight_filter_radius: float,
        weight_filter_is_cell_field: bool,
        result_dir: str,
        obstacle_cost_weight: float,
        alpha_fun_max: float = 1000.0,
        alpha_fun_min: float = 0.01,
        alpha_fun_p: float = 3.0,
    ):
        super().__init__(
            name,
            mesh,
            fluid_type,
            nut_init,
            cost_item,
            regularization_item,
            topo_weight_init,
            weight_filter_radius,
            weight_filter_is_cell_field,
            result_dir,
            alpha_fun_max,
            alpha_fun_min,
            alpha_fun_p,
        )

        self.obstacle_cost_weight = obstacle_cost_weight
        self.obstacle_source = dolfinx.fem.Function(
            self.point_scalar_func_space, name="obstacle_source"
        )
        self.obstacle_source.x.array[:] = 0.0  # type: ignore
