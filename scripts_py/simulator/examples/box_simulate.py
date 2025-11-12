import os

import dolfinx
from mpi4py import MPI

import numpy as np
import pyvista
from functools import partial
import time

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.simulator.dolfin_simulator import StokeSimulator, FluidBoundary


def generate_mesh(xdmf_file):
    def dist_pcd2center(pcd: np.ndarray, center_point: np.ndarray):
        return np.linalg.norm(pcd - center_point, axis=1)
        
        
    def location_func(x: np.ndarray, face_center: np.ndarray, face_norm_vector: np.ndarray, face_radius: float):
        if x.shape[0] in [2, 3]:
            position = x.T
        else:
            position = x
        pcd2center_dist = dist_pcd2center(position, face_center)
        is_on_surface = pcd2center_dist <= face_radius
        return is_on_surface
        

    domain = dolfinx.mesh.create_box(
        MPI.COMM_WORLD, 
        [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]], 
        [30, 30, 30], 
        # cell_type=dolfinx.mesh.CellType.hexahedron  # type: ignore
        cell_type=dolfinx.mesh.CellType.tetrahedron  # type: ignore
    )
    grid = MeshUtils.mesh_to_vtk(domain, domain.topology.dim)
    
    inlet_meshtag = MeshUtils.define_meshTag(
        domain=domain,
        marker=2,
        dim=domain.topology.dim - 1,
        location_func=partial(
            location_func, 
            face_center=np.array([0.0, 1.0, 1.0]), 
            face_norm_vector=np.array([1.0, 0.0, 0.0]), 
            face_radius=1.0,
        )
    )
    assert len(inlet_meshtag.indices) > 0

    outlet_meshtag = MeshUtils.define_meshTag(
        domain=domain,
        marker=3,
        dim=domain.topology.dim - 1,
        location_func=partial(
            location_func, 
            face_center=np.array([5.0, 4.0, 4.0]), 
            face_norm_vector=np.array([1.0, 0.0, 0.0]), 
            face_radius=1.0,
        )
    )
    assert len(outlet_meshtag.indices) > 0
    
    inout_tag_indices = np.concatenate([inlet_meshtag.indices, outlet_meshtag.indices])
    inout_tag_values = np.concatenate([inlet_meshtag.values, outlet_meshtag.values])
    
    wall_indices = MeshUtils.extract_entity_indices(domain, style='facet', marker=None)
    wall_indices = np.array(list(
        set(wall_indices.tolist()) - set(inout_tag_indices.tolist())
    )).astype(np.int32)
    
    mesh_tag_indices = np.concatenate([inout_tag_indices, wall_indices]).astype(np.int32)
    mesh_tag_values = np.concatenate([inout_tag_values, np.full_like(wall_indices, 1)]).astype(np.int32)
    facet_tags = dolfinx.mesh.meshtags(domain, domain.topology.dim - 1, mesh_tag_indices, mesh_tag_values)
    cell_tags = dolfinx.mesh.meshtags(
        domain,
        domain.topology.dim,
        np.arange(0, grid.n_cells, 1).astype(np.int32),
        np.full(shape=(grid.n_cells, ), fill_value=9).astype(np.int32),
    )
    
    MeshUtils.save_XDMF(
        xdmf_file=xdmf_file,
        domain=domain,
        facet_tags=facet_tags,
        cell_tags=cell_tags
    )


xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t23/test.xdmf'
generate_mesh(xdmf_file)

mesh = MeshObj(xdmf_file=xdmf_file)
print(mesh.grid)

inlet_indices = mesh.extract_surface_indices(2)
outlet_indices = mesh.extract_surface_indices(3)
wall_indices = mesh.extract_surface_indices(1)
plotter = pyvista.Plotter()
plotter.add_mesh(mesh.grid, color='black', style='wireframe')
plotter.add_mesh(pyvista.PointSet(mesh.grid.points[inlet_indices]), color='red', point_size=10)
plotter.add_mesh(pyvista.PointSet(mesh.grid.points[outlet_indices]), color='blue', point_size=10)
plotter.add_mesh(pyvista.PointSet(mesh.grid.points[wall_indices]), color='green', point_size=10)
plotter.show()

simulator = StokeSimulator('stoke', mesh, velocity_degree=2, pressure_degree=1)
simulator.define_equation(nu_value=1e-5)
    
inlet_bry = FluidBoundary(
    name='inlet',
    boundary_type='velocity',
    value=[5.0, 0.0, 0.0],
    marker=2,
    face_center=[0.0, 1.0, 1.0],
    face_norm_vector=[1.0, 0.0, 0.0],
    face_radius=1.0
)
outlet_bry = FluidBoundary(
    name='outlet',
    boundary_type='pressure',
    value=0.0,
    marker=3
)
wall_bry = FluidBoundary(
    name='wall',
    boundary_type='wall',
    value=0.0,
    marker=1
)
    
simulator.add_boundary(inlet_bry)
simulator.add_boundary(outlet_bry)
simulator.add_boundary(wall_bry)
    
simulator.define_boundary()

start_time = time.time()
simulator.run(with_residual=True)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

simulator.save('/home/admin123456/Desktop/work/gmsh_test/t23/stoke_res_tet.vtu')
