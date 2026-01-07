import os
import shutil
from typing import Literal, Union, Callable

import dolfinx
from mpi4py import MPI

import numpy as np
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.vtu_recorder import TensorBoardRecorder

from scripts_py.auto_dolfinx.cbs_topo.utils import generate_rectangle_mesh
from scripts_py.auto_dolfinx.cbs_topo.topo_fluid_fipy_optimizer import TopoFluidOptimizerFipy
from scripts_py.simulator.fipy_simulator import FipyStokeSimulator
from scripts_py.simulator.dolfin_simulator import StokeSimulator

from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from thirdparty.pyadjoint.pyadjoint import Tape


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t42/'


def define_boundary(cfg, boundary_type: Literal['velocity', 'pressure', 'wall']):
    return AutoFluidBoundary(
        name=cfg['name'],
        boundary_type=boundary_type,
        value=0.0 if boundary_type == 'wall' else cfg['value'],
        marker=cfg['marker'],
        face_center=None if boundary_type == 'wall' else cfg['face_center'],
        face_norm_vector=None if boundary_type == 'wall' else cfg['face_norm_vector'],
        face_radius=None if boundary_type == 'wall' else cfg['face_radius'],
    )
        

boundary_cfgs = [
    {
        'name': 'inlet1',
        'boundary_type': 'velocity',
        'value': [1.0, 0.0],
        'marker': 1,
        'face_center': np.array([-0.05, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'outlet1',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 2,
        'face_center': np.array([1.05, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'wall',
        'boundary_type': 'wall',
        'value': 0.0,
        'marker': 3,
    }
]


def prepare_mesh():
    xdmf_file = os.path.join(base_dir, 'model.xdmf')
    msh_file = os.path.join(base_dir, 'model.msh')
    
    boundary_list = [define_boundary(bry_cfg, boundary_type=bry_cfg['boundary_type']) for bry_cfg in boundary_cfgs]
    mesh = generate_rectangle_mesh(
        boundary_list=boundary_list, 
        left_bottom_corner=np.array([0.0, 0.0]), 
        right_top_corner=np.array([1.0, 1.0]), 
        num_cells=[60, 60], 
        xdmf_file=xdmf_file, 
        cell_type=dolfinx.mesh.CellType.triangle, comm=MPI.COMM_WORLD
    )
    print('Mesh. Info:')
    print(mesh.grid)
    
    face_tags = {}
    for boundary in boundary_list:
        face_tags[boundary.name] = boundary.marker
    
    MeshUtils.generate_msh22_file(
        mesh=mesh, filename=msh_file, face_tags=face_tags, cell_tags={"volume": 9}
    )
    FipyStokeSimulator.msh_version_convert(
        source_msh_file=msh_file, target_msh_file=msh_file, target_version='msh4', num_partitions=1
    )
    
    MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=mesh.tdim)
    mesh = MeshObj(xdmf_file=os.path.join(base_dir, 'model.xdmf'))
    
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh.grid, style='wireframe')
    for boundary in boundary_list:
        idxs = mesh.extract_surface_indices(boundary.marker)
        if boundary.boundary_type == 'velocity':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='red', style='points')
        elif boundary.boundary_type == 'pressure':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='blue', style='points')
        elif boundary.boundary_type == 'wall':
            plotter.add_mesh(pyvista.PointSet(mesh.grid.points[idxs]), color='green', style='points')
    plotter.show()
    
    return mesh


def run_topo_opt():
    msh_file = os.path.join(base_dir, 'model.msh')
    
    # --- 1. define mesh ---
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    cost_item = AutoFluidCost(name='obj_cost', style='MinDissipation', weight=1.0)
    regularization_item = AutoFluidRegularization(name='reg_cost', style='Volume', weight=0.5, target_ratio=0.05)
    
    result_dir = os.path.join(base_dir, 'topo_opt')
    optim = TopoFluidOptimizerFipy(
        name='test_example_42',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        topo_weight_init=0.1,
        obstacle_func=None,
        weight_filter_radius=0.06,
        result_dir=result_dir,
        velocity_degree=1,
        pressure_degree=1,
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    boundary_list = [define_boundary(bry_cfg, boundary_type=bry_cfg['boundary_type']) for bry_cfg in boundary_cfgs]
    for boundary in boundary_list:
        optim.add_boundary(boundary)

    optim.define_problem(
        adj_ksp_option={
            'ksp_type': 'richardson',
            'ksp_max_it': 1, # only one iteration = equivalent to preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,
            'mat_mumps_icntl_25': 0,
        },
        simulation_iterations=300,
        convergence_threshold=1e-4,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        hot_steps=10,
        log_step=2,
        parallel_num=4,
        use_tqdm=True,
        use_log=True,
    )
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(result_dir, 'log'))
    optim.run(
        line_search_steps=[0.1, 0.05, 0.01],
        tensorboard_recorder=recorder
    )


if __name__ == '__main__':
    # prepare_mesh()
    run_topo_opt()
