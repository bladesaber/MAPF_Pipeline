import os
import shutil
from typing import Literal, Union, Callable

import dolfinx
from mpi4py import MPI

import numpy as np
import pandas as pd
import pyvista
import math
from sklearn.neighbors import KDTree

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.vtu_recorder import TensorBoardRecorder
from scripts_py.dolfinx_utils import DolfinUtils

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
from scripts_py.dolfinx_utils import AssembleUtils


base_dir = '/home/admin123456/Desktop/work/gmsh_test/compare_2d'


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
        'name': 'inlet',
        'boundary_type': 'velocity',
        'value': [1.0, 0.0],
        'marker': 1,
        'face_center': np.array([-0.05, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'outlet',
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

    
def eval_res():
    msh_file = os.path.join(base_dir, 'model.msh')
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    topo_weight_init = 0.1
    
    name = 'topo_exam_01'
    alpha_fun_max = 1000.0
    reg_weight = 100.0
    reg_target_ratio = 0.2
    reg_rescale = True

    cost_item = AutoFluidCost(name='obj_cost', style='MinDissipation', weight=1.0)
    regularization_item = AutoFluidRegularization(name='reg_cost', style='Volume', weight=reg_weight, target_ratio=reg_target_ratio)
    
    result_dir = os.path.join(base_dir, name)
    optim = TopoFluidOptimizerFipy(
        name=name,
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        topo_weight_init=topo_weight_init,
        obstacle_func=None,
        weight_filter_radius=0.075,
        weight_filter_is_cell_field=False,
        result_dir=result_dir,
        velocity_degree=1,
        pressure_degree=1,
        alpha_fun_max=alpha_fun_max,
        alpha_fun_min=1e-12,
        alpha_fun_p=3.0
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
        simulation_iterations=200,
        convergence_threshold=1e-4,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        hot_steps=10,
        log_step=2,
        parallel_num=1,
        use_tqdm=True,
        use_log=True,
        activate_init_method='boundary_radius',
        wall_init_method='boundary_raw',
        constraint_method='fix',
        reg_rescale=reg_rescale
    )
    
    print(f'{name}_2d: alpha_fun_max: {alpha_fun_max}, reg_weight: {reg_weight}, reg_target_ratio: {reg_target_ratio}')
    print(f'{name}_2d: topo volume: {AssembleUtils.assemble_scalar(optim.vol_ratio_form)}')
    print(f'{name}_2d: loss: {optim.eval()}')
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(result_dir, 'log'))
    optim.run(
        max_iteration=200,
        line_search_steps=[0.05],
        tensorboard_recorder=recorder,
        full_run=True,
        momentum_weight=0.8,
    )


if __name__ == '__main__':
    eval_res()
    