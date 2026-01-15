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


"""
拓扑方法只做后续修正,使用BESO生成原型,我猜是优化路径问题(但reg_cost这么小,却能影响优化结果,说明reg_cost的敏度/优化斜率很大)
可不可以在二维复现这个问题
"""

base_dir = '/home/admin123456/Desktop/work/gmsh_test/compare'


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
        'value': [1.0, 0.0, 0.0],
        'marker': 1,
        'face_center': np.array([-0.05, 0.5, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'outlet',
        'boundary_type': 'pressure',
        'value': 0.0,
        'marker': 2,
        'face_center': np.array([1.05, 0.5, 0.5]),
        'face_norm_vector': np.array([1.0, 0.0, 0.0]),
        'face_radius': 0.1,
    },
    {
        'name': 'wall',
        'boundary_type': 'wall',
        'value': 0.0,
        'marker': 3,
    }
]


def get_init_topo_weight(mesh: SeqMeshObj, is_cell_field: bool, radius: float):
    if is_cell_field:
        cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(mesh.domain)
        point_coords = DolfinUtils.tabulate_dof_coordinates(cell_scalar_func_space)
    else:
        point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
        point_coords = DolfinUtils.tabulate_dof_coordinates(point_scalar_func_space)
    
    from_center = boundary_cfgs[0]['face_center']
    to_center = boundary_cfgs[1]['face_center']
    path = np.array([from_center, to_center])
    
    detail_step = 0.025
    detail_path = np.linspace(path[0], path[1], math.ceil(np.linalg.norm(path[1] - path[0]) / detail_step))
    
    tree = KDTree(point_coords)
    idxs_list, dists_list = tree.query_radius(detail_path, r=radius, return_distance=True)
    idxs_list = np.concatenate(idxs_list, axis=-1)
    dists_list = np.concatenate(dists_list, axis=-1)
        
    df = pd.DataFrame({'idxs': idxs_list, 'dists': dists_list})
    df.sort_values(by='dists', inplace=True)
    df.drop_duplicates(subset=['idxs'], keep='first', inplace=True)
    idxs_list = df['idxs'].values
    dists_list = df['dists'].values
    
    topo_weight = np.ones(shape=mesh.num_points) * 0.3
    topo_weight[idxs_list] = 1.0 - 0.3  # type: ignore
    
    plotter = pyvista.Plotter()
    point_mesh = pyvista.PointSet(point_coords)
    point_mesh.point_data['color'] = topo_weight
    point_mesh.warp_by_scalar('color')
    plotter.add_mesh(point_mesh)
    plotter.show()
    
    return topo_weight

    
def eval_res():
    # vtu_file = os.path.join(base_dir, 'topo_ref/report_00194.vtu')
    # vtu_mesh = pyvista.read(vtu_file)
    # topo_weight_init = vtu_mesh.point_data['topo_weight']
        
    msh_file = os.path.join(base_dir, 'model.msh')
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=3)
    mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    # topo_weight_init = get_init_topo_weight(mesh, is_cell_field=False, radius=0.1)
    topo_weight_init = 0.1
    
    name = 'topo_exam_09'
    alpha_fun_max = 10.0
    reg_weight = 1.0
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
        parallel_num=4,
        use_tqdm=True,
        use_log=True,
        activate_init_method='boundary_radius',
        wall_init_method='boundary_raw',
        constraint_method='fix',
        reg_rescale=reg_rescale
    )
    
    print(f'{name}: alpha_fun_max: {alpha_fun_max}, reg_weight: {reg_weight}, reg_target_ratio: {reg_target_ratio}')
    print(f'{name}: topo volume: {AssembleUtils.assemble_scalar(optim.vol_ratio_form)}')
    print(f'{name}: loss: {optim.eval()}')
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(result_dir, 'log'))
    optim.run(
        max_iteration=400,
        line_search_steps=[0.05],
        tensorboard_recorder=recorder,
        full_run=True,
        momentum_weight=0.8,
    )


if __name__ == '__main__':
    eval_res()
    