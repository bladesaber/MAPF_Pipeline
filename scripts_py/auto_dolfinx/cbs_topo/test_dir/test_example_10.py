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


"""
FIXME:
1. 结果还是有不符合我们想象的地方,进出口处可能由于崩角原因,速度会在进出口处变得很大,导致进出口体积大,中间体积小. (似乎还是由于初始化原因造成的)
2. 为什么结果存在崩塌,为什么崩塌后obj_cost反而更小.我先创建一个我认为好的初始化topo weight然后再看看其是否崩塌

这两个问题都可以归结为: 
    a.当topo weight普遍小时,仿真结果不准确 
    b.不准确的仿真结果导致前期错误的topo weight演变,但由于topology optimization的局部优化特性,导致前期错误的topo weight演变可能难以修复(低效)
    c.仿真器有问题或者目标函数有问题

我计划放弃这些方法
"""

base_dir = '/home/admin123456/Desktop/work/gmsh_test/t43/'


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


def get_init_topo_weight(mesh: SeqMeshObj, is_cell_field: bool):
    if is_cell_field:
        cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(mesh.domain)
        point_coords = DolfinUtils.tabulate_dof_coordinates(cell_scalar_func_space)
    else:
        point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
        point_coords = DolfinUtils.tabulate_dof_coordinates(point_scalar_func_space)
    
    from_center = boundary_cfgs[0]['face_center']
    to_center = boundary_cfgs[1]['face_center']
    radius = 0.2
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
    dists_list = (radius - dists_list) / radius * 0.5 + 0.5  # type: ignore
    
    if is_cell_field:
        topo_weight = np.ones(shape=mesh.num_cells) * 0.25
    else:
        topo_weight = np.ones(shape=mesh.num_points) * 0.25
    topo_weight[idxs_list] = dists_list  # type: ignore
    
    # plotter = pyvista.Plotter()
    # point_mesh = pyvista.PointSet(point_coords)
    # point_mesh.point_data['color'] = topo_weight
    # point_mesh.warp_by_scalar('color')
    # plotter.add_mesh(point_mesh)
    # plotter.show()
    
    return topo_weight

    
def run_topo_opt():
    msh_file = os.path.join(base_dir, 'model.msh')
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=3)
    mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    # topo_weight_init = 1.0
    # FIXME: 仿真器有问题或者目标函数有问题,形变错误后,obj cost仍然下降，还是我的想象是错的呢(奇怪形状是对的?)
    topo_weight_init = get_init_topo_weight(mesh, is_cell_field=False)
    
    cost_item = AutoFluidCost(name='obj_cost', style='MinDissipation', weight=1.0)
    regularization_item = AutoFluidRegularization(name='reg_cost', style='Volume', weight=1.0, target_ratio=0.2)
    
    result_dir = os.path.join(base_dir, 'topo_opt_with_init')
    optim = TopoFluidOptimizerFipy(
        name='test_example_10',
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
        parallel_num=8,
        use_tqdm=True,
        use_log=True,
        activate_init_method='boundary_radius',
        wall_init_method='boundary_raw',
        constraint_method='fix',
    )
            
    recorder = TensorBoardRecorder(log_dir=os.path.join(result_dir, 'log'))
    optim.run(
        line_search_steps=[0.02],
        tensorboard_recorder=recorder,
        full_run=True,
        momentum_weight=0.8,
    )


def run_simulation():
    from scripts_py.auto_dolfinx.cbs_topo.fipy_obs_simulator import FipyStokeObsTopoSimulator
    
    init_file_1 = '/home/admin123456/Desktop/work/gmsh_test/t43/topo_opt_with_init/report_00032.vtu'
    init_file_2 = '/home/admin123456/Desktop/work/gmsh_test/t43/topo_opt_with_init/report_00194.vtu'
    for idx, init_file in enumerate([init_file_1, init_file_2]):
        source_msh_file = os.path.join(base_dir, 'model.msh')
        
        simulation_dir = os.path.join(base_dir, f'simulztion_{idx}')
        if os.path.exists(simulation_dir):
            shutil.rmtree(simulation_dir)
        os.mkdir(simulation_dir)
        
        msh_file = os.path.join(simulation_dir, 'model.msh')
        shutil.copy(source_msh_file, msh_file)
        
        domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=3)
        mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        
        init_vtu = pyvista.read(init_file)
        obs_weight = (init_vtu.cell_data['topo_weight'] < 0.5).astype(float)

        mesh.grid.cell_data.clear()
        mesh.grid.cell_data['obstacle'] = obs_weight
        init_vtu_file = os.path.join(simulation_dir, 'init.vtu')
        mesh.grid.save(init_vtu_file)
        
        boundary_list = [define_boundary(bry_cfg, boundary_type=bry_cfg['boundary_type']) for bry_cfg in boundary_cfgs]
        FipyStokeObsTopoSimulator.create_simulation_config(
            cfg_file=os.path.join(simulation_dir, 'config.json'),
            source_msh_file=msh_file,
            dim=3,
            viscosity=1.0 / 400.0,
            velocityRelaxation=0.5,
            pressureRelaxation=0.8,
            boundary_list=boundary_list,  # type: ignore
            iterations=200,
            convergence_threshold=1e-4,
            hot_steps=10,
            log_step=2,
            save_dir=os.path.join(simulation_dir, 'cache'),
            use_tqdm=False,
            use_log=True,
            structure_init_vtu_file=init_vtu_file,
            vtu_velocity_field='velocity',
            vtu_pressure_field='pressure',
            vtu_alpha_field='alpha',
            vtu_obs_weight_field='obstacle',
        )


if __name__ == '__main__':
    # run_topo_opt()
    run_simulation()
    