import os
import shutil
from typing import Literal, Union, Callable

import numpy as np
import pyvista
import copy
import json

from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.opt_solver.grid_mesher import GridMesher
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import TopoFluidOptimizer
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_fipy import TopoFluidOptimizerFipy
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import AutoFluidBoundary

from thirdparty.pyadjoint.pyadjoint import Tape
from scripts_py.vtu_recorder import TensorBoardRecorder


class TopoFluidChainOptimizer(object):
    def __init__(
        self,
        name: str,
        run_dir: str,
        msh_file: str,
        mesh: MeshObj,
        boundary_list: list[AutoFluidBoundary],
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float|np.ndarray,
        topo_weight_init: float|np.ndarray,
        init_topo_weight_method: Literal['poisson', 'filter', 'none', 'raw'],
        weight_filter_radius: float,
        alpha_fun_max: float = 100.0,
        alpha_fun_min: float = 1e-12,
        alpha_fun_p: float = 3.0,
        us_fipy=False
    ):
        self.name = name
        self.run_dir = run_dir
        
        # if os.path.exists(self.run_dir):
        #     shutil.rmtree(self.run_dir)
        # os.makedirs(self.run_dir)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)
        
        self.us_fipy = us_fipy
        self.msh_file = msh_file
        
        self.mesh = mesh
        self.boundary_list = boundary_list
        self.fluid_type: Literal['stoke', 'naiver_stoke', 'rans'] = fluid_type
        self.viscosity_init = viscosity_init
        self.topo_weight_init = topo_weight_init
        self.init_topo_weight_method: Literal['poisson', 'filter', 'none', 'raw'] = init_topo_weight_method
        self.weight_filter_radius = weight_filter_radius
        self.alpha_fun_max = alpha_fun_max
        self.alpha_fun_min = alpha_fun_min
        self.alpha_fun_p = alpha_fun_p
    
    def optimize(
        self, 
        run_cfgs: list[dict] = [
            {
                'obj_cost_args_list': [{
                    'style': 'MinDissipation+BrinkmanPenalized',
                    'with_viscosity': True,
                    'dissipation_weight': 1.0,
                    'penalized_weight': 1.0,
                }],
                'reg_cost_args': {
                    'style': 'Volume',
                    'weight': 0.1,
                    'target_volume': 0.7,  # TODO: 在相互竞争下,这里会产生很坏的影响
                    'reg_rescale': True,
                },
                'obstacle_cost_args': None,
                'fipy_simulator_cfg': {
                    'simulation_iterations': 200,
                    'convergence_threshold': 1e-4,
                    'velocityRelaxation': 0.6,
                    'pressureRelaxation': 0.8,
                    'hot_steps': 10,
                    'log_step': 2,
                    'parallel_num': 3,
                    'use_tqdm': False,
                    'use_log': True,
                    'max_iteration': 300,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                },
                'dolfinx_simulator_cfg': {
                    'max_iteration': 100,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                    'hot_steps': 10,
                }
            }
        ]
    ):
        topo_weight_init = self.topo_weight_init
        init_topo_weight_method = self.init_topo_weight_method
        
        last_obj_cost_args_list: list[dict] = None  # type: ignore
        last_reg_cost_args: dict = None  # type: ignore
        last_obstacle_cost_args: dict = None  # type: ignore
        last_fipy_simulator_cfg: dict = None  # type: ignore
        last_dolfinx_simulator_cfg: dict = None  # type: ignore
        
        for chain_idx, run_cfg in enumerate(run_cfgs):
            chain_dir = os.path.join(self.run_dir, f'chain_{chain_idx}')
            
            # if os.path.exists(chain_dir):
            #     shutil.rmtree(chain_dir)
            # os.makedirs(chain_dir)
            if not os.path.exists(chain_dir):
                os.makedirs(chain_dir)
            
            result_opt_vtu = os.path.join(chain_dir, 'opt.vtu')
            
            # ------ define optimizer ------
            obj_cost_args_list = run_cfg.get('obj_cost_args_list', last_obj_cost_args_list)
            reg_cost_args = run_cfg.get('reg_cost_args', last_reg_cost_args)
            obstacle_cost_args = run_cfg.get('obstacle_cost_args', last_obstacle_cost_args)
            fipy_simulator_cfg = run_cfg.get('fipy_simulator_cfg', last_fipy_simulator_cfg)
            dolfinx_simulator_cfg = run_cfg.get('dolfinx_simulator_cfg', last_dolfinx_simulator_cfg)
            assert obj_cost_args_list is not None, 'obj_cost_args_list is required'
            assert reg_cost_args is not None, 'reg_cost_args is required'
            log_run_json = {
                'obj_cost_args_list': obj_cost_args_list,
                'reg_cost_args': reg_cost_args,
                'obstacle_cost_args': obstacle_cost_args,
                'fipy_simulator_cfg': fipy_simulator_cfg,
                'dolfinx_simulator_cfg': dolfinx_simulator_cfg,
            }
            
            # ------ define optimizer ------
            if self.us_fipy:
                assert fipy_simulator_cfg is not None, 'fipy_simulator_cfg is required'
                
                optim = TopoFluidOptimizerFipy(
                    name=self.name,
                    msh_file=self.msh_file,
                    mesh=self.mesh,
                    fluid_type=self.fluid_type,
                    viscosity_init=self.viscosity_init,
                    topo_weight_init=topo_weight_init,
                    init_topo_weight_method=init_topo_weight_method,
                    weight_filter_radius=self.weight_filter_radius,
                    result_dir=chain_dir,
                    alpha_fun_max=self.alpha_fun_max,
                    alpha_fun_min=self.alpha_fun_min,
                    alpha_fun_p=self.alpha_fun_p,
                    velocity_degree=1,
                    pressure_degree=1
                )
                optim.setup_tape(Tape())
                for boundary in self.boundary_list:
                    optim.add_boundary(boundary)
                
                optim.define_problem(
                    obj_cost_args_list=copy.deepcopy(obj_cost_args_list),
                    reg_cost_args=copy.deepcopy(reg_cost_args),
                    obstacle_cost_args=copy.deepcopy(obstacle_cost_args) if obstacle_cost_args is not None else None,
                    simulation_iterations=fipy_simulator_cfg['simulation_iterations'],
                    convergence_threshold=fipy_simulator_cfg['convergence_threshold'],
                    velocityRelaxation=fipy_simulator_cfg['velocityRelaxation'],
                    pressureRelaxation=fipy_simulator_cfg['pressureRelaxation'],
                    hot_steps=10,
                    log_step=fipy_simulator_cfg['log_step'],
                    parallel_num=fipy_simulator_cfg['parallel_num'],
                    use_tqdm=fipy_simulator_cfg['use_tqdm'],
                    use_log=fipy_simulator_cfg['use_log'],
                    adj_kwargs={
                        'critical_method': 'residual',
                        'mat_assemble_method': 'lift',
                        'with_residual': True,
                        'residual_error_threshold': 1e-3
                    },
                )
                recorder = TensorBoardRecorder(os.path.join(chain_dir, 'log'))
                result = optim.run(
                    max_iteration=fipy_simulator_cfg['max_iteration'],
                    line_search_steps=fipy_simulator_cfg['line_search_steps'],
                    momentum_weight=fipy_simulator_cfg['momentum_weight'],
                    recorder=recorder,
                    full_run=fipy_simulator_cfg['full_run'],
                    result_vtu_file=result_opt_vtu,
                    hot_step=fipy_simulator_cfg['hot_steps']
                )
                last_fipy_simulator_cfg = fipy_simulator_cfg
                
            else:
                assert dolfinx_simulator_cfg is not None, 'dolfinx_simulator_cfg is required'
                
                optim = TopoFluidOptimizer(
                    name=self.name,
                    mesh=self.mesh,
                    fluid_type=self.fluid_type,
                    viscosity_init=self.viscosity_init,
                    topo_weight_init=topo_weight_init,
                    init_topo_weight_method=init_topo_weight_method,
                    weight_filter_radius=self.weight_filter_radius,
                    result_dir=chain_dir,
                    alpha_fun_max=self.alpha_fun_max,
                    alpha_fun_min=self.alpha_fun_min,
                    alpha_fun_p=self.alpha_fun_p,
                )
                optim.setup_tape(Tape())
                for boundary in self.boundary_list:
                    optim.add_boundary(boundary)
            
                optim.define_problem(
                    obj_cost_args_list=copy.deepcopy(obj_cost_args_list),
                    reg_cost_args=copy.deepcopy(reg_cost_args),
                    obstacle_cost_args=copy.deepcopy(obstacle_cost_args) if obstacle_cost_args is not None else None,
                )
                recorder = TensorBoardRecorder(os.path.join(chain_dir, 'log'))
                result = optim.run(
                    max_iteration=dolfinx_simulator_cfg['max_iteration'],
                    line_search_steps=dolfinx_simulator_cfg['line_search_steps'],
                    momentum_weight=dolfinx_simulator_cfg['momentum_weight'],
                    recorder=recorder,
                    full_run=dolfinx_simulator_cfg['full_run'],
                    result_vtu_file=result_opt_vtu,
                    hot_step=dolfinx_simulator_cfg['hot_steps']
                )
                last_dolfinx_simulator_cfg = dolfinx_simulator_cfg
            
            last_obj_cost_args_list = obj_cost_args_list
            last_reg_cost_args = reg_cost_args
            last_obstacle_cost_args = obstacle_cost_args
            
            with open(os.path.join(chain_dir, 'run_cfg.json'), 'w') as f:
                json.dump(log_run_json, f, indent=4)

            res_vtu = pyvista.read(result_opt_vtu)
            topo_weight_init = res_vtu.point_data['topo_weight']
            # TODO: topo_weight_init 是不是应该弱化一下, check
            topo_weight_init = (topo_weight_init - np.min(topo_weight_init)) / (np.max(topo_weight_init) - np.min(topo_weight_init)) * 0.1 + 0.1
            init_topo_weight_method = self.init_topo_weight_method
            
        return result


def debug():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t47'
    msh_file = os.path.join(base_dir, 'model.msh')

    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
 
    boundaries = [
        AutoFluidBoundary(
            name='inlet',
            boundary_type='velocity',
            value=[0.0, 1.0],
            marker=1,
            face_center=np.array([1.0, -0.05]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet1',
            boundary_type='pressure',
            value=0.0,
            marker=2,
            face_center=np.array([0.2, 1.05]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet2',
            boundary_type='pressure',
            value=0.0,
            marker=3,
            face_center=np.array([0.6, 1.05]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet3',
            boundary_type='pressure',
            value=0.0,
            marker=4,
            face_center=np.array([1.0, 1.05]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet4',
            boundary_type='pressure',
            value=0.0,
            marker=5,
            face_center=np.array([1.4, 1.05]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet5',
            boundary_type='pressure',
            value=0.0,
            marker=6,
            face_center=np.array([1.8, 1.05]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='wall',
            boundary_type='wall',
            value=0.0,
            marker=7
        )
    ]
    
    result_dir = os.path.join(base_dir, 'dolfinx_chain_topo')
    optim = TopoFluidChainOptimizer(
        name='compare',
        run_dir=result_dir,
        msh_file=msh_file,
        mesh=mesh,
        boundary_list=boundaries,
        fluid_type='stoke',
        viscosity_init=1.0 / 400.,
        topo_weight_init=0.1,
        init_topo_weight_method='filter',
        weight_filter_radius=0.05,
        alpha_fun_max=100.0,
        alpha_fun_min=1e-12,
        alpha_fun_p=3.0,
        us_fipy=False,
    )
    
    optim.optimize(
        run_cfgs=[
            {
                'obj_cost_args_list': [
                    {
                        'style': 'MinDissipation+BrinkmanPenalized',
                        'with_viscosity': True,
                        'dissipation_weight': 1.0,
                        'penalized_weight': 1.0,
                    },
                    {
                        'style': 'FixedFlux',
                        'inlet_markers': [1],
                        'outlet_markers': [2, 3, 4, 5, 6],
                        'outlet_marker_ratios': [0.2, 0.2, 0.2, 0.2, 0.2],
                        'outlet_marker_weights': [0.1, 0.1, 0.1, 0.1, 0.1],
                        'with_rescale': True
                    }
                ],
                'reg_cost_args': {
                    'style': 'Volume',
                    'weight': 0.1,
                    'target_volume': 0.7,
                    'reg_rescale': True,
                },
                'obstacle_cost_args': None,
                'dolfinx_simulator_cfg': {
                    'max_iteration': 100,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                }
            },
            {
                'reg_cost_args': {
                    'style': 'Volume',
                    'weight': 0.1,
                    'target_volume': 0.4,
                    'reg_rescale': True,
                },
            },
            {
                'reg_cost_args': {
                    'style': 'Volume',
                    'weight': 0.1,
                    'target_volume': 0.2,  # TODO: 似乎说明不可实现
                    'reg_rescale': True,
                }
            }
        ]
    )


if __name__ == '__main__':
    debug()