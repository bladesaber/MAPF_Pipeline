import os, shutil
from typing import Tuple, Literal
from functools import partial
from tqdm import tqdm

import dolfinx

import json
from meshio import dolfin
import numpy as np
from pandas._config.dates import cf
import pyvista
from sklearn.neighbors import KDTree

from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.mesh_utils import MeshUtils
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.vtu_recorder import TensorBoardRecorder
from scripts_py.auto_dolfinx.cbs_topo.utils import extract_grid_info_from_standard_mesh

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from thirdparty.pyadjoint.pyadjoint import Tape
from scripts_py.auto_dolfinx.cbs_topo.topo_fluid_fipy_optimizer import TopoFluidOptimizerFipy


class CbsMeshObj(SeqMeshObj):
    def __init__(
        self, 
        msh_file: str,
        dim: int,
        cell_mode: bool,
    ):
        domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=dim)
        super().__init__(None, 'model', domain, cell_tags, facet_tags, cell_mode)
        self.msh_file = msh_file


class TopoFluidNode(object):
    @staticmethod
    def bry_to_dict(bry: AutoFluidBoundary):
        cfg = {
            'name': bry.name,
            'boundary_type': bry.boundary_type,
            'marker': bry.marker,
        }
        if isinstance(bry.value, np.ndarray):
            cfg.update({'value': bry.value.tolist()})
        else:
            cfg.update({'value': bry.value})
        
        if bry.face_center is not None:
            if isinstance(bry.face_center, np.ndarray):
                cfg.update({'face_center': bry.face_center.tolist()})
            else:
                cfg.update({'face_center': bry.face_center})
        if bry.face_norm_vector is not None:
            if isinstance(bry.face_norm_vector, np.ndarray):
                cfg.update({'face_norm_vector': bry.face_norm_vector.tolist()})
            else:
                cfg.update({'face_norm_vector': bry.face_norm_vector})
        if bry.face_radius is not None:
            cfg.update({'face_radius': bry.face_radius})
        return cfg
    
    def __init__(
        self, 
        mesh: CbsMeshObj, 
        name: str, 
        save_dir: str, 
        node_idx: int,
        weight_filter_radius: float = 0.06,
        topo_weight_init: float = 1.0,
        
        # ------ simulator related ------
        viscosity: float = 1.0 / 400.,
        
        # ------ computational related ------
        simulation_iterations: int = 400,
        convergence_threshold: float = 1e-4,
        velocityRelaxation: float = 0.6,
        pressureRelaxation: float = 0.8,
        log_step: int = 2,
        hot_steps: int = 10,
        parallel_num: int = 1,
        use_tqdm: bool = True,
        use_log: bool = True,
    ):
        self.name = name
        self.node_idx = node_idx
        
        self.save_dir = save_dir
        self.fluid_dir = os.path.join(save_dir, f'{self.name}_{self.node_idx}')
        if not os.path.exists(self.fluid_dir):
            os.mkdir(self.fluid_dir)
        
        self.mesh: CbsMeshObj = mesh
        self.inlet_brys: dict[str, AutoFluidBoundary] = {}
        self.outlet_brys: dict[str, AutoFluidBoundary] = {}
        self.wall_brys: dict[str, AutoFluidBoundary] = {}
        
        self.object_cost: AutoFluidCost
        self.regularization_cost: AutoFluidRegularization
        self.loss: float
        
        self.obstacle_center: np.ndarray = np.array([])
        self.obstacle_radius: float = 0.02  # TODO: please check
        
        # self.body: np.ndarray = None  # type: ignore
        # self.body_file = os.path.join(self.fluid_dir, 'result.npy')
        # self.point_body_mask: np.ndarray = None  # type: ignore
        
        self.is_clear = True
        
        # ------ topology optimization related ------
        self.weight_filter_radius = weight_filter_radius
        self.topo_weight_init = topo_weight_init
        
        # ------ simulator related ------
        self.viscosity = viscosity
        
        # ------ computational related ------
        self.simulation_iterations = simulation_iterations
        self.convergence_threshold = convergence_threshold
        self.velocityRelaxation = velocityRelaxation
        self.pressureRelaxation = pressureRelaxation
        self.log_step = log_step
        self.hot_steps = hot_steps
        self.parallel_num = parallel_num
        self.use_tqdm = use_tqdm
        self.use_log = use_log
        
        # ------ save related ------
        self.cfg_file = os.path.join(self.fluid_dir, 'node_cfg.json')
        self.obstacle_center_file = os.path.join(self.fluid_dir, 'obstacle_center.npy')
        self.topo_res_dir = os.path.join(self.fluid_dir, 'topo_res')

    def solve(self):
        define_obstacle_func = partial(
            TopoFluidNode.define_obstacle_func, 
            obstacle_centers=self.obstacle_center,
            obstacle_radius = self.obstacle_radius
        )
        
        optim = TopoFluidOptimizerFipy(
            name=self.name,
            mesh=self.mesh,
            msh_file=self.mesh.msh_file,
            nut_init=self.viscosity,
            cost_item=self.object_cost,
            regularization_item=self.regularization_cost,
            topo_weight_init=self.topo_weight_init,
            obstacle_func=define_obstacle_func,
            weight_filter_radius=self.weight_filter_radius,
            weight_filter_is_cell_field=False,
            result_dir=self.topo_res_dir,
            velocity_degree=1,
            pressure_degree=1,
        )
        
        tape = Tape()
        optim.setup_tape(tape)
        
        for _, inlet_bry in self.inlet_brys.items():
            optim.add_boundary(inlet_bry)
        for _, outlet_bry in self.outlet_brys.items():
            optim.add_boundary(outlet_bry)
        for _, wall_bry in self.wall_brys.items():
            optim.add_boundary(wall_bry)
        optim.define_problem(
            simulation_iterations=self.simulation_iterations,
            convergence_threshold=self.convergence_threshold,
            velocityRelaxation=self.velocityRelaxation,
            pressureRelaxation=self.pressureRelaxation,
            adj_ksp_option={
                'ksp_type': 'richardson',
                'ksp_max_it': 1,
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps',
                'mat_mumps_icntl_24': 1,
                'mat_mumps_icntl_25': 0,
            },
            adj_kwargs={
                'mat_assemble_method': 'lift',
                'with_residual': True,
                'critical_method': 'residual'
            },
            log_step=self.log_step,
            hot_steps=self.hot_steps,
            parallel_num=self.parallel_num,
            use_tqdm=self.use_tqdm,
            use_log=self.use_log,
            activate_init_method='boundary_radius',
            wall_init_method='boundary_raw',
            constraint_method='fix',
        )
        
        recorder = TensorBoardRecorder(log_dir=os.path.join(self.fluid_dir, 'log'))
        
        optim.run(
            # max_iteration=200,
            # line_search_steps=[0.1, 0.05, 0.01],
            max_iteration=200,
            line_search_steps=[0.02],
            tensorboard_recorder=recorder,
            hot_step=20,
            full_run=True
        )
        
        loss_dict = optim.eval()
        self.loss = loss_dict[self.object_cost.name] + loss_dict[self.regularization_cost.name]

    def save(self):
        inlet_bry_dict = {}
        for name, bry in self.inlet_brys.items():
            inlet_bry_dict[name] = self.bry_to_dict(bry)
        
        outlet_bry_dict = {}
        for name, bry in self.outlet_brys.items():
            outlet_bry_dict[name] = self.bry_to_dict(bry)
        
        wall_bry_dict = {}
        for name, bry in self.wall_brys.items():
            wall_bry_dict[name] = self.bry_to_dict(bry)
        
        np.save(self.obstacle_center_file, self.obstacle_center)
        # if self.body is not None:
        #     np.save(self.body_file, self.body)
        
        cfg = {
            'name': self.name,
            'msh_file': self.mesh.msh_file,
            'tdim': self.mesh.tdim,
            'cell_mode': self.mesh.cell_mode,
            
            'wall_brys': wall_bry_dict,
            'inlet_brys': inlet_bry_dict,
            'outlet_brys': outlet_bry_dict,
            
            'object_cost': {
                'name': self.object_cost.name,
                'style': self.object_cost.style,
                'weight': self.object_cost.weight
            },
            'regularization_cost': {
                'name': self.regularization_cost.name,
                'style': self.regularization_cost.style,
                'weight': self.regularization_cost.weight,
                'kwargs': self.regularization_cost.kwargs
            },
            'loss': self.loss,
            
            'weight_filter_radius': self.weight_filter_radius,
            'topo_weight_init': self.topo_weight_init,
            'viscosity': self.viscosity,
            'simulation_iterations': self.simulation_iterations,
            'convergence_threshold': self.convergence_threshold,
            'velocityRelaxation': self.velocityRelaxation,
            'pressureRelaxation': self.pressureRelaxation,
            'log_step': self.log_step,
            'hot_steps': self.hot_steps,
            'parallel_num': self.parallel_num,
            'use_tqdm': self.use_tqdm,
            'use_log': self.use_log,
            
            'obstacle_radius': self.obstacle_radius,
            'obstacle_file': self.obstacle_center_file,
        }
        
        with open(self.cfg_file, 'w') as f:
            json.dump(cfg, f, indent=4)
    
    def load(self):
        if not os.path.exists(self.cfg_file):
            raise ValueError('[Error]: No cfg file.')
        with open(self.cfg_file, 'r') as f:
            cfg = json.load(f)
        assert self.name == cfg['name']
        
        # if os.path.exists(self.body_file):
        #     self.body = np.load(self.body_file, allow_pickle=True)
        
        for _, item in cfg['wall_brys'].items():
            self.wall_brys[item['name']] = AutoFluidBoundary(
                name=item['name'],
                boundary_type=item['boundary_type'],
                value=item['value'],
                marker=item['marker']
            )

        for _, item in cfg['inlet_brys'].items():
            self.inlet_brys[item['name']] = AutoFluidBoundary(
                name=item['name'],
                boundary_type=item['boundary_type'],
                value=item['value'],
                marker=item['marker'],
                face_center=item['face_center'],
                face_norm_vector=item['face_norm_vector'],
                face_radius=item['face_radius'],
            )
        
        for _, item in cfg['outlet_brys'].items():
            self.outlet_brys[item['name']] = AutoFluidBoundary(
                name=item['name'],
                boundary_type=item['boundary_type'],
                value=item['value'],
                marker=item['marker'],
                face_center=item['face_center'],
                face_norm_vector=item['face_norm_vector'],
                face_radius=item['face_radius'],
            )
        
        self.object_cost = AutoFluidCost(
            name=cfg['object_cost']['name'],
            style=cfg['object_cost']['style'],
            weight=cfg['object_cost']['weight'],
        )
        self.regularization_cost = AutoFluidRegularization(
            name=cfg['regularization_cost']['name'],
            style=cfg['regularization_cost']['style'],
            weight=cfg['regularization_cost']['weight'],
            **cfg['regularization_cost']['kwargs']
        )
        
        self.is_clear = False
        self.loss = cfg['loss']
        
        # ------ obstacle related ------
        self.obstacle_center = np.load(self.obstacle_center_file, allow_pickle=True)
        self.obstacle_radius = cfg['obstacle_radius']
        
        # ------ topology optimization related ------
        self.weight_filter_radius = cfg['weight_filter_radius']
        self.topo_weight_init = cfg['topo_weight_init']
        
        # ------ simulator related ------
        self.viscosity = cfg['viscosity']
        
        # ------ computational related ------
        self.simulation_iterations = cfg['simulation_iterations']
        self.convergence_threshold = cfg['convergence_threshold']
        self.velocityRelaxation = cfg['velocityRelaxation']
        self.pressureRelaxation = cfg['pressureRelaxation']
        self.log_step = cfg['log_step']
        self.hot_steps = cfg['hot_steps']
        self.parallel_num = cfg['parallel_num']
        self.use_tqdm = cfg['use_tqdm']
        self.use_log = cfg['use_log']
    
    def clear(self, force_clear=False):
        if not self.is_clear or force_clear:
            self.inlet_brys.clear()
            self.outlet_brys.clear()
            self.wall_brys.clear()
            del self.obstacle_center
            del self.object_cost
            del self.regularization_cost
            # del self.body
            self.is_clear = True
    
    @staticmethod
    def define_obstacle_func(cell_centers: np.ndarray, obstacle_centers: np.ndarray, obstacle_radius: float):
        if obstacle_centers.shape[0] == 0:
            return np.array([], dtype=int)
        
        tree = KDTree(obstacle_centers)
        dists_list, idxs_list = tree.query(cell_centers, k=1, return_distance=True)
        dists_list = np.concatenate(dists_list, axis=-1)
        obs_dofs = np.where(dists_list < obstacle_radius)[0]
        obs_dofs = np.unique(obs_dofs)
        return obs_dofs

    def copy(self, node_idx: int):
        new_node = TopoFluidNode(self.mesh, self.name, self.save_dir, node_idx)
        for file_name in os.listdir(self.fluid_dir):
            if os.path.isfile(os.path.join(self.fluid_dir, file_name)):
                shutil.copy(
                    os.path.join(self.fluid_dir, file_name),
                    os.path.join(new_node.fluid_dir, file_name),
                )
            else:
                if file_name == 'topo_res':
                    continue
                shutil.copytree(
                    os.path.join(self.fluid_dir, file_name),
                    os.path.join(new_node.fluid_dir, file_name),
                )
        return new_node

    def update_obstacle_center(self, obstacle_center: np.ndarray):
        self.obstacle_center = obstacle_center
        np.save(self.obstacle_center_file, obstacle_center)

