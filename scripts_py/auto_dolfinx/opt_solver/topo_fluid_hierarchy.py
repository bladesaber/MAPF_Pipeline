import os
import shutil
from typing import Literal, Union, Callable

import dolfinx

import numpy as np
import pyvista
from functools import partial
import copy
import json

from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.opt_solver.grid_mesher import GridMesher
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import TopoFluidOptimizer
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_fipy import TopoFluidOptimizerFipy
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import AutoFluidBoundary
from scripts_py.auto_dolfinx.opt_solver.topo_fluid_chain import TopoFluidChainOptimizer
from scripts_py.dolfinx_adjoint.topology_utils.topology_filter import DistanceBandFilter

from thirdparty.pyadjoint.pyadjoint import Tape
from scripts_py.vtu_recorder import TensorBoardRecorder


class TopoFluidGridOptimizer(object):
    def __init__(
        self,
        name: str,
        run_dir: str,
        space_dim: int,
        space_left_bottom_corner: np.ndarray,
        space_right_top_corner: np.ndarray,
        space_resolution: float,
        mesh_min_size: float,
        mesh_max_size: float,
        boundary_list: list[AutoFluidBoundary],
        obj_cost_args_list: list[dict],
        reg_cost_args: dict,
        obstacle_cost_args: dict|None,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float|np.ndarray,
        topo_weight_init: float|np.ndarray,
        init_topo_weight_method: Literal['poisson', 'filter', 'none', 'raw'],
        alpha_fun_max: float = 100.0,
        alpha_fun_min: float = 1e-12,
        alpha_fun_p: float = 3.0,
        weight_filter_ratio: float = 1.5,
        dist_max_ratio: float = 1.0,
        us_fipy=False,
        fipy_simulator_dict = {
            'simulation_iterations': 200,
            'convergence_threshold': 1e-4,
            'velocityRelaxation': 0.6,
            'pressureRelaxation': 0.8,
            'hot_steps': 10,
            'log_step': 2,
            'parallel_num': 8,
            'use_tqdm': True,
            'use_log': True,
            'max_iteration': 150,
            'line_search_steps': [0.05],
            'momentum_weight': 0.8,
            'full_run': True,
        },
        dolfinx_simulator_dict = {
            'max_iteration': 150,
            'line_search_steps': [0.05],
            'momentum_weight': 0.8,
            'full_run': True,
        },
        inout_pipe_length: float = 0.05
    ):
        self.name = name
        self.run_dir = run_dir
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
        os.makedirs(self.run_dir)
        self.opt_dir = os.path.join(run_dir, 'opt')
        
        self.geo_file_path = os.path.join(self.run_dir, 'design_space.geo')
        self.msh_file_path = os.path.join(self.run_dir, 'design_space.msh')
        self.result_opt_vtu = os.path.join(self.run_dir, 'opt.vtu')
                
        self.obj_cost_args_list = obj_cost_args_list
        self.reg_cost_args = reg_cost_args
        assert 'target_volume' in self.reg_cost_args.keys(), 'Target volume is required for Volume regularization'
        self.obstacle_cost_args = obstacle_cost_args
        
        self.us_fipy = us_fipy
        self.fipy_simulator_dict = fipy_simulator_dict
        self.dolfinx_simulator_dict = dolfinx_simulator_dict
        
        self.space_dim = space_dim
        self.space_left_bottom_corner = space_left_bottom_corner
        self.space_right_top_corner = space_right_top_corner
        self.space_resolution = space_resolution
        self.mesh_min_size = mesh_min_size
        self.mesh_max_size = mesh_max_size
        self.dist_max_ratio = dist_max_ratio
        
        self.inout_pipe_length = inout_pipe_length
        if self.inout_pipe_length > 0.0 and self.space_dim == 3:
            # TODO: 这里默认了velocity是inlet,pressure是outlet,日后需要更改
            self.boundary_list = []
            for boundary in boundary_list:
                boundary_copy = copy.deepcopy(boundary)
                boundary_copy.origin_face_center = boundary_copy.face_center  # type: ignore
                if boundary_copy.boundary_type == 'velocity':
                    boundary_copy.face_center = boundary_copy.face_center - boundary_copy.face_norm_vector * self.inout_pipe_length
                elif boundary_copy.boundary_type == 'pressure':
                    boundary_copy.face_center = boundary_copy.face_center + boundary_copy.face_norm_vector * self.inout_pipe_length
                self.boundary_list.append(boundary_copy)
        else:
            self.boundary_list = boundary_list
        
        # ------ optimizer related ------
        self.fluid_type: Literal['stoke', 'naiver_stoke', 'rans'] = fluid_type
        self.viscosity_init = viscosity_init
        self.topo_weight_init = topo_weight_init
        self.init_topo_weight_method: Literal['poisson', 'filter', 'none', 'raw'] = init_topo_weight_method
        self.weight_filter_ratio = weight_filter_ratio
        self.weight_filter_radius = self.mesh_max_size * self.weight_filter_ratio
        self.alpha_fun_max = alpha_fun_max
        self.alpha_fun_min = alpha_fun_min
        self.alpha_fun_p = alpha_fun_p
        
        # ------
        self.mesh: MeshObj = None  # type: ignore
        self.topo_vtu: pyvista.UnstructuredGrid = None  # type: ignore
        self.topo_function: dolfinx.fem.Function = None  # type: ignore
        
        self.grid_mesher: GridMesher
        self.field_value: np.ndarray

    def prepare_design_space(
        self, reference_optimizer: Union["TopoFluidGridOptimizer", None] = None,
        threshold: float = 0.4, diffusion_step = 0, compress_mesh=False
    ):
        grid_mesher = GridMesher(
            dim=self.space_dim,
            left_bottom_corner=self.space_left_bottom_corner,
            right_top_corner=self.space_right_top_corner,
            resolution=self.space_resolution,
            boundaries=self.boundary_list,  # type: ignore
            inout_pipe_length=self.inout_pipe_length,
        )
        
        if reference_optimizer is None:
            init_field_value = np.ones(shape=(grid_mesher.cell_num,))
        else:
            grid_points = grid_mesher.points
            if grid_points.shape[1] == 2:
                grid_points = np.concatenate([grid_points, np.zeros(shape=(grid_points.shape[0], 1))], axis=1)
            
            point_field_value = np.zeros(shape=(grid_mesher.point_num,))
            point_to_ref_cell_idxs: np.ndarray = MeshUtils.cell_indices_of_points(
                reference_optimizer.prepare_mesh().grid, grid_points  # type: ignore
            )  
            inside_ref_cell_bool = point_to_ref_cell_idxs >= 0
            point_field_value[inside_ref_cell_bool] = DolfinUtils.function_eval(
                uh=reference_optimizer.prepare_topo_function(diffusion_step=diffusion_step),
                pcd_xyz=grid_points[inside_ref_cell_bool],
                cell_indices=point_to_ref_cell_idxs[inside_ref_cell_bool].tolist()
            ).reshape(-1)
            
            # ------ add pipe include plane points ------
            if self.inout_pipe_length > 0.0:
                for bry in self.boundary_list:
                    if bry.boundary_type == 'wall':
                        continue
                    face_center = bry.origin_face_center[:grid_mesher.dim]  # type: ignore
                    in_cylinder_bool = np.bitwise_and(
                        np.linalg.norm(grid_mesher.points - face_center, axis=1) < bry.face_radius,
                        np.abs(np.dot(grid_mesher.points - face_center, bry.face_norm_vector)) < grid_mesher.resolution * 0.5
                    )
                    point_field_value[in_cylinder_bool] = 1.0
            # ------ add pipe include plane points ------
            
            point_field_value = (point_field_value >= threshold).astype(float)
            
            # # ----- debug visualization ------
            # plotter = pyvista.Plotter()
            # plotter.add_mesh(reference_optimizer.prepare_mesh().grid, color='black', style='wireframe')
            # point_mesh = pyvista.PointSet(grid_points)
            # point_mesh.point_data['field_value'] = point_field_value
            # point_mesh.warp_by_scalar('field_value')
            # plotter.add_mesh(point_mesh, point_size=8)
            # plotter.show()
            # # ------ debug visualization ------
            
            inside_line_bool = np.sum(point_field_value[grid_mesher.line2points], axis=1) > 0.0
            if self.space_dim == 2:
                inside_cell_bool = np.sum(inside_line_bool[grid_mesher.surface2lines], axis=1) > 0.0
            else:
                inside_surface_bool = np.sum(inside_line_bool[grid_mesher.surface2lines], axis=1) > 0.0
                inside_cell_bool = np.sum(inside_surface_bool[grid_mesher.volume2surfaces], axis=1) > 0.0
            init_field_value = np.zeros(shape=(grid_mesher.cell_num,))
            init_field_value[inside_cell_bool] = 1.0
            
            # # ------ debug visualization ------
            # plotter = pyvista.Plotter()
            # plotter.add_mesh(reference_optimizer.prepare_mesh().grid, color='black', style='wireframe')
            
            # if grid_mesher.dim == 2:
            #     unique_lines_idxs = np.unique(grid_mesher.surface2lines[init_field_value == 1.0])
            #     unique_points_idxs= np.unique(grid_mesher.line2points[unique_lines_idxs])
            #     unique_points = grid_mesher.points[unique_points_idxs]
            # else:
            #     unique_surfaces_idxs = np.unique(grid_mesher.volume2surfaces[init_field_value == 1.0])
            #     unique_lines_idxs = np.unique(grid_mesher.surface2lines[unique_surfaces_idxs])
            #     unique_points_idxs= np.unique(grid_mesher.line2points[unique_lines_idxs])
            #     unique_points = grid_mesher.points[unique_points_idxs]
            
            # if unique_points.shape[1] == 2:
            #     unique_points = np.concatenate([unique_points, np.zeros(shape=(unique_points.shape[0], 1))], axis=1)
            
            # plotter.add_mesh(pyvista.PointSet(grid_points), color='blue', point_size=5)
            # plotter.add_mesh(pyvista.PointSet(unique_points), color='red', point_size=8)
            # plotter.show()
            # # ------ debug visualization ------
        
        grid_mesher.extract_field(
            field_value=init_field_value,
            threshold=0.5,
            output_type='geo',
            geo_file_path=self.geo_file_path,
            min_size=self.mesh_min_size,
            max_size=max(self.mesh_max_size, self.mesh_min_size * 1.1),
            dist_min=self.mesh_min_size,
            dist_max=self.mesh_min_size * self.dist_max_ratio,  # self.dist_max_ratio larger than 1.0 make flux optimization more stable
            msh_file_path=self.msh_file_path,
            compress_mesh_extend_from_boundary=compress_mesh
        )
        assert os.path.exists(self.msh_file_path), "[Error]: gmsh failed"
        
        self.grid_mesher = grid_mesher
        self.field_value = init_field_value

    def prepare_mesh(self):
        if self.mesh is not None:
            return self.mesh
        domain, cell_tags, facet_tags = MeshUtils.read_msh(self.msh_file_path, self.space_dim)
        self.mesh = MeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        return self.mesh

    def prepare_topo_vtu(self):
        if self.topo_vtu is not None:
            return self.topo_vtu
        self.topo_vtu = pyvista.read(self.result_opt_vtu)  # type: ignore
        return self.topo_vtu
        
    def prepare_topo_function(self, diffusion_step=0):
        if self.topo_function is not None:
            return self.topo_function
        mesh = self.prepare_mesh()
        point_scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
        topo_function = dolfinx.fem.Function(point_scalar_func_space)
        topo_vtu = self.prepare_topo_vtu()
        
        topo_weight = topo_vtu.point_data['topo_weight']
        if diffusion_step > 0:
            weight_filter = DistanceBandFilter(topo_vtu, self.weight_filter_radius, False, True)
            for _ in range(diffusion_step):
                topo_weight = np.asarray(weight_filter.run(topo_weight.reshape((-1, 1)))).reshape(-1)
        
        topo_function.x.array[:] = topo_weight  # type: ignore
        self.topo_function = topo_function  # type: ignore
        return self.topo_function

    def optimize(self, residual_error_threshold=1e-4):
        if self.us_fipy:   
            optim = TopoFluidOptimizerFipy(
                name=self.name,
                msh_file=self.msh_file_path,
                mesh=self.prepare_mesh(),
                fluid_type=self.fluid_type,
                viscosity_init=self.viscosity_init,
                topo_weight_init=self.topo_weight_init,
                init_topo_weight_method=self.init_topo_weight_method,
                weight_filter_radius=self.weight_filter_radius,
                result_dir=self.opt_dir,
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
                obj_cost_args_list=self.obj_cost_args_list,
                reg_cost_args=self.reg_cost_args,
                obstacle_cost_args=self.obstacle_cost_args,
                simulation_iterations=self.fipy_simulator_dict['simulation_iterations'],
                convergence_threshold=self.fipy_simulator_dict['convergence_threshold'],
                velocityRelaxation=self.fipy_simulator_dict['velocityRelaxation'],
                pressureRelaxation=self.fipy_simulator_dict['pressureRelaxation'],
                hot_steps=self.fipy_simulator_dict.get('simulation_hot_steps', 10),
                log_step=self.fipy_simulator_dict['log_step'],
                parallel_num=self.fipy_simulator_dict['parallel_num'],
                use_tqdm=self.fipy_simulator_dict['use_tqdm'],
                use_log=self.fipy_simulator_dict['use_log'],
                adj_kwargs={
                    'critical_method': 'residual',
                    'mat_assemble_method': 'lift',
                    'with_residual': True,
                    'residual_error_threshold': residual_error_threshold  # 1e-3 NOTE: residual_error_threshold在momentum_weight作用下能允许较大的误差
                },
                alpha_filter_ratio=self.fipy_simulator_dict.get('alpha_filter_ratio', -1.0),
            )
            
            recorder = TensorBoardRecorder(os.path.join(self.opt_dir, 'log'))
            use_lr_scheduler = self.fipy_simulator_dict.get('use_lr_scheduler', False)     
            if not use_lr_scheduler:
                result = optim.run(
                    max_iteration=self.fipy_simulator_dict['max_iteration'],
                    line_search_steps=self.fipy_simulator_dict['line_search_steps'],
                    momentum_weight=self.fipy_simulator_dict['momentum_weight'],
                    recorder=recorder,
                    full_run=self.fipy_simulator_dict['full_run'],
                    result_vtu_file=self.result_opt_vtu,
                    hot_step=self.fipy_simulator_dict['hot_steps'],
                    loss_decay_weight=self.fipy_simulator_dict.get('loss_decay_weight', 0.0)
                )
            else:
                result = optim.run_with_lr_scheduler(
                    min_lr=self.fipy_simulator_dict['min_lr'],
                    max_lr=self.fipy_simulator_dict['max_lr'],
                    max_iteration=self.fipy_simulator_dict['max_iteration'],
                    hot_step=self.fipy_simulator_dict['hot_steps'],
                    momentum_weight=self.fipy_simulator_dict['momentum_weight'],
                    full_run=self.fipy_simulator_dict['full_run'],
                    recorder=recorder,
                    result_vtu_file=self.result_opt_vtu,
                    loss_decay_weight=self.fipy_simulator_dict.get('loss_decay_weight', 0.0)
                )
            
        else:
            optim = TopoFluidOptimizer(
                name=self.name,
                mesh=self.prepare_mesh(),
                fluid_type=self.fluid_type,
                viscosity_init=self.viscosity_init,
                topo_weight_init=self.topo_weight_init,
                init_topo_weight_method=self.init_topo_weight_method,
                weight_filter_radius=self.weight_filter_radius,
                result_dir=self.opt_dir,
                alpha_fun_max=self.alpha_fun_max,
                alpha_fun_min=self.alpha_fun_min,
                alpha_fun_p=self.alpha_fun_p,
            )
            optim.setup_tape(Tape())
            for boundary in self.boundary_list:
                optim.add_boundary(boundary)
        
            optim.define_problem(
                obj_cost_args_list=self.obj_cost_args_list,
                reg_cost_args=self.reg_cost_args,
                obstacle_cost_args=self.obstacle_cost_args,
            )
            recorder = TensorBoardRecorder(os.path.join(self.opt_dir, 'log'))
            use_lr_scheduler = self.dolfinx_simulator_dict.get('use_lr_scheduler', False)     
            if not use_lr_scheduler:
                result = optim.run(
                    max_iteration=self.dolfinx_simulator_dict['max_iteration'],
                    line_search_steps=self.dolfinx_simulator_dict['line_search_steps'],
                    momentum_weight=self.dolfinx_simulator_dict['momentum_weight'],
                    recorder=recorder,
                    full_run=self.dolfinx_simulator_dict['full_run'],
                    result_vtu_file=self.result_opt_vtu,
                    hot_step=self.dolfinx_simulator_dict['hot_steps'],
                    loss_decay_weight=self.dolfinx_simulator_dict.get('loss_decay_weight', 0.0)
                )
            else:
                result = optim.run_with_lr_scheduler(
                    min_lr=self.dolfinx_simulator_dict['min_lr'],
                    max_lr=self.dolfinx_simulator_dict['max_lr'],
                    max_iteration=self.dolfinx_simulator_dict['max_iteration'],
                    hot_step=self.dolfinx_simulator_dict['hot_steps'],
                    momentum_weight=self.dolfinx_simulator_dict['momentum_weight'],
                    full_run=self.dolfinx_simulator_dict['full_run'],
                    recorder=recorder,
                )
        
        return result

    @staticmethod
    def bry_to_dict(bry: AutoFluidBoundary):
        return {
            'name': bry.name,
            'boundary_type': bry.boundary_type,
            'marker': bry.marker,
            'value': list(bry.value) if isinstance(bry.value, np.ndarray) else bry.value,
            'face_center': list(bry.face_center) if bry.face_center is not None else None,
            'face_norm_vector': list(bry.face_norm_vector) if bry.face_norm_vector is not None else None,
            'face_radius': bry.face_radius if bry.face_radius is not None else None,
            'is_free': bry.is_free,
            'kwargs': bry.kwargs,
        }

    @staticmethod
    def dict_to_bry(bry_dict: dict):
        bry_value = bry_dict['value']
        if isinstance(bry_value, list):
            bry_value = np.array(bry_value)
        return AutoFluidBoundary(
            name=bry_dict['name'],
            boundary_type=bry_dict['boundary_type'],
            value=bry_value,
            marker=bry_dict['marker'],
            face_center=np.array(bry_dict['face_center']) if bry_dict['face_center'] is not None else None,
            face_norm_vector=np.array(bry_dict['face_norm_vector']) if bry_dict['face_norm_vector'] is not None else None,
            face_radius=bry_dict['face_radius'] if bry_dict['face_radius'] is not None else None,
            is_free=bry_dict['is_free'],
            **bry_dict['kwargs'],
        )

    def save_cfg(self):
        if isinstance(self.viscosity_init, np.ndarray):
            viscosity_init_file = os.path.join(self.run_dir, 'viscosity_init.npy')
            np.save(viscosity_init_file, self.viscosity_init)
            viscosity_init = viscosity_init_file
        else:
            viscosity_init = self.viscosity_init
        
        if isinstance(self.topo_weight_init, np.ndarray):
            topo_weight_init_file = os.path.join(self.run_dir, 'topo_weight_init.npy')
            np.save(topo_weight_init_file, self.topo_weight_init)
            topo_weight_init = topo_weight_init_file
        else:
            topo_weight_init = self.topo_weight_init
        
        cfg = {
            'name': self.name,
            'run_dir': self.run_dir,
            'space_dim': self.space_dim,
            'space_left_bottom_corner': self.space_left_bottom_corner.tolist(),
            'space_right_top_corner': self.space_right_top_corner.tolist(),
            'space_resolution': self.space_resolution,
            'mesh_min_size': self.mesh_min_size,
            'mesh_max_size': self.mesh_max_size,
            'boundary_list': [self.bry_to_dict(bry) for bry in self.boundary_list],
            'obj_cost_args_list': self.obj_cost_args_list,
            'reg_cost_args': self.reg_cost_args,
            'obstacle_cost_args': self.obstacle_cost_args,
            'fluid_type': self.fluid_type,
            'viscosity_init': viscosity_init,
            'topo_weight_init': topo_weight_init,
            'init_topo_weight_method': self.init_topo_weight_method,
            'alpha_fun_max': self.alpha_fun_max,
            'alpha_fun_min': self.alpha_fun_min,
            'alpha_fun_p': self.alpha_fun_p,
            'weight_filter_ratio': self.weight_filter_ratio,
            'dist_max_ratio': self.dist_max_ratio,
            'us_fipy': self.us_fipy,
            'fipy_simulator_dict': self.fipy_simulator_dict,
            'dolfinx_simulator_dict': self.dolfinx_simulator_dict,
            'inout_pipe_length': self.inout_pipe_length,
        }
        with open(os.path.join(self.run_dir, 'settings.json'), 'w') as f:
            json.dump(cfg, f, indent=4)
    
    @staticmethod
    def load_from_json(json_file: str):
        with open(json_file, 'r') as f:
            cfg = json.load(f)
        
        boundary_list = []
        for bry_dict in cfg['boundary_list']:
            boundary_list.append(TopoFluidGridOptimizer.dict_to_bry(bry_dict))
        
        if isinstance(cfg['viscosity_init'], str):
            viscosity_init = np.load(cfg['viscosity_init'], allow_pickle=True)
        else:
            viscosity_init = cfg['viscosity_init']
        if isinstance(cfg['topo_weight_init'], str):
            topo_weight_init = np.load(cfg['topo_weight_init'], allow_pickle=True)
        else:
            topo_weight_init = cfg['topo_weight_init']
        
        optim = TopoFluidGridOptimizer(
            name=cfg['name'],
            run_dir=cfg['run_dir'],
            space_dim=cfg['space_dim'],
            space_left_bottom_corner=cfg['space_left_bottom_corner'],
            space_right_top_corner=cfg['space_right_top_corner'],
            space_resolution=cfg['space_resolution'],
            mesh_min_size=cfg['mesh_min_size'],
            mesh_max_size=cfg['mesh_max_size'],
            boundary_list=boundary_list,
            obj_cost_args_list=cfg['obj_cost_args_list'],
            reg_cost_args=cfg['reg_cost_args'],
            obstacle_cost_args=cfg['obstacle_cost_args'],
            fluid_type=cfg['fluid_type'],
            viscosity_init=viscosity_init,
            topo_weight_init=topo_weight_init,
            init_topo_weight_method=cfg['init_topo_weight_method'],
            alpha_fun_max=cfg['alpha_fun_max'],
            alpha_fun_min=cfg['alpha_fun_min'],
            alpha_fun_p=cfg['alpha_fun_p'],
            weight_filter_ratio=cfg['weight_filter_ratio'],
            dist_max_ratio=cfg['dist_max_ratio'],
            us_fipy=cfg['us_fipy'],
            fipy_simulator_dict=cfg['fipy_simulator_dict'],
            dolfinx_simulator_dict=cfg['dolfinx_simulator_dict'],
            inout_pipe_length=cfg['inout_pipe_length'],
        )
        return optim


class TopoFluidHierarchyOptimizer(TopoFluidChainOptimizer):
    def __init__(
        self,
        name: str,
        run_dir: str,
        space_dim: int,
        space_left_bottom_corner: np.ndarray,
        space_right_top_corner: np.ndarray,
        space_resolutions: list[float],
        mesh_min_sizes: list[float],
        mesh_max_sizes: list[float],
        boundary_list: list[AutoFluidBoundary],        
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float|np.ndarray,
        topo_weight_init: float|np.ndarray,
        init_topo_weight_method: Literal['poisson', 'filter', 'none', 'raw'],
        alpha_fun_max: float = 100.0,
        alpha_fun_min: float = 1e-12,
        alpha_fun_p: float = 3.0,
        us_fipy=False,
        extract_threshold: float = 0.4,
        diffusion_step: int = 0,
    ):
        super().__init__(
            name=name,
            run_dir=run_dir,
            msh_file=None,  # type: ignore
            mesh=None,  # type: ignore
            boundary_list=boundary_list,
            fluid_type=fluid_type,
            viscosity_init=viscosity_init,
            topo_weight_init=topo_weight_init,
            init_topo_weight_method=init_topo_weight_method,
            weight_filter_radius=None,  # type: ignore
            alpha_fun_max=alpha_fun_max,
            alpha_fun_min=alpha_fun_min,
            alpha_fun_p=alpha_fun_p,
            us_fipy=us_fipy
        )
        self.space_dim = space_dim
        self.space_left_bottom_corner = space_left_bottom_corner
        self.space_right_top_corner = space_right_top_corner
        self.space_resolutions = space_resolutions
        self.mesh_min_sizes = mesh_min_sizes
        self.mesh_max_sizes = mesh_max_sizes
        
        self.extract_threshold = extract_threshold
        self.diffusion_step = diffusion_step
        self.grid_optimizers: list[TopoFluidGridOptimizer] = []
    
    def append_optimizer_from_json(self, json_file: str):
        optim = TopoFluidGridOptimizer.load_from_json(json_file)
        self.grid_optimizers.append(optim)
    
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
                    'target_volume': 0.5,
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
                    'parallel_num': 8,
                    'use_tqdm': True,
                    'use_log': True,
                    'max_iteration': 150,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                },
                'dolfinx_simulator_cfg': {
                    'max_iteration': 150,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                }
            }
        ],
        residual_error_threshold=1e-4,
        inout_pipe_length=0.05,
        weight_filter_ratio: float|list[float]=1.5,
        dist_max_ratio: float|list[float]=2.0,
        compress_mesh: bool = False,
        optimizer_jsons: list[str] = []
    ):
        self.grid_optimizers.clear()
                
        last_obj_cost_args_list: list[dict] = None  # type: ignore
        last_reg_cost_args: dict = None  # type: ignore
        last_obstacle_cost_args: dict = None  # type: ignore
        last_fipy_simulator_cfg: dict = None  # type: ignore
        last_dolfinx_simulator_cfg: dict = None  # type: ignore
        
        if isinstance(weight_filter_ratio, float):
            weight_filter_ratio_list = [weight_filter_ratio] * len(self.space_resolutions)
        else:
            weight_filter_ratio_list: list[float] = weight_filter_ratio  # type: ignore
        if isinstance(dist_max_ratio, float):
            dist_max_ratio_list = [dist_max_ratio] * len(self.space_resolutions)
        else:
            dist_max_ratio_list: list[float] = dist_max_ratio  # type: ignore
        
        # --------------------------------------
        for layer_id, (space_resolution, mesh_min_size, mesh_max_size, weight_filter_ratio, dist_max_ratio) in enumerate(
            zip(self.space_resolutions, self.mesh_min_sizes, self.mesh_max_sizes, 
                weight_filter_ratio_list, dist_max_ratio_list)
        ):
            if layer_id < len(optimizer_jsons):
                self.append_optimizer_from_json(optimizer_jsons[layer_id])
                continue
            
            if len(self.grid_optimizers) > 0:
                reference_optimizer = self.grid_optimizers[-1]
            else:
                reference_optimizer = None
            
            layer_dir = os.path.join(self.run_dir, f'layer_{layer_id}')
            
            # ------ define optimizer ------
            run_cfg = run_cfgs[min(layer_id, len(run_cfgs) - 1)]
            obj_cost_args_list = run_cfg.get('obj_cost_args_list', last_obj_cost_args_list)
            reg_cost_args = run_cfg.get('reg_cost_args', last_reg_cost_args)
            obstacle_cost_args = run_cfg.get('obstacle_cost_args', last_obstacle_cost_args)
            fipy_simulator_cfg = run_cfg.get('fipy_simulator_cfg', last_fipy_simulator_cfg)
            dolfinx_simulator_cfg = run_cfg.get('dolfinx_simulator_cfg', last_dolfinx_simulator_cfg)
            assert obj_cost_args_list is not None, 'obj_cost_args_list is required'
            assert reg_cost_args is not None, 'reg_cost_args is required'
            assert 'target_volume' in reg_cost_args.keys(), 'Target volume is required for Volume regularization'
            log_run_json = {
                'obj_cost_args_list': obj_cost_args_list,
                'reg_cost_args': reg_cost_args,
                'obstacle_cost_args': obstacle_cost_args,
                'fipy_simulator_cfg': fipy_simulator_cfg,
                'dolfinx_simulator_cfg': dolfinx_simulator_cfg,
            }
                        
            grid_optim = TopoFluidGridOptimizer(
                name=f'{self.name}_layer_{layer_id}',
                run_dir=layer_dir,
                space_dim=self.space_dim,
                space_left_bottom_corner=self.space_left_bottom_corner,
                space_right_top_corner=self.space_right_top_corner,
                space_resolution=space_resolution,
                mesh_min_size=mesh_min_size,
                mesh_max_size=mesh_max_size,
                boundary_list=self.boundary_list,
                obj_cost_args_list=copy.deepcopy(obj_cost_args_list),
                reg_cost_args=copy.deepcopy(reg_cost_args),
                obstacle_cost_args=copy.deepcopy(obstacle_cost_args) if obstacle_cost_args is not None else None,
                fluid_type=self.fluid_type,
                viscosity_init=self.viscosity_init,
                topo_weight_init=self.topo_weight_init if layer_id == 0 else 0.8,
                init_topo_weight_method=self.init_topo_weight_method,
                alpha_fun_max=self.alpha_fun_max,
                alpha_fun_min=self.alpha_fun_min,
                alpha_fun_p=self.alpha_fun_p,
                weight_filter_ratio=weight_filter_ratio,
                dist_max_ratio=dist_max_ratio,
                us_fipy=self.us_fipy,
                fipy_simulator_dict=fipy_simulator_cfg,
                dolfinx_simulator_dict=dolfinx_simulator_cfg,
                inout_pipe_length=inout_pipe_length,
            )           
            grid_optim.prepare_design_space(
                reference_optimizer=reference_optimizer,
                threshold=self.extract_threshold,
                diffusion_step=self.diffusion_step,
                compress_mesh=compress_mesh
            )
            grid_optim.optimize(residual_error_threshold=residual_error_threshold)
            self.grid_optimizers.append(grid_optim)
            
            with open(os.path.join(layer_dir, 'run_cfg.json'), 'w') as f:
                json.dump(log_run_json, f, indent=4)
            grid_optim.save_cfg()
            
            last_obj_cost_args_list = obj_cost_args_list
            last_reg_cost_args = reg_cost_args
            last_obstacle_cost_args = obstacle_cost_args
            last_fipy_simulator_cfg = fipy_simulator_cfg
            last_dolfinx_simulator_cfg = dolfinx_simulator_cfg
    

def debug_hierarchy_optimizer_2d():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t44'
    
    optim = TopoFluidHierarchyOptimizer(
        name='t44',
        run_dir=base_dir,
        space_dim=2,
        space_left_bottom_corner=np.array([0.0, 0.0]),
        space_right_top_corner=np.array([1.0, 1.0]),
        space_resolutions=[0.2, 0.1, 0.06],
        mesh_min_sizes=[0.03, 0.02, 0.01],
        mesh_max_sizes=[0.08, 0.05, 0.02],
        boundary_list=[
            AutoFluidBoundary(
                name='inlet',
                boundary_type='velocity',
                value=[1.0, 0.0],
                marker=1,
                face_center=np.array([0.0, 0.2]),
                face_norm_vector=np.array([1.0, 0.0]),
                face_radius=0.1
            ),
            AutoFluidBoundary(
                name='outlet',
                boundary_type='pressure',
                value=0.0,
                marker=2,
                face_center=np.array([1.0, 0.8]),
                face_norm_vector=np.array([1.0, 0.0]),
                face_radius=0.1
            ),
            AutoFluidBoundary(
                name='wall',
                boundary_type='wall',
                value=0.0,
                marker=3
            )
        ],
        fluid_type='stoke',
        viscosity_init=1.0 / 400.,
        topo_weight_init=0.1,
        init_topo_weight_method='filter',
        us_fipy=False,
    )
    optim.optimize(
        run_cfgs=[
            {
                'obj_cost_args_list':[{
                    'style': 'MinDissipation+BrinkmanPenalized',
                    'with_viscosity': True,
                    'dissipation_weight': 1.0,
                    'penalized_weight': 1.0,
                }],
                'reg_cost_args':{
                    'style': 'Volume',
                    'weight': 1.0,
                    'target_volume': 0.1,
                    'reg_rescale': True,
                },
                'obstacle_cost_args':None,
                'fipy_simulator_cfg': {
                    'simulation_iterations': 200,
                    'convergence_threshold': 1e-4,
                    'velocityRelaxation': 0.6,
                    'pressureRelaxation': 0.8,
                    'hot_steps': 10,
                    'log_step': 2,
                    'parallel_num': 8,
                    'use_tqdm': True,
                    'use_log': True,
                    'max_iteration': 150,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                },
                'dolfinx_simulator_cfg': {
                    'max_iteration': 100,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                }
            }
        ]
    )


def debug_hierarchy_optimizer_3d():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t44'
    
    optim = TopoFluidHierarchyOptimizer(
        name='t44_compare',
        run_dir=base_dir,
        space_dim=3,
        space_left_bottom_corner=np.array([0.0, 0.0, 0.0]),
        space_right_top_corner=np.array([1.0, 1.0, 1.0]),
        space_resolutions=[0.2, 0.1, 0.06],
        mesh_min_sizes=[0.02, 0.01, 0.01],
        mesh_max_sizes=[0.08, 0.03, 0.01],
        boundary_list=[
            AutoFluidBoundary(
                name='inlet',
                boundary_type='velocity',
                value=[1.0, 0.0, 0.0],
                marker=1,
                face_center=np.array([0.0, 0.2, 0.2]),
                face_norm_vector=np.array([1.0, 0.0, 0.0]),
                face_radius=0.1
            ),
            AutoFluidBoundary(
                name='outlet',
                boundary_type='pressure',
                value=0.0,
                marker=2,
                face_center=np.array([0.8, 1.0, 0.8]),
                face_norm_vector=np.array([0.0, 1.0, 0.0]),
                face_radius=0.1
            ),
            AutoFluidBoundary(
                name='wall',
                boundary_type='wall',
                value=0.0,
                marker=3
            )
        ],
        fluid_type='stoke',
        viscosity_init=1.0 / 400.,
        topo_weight_init=0.1,
        init_topo_weight_method='poisson',
        us_fipy=True,
        extract_threshold=0.4,
        diffusion_step=0,
    )
    optim.optimize(
        run_cfgs=[
            {
                'obj_cost_args_list':[{
                    'style': 'MinDissipation+BrinkmanPenalized',
                    'with_viscosity': True,
                    'dissipation_weight': 1.0,
                    'penalized_weight': 1.0,
                }],
                'reg_cost_args':{
                    'style': 'Volume',
                    'weight': 0.1,
                    'target_volume': 0.06,
                    'reg_rescale': True,
                },
                'obstacle_cost_args':None,
                'fipy_simulator_cfg': {
                    'simulation_iterations': 200,
                    'convergence_threshold': 1e-4,
                    'velocityRelaxation': 0.6,
                    'pressureRelaxation': 0.8,
                    'hot_steps': 10,
                    'log_step': 2,
                    'parallel_num': 4,
                    'use_tqdm': True,
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
                }
            },
            {
                'fipy_simulator_cfg': {
                    'simulation_iterations': 200,
                    'convergence_threshold': 1e-4,
                    'velocityRelaxation': 0.6,
                    'pressureRelaxation': 0.8,
                    'hot_steps': 10,
                    'log_step': 2,
                    'parallel_num': 8,
                    'use_tqdm': True,
                    'use_log': True,
                    'max_iteration': 300,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                },
            },
            {
                'fipy_simulator_cfg': {
                    'simulation_iterations': 200,
                    'convergence_threshold': 1e-4,
                    'velocityRelaxation': 0.6,
                    'pressureRelaxation': 0.8,
                    'hot_steps': 10,
                    'log_step': 2,
                    'parallel_num': 8,
                    'use_tqdm': True,
                    'use_log': True,
                    'max_iteration': 300,
                    'line_search_steps': [0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                },
            }
        ],
        residual_error_threshold=1e-3,
    )


def debug_multi_2d_pipes():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t49'

    boundaries = [
        AutoFluidBoundary(
            name='inlet',
            boundary_type='velocity',
            value=[0.0, 1.0],
            marker=1,
            face_center=np.array([1.0, 0.0]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet1',
            boundary_type='pressure',
            value=0.0,
            marker=2,
            face_center=np.array([0.2, 1.0]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet2',
            boundary_type='pressure',
            value=0.0,
            marker=3,
            face_center=np.array([0.6, 1.0]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet3',
            boundary_type='pressure',
            value=0.0,
            marker=4,
            face_center=np.array([1.0, 1.0]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet4',
            boundary_type='pressure',
            value=0.0,
            marker=5,
            face_center=np.array([1.4, 1.0]),
            face_norm_vector=np.array([0.0, 1.0]),
            face_radius=0.1
        ),
        AutoFluidBoundary(
            name='outlet5',
            boundary_type='pressure',
            value=0.0,
            marker=6,
            face_center=np.array([1.8, 1.0]),
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

    optim = TopoFluidHierarchyOptimizer(
        name='t48',
        run_dir=base_dir,
        space_dim=2,
        space_left_bottom_corner=np.array([0.0, 0.0]),
        space_right_top_corner=np.array([2.0, 1.0]),
        space_resolutions=[0.1, 0.06, 0.03, 0.03],
        mesh_min_sizes=[0.03, 0.02, 0.01, 0.005],
        mesh_max_sizes=[0.06, 0.04, 0.02, 0.01],
        boundary_list=boundaries,
        fluid_type='stoke',
        viscosity_init=1.0 / 400.,
        topo_weight_init=0.1,
        init_topo_weight_method='poisson',
        us_fipy=False,
        extract_threshold=0.4,
        diffusion_step=0,
    )
    optim.optimize(
        run_cfgs=[
            {
                'obj_cost_args_list':[
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
                        'outlet_marker_weights': [0.1, 0.1, 0.1, 0.1, 0.1],  # TODO: shouldn't be main cost
                        'with_rescale': True
                    }
                ],
                'reg_cost_args':{
                    'style': 'Volume',
                    'weight': 1.0,
                    'target_volume': 0.6,
                    'reg_rescale': True,
                },
                'obstacle_cost_args':None,
                'dolfinx_simulator_cfg': {
                    'max_iteration': 200,
                    'line_search_steps': [0.1, 0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                }
            },
            {
                'reg_cost_args': {
                    'style': 'Volume',
                    'weight': 1.0,
                    'target_volume': 0.4,
                    'reg_rescale': True,
                },
            },
            {
                'reg_cost_args': {
                    'style': 'Volume',
                    'weight': 1.0,
                    'target_volume': 0.2,
                    'reg_rescale': True,
                },
                'dolfinx_simulator_cfg': {
                    'max_iteration': 200,
                    'line_search_steps': [0.05],
                    'momentum_weight': 0.8,
                    'full_run': False,
                }
            },
            {
                'reg_cost_args': {
                    'style': 'Volume',
                    'weight': 1.0,
                    'target_volume': 0.1,
                    'reg_rescale': True,
                },
                'dolfinx_simulator_cfg': {
                    'max_iteration': 200,
                    'line_search_steps': [0.05],
                    'momentum_weight': 0.8,
                    'full_run': True,
                }
            }
        ]
    )


if __name__ == '__main__':
    # debug_hierarchy_optimizer_2d()
    # debug_hierarchy_optimizer_3d()

    debug_multi_2d_pipes()