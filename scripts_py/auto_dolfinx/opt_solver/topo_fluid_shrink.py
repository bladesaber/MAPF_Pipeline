from typing import Literal
import os
import shutil

import dolfinx
import ufl

import math
import numpy as np
import pandas as pd
from copy import deepcopy
import pyvista

from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils
from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.auto_dolfinx.opt_solver.seq_mesh import SequenceMeshObj
from scripts_py.auto_dolfinx.opt_solver.topo_fluid import TopoFluidOptimizer, AutoFluidBoundary
from scripts_py.vtu_recorder import VtuRecorder, TensorBoardRecorder
from scripts_py.dolfinx_adjoint.topology_utils.topology_filter import DistanceBandFilter

from thirdparty.pyadjoint.pyadjoint import Tape


class TopoFluidShrinkOptimizer(TopoFluidOptimizer):
    def __init__(
        self, 
        name: str,
        run_dir: str,
        msh_file_name: str,
        mesh_dimension: int,
        boundary_dict: dict[str, AutoFluidBoundary],
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float|np.ndarray,
        weight_filter_radius: float,
        
        obj_cost_args_list: list[dict],
        reg_cost_args: dict,
        
        alpha_fun: dict = {
            'name': 'exponential',
            'args': {
                'alpha_min': 1e-12,
                'alpha_max': 100.0,
                'p': 3.0,
            }
        },
        velocity_degree = 2,
        pressure_degree = 1,
        velocity_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        pressure_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        
        tlm_ksp_option: dict = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        adj_ksp_option: dict = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        forward_ksp_option: dict = {
            'ksp_setting': {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1
            },
            'snes_setting': {}
        },
        tlm_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        adj_kwargs: dict = {
            'critical_method': 'residual',
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        forward_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        
        min_lr: float = 1e-2,
        max_lr: float = 1e-1,
    ):
        self.name = name
        self.run_dir = run_dir
        if not os.path.exists(self.run_dir):
            raise FileNotFoundError(f'Run directory {self.run_dir} not found')
        
        self.mesh_dimension = mesh_dimension
        self.msh_file_name = msh_file_name
        self.msh_file = os.path.join(self.run_dir, self.msh_file_name)
        if not os.path.exists(self.msh_file):
            raise FileNotFoundError(f'Mesh file {self.msh_file} not found')
        
        self.boundary_dict = boundary_dict
        self.fluid_type = fluid_type
        self.viscosity_init = viscosity_init
        self.weight_filter_radius = weight_filter_radius
        self.alpha_fun = alpha_fun
        self.velocity_degree = velocity_degree
        self.pressure_degree = pressure_degree
        self.velocity_space: Literal['CG', 'DG', 'Lagrange'] = velocity_space
        self.pressure_space: Literal['CG', 'DG', 'Lagrange'] = pressure_space
        
        self.obj_cost_args_list = obj_cost_args_list
        self.reg_cost_args = reg_cost_args
        self.tlm_ksp_option = tlm_ksp_option
        self.adj_ksp_option = adj_ksp_option
        self.forward_ksp_option = forward_ksp_option
        self.tlm_kwargs = tlm_kwargs
        self.adj_kwargs = adj_kwargs
        self.forward_kwargs = forward_kwargs
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.initial_volume = self.estimate_volume()
        print(f'Initial volume: {self.initial_volume}')
    
    def estimate_volume(self):
        domain, cell_tags, facet_tags = MeshUtils.read_msh(self.msh_file, self.mesh_dimension)
        mesh = MeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        
        vol_weight = np.ones(shape=(mesh.num_points,))
        for boundary in self.boundary_dict.values():
            if boundary.boundary_type == 'wall':
                if boundary.boundary_dim == 'facet':
                    vol_weight[mesh.extract_surface_indices(boundary.marker)] = 0.0
                elif boundary.boundary_dim == 'cell':
                    vol_weight[mesh.extract_cell_indices(boundary.marker, point_space=True)] = 0.0
        
        vol_weight_func = dolfinx.fem.Function(DolfinUtils.point_scalar_function_space(domain))
        vol_weight_func.x.array[:] = vol_weight  # type: ignore
        volume = AssembleUtils.assemble_scalar(vol_weight_func * ufl.dx)
        return volume
    
    def run_step(
        self, 
        step: int, 
        run_iteration: int, 
        target_volume: float, 
        obs_cell_indices: np.ndarray|None = None,
        positive_point_indices: np.ndarray|None = None,
        fuzzy_alpha_boundary: float = -1.0,
    ):
        run_name = f'{self.name}_run_{step}'
        run_dir = os.path.join(self.run_dir, run_name)
        os.makedirs(run_dir)
        
        run_msh_file = os.path.join(run_dir, self.msh_file_name)
        shutil.copy(self.msh_file, run_msh_file)
        
        domain, cell_tags, facet_tags = MeshUtils.read_msh(run_msh_file, self.mesh_dimension)
        mesh = SequenceMeshObj(domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
        
        has_obstacle = False
        boundaries = []
        for _, bry_info in self.boundary_dict.items():
            if bry_info.boundary_type == 'wall' and bry_info.boundary_dim == 'cell':
                has_obstacle = True
                if obs_cell_indices is not None:
                    cell_tag_values = mesh.cell_tags.values.copy()
                    cell_tag_values[obs_cell_indices] = bry_info.marker
                    cell_tags = MeshUtils.define_meshTag_from_numpy(
                        domain, self.mesh.tdim, 
                        mesh.cell_tags.indices.copy().astype(np.int32), 
                        cell_tag_values.astype(np.int32)
                    )
                    mesh.cell_tags = cell_tags
                boundaries.append(deepcopy(bry_info))
            
            else:
                boundaries.append(deepcopy(bry_info))
        
        if (not has_obstacle) and (obs_cell_indices is not None):
            cell_tag_values = mesh.cell_tags.values.copy()
            new_marker = np.max(np.unique(cell_tag_values)) + 1
            cell_tag_values[obs_cell_indices] = new_marker
            cell_tags = MeshUtils.define_meshTag_from_numpy(
                domain, mesh.tdim, 
                mesh.cell_tags.indices.copy().astype(np.int32), 
                cell_tag_values.astype(np.int32)
            )
            mesh.cell_tags = cell_tags

            boundaries.append(AutoFluidBoundary(
                name='obs',
                boundary_type='wall',
                boundary_dim='cell',
                value=0.0,
                marker=new_marker,
            ))
        
        optim = TopoFluidOptimizer(
            name=run_name,
            mesh=mesh,
            fluid_type=self.fluid_type,
            viscosity_init=self.viscosity_init,
            weight_filter_radius=self.weight_filter_radius,
            result_dir=os.path.join(run_dir, 'process_record'),
            alpha_fun=self.alpha_fun,
            velocity_degree=self.velocity_degree,
            pressure_degree=self.pressure_degree,
            velocity_space=self.velocity_space,
            pressure_space=self.pressure_space,
        )
        
        optim.setup_tape(Tape())
        for boundary in boundaries:
            optim.add_boundary(boundary)
        
        if positive_point_indices is not None:
            optim.add_topo_weight_constraint_term(
                name='positive_weight',
                init_priority=2,
                set_idxs=positive_point_indices,
                set_value=np.ones(shape=(positive_point_indices.shape[0],)),
                init_method='fluid_area',
                constraint_method='fix',
            )
            fixed_pos = np.arange(mesh.num_points)
            fixed_pos[positive_point_indices] = 1.0
            mesh.grid.point_data['fixed_pos'] = fixed_pos  # type: ignore
        
        reg_cost_args = self.reg_cost_args.copy()
        reg_cost_args.update({'target_volume': target_volume})
        
        optim.define_fluid_problem(
            obj_cost_args_list=self.obj_cost_args_list,
            reg_cost_args=reg_cost_args,
            tlm_ksp_option=self.tlm_ksp_option,
            adj_ksp_option=self.adj_ksp_option,
            forward_ksp_option=self.forward_ksp_option,
            tlm_kwargs=self.tlm_kwargs,
            adj_kwargs=self.adj_kwargs,
            forward_kwargs=self.forward_kwargs,
        )
        
        recorder = TensorBoardRecorder(os.path.join(run_dir, 'log'))
        result_vtu_file = os.path.join(run_dir, 'result.vtu')
        optim.run_along_volume(
            min_lr=self.min_lr, max_lr=self.max_lr,
            max_iteration=run_iteration, recorder=recorder,
            result_vtu_file=result_vtu_file,
        )
        
        res_vtu = pyvista.read(result_vtu_file)
        
        # TODO: still need to fix this
        # if fuzzy_alpha_boundary > 0.0:
        #     filter = DistanceBandFilter(mesh.grid, self.weight_filter_radius, True, True)
        #     alpha_np = res_vtu.cell_data['alpha']
        #     fixed_indices = alpha_np >= fuzzy_alpha_boundary
        #     fixed_alpha = alpha_np[fixed_indices]
            
        #     for _ in range(1):
        #         alpha_np = alpha_np.reshape((-1, 1))
        #         alpha_np = filter.run(alpha_np)
        #         alpha_np = np.asarray(alpha_np).reshape((-1))
        #         alpha_np[fixed_indices] = fixed_alpha
        #     res_vtu.cell_data['alpha'] = alpha_np
        
        return {
            'res_vtu': res_vtu,
            'mesh': mesh,
            'optim': optim,
        }
    
    def run_optimize_alone_volume(
        self, 
        volume_stages: list[float],
        step_iterations: list[int],
        filter_threshold: float = 0.4, fuzzy_alpha_boundary: float = -1.0
    ):        
        obs_cell_indices = None
        last_conn_map: np.ndarray = None  # type: ignore
        positive_pcd_indices: np.ndarray = None  # type: ignore
        
        for step, volume_stage, each_step_iteration in zip(range(len(volume_stages)), volume_stages, step_iterations):
            print(f'Running step {step} with target volume {volume_stage}')
            result_dict = self.run_step(
                step, each_step_iteration, volume_stage, 
                obs_cell_indices=obs_cell_indices, 
                positive_point_indices=positive_pcd_indices,
                fuzzy_alpha_boundary=fuzzy_alpha_boundary,
            )  # type: ignore
            
            res_vtu: pyvista.UnstructuredGrid = result_dict['res_vtu']
            mesh: SequenceMeshObj = result_dict['mesh']
            # optim: TopoFluidOptimizer = result_dict['optim']
            
            cell_to_points = mesh.get_cell_to_points()
            point_alpha = res_vtu.point_data['alpha']
            cell_to_points = mesh.get_cell_to_points()
            cell_alphas = point_alpha[cell_to_points]
            remove_bool = np.sum(cell_alphas <= filter_threshold, axis=1) == cell_alphas.shape[1]
            
            current_conn_map = np.ones(shape=(mesh.num_cells,))
            current_conn_map[remove_bool] = 0.0
            
            clusters = mesh.extract_cell_clusters(current_conn_map)
            if len(clusters) > 1:
                for cluster in clusters:
                    mesh.extract_cluster_feature(cluster)
                conn_idxs_dict = mesh.compute_connectivity_map(clusters, last_conn_map, relax_factor=2)
                
                conn_idxs_list = []
                for _, conn_idxs in conn_idxs_dict.items():
                    current_conn_map[conn_idxs] = 1.0
                    conn_idxs_list.append(conn_idxs)
                positive_cell_indices = np.concatenate(conn_idxs_list, axis=-1)
                
                cell_to_points = mesh.get_cell_to_points()
                point_count = np.zeros(shape=(mesh.num_points,))
                for cell_idx in positive_cell_indices:
                    point_count[cell_to_points[cell_idx]] += 1
                current_pos_point_indices = np.arange(mesh.num_points)[point_count > 1]
                if current_pos_point_indices.shape[0] == 0:
                    current_pos_point_indices = np.arange(mesh.num_points)[point_count > 0]
            else:
                current_pos_point_indices = None  # type: ignore
            
            if current_pos_point_indices is not None:
                if positive_pcd_indices is None:
                    positive_pcd_indices = current_pos_point_indices
                else:
                    positive_pcd_indices = np.concatenate([positive_pcd_indices, current_pos_point_indices])
                    positive_pcd_indices = np.unique(positive_pcd_indices)
            
            if (current_conn_map == 0.0).sum() > 0:
                obs_cell_indices = np.arange(res_vtu.n_cells)[current_conn_map == 0.0]
            else:
                obs_cell_indices = None
            last_conn_map = current_conn_map
            
            print('\n')
            