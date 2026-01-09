import os
from typing import Literal, Union, Callable

import dolfinx
import ufl
from ufl import inner, grad, div

from functools import partial
import numpy as np
import pyvista
from sklearn.neighbors import KDTree

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import TopoFluidOptimizer
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils
from scripts_py.vtu_recorder import TensorBoardRecorder

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control, stop_annotating

from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from scripts_py.auto_dolfinx.cbs_topo.fipy_eqn_block import equation_stokeEqn_solve
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve


class TopoWeightTerm(object):
    def __init__(
        self,
        name: str,
        mesh: MeshObj,
        is_cell_field: bool,
        constraint_method: Literal['fix', 'none'],
        set_idxs: np.ndarray|None = None,
        set_value: np.ndarray|None = None,
        **kwargs
    ):
        self.name = name
        self.mesh = mesh
        self.is_cell_field = is_cell_field
        
        self.init_priority: int
        self.init_method: str
        
        self.constraint_priority: int
        self.constraint_method: str = constraint_method
        
        self.set_idxs: np.ndarray|None = set_idxs
        self.set_value: np.ndarray|None = set_value
        self.define_init_weight(**kwargs)
    
    def define_init_weight(
        self, 
        init_method: Literal['boundary_raw', 'boundary_radius', 'flatten', 'obstacle', 'init'], 
        **kwargs
    ):
        self.init_method = init_method
        if init_method == 'boundary_raw':
            assert not self.is_cell_field, "boundary_raw is not supported for cell field"
            bry_info = kwargs['bry_info']
            assert bry_info.boundary_type in ['velocity', 'pressure', 'wall']
            
            self.set_idxs = self.mesh.extract_surface_indices(bry_info.marker)
            if bry_info.boundary_type in ['velocity', 'pressure']:
                self.set_value = np.ones(shape=self.set_idxs.shape[0])
            elif bry_info.boundary_type == 'wall':
                self.set_value = np.zeros(shape=self.set_idxs.shape[0])
            
            if bry_info.boundary_type in ['velocity', 'pressure']:
                self.init_priority = 1
            elif bry_info.boundary_type == 'wall':
                self.init_priority = 2
        
        elif init_method == 'boundary_radius':
            assert not self.is_cell_field, "boundary_radius is not supported for cell field"
            bry_info = kwargs['bry_info']
            assert bry_info.boundary_type in ['velocity', 'pressure']
            assert bry_info.face_center is not None
            assert bry_info.face_radius is not None
            
            face_center = bry_info.face_center
            points = self.mesh.points[:, :self.mesh.tdim]
            face_radius = bry_info.face_radius
            
            dists = np.linalg.norm(points - face_center.reshape((1, -1)), axis=1)
            
            set_idxs = np.arange(0, dists.shape[0], 1)
            # adding 1e-4 to guarantee the boundary is set to 1.0 and symmetry
            self.set_idxs = set_idxs[dists < (face_radius + 1e-4)]
            self.set_value = np.ones(shape=self.set_idxs.shape[0])  # type: ignore
            self.init_priority = 1
        
        elif init_method == 'flatten':
            if self.is_cell_field:
                self.set_idxs = np.arange(0, self.mesh.num_cells, 1)
            else:
                self.set_idxs = np.arange(0, self.mesh.num_points, 1)
            self.set_value = kwargs['init_value'] * np.ones(shape=self.set_idxs.shape[0])  # type: ignore
            self.init_priority = 0
        
        elif init_method == 'obstacle':
            obs_func: Callable = kwargs['obstacle_func']
            if obs_func is not None:
                if self.is_cell_field:
                    cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(self.mesh.domain)
                    self.set_idxs = obs_func(DolfinUtils.tabulate_dof_coordinates(cell_scalar_func_space))
                else:
                    point_scalar_func_space = DolfinUtils.point_scalar_function_space(self.mesh.domain)
                    self.set_idxs = obs_func(DolfinUtils.tabulate_dof_coordinates(point_scalar_func_space))
                if self.set_idxs.shape[0] > 0:  # type: ignore
                    self.set_value = np.zeros(shape=self.set_idxs.shape[0])  # type: ignore
            self.init_priority = 2
        
        elif init_method == 'init':
            if self.is_cell_field:
                self.set_idxs = np.arange(0, self.mesh.num_cells, 1)
            else:
                self.set_idxs = np.arange(0, self.mesh.num_points, 1)
            self.set_value = kwargs['topo_weight_init']
            self.init_priority = 0
        
        else:
            raise ValueError(f"Invalid init method: {init_method}")
    
    def apply_constraint_gradient(self, gradient_np: np.ndarray):
        if self.init_method in ['boundary_raw', 'boundary_radius']:
            if self.constraint_method == 'fix':
                gradient_np[self.set_idxs] = 0.0
                
        elif self.init_method == 'obstacle':
            gradient_np[self.set_idxs] = 0.0
            
        return gradient_np

    def num_idxs(self):
        return self.set_idxs.shape[0] if self.set_idxs is not None else 0


class TopoFluidOptimizerFipy(TopoFluidOptimizer):
    def __init__(
        self, 
        name: str,
        msh_file: str,
        mesh: SeqMeshObj,
        nut_init: float,
        cost_item: AutoFluidCost,
        regularization_item: AutoFluidRegularization,
        topo_weight_init: float|np.ndarray,
        obstacle_func: Callable|None,
        weight_filter_radius: float,
        weight_filter_is_cell_field: bool,
        result_dir: str,
        alpha_fun_max: float = 1000.0,
        alpha_fun_min: float = 1e-12,
        alpha_fun_p: float = 3.0,
        velocity_degree = 1,
        pressure_degree = 1,
        velocity_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        pressure_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
    ):
        """1.topo_weight_init预设在相对小的数值通常优化路径更短
        2.Topology优化的问题在于一旦形成初始结果就会陷入局部最优点,
            因此分布优化效果更好,先使用小权重reg_cost热身优化,再使用大权重reg_cost进行收缩优化.
        """
        super().__init__(
            name,
            mesh,
            fluid_type='stoke',
            nut_init=nut_init,
            cost_item=cost_item,
            regularization_item=regularization_item,
            topo_weight_init=topo_weight_init,
            weight_filter_radius=weight_filter_radius,
            weight_filter_is_cell_field=weight_filter_is_cell_field,
            result_dir=result_dir,
            alpha_fun_max=alpha_fun_max,
            alpha_fun_min=alpha_fun_min,
            alpha_fun_p=alpha_fun_p,
            velocity_degree=velocity_degree,
            pressure_degree=pressure_degree,
            velocity_space=velocity_space,
            pressure_space=pressure_space
        )
        self.msh_file = msh_file
        
        if self.weight_filter_is_cell_field:
            self.topo_phi_input = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_phi_input')
        else:
            self.topo_phi_input = DolfinxFunction(self.point_scalar_func_space, name=f'topo_phi_input')
        
        # ------ define topo weight terms ------
        self.topo_weight_terms: dict[str, TopoWeightTerm] = {}
        
        obs_term = TopoWeightTerm(
            name='obstacle', mesh=self.mesh, 
            is_cell_field=self.weight_filter_is_cell_field, 
            constraint_method='none',
            init_method='obstacle', obstacle_func=obstacle_func
        )
        if obs_term.num_idxs() > 0:
            self.topo_weight_terms[obs_term.name] = obs_term
        
        if isinstance(self.topo_weight_init, (float, int)):
            init_term = TopoWeightTerm(
                name='init_flatten', mesh=self.mesh, 
                is_cell_field=self.weight_filter_is_cell_field, 
                constraint_method='none',
                init_method='flatten', init_value=self.topo_weight_init
            )
        elif isinstance(self.topo_weight_init, np.ndarray):
            init_term = TopoWeightTerm(
                name='init_term', mesh=self.mesh, 
                is_cell_field=self.weight_filter_is_cell_field, 
                constraint_method='none',
                init_method='init', topo_weight_init=self.topo_weight_init
            )
        else:
            raise ValueError(f"Invalid topo weight init type: {type(self.topo_weight_init)}")
        self.topo_weight_terms[init_term.name] = init_term
        # ------ define topo weight terms ------
        
        self.obs_cell_mask = np.zeros(self.mesh.num_cells)
        if obstacle_func is not None:
            obstacle_dofs = obstacle_func(DolfinUtils.tabulate_dof_coordinates(self.cell_scalar_func_space))
            if obstacle_dofs.shape[0] > 0:
                self.obs_cell_mask[obstacle_dofs] = 1.0
    
    @staticmethod
    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_min=1e-12, alpha_max=1000.0, p=3.0, pow_operator=ufl.elem_pow):
        return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)  # type: ignore

    def init_topo_weight(self, topo_weight_fun: DolfinxFunction):
        """当整个场的topo_weight都很少时,速度与压力会汇聚在出口的边角区域,造成崩角的效果.
        崩角效果会造成优化畸形(因为拓扑优化易于陷入局部最优点),因此构造一个球形区域,使出口区域比边缘更流畅,减缓崩角的影响.
        当然也可以通过减少reg_cost的权重来减缓崩角的影响.
        """
        init_weight = dolfinx.fem.Function(self.point_scalar_func_space)
        init_weight_value = np.zeros(shape=self.mesh.num_points)
        
        priority_list = []
        for name, term in self.topo_weight_terms.items():
            priority_list.append(term.init_priority)
        priority_list = np.unique(np.array(priority_list))
        priority_list = np.sort(priority_list, axis=0)
                
        for priority in priority_list:
            for name, term in self.topo_weight_terms.items():
                if term.init_priority == priority:
                    init_weight_value[term.set_idxs] = term.set_value
        init_weight.x.array[:] = init_weight_value  # type: ignore
        topo_weight_fun.interpolate(init_weight)

    def apply_constraint_to_gradient(self, direction_np: np.ndarray):
        for name, term in self.topo_weight_terms.items():
            direction_np = term.apply_constraint_gradient(direction_np)
        self.weight_filter.run(direction_np.reshape((-1, 1)))
        direction_np = np.asarray(direction_np).reshape((-1))
        for name, term in self.topo_weight_terms.items():
            direction_np = term.apply_constraint_gradient(direction_np)

    def define_problem(
        self,
        simulation_iterations: int = 100,
        convergence_threshold: float = 1e-5,
        velocityRelaxation: float = 0.6,
        pressureRelaxation: float = 0.8,
        tlm_ksp_option: dict = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
        adj_ksp_option: dict = {
            'ksp_type': 'richardson',
            'ksp_max_it': 1, # only one iteration = equivalent to preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,
            'mat_mumps_icntl_25': 0,
        },
        tlm_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        adj_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        log_step: int = 2,
        hot_steps: int = 10,
        parallel_num: int = 1,
        use_tqdm: bool = True,
        use_log: bool = True,
        activate_init_method: str = 'boundary_radius',
        wall_init_method: str = 'boundary_raw',
        constraint_method: Literal['fix', 'none'] = 'fix',
    ):
        self.define_boundary()
        if isinstance(self.topo_weight_init, (float, int)):
            for name, bry_info in self.boundaries.items():
                if bry_info.boundary_type == 'velocity':
                    term = TopoWeightTerm(
                        name=name, mesh=self.mesh, 
                        is_cell_field=self.weight_filter_is_cell_field, 
                        init_method=activate_init_method, bry_info=bry_info,
                        constraint_method=constraint_method,
                    )
                    self.topo_weight_terms[name] = term
                elif bry_info.boundary_type == 'pressure':
                    term = TopoWeightTerm(
                        name=name, mesh=self.mesh, 
                        is_cell_field=self.weight_filter_is_cell_field, 
                        init_method=activate_init_method, bry_info=bry_info,
                        constraint_method='none',
                    )
                    self.topo_weight_terms[name] = term
                elif bry_info.boundary_type == 'wall':
                    term = TopoWeightTerm(
                        name=name, mesh=self.mesh, 
                        is_cell_field=self.weight_filter_is_cell_field, 
                        init_method=wall_init_method, bry_info=bry_info,
                        constraint_method=constraint_method,
                    )
                    self.topo_weight_terms[name] = term
                else:
                    raise ValueError(f"[Error]: Invalid boundary type: {bry_info.boundary_type}")
        
        with start_annotation():
            uh = DolfinxFunction(self.W, name=f"{self.name}_state")
            u, p = ufl.split(uh)  # type: ignore # please don't use up.split()
            vq = DolfinxFunction(self.W, name=f"{self.name}_adjoint")
            v, q = ufl.split(vq)  # type: ignore # please don't use up.split()
            
            f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
            nut = DolfinxFunction(self.point_scalar_func_space, name=f"{self.name}_nut")
            nut.x.array[:] = self.nut_init  # type: ignore
                    
            govern_form = self.define_govern_equation(u, p, v, q, f, nut)  # type: ignore
            
            if self.weight_filter_is_cell_field:
                self.topo_weight = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_weight')  # type: ignore
            else:
                self.topo_weight = DolfinxFunction(self.point_scalar_func_space, name=f'topo_weight')  # type: ignore
            
            topo_weight = self.topo_weight
            self.init_topo_weight(topo_weight)
            
            self.topo_phi_input.x.array[:] = topo_weight.x.array.copy()  # type: ignore
            alpha = self.alpha_fun(
                rho=topo_weight,  # type: ignore
                alpha_min=self.alpha_fun_min, 
                alpha_max=self.alpha_fun_max, 
                p=self.alpha_fun_p, 
                pow_operator=np.power
            )
            govern_form += alpha * inner(u, v) * ufl.dx
            
            uh = equation_stokeEqn_solve(
                # ------ simulator related ------
                boundaries=[bry_info for bry_info in self.boundaries.values()],
                msh_file=self.msh_file,
                viscosity=self.nut_init,
                velocityRelaxation=velocityRelaxation,
                pressureRelaxation=pressureRelaxation,
                structure_mesh=self.mesh,  # type: ignore
                structure_obs_weight=None if self.obs_cell_mask.sum() == 0 else self.obs_cell_mask,
                log_step=log_step,
                hot_steps=hot_steps,
                simulation_iterations=simulation_iterations,
                convergence_threshold=convergence_threshold,
                save_dir=os.path.join(self.result_dir, 'simulation'),
                parallel_num=parallel_num,
                use_tqdm=use_tqdm,
                use_log=use_log,
                with_cache=True,
                is_cell_field=self.weight_filter_is_cell_field,
                is_vector_field=False,
                
                # ------ result convert related ------
                V_to_W_dofs=self.V_to_W_dofs,
                Q_to_W_dofs=self.Q_to_W_dofs,
                V_func_space = self.V,
                Q_func_space = self.Q,
                
                # ------ alpha related ------
                topo_weight_name=self.topo_weight.name,
                alpha_fun=self.alpha_fun,
                alpha_min=self.alpha_fun_min,
                alpha_max=self.alpha_fun_max,
                alpha_p=self.alpha_fun_p,
                
                # ------ autdolfinx related ------
                F_form=govern_form,
                state=uh,  # type: ignore
                adjoint=vq,  # type: ignore
                tlm_ksp_option=tlm_ksp_option,
                adj_ksp_option=adj_ksp_option,
                tlm_kwargs=tlm_kwargs,
                adj_kwargs=adj_kwargs,
                tape=self.tape,
            )
            
            self.uh = uh  # type: ignore
            self.f = f  # type: ignore
            self.nut = nut  # type: ignore
            self.record_Upw(0)
            
            # ------ fluid object related ------
            u, p = ufl.split(self.uh)  # type: ignore
            self.define_cost(self.cost_item, u, p, alpha, nut=nut)  # type: ignore
            J_form = self.cost_item.cost_form
            
            """
            # ------ obstacle related ------
            J_form = self.cost_item.cost_form
            if self.obs_cell_mask.sum() > 0:
                self.obs_source = DolfinxFunction(self.cell_scalar_func_space, name="obs")  # type: ignore
                self.obs_source.x.array[:] = self.obs_cell_mask  # type: ignore
                self.obstacle_cost_form = self.obstacle_cost_weight * inner(self.obs_source, self.topo_weight) * ufl.dx  # type: ignore                
                J_form += self.obstacle_cost_form
            """
            
            fluid_obj_cost = assemble_scalar(J_form, domain=self.domain, tape=self.tape)  # type: ignore
            
            # ------ regularization related ------
            target_volume = self.regularization_item.kwargs['target_ratio'] * self.volume
            
            self.vol_ratio_form = self.topo_weight * ufl.dx            
            topo_volume = assemble_scalar(self.vol_ratio_form, domain=self.domain, tape=self.tape)            
            volume_dif = (target_volume - topo_volume) / target_volume
            reg_cost = self.regularization_item.weight * (volume_dif ** 2)
            
            # ------ final cost related ------
            fin_cost = fluid_obj_cost + reg_cost
            self.control_phi = Control(self.topo_weight)
            self.opt_problem = ReducedFunctional(fin_cost, [self.control_phi])
        
        print('[Log]: Initial loss:', self.eval())
    
    def run(
        self,
        line_search_steps: list[float] = [0.1, 0.05, 0.01, 0.005, 0.001, 0.00025],
        max_iteration=200,
        tensorboard_recorder: TensorBoardRecorder|None = None,
        hot_step: int = 20,
        momentum_weight: float = 0.8,
        full_run: bool = False,
    ) -> dict:
        is_solved, step = False, 0
        loss_current, loss_last = 0.0, 0.0
        
        with_line_search = len(line_search_steps) > 1
        line_search_steps_np: np.ndarray = np.array(line_search_steps)
        step_length_last = line_search_steps_np[0]
        
        topo_phi = DolfinxFunction(self.topo_weight.function_space, name=f'topo_phi')  # type: ignore
        topo_phi.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        
        loss_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(step)
        
        with_momentum = momentum_weight > 0.0
        if with_momentum:
            cum_direction_np = np.zeros_like(self.topo_weight.x.array)
        
        while True:
            step += 1
                    
            if step > 1:
                loss_dict = self.eval()
                log = f"[step/{step} loss:{loss_current:.5f} last_step_length:{step_length_last:.5f}]"
                for name, cost_value in loss_dict.items():
                    log += f" {name}:{cost_value:.5f}"
                print(log)
                
                if tensorboard_recorder is not None:
                    tensorboard_recorder.write_scalars(tag=f'{self.name}/loss', scalar_dict=loss_dict, step=step)
                
                loss_residual = loss_current - loss_last
                if np.abs(loss_residual) < 1e-6:
                    is_solved = True
                    break
                
            adj_input = np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2)
            cur_direction_np = self.compute_gradient(adj_input=adj_input, with_filter=True, with_norm=True)
            self.apply_constraint_to_gradient(cur_direction_np)
            if with_momentum:
                cum_direction_np = cum_direction_np * momentum_weight + cur_direction_np * (1.0 - momentum_weight)
                direction_np = cum_direction_np / np.max(np.abs(cum_direction_np))
            else:
                direction_np = cur_direction_np
            
            find_best_step = False
            if with_line_search:
                topo_phi_origin: np.array = topo_phi.x.array.copy()  # type: ignore
            
                last_step_index = np.where(line_search_steps_np == step_length_last)[0][0]
                line_search_start_index = np.maximum(last_step_index - 1, 0)
                for i in range(line_search_start_index, line_search_steps_np.shape[0]):
                    step_length = line_search_steps_np[i]
                    
                    new_topo_phi = topo_phi_origin - direction_np * step_length
                    new_topo_phi = np.maximum(np.minimum(new_topo_phi, 1.0 - 1e-16), 1e-16)
                    topo_phi.x.array[:] = new_topo_phi  # type: ignore
                    
                    loss_new = self.opt_problem(values=[topo_phi])  # type: ignore
                    if loss_new < loss_current or step < hot_step:
                        find_best_step = True
                        loss_last = loss_current
                        loss_current = loss_new                    
                        step_length_last = step_length
                        # self.topo_weight.x.array[:] = topo_phi.x.array  # type: ignore          
                        self.record_Upw(step, gradient=direction_np)
                        break

            else:
                new_topo_phi = topo_phi.x.array - direction_np * line_search_steps[0]  # type: ignore
                new_topo_phi = np.maximum(np.minimum(new_topo_phi, 1.0 - 1e-16), 1e-16)
                topo_phi.x.array[:] = new_topo_phi  # type: ignore
                loss_new = self.opt_problem(values=[topo_phi])  # type: ignore
                if loss_new < loss_current or step < hot_step or full_run:
                    find_best_step = True
                    loss_last = loss_current
                    loss_current = loss_new
                    self.record_Upw(step, gradient=direction_np)
                    
            if not find_best_step:
                print(f"[Warning]: line search fail.")
                is_solved = False
                break
                
            if step > max_iteration:
                print(f"[Warning]: max iteration reached.")
                is_solved = False
                break
        
        return {'is_success': is_solved, 'topo_weight': self.topo_weight.x.array.copy()} 
    
    def eval(self):
        fluid_obj_cost = AssembleUtils.assemble_scalar(self.cost_item.cost_form)
        
        target_volume = self.regularization_item.kwargs['target_ratio'] * self.volume
        topo_volume = AssembleUtils.assemble_scalar(self.vol_ratio_form)
        volume_dif = (target_volume - topo_volume) / target_volume
        reg_cost = self.regularization_item.weight * (volume_dif ** 2)
        
        loss_dict = {
            self.cost_item.name: fluid_obj_cost,
            self.regularization_item.name: reg_cost,
        }
        return loss_dict
    
    def record_Upw(self, step: int, **kwargs):
        self.recorder.clear(clear_point_data=True, clear_cell_data=True)
        
        vel_ref = dolfinx.fem.Function(self.point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(self.uh.sub(0).collapse())  # type: ignore
        p_ref = self.uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

        self.recorder.update_point_data('vel', vel_ref_np)
        self.recorder.update_point_data('vel_mag', np.linalg.norm(vel_ref_np, axis=1))
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        
        if self.weight_filter_is_cell_field:
            self.recorder.update_cell_data('topo_weight', self.topo_weight.x.array.copy())
            point_scalar_fun = dolfinx.fem.Function(self.point_scalar_func_space)
            point_scalar_fun.interpolate(self.topo_weight)  # type: ignore
            self.recorder.update_point_data('topo_weight', point_scalar_fun.x.array.copy())  # type: ignore
        else:
            self.recorder.update_point_data('topo_weight', self.topo_weight.x.array.copy())  # type: ignore
            cell_scalar_fun = dolfinx.fem.Function(self.cell_scalar_func_space)
            cell_scalar_fun.interpolate(self.topo_weight)  # type: ignore
            self.recorder.update_cell_data('topo_weight', cell_scalar_fun.x.array.copy())  # type: ignore
        
        if 'obstacle' in self.topo_weight_terms.keys():
            obs_term = self.topo_weight_terms['obstacle']
            if obs_term.is_cell_field:
                obs_cell_mask = np.zeros(self.mesh.num_cells)
                obs_cell_mask[obs_term.set_idxs] = obs_term.set_value
                self.recorder.update_cell_data('obs', obs_cell_mask)  # type: ignore
            else:
                obs_point_mask = np.zeros(self.mesh.num_points)
                obs_point_mask[obs_term.set_idxs] = obs_term.set_value
                self.recorder.update_point_data('obs', obs_point_mask)  # type: ignore
            
        if kwargs.get('gradient', None) is not None:
            if self.weight_filter_is_cell_field:
                self.recorder.update_cell_data('gradient', kwargs['gradient'])
            else:
                self.recorder.update_point_data('gradient', kwargs['gradient'])
        
        self.recorder.save(step)


def debug_example1():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t30/'
    msh_file = os.path.join(base_dir, 't30_model.msh')
    
    # --- 1. define mesh ---
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    cost_item = AutoFluidCost(name='obj_cost', style='MinDissipation', weight=1.0)
    regularization_item = AutoFluidRegularization(name='reg_cost', style='Volume', weight=1.0, target_ratio=0.1)
    
    result_dir = os.path.join(base_dir, 'topo_opt')
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_30',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        topo_weight_init=1.0,
        obstacle_func=None,
        weight_filter_radius=0.06,
        weight_filter_is_cell_field=True,
        result_dir=result_dir,
        velocity_degree=1,
        pressure_degree=1,
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0],
        marker=22,
        face_center=[-0.05, 0.2],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.1
    )
    outlet_bry = AutoFluidBoundary(name='outlet', boundary_type='pressure', value=0.0, marker=23)
    wall_bry = AutoFluidBoundary(name='wall', boundary_type='wall', value=0.0, marker=24)

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    optim.define_problem(
        # TODO: i don't know why, but need to change ksp option in richardson
        adj_ksp_option={
            'ksp_type': 'richardson',
            'ksp_max_it': 1, # only one iteration = equivalent to preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,
            'mat_mumps_icntl_25': 0,
        },
        simulation_iterations=60,
        convergence_threshold=1e-3,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        hot_steps=10,
        log_step=2,
        parallel_num=1,
        use_tqdm=False,
        use_log=True,
    )
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(result_dir, 'log'))
    optim.run(
        line_search_steps=[0.1, 0.05, 0.01],
        tensorboard_recorder=recorder
    )


def debug_example2():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t41/'
    msh_file = os.path.join(base_dir, 'model.msh')
    
    # --- 1. define mesh ---
    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = SeqMeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    cost_item = AutoFluidCost(name='obj_cost', style='MinDissipation', weight=1.0,)
    regularization_item = AutoFluidRegularization(name='reg_cost', style='Volume', weight=1.0, target_ratio=0.1)
    
    result_dir = os.path.join(base_dir, 'topo_opt')
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_30',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        topo_weight_init=1.0,
        obstacle_func=None,
        weight_filter_radius=0.06,
        weight_filter_is_cell_field=True,
        result_dir=result_dir,
        velocity_degree=1,
        pressure_degree=1,
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0],
        marker=22,
        face_center=[-0.05, 0.5],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.05
    )
    outlet_bry = AutoFluidBoundary(name='outlet', boundary_type='pressure', value=0.0, marker=23)
    wall_bry = AutoFluidBoundary(name='wall', boundary_type='wall', value=0.0,marker=24)

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    optim.define_problem(
        adj_ksp_option={
            'ksp_type': 'richardson',
            'ksp_max_it': 1, # only one iteration = equivalent to preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,
            'mat_mumps_icntl_25': 0,
        },
        simulation_iterations=100,
        convergence_threshold=1e-3,
        velocityRelaxation=0.6,
        pressureRelaxation=0.8,
        hot_steps=10,
        log_step=2,
        parallel_num=4,
        use_tqdm=False,
        use_log=True,
    )
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(result_dir, 'log'))
    optim.run(
        line_search_steps=[0.1, 0.05, 0.01],
        tensorboard_recorder=recorder
    )


if __name__ == '__main__':
    # debug_example1()
    # debug_example2()
    
    pass
    
  