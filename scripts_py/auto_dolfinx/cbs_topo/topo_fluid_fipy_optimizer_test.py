import os
from typing import Literal, Union, Callable

import dolfinx
import ufl
from ufl import inner, grad, div

import numpy as np
import pyvista
from sklearn.neighbors import KDTree

from scripts_py.mesh_utils import MeshUtils
from scripts_py.auto_dolfinx.cbs_topo.seq_mesh_utils import SeqMeshObj
from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import TopoFluidOptimizer
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils
from scripts_py.vtu_recorder import TensorBoardRecorder

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control, stop_annotating

from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import (
    AutoFluidBoundary,
    AutoFluidCost,
    AutoFluidRegularization,
)
from scripts_py.auto_dolfinx.cbs_topo.fipy_eqn_block import equation_stokeEqn_solve


"""
TODO: Critical problems:
2. obstacle过于零碎, 碎片化的obstacle导致优化不好
3. 拓扑优化结果需要一个更好的映射方法,转换为形状
"""


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
        result_dir: str,
        alpha_fun_max: float = 1000.0,
        alpha_fun_min: float = 1e-12,
        alpha_fun_p: float = 3.0,
        velocity_degree = 1,
        pressure_degree = 1,
        velocity_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        pressure_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        obstacle_cost_weight: float|None = None,
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
            weight_filter_is_cell_field=True,
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
        
        # self.obstacle_cost_weight = obstacle_cost_weight
        obstacle_mask = np.zeros(self.mesh.num_cells)
        if obstacle_func is not None:
            obstacle_dofs = obstacle_func(DolfinUtils.tabulate_dof_coordinates(self.cell_scalar_func_space))
            if obstacle_dofs.shape[0] > 0:
                obstacle_mask[obstacle_dofs] = 1.0
        self.obs_cell_mask = obstacle_mask
        self.obs_source: DolfinxFunction = None  # type: ignore
        
        self.topo_phi_input = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_phi_input')

    @staticmethod
    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_min=1e-12, alpha_max=1000.0, p=3.0, pow_operator=ufl.elem_pow):
        return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)  # type: ignore

    def init_topo_weight(
        self, 
        init_value: float|np.ndarray, 
        interpolate_shape: Literal['raw', 'circle', 'all'] = 'circle',
    ) -> dolfinx.fem.Function:
        """当整个场的topo_weight都很少时,速度与压力会汇聚在出口的边角区域,造成崩角的效果.
        崩角效果会造成优化畸形(因为拓扑优化易于陷入局部最优点),因此构造一个球形区域,使出口区域比边缘更流畅,减缓崩角的影响.
        当然也可以通过减少reg_cost的权重来减缓崩角的影响.
        """
        topo_weight = dolfinx.fem.Function(self.point_scalar_func_space)
        if isinstance(init_value, np.ndarray):
            topo_weight.x.array[:] = init_value  # type: ignore
            
        elif isinstance(init_value, float):
            topo_weight_value = init_value * np.ones(shape=topo_weight.x.array.shape)  # type: ignore
            for bry_info in self.boundaries.values():
                if bry_info.boundary_type in ['velocity', 'pressure']:
                    if interpolate_shape == 'raw':
                        marker_indices = self.mesh.extract_surface_indices(bry_info.marker)
                        topo_weight_value[marker_indices] = 1.0
                    elif interpolate_shape == 'circle':
                        dists = np.linalg.norm(
                            self.mesh.grid.points[:, :self.mesh.tdim] - bry_info.face_center.reshape((1, -1)), axis=1  # type: ignore
                        )
                        topo_weight_value[dists < bry_info.face_radius] = 1.0
                    elif interpolate_shape == 'all':
                        pass
                    else:
                        raise ValueError(f"Invalid interpolate shape: {interpolate_shape}")
                                    
                elif bry_info.boundary_type == 'wall':
                    marker_indices = self.mesh.extract_surface_indices(bry_info.marker)
                    topo_weight_value[marker_indices] = 0.0
            topo_weight.x.array[:] = topo_weight_value  # type: ignore
            
        else:
            raise ValueError(f"Invalid init value type: {type(init_value)}")
        return topo_weight # type: ignore

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
    ):
        with start_annotation():
            uh = DolfinxFunction(self.W, name=f"{self.name}_state")
            u, p = ufl.split(uh)  # type: ignore # please don't use up.split()
            vq = DolfinxFunction(self.W, name=f"{self.name}_adjoint")
            v, q = ufl.split(vq)  # type: ignore # please don't use up.split()
            
            f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
            nut = DolfinxFunction(self.point_scalar_func_space, name=f"{self.name}_nut")
            nut.x.array[:] = self.nut_init  # type: ignore
                    
            govern_form = self.define_govern_equation(u, p, v, q, f, nut)  # type: ignore
            
            self.topo_weight = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_weight')  # type: ignore
            self.topo_weight.interpolate(
                self.init_topo_weight(self.topo_weight_init, interpolate_shape='circle')
            )
            
            if self.obs_cell_mask.sum() > 0:
                self.topo_weight.x.array[self.obs_cell_mask > 0.0] = 0.0
            self.topo_phi_input.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
            
            alpha = self.alpha_fun(
                rho=self.topo_weight,  # type: ignore
                alpha_min=self.alpha_fun_min, 
                alpha_max=self.alpha_fun_max, 
                p=self.alpha_fun_p, 
                pow_operator=np.power
            )
            govern_form += alpha * inner(u, v) * ufl.dx
            
            self.define_boundary()
            
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
            if with_momentum:
                current_direction_np = self.compute_gradient(adj_input=adj_input, with_filter=True, with_norm=True)
                cum_direction_np = cum_direction_np * momentum_weight + current_direction_np * (1.0 - momentum_weight)
                direction_np = cum_direction_np / np.max(np.abs(cum_direction_np))
            else:
                direction_np = self.compute_gradient(adj_input=adj_input, with_filter=True, with_norm=True)
            
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
                if loss_new < loss_current or step < hot_step:
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
        
        self.recorder.update_cell_data('topo_weight', self.topo_weight.x.array.copy())
        cell_scalar_fun = dolfinx.fem.Function(self.cell_scalar_func_space)
        cell_scalar_fun.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        point_scalar_fun = dolfinx.fem.Function(self.point_scalar_func_space)
        point_scalar_fun.interpolate(cell_scalar_fun)  # type: ignore
        self.recorder.update_point_data('topo_weight', point_scalar_fun.x.array.copy())  # type: ignore
        
        if self.obs_cell_mask is not None:
            self.recorder.update_cell_data('obs', self.obs_cell_mask)  # type: ignore
        if kwargs.get('gradient', None) is not None:
            self.recorder.update_cell_data('gradient', kwargs['gradient'])
        
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
    debug_example2()
    
    # debug_2D_with_obstacle()
    # debug_3D_without_obstacle()
    # debug_3D_with_obstacle()
    # debug_3D_2_without_obstacle()
    # debug_3D_2_with_obstacle()
    
    pass
    
  