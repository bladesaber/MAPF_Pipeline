import os
from typing import Literal, Union, Callable

import dolfinx
import pyvista
import ufl
from ufl import inner, grad, div

import numpy as np

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
1. 拓扑优化精度不够, 收敛缓慢, 接近头部很大, 原因不明, 需要更好的参考与方法(openPisco)
2. obstacle过于零碎, 碎片化的obstacle导致优化不好
3. 拓扑优化结果需要一个更好的映射方法,转换为形状
4. 这里的失败表明我的优化路径设计并不合理, reg_cost与obj_cost的变化速度不一样, 需要调整优化路径. !!!!
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
        topo_weight_init: float,
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
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
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
            self.topo_weight.x.array[:] = self.topo_weight_init
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
                with_cache=True,  #  FIXME
                
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
            
            # volume_dif = target_volume - topo_volume
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
    ) -> dict:
        is_solved, step = False, 0
        loss_current, loss_last = 0.0, 0.0
        line_search_steps_np: np.ndarray = np.array(line_search_steps)
        step_length_last = line_search_steps_np[0]
        
        topo_phi = DolfinxFunction(self.topo_weight.function_space, name=f'topo_phi')  # type: ignore
        topo_phi.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        
        loss_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(step)
        
        # last_direction_np: np.ndarray
        
        while True:
            step += 1
                    
            if step > 1:
                loss_dict = self.eval()
                log = f"[step/{step} loss:{loss_current:.5f} last_step_length:{step_length_last:.5f}]"
                for name, cost_value in loss_dict.items():
                    log += f" {name}:{cost_value:.5f}"
                print(log)
                
                if tensorboard_recorder is not None:
                    tensorboard_recorder.write_scalars(
                        tag=f'{self.name}/loss',
                        scalar_dict=loss_dict,
                        step=step
                    )
                
                loss_residual = loss_current - loss_last
                if np.abs(loss_residual) < 1e-6:
                    is_solved = True
                    break
                
            adj_input = np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2)
            direction_np = self.compute_gradient(adj_input=adj_input, with_filter=True, with_norm=True)
            
            # TODO: please check later,it seem it will cause bias
            # if step == 1:
            #     last_direction_np = direction_np
            # else:
            #     direction_np = direction_np * 0.2 + last_direction_np * 0.8
            #     last_direction_np = direction_np
            
            find_best_step = False
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
                    print(f'[Log: step/{step} loss_new:{loss_new:.5f}]')
                
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


"""
def debug_2D_with_obstacle():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t30/'
    msh_file = os.path.join(base_dir, 't30_model.msh')
    xdmf_file = os.path.join(base_dir, 't30_model.xdmf')
    MeshUtils.msh_to_XDMF(msh_file, xdmf_file, dim=2)
    
    # --- 1. define mesh ---
    mesh = SeqMeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_cost',
        style='Volume',
        weight=2.0,
        target_ratio=0.2
    )
    
    def define_obstacle_func(cell_centers: np.ndarray):
        obstacle_center = np.array([0.5, 0.5])
        cell2obs_dist = np.linalg.norm(cell_centers[:, :2] - obstacle_center[:2], axis=1)
        obs_dofs = np.where(cell2obs_dist < 0.15)[0]
        return obs_dofs
        
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_30',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        # topo_weight_init=regularization_item.kwargs['target_ratio'] * 1.05,
        topo_weight_init=0.5,  # TODO: it seems source term cause result wrong
        obstacle_func=define_obstacle_func,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.06,
        result_dir=os.path.join(base_dir, 'topo_opt_withobs_cache'),
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
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=23
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=24
    )

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    optim.define_problem(
        simulation_iterations=66,
        adj_ksp_option={
            'ksp_type': 'richardson',
            'ksp_max_it': 1,                  # 只迭代一次 = 相当于 preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            # 可选：提高 MUMPS 容错性
            'mat_mumps_icntl_24': 1,          # 检测奇异矩阵
            'mat_mumps_icntl_25': 0,          # 可选：缺陷矩阵处理
        }
    )
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(base_dir, 'topo_opt_withobs_cache', 'tensorboardx_record'))
    optim.run(
        line_search_steps=[0.2, 0.1, 0.05, 0.01, 0.005],
        tensorboard_recorder=recorder
    )


def debug_3D_without_obstacle():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t29/'
    xdmf_file = os.path.join(base_dir, 't29_model.xdmf')
    msh_file = os.path.join(base_dir, 't29_model.msh')
    MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=3)

    # --- 1. define mesh ---
    mesh = SeqMeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_cost',
        style='Volume',
        weight=10.0,
        target_ratio=0.2
    )
    
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_29',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        # topo_weight_init=regularization_item.kwargs['target_ratio'] * 1.05,
        topo_weight_init=0.5,  # TODO: it seems source term cause result wrong
        obstacle_func=None,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.06,
        result_dir=os.path.join(base_dir, 'topo_opt_cache'),
        velocity_degree=1,
        pressure_degree=1,
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=20,
        face_center=[-0.05, 0.15, 0.15],
        face_norm_vector=[1.0, 0.0, 0.0],
        face_radius=0.1
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=21
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=22
    )

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    optim.define_problem(
        # TODO: i don't know why, but if i use my ksp in the code, i need to change ksp option in richardson
        adj_ksp_option={
            'ksp_type': 'richardson',
            'ksp_max_it': 1,                  # 只迭代一次 = 相当于 preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,          # 检测奇异矩阵
            'mat_mumps_icntl_25': 0,          # 可选：缺陷矩阵处理
        }
    )
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(base_dir, 'topo_opt_cache', 'tensorboardx_record'))
    optim.run(
        line_search_steps=[0.2, 0.1, 0.05, 0.01, 0.005],
        tensorboard_recorder=recorder
    )


def debug_3D_with_obstacle():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t33/'
    xdmf_file = os.path.join(base_dir, 't33_model.xdmf')
    msh_file = os.path.join(base_dir, 't33_model.msh')
    MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=3)

    # --- 1. define mesh ---
    mesh = SeqMeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_cost',
        style='Volume',
        weight=10.0,
        target_ratio=0.2
    )
    
    def define_obstacle_func(cell_centers: np.ndarray):
        obstacle_center = np.array([0.5, 0.5, 0.15])
        cell2obs_dist = np.linalg.norm(cell_centers[:, :2] - obstacle_center[:2], axis=1)
        obs_dofs = np.where(cell2obs_dist < 0.05)[0]
        return obs_dofs
    
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_33',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        # topo_weight_init=regularization_item.kwargs['target_ratio'] * 1.05,
        topo_weight_init=0.5,  # TODO: it seems source term cause result wrong
        obstacle_func=define_obstacle_func,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.06,
        result_dir=os.path.join(base_dir, 'topo_opt_withobs_cache'),
        velocity_degree=1,  # TODO: can it be 1?
        pressure_degree=1,
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=20,
        face_center=[-0.05, 0.15, 0.15],
        face_norm_vector=[1.0, 0.0, 0.0],
        face_radius=0.1
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=21
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=22
    )

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    optim.define_problem(
        simulation_iterations=70,
        adj_ksp_option={
            'ksp_type': 'richardson',
            'ksp_max_it': 1,                  # 只迭代一次 = 相当于 preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,          # 检测奇异矩阵
            'mat_mumps_icntl_25': 0,          # 可选：缺陷矩阵处理
        }
    )
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(base_dir, 'topo_opt_withobs_cache', 'tensorboardx_record'))
    optim.run(
        line_search_steps=[0.2, 0.1, 0.05, 0.01, 0.005],
        tensorboard_recorder=recorder
    )


def debug_3D_2_without_obstacle():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t31'
    xdmf_file = os.path.join(base_dir, 'model.xdmf')
    msh_file = os.path.join(base_dir, 'model.msh')
    MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=3)

    # --- 1. define mesh ---
    mesh = SeqMeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_cost',
        style='Volume',
        weight=10.0,
        target_ratio=0.2
    )
    
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_31',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        # topo_weight_init=regularization_item.kwargs['target_ratio'] * 1.05,
        topo_weight_init=0.5,  # TODO: it seems source term cause result wrong
        obstacle_func=None,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.06,
        result_dir=os.path.join(base_dir, 'topo_opt_cache'),
        velocity_degree=1,
        pressure_degree=1,
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=20,
        face_center=[-0.05, 0.15, 0.15],
        face_norm_vector=[1.0, 0.0, 0.0],
        face_radius=0.1
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=21
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=22
    )

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    optim.define_problem(
        # TODO: i don't know why, but if i use my ksp in the code, i need to change ksp option in richardson
        adj_ksp_option={
            'ksp_type': 'richardson',
            'ksp_max_it': 1,                  # 只迭代一次 = 相当于 preonly
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            # 可选：提高 MUMPS 容错性
            'mat_mumps_icntl_24': 1,          # 检测奇异矩阵
            'mat_mumps_icntl_25': 0,          # 可选：缺陷矩阵处理
        }
    )
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(base_dir, 'topo_opt_cache', 'tensorboardx_record'))
    optim.run(
        line_search_steps=[0.2, 0.1, 0.05, 0.01, 0.005],
        tensorboard_recorder=recorder
    )


def debug_3D_2_with_obstacle():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t34'
    xdmf_file = os.path.join(base_dir, 'model.xdmf')
    msh_file = os.path.join(base_dir, 'model.msh')
    MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=3)

    # --- 1. define mesh ---
    mesh = SeqMeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_cost',
        style='Volume',
        weight=10.0,
        target_ratio=0.2
    )
    
    def define_obstacle_func(cell_centers: np.ndarray):
        obstacle_center = np.array([0.5, 0.5, 0.5])
        cell2obs_dist = np.linalg.norm(cell_centers - obstacle_center, axis=1)
        obs_dofs = np.where(cell2obs_dist < 0.1)[0]
        return obs_dofs
    
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_31',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        # topo_weight_init=regularization_item.kwargs['target_ratio'] * 1.05,
        topo_weight_init=0.5,  # TODO: it seems source term cause result wrong
        obstacle_func=define_obstacle_func,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.06,
        result_dir=os.path.join(base_dir, 'topo_opt_cache'),
        velocity_degree=1,
        pressure_degree=1,
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=20,
        face_center=[-0.05, 0.15, 0.15],
        face_norm_vector=[1.0, 0.0, 0.0],
        face_radius=0.1
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=21
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=22
    )

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    optim.define_problem(
        simulation_iterations=70,
        adj_ksp_option={
            'ksp_type': 'richardson',
            'ksp_max_it': 1,
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_mumps_icntl_24': 1,
            'mat_mumps_icntl_25': 0,
        }
    )
    
    recorder = TensorBoardRecorder(log_dir=os.path.join(base_dir, 'topo_opt_cache', 'tensorboardx_record'))
    optim.run(
        line_search_steps=[0.2, 0.1, 0.05, 0.01, 0.005],
        tensorboard_recorder=recorder
    )
"""

if __name__ == '__main__':
    # debug_example1()
    debug_example2()
    
    # debug_2D_with_obstacle()
    # debug_3D_without_obstacle()
    # debug_3D_with_obstacle()
    # debug_3D_2_without_obstacle()
    # debug_3D_2_with_obstacle()
    
    pass
    
  