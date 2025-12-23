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
from thirdparty.pyadjoint.pyadjoint.block import Block
from thirdparty.pyadjoint.pyadjoint.block_variable import BlockVariable

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
        obstacle_cost_weight: float,
        weight_filter_radius: float,
        result_dir: str,
        alpha_fun_max: float = 200.0,
        alpha_fun_min: float = 0.01,
        alpha_fun_p: float = 3.0,
        velocity_degree = 1,
        pressure_degree = 1,
        velocity_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        pressure_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
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
        
        self.obstacle_cost_weight = obstacle_cost_weight
        obstacle_mask = np.zeros(self.mesh.num_cells)
        if obstacle_func is not None:
            obstacle_dofs = obstacle_func(DolfinUtils.tabulate_dof_coordinates(self.cell_scalar_func_space))
            if obstacle_dofs.shape[0] > 0:
                obstacle_mask[obstacle_dofs] = 1.0
        self.obs_cell_mask = obstacle_mask
        self.obs_source: DolfinxFunction = None  # type: ignore
        
        self.topo_phi_input = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_phi_input')
        
        """
        self.reg_mul_block: Block = None # type: ignore
        self.reg_mul_block_dep: BlockVariable = None # type: ignore
        self.reg_ratio = 5.0
        """

    def define_problem(
        self,
        simulation_iterations: int = 100,
        velocityRelaxation: float = 0.5,
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
        }
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
                log_step=2,
                simulation_iterations=simulation_iterations,
                convergence_threshold=1e-5,
                save_dir=os.path.join(self.result_dir, 'progress'),
                
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
            
            u, p = ufl.split(self.uh)  # type: ignore
            self.define_cost(self.cost_item, u, p, alpha, nut=nut)  # type: ignore
            # self.define_regularization(self.regularization_item)
            
            J_form = self.cost_item.cost_form
            
            if self.obs_cell_mask.sum() > 0:
                self.obs_source = DolfinxFunction(self.cell_scalar_func_space, name="obstacle_source")  # type: ignore
                self.obs_source.x.array[:] = self.obs_cell_mask  # type: ignore
                # we don't need these
                # self.obstacle_cost_form = self.obstacle_cost_weight * inner(self.obs_source, self.topo_weight) * ufl.dx  # type: ignore                
                # J_form += self.obstacle_cost_form
            
            form_cost = assemble_scalar(J_form, domain=self.domain, tape=self.tape)  # type: ignore
            
            self.vol_ratio_form = self.topo_weight * ufl.dx            
            topo_volume = assemble_scalar(self.vol_ratio_form, domain=self.domain, tape=self.tape)
            volume_dif = self.regularization_item.kwargs['target_ratio'] * self.volume - topo_volume
            
            """
            # TODO: not good, please improve it
            # ------ adjust regularization weight ------
            with stop_annotating():
                if np.isclose(volume_dif, 0.0, atol=1e-8):
                    raise ValueError(f'[Error] Volume difference is too small: {volume_dif}')
                
                volume_cost_tmp = self.regularization_item.weight * volume_dif ** 2
                limit_form_cost_tmp = form_cost * self.reg_ratio   # type: ignore
                if volume_cost_tmp < limit_form_cost_tmp:
                    self.regularization_item.weight = limit_form_cost_tmp / (volume_dif ** 2)
            # ------ adjust regularization weight ------
            """
             
            reg_cost = self.regularization_item.weight * volume_dif ** 2
            
            """
            # ------ extract reg_mul_block and reg_mul_block_dep ------
            self.reg_mul_block = self.tape.get_blocks()[-1]
            for dep in self.reg_mul_block.get_dependencies():
                if dep.saved_output == self.regularization_item.weight:
                    self.reg_mul_block_dep = dep
                    self.reg_mul_block_dep.save_output()
            # ------ extract reg_mul_block and reg_mul_block_dep ------
            """
            
            fin_cost = form_cost + reg_cost
            self.control_phi = Control(self.topo_weight)
            self.opt_problem = ReducedFunctional(fin_cost, [self.control_phi])
        
        print('[Log]: Initial loss:', self.eval())

    """
    # TODO: please improve this
    def update_reg_mul_weight(self, loss_dict: dict):
        form_cost = loss_dict[self.cost_item.name]
        volume_cost = loss_dict[self.regularization_item.name]
        if volume_cost < form_cost * self.reg_ratio:
            weight = form_cost * self.reg_ratio / (volume_cost / self.reg_mul_block_dep.saved_output)
            self.reg_mul_block_dep.checkpoint = weight
            self.reg_mul_block_dep.saved_output
    """

    """
    def eval(self):
        loss_dict = {
            self.cost_item.name: self.cost_item.eval(),
            self.regularization_item.name: self.reg_mul_block_dep.saved_output * (
                self.regularization_item.kwargs['target_ratio'] * self.volume - AssembleUtils.assemble_scalar(self.vol_ratio_form)
            ) ** 2,
            'abs_volume': float(np.abs(
                AssembleUtils.assemble_scalar(self.vol_ratio_form) - 
                self.regularization_item.kwargs['target_ratio'] * self.volume
            )) / self.volume,
            'reg_mul_weight': self.reg_mul_block_dep.saved_output,
        }
        return loss_dict
    """
    
    def eval(self):
        loss_dict = {
            self.cost_item.name: self.cost_item.eval(),
            self.regularization_item.name: self.regularization_item.weight * (
                self.regularization_item.kwargs['target_ratio'] * self.volume - AssembleUtils.assemble_scalar(self.vol_ratio_form)
            ) ** 2,
            'abs_volume': float(np.abs(
                AssembleUtils.assemble_scalar(self.vol_ratio_form) - 
                self.regularization_item.kwargs['target_ratio'] * self.volume
            )) / self.volume,
        }
        return loss_dict
    
    def record_Upw(self, step: int, **kwargs):
        vel_ref = dolfinx.fem.Function(self.point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(self.uh.sub(0).collapse())  # type: ignore
        p_ref = self.uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

        self.recorder.update_point_data('U', vel_ref_np)
        self.recorder.update_point_data('U_mag', np.linalg.norm(vel_ref_np, axis=1))
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        self.recorder.update_cell_data('topo_weight', self.topo_weight.x.array.copy())  # type: ignore
        
        cell_scalar_fun = dolfinx.fem.Function(self.cell_scalar_func_space)
        cell_scalar_fun.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        point_scalar_fun = dolfinx.fem.Function(self.point_scalar_func_space)
        point_scalar_fun.interpolate(cell_scalar_fun)  # type: ignore
        self.recorder.update_point_data('topo_weight_pcd', point_scalar_fun.x.array.copy())  # type: ignore
        
        if self.obs_source is not None:
            self.recorder.update_cell_data('obstacle_source', self.obs_source.x.array.copy())  # type: ignore
        if kwargs.get('gradient', None) is not None:
            self.recorder.update_cell_data('gradient', kwargs['gradient'])
        
        self.recorder.save(step)

    def corase_step_run(self, step: float, iter: int = -1):
        # TODO: adjust 
        gradient_np = self.compute_gradient(adj_input=1.0, with_filter=True, with_norm=True)
        # direction_np = np.sign(gradient_np)  # TODO: still unstable
        direction_np = gradient_np
        
        # # ------ visulize obstacle and gradient
        # plotter = pyvista.Plotter()
        # plotter.add_mesh(self.mesh.grid, color='black', style='wireframe')
        
        # # pcd_mesh = pyvista.PointSet(self.mesh.grid.cell_centers().points)
        # # pcd_mesh.point_data['gradient'] = gradient_np
        # # pcd_mesh.warp_by_scalar(scalars='gradient')
        # # plotter.add_mesh(pcd_mesh, point_size=3)
        
        # # obs_pcd = self.mesh.points[self.obstacle_source.x.array > 0, :]  # type: ignore
        # # if obs_pcd.shape[0] > 0:
        # #     plotter.add_mesh(pyvista.PointSet(obs_pcd), color='black', point_size=3)
            
        # descend_pcd = self.mesh.grid.cell_centers().points[direction_np > 0.0, :]
        # if descend_pcd.shape[0] > 0:
        #     plotter.add_mesh(pyvista.PointSet(descend_pcd), color='blue', point_size=3)
        # ascend_pcd = self.mesh.grid.cell_centers().points[direction_np < 0.0, :]
        # if ascend_pcd.shape[0] > 0:
        #     plotter.add_mesh(pyvista.PointSet(ascend_pcd), color='red', point_size=3)
        # zero_pcd = self.mesh.grid.cell_centers().points[direction_np == 0.0, :]
        # if zero_pcd.shape[0] > 0:
        #     plotter.add_mesh(pyvista.PointSet(zero_pcd), color='green', point_size=3)
            
        # plotter.show()
        # # ------ visulize obstacle and gradient
        
        topo_phi_origin: np.array = self.topo_phi_input.x.array.copy()  # type: ignore
        new_topo_phi = topo_phi_origin - direction_np * step
        new_topo_phi = np.maximum(np.minimum(new_topo_phi, 1.0 - 1e-16), 1e-16)
        self.topo_phi_input.x.array[:] = new_topo_phi  # type: ignore
        
        loss_new = self.opt_problem(values=[self.topo_phi_input])  # type: ignore
        if iter > 0:
            self.record_Upw(step=iter, gradient=direction_np)
        
        detail_loss = self.eval()
        detail_loss['total_loss'] = loss_new
        
        return detail_loss

    """
    # TODO: please improve this
    def run(
        self,
        line_search_steps: list[float] = [0.1, 0.05, 0.01, 0.005, 0.001, 0.00025],
        max_iteration=200,
        tensorboard_recorder: TensorBoardRecorder|None = None,
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
        
        last_direction_np: np.ndarray
        
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
            
            # ------ update regularization weight ------
            self.update_reg_mul_weight(loss_dict)  # FIXME: terrible
            # ------ update regularization weight ------
            
            adj_input = np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2)
            direction_np = self.compute_gradient(adj_input=adj_input, with_filter=True, with_norm=True)
            if step == 1:
                last_direction_np = direction_np
            else:
                direction_np = direction_np * 0.2 + last_direction_np * 0.8
                last_direction_np = direction_np
            
            topo_phi_origin: np.array = topo_phi.x.array.copy()  # type: ignore
            new_topo_phi = topo_phi_origin - direction_np * 0.05
            new_topo_phi = np.maximum(np.minimum(new_topo_phi, 0.999), 0.001)
            topo_phi.x.array[:] = new_topo_phi  # type: ignore
            loss_current = self.opt_problem(values=[topo_phi])  # type: ignore
            self.record_Upw(step, shape_gradient=direction_np)
                
            if step > max_iteration:
                print(f"[Warning]: max iteration reached.")
                is_solved = False
                break
        
        return {'is_success': is_solved, 'topo_weight': self.topo_weight.x.array.copy()} 
    """


def debug_2D_without_obstacle():
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
        
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_30',
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
        velocity_degree=1,  # TODO: can it be 1?
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


if __name__ == '__main__':
    # debug_2D_without_obstacle()
    debug_2D_with_obstacle()
    # debug_3D_without_obstacle()
    # debug_3D_with_obstacle()
    # debug_3D_2_without_obstacle()
    # debug_3D_2_with_obstacle()
    
    pass
    
  