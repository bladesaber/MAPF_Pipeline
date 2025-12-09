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

from scripts_py.auto_dolfinx.cbs_topo.fipy_topo_simulator import FipyTopologySimulator


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
        self.obstacle_source = DolfinxFunction(self.point_scalar_func_space, name="obstacle_source")
        obstacle_source = np.zeros(self.mesh.num_points)
        
        if obstacle_func is not None:
            obstacle_dofs = obstacle_func(DolfinUtils.tabulate_dof_coordinates(self.point_scalar_func_space))
            if obstacle_dofs.shape[0] > 0:
                obstacle_source[obstacle_dofs] = 1.0
        
        self.obstacle_source.x.array[:] = obstacle_source  # type: ignore
        
        self.topo_phi_input = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_phi_input')

    def define_problem(
        self,
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
                velocityRelaxation=0.5,
                pressureRelaxation=0.8,
                structure_mesh=self.mesh,  # type: ignore
                log_step=2,
                simulation_iterations=100,
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
            
            self.vol_ratio_form = self.topo_weight * ufl.dx            
            volume = assemble_scalar(self.vol_ratio_form, domain=self.domain, tape=self.tape)
            reg_cost = self.regularization_item.weight * (
                self.regularization_item.kwargs['target_ratio'] * self.volume - volume
            ) ** 2
            
            # self.obstacle_cost_form = self.obstacle_cost_weight * inner(self.obstacle_source, self.topo_weight) * ufl.dx  # type: ignore
            
            J_form = (
                self.cost_item.cost_form 
                # + self.obstacle_cost_form
            )
            Jhat = assemble_scalar(J_form, domain=self.domain, tape=self.tape)  # type: ignore
            Jhat = Jhat + reg_cost
            
            self.control_phi = Control(self.topo_weight)
            self.opt_problem = ReducedFunctional(Jhat, [self.control_phi])
        
        print(self.eval())
                
    def eval(self):
        return {
            self.cost_item.name: self.cost_item.eval(),
            # self.regularization_item.name: self.regularization_item.eval(),
            self.regularization_item.name: self.regularization_item.weight * (
                self.regularization_item.kwargs['target_ratio'] * self.volume - AssembleUtils.assemble_scalar(self.vol_ratio_form)
            ) ** 2,
            'abs_volume': float(np.abs(AssembleUtils.assemble_scalar(self.vol_ratio_form) - self.regularization_item.kwargs['target_ratio'] * self.volume)) / self.volume,
            # 'obstacle_cost': AssembleUtils.assemble_scalar(self.obstacle_cost_form),
        }
    
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
        self.recorder.update_point_data('obstacle_source', self.obstacle_source.x.array.copy())  # type: ignore
        if kwargs.get('gradient', None) is not None:
            self.recorder.update_cell_data('gradient', kwargs['gradient'])
        
        self.recorder.save(step)

    def corase_step_run(self, step: float, iter: int = -1):
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


def debug_2D():
    msh_file = '/home/admin123456/Desktop/work/gmsh_test/t30/t30_model.msh'
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t30/t30_model.xdmf'
    MeshUtils.msh_to_XDMF(msh_file, xdmf_file, dim=2)
    
    # --- 1. define mesh ---
    mesh = SeqMeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_weight',
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
        # obstacle_func=define_obstacle_func,
        obstacle_func=None,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.1,
        result_dir='/home/admin123456/Desktop/work/gmsh_test/t30/topo_opt_cache',
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
    
    recorder = TensorBoardRecorder(log_dir='/home/admin123456/Desktop/work/gmsh_test/t29/cache/tensorboardx_record')
    optim.run(
        tensorboard_recorder=recorder
    )


def debug_3D():
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t29/t29_model.xdmf'
    msh_file = '/home/admin123456/Desktop/work/gmsh_test/t29/t29_model.msh'
    MeshUtils.msh_to_XDMF(
        msh_file=msh_file,
        xdmf_file=xdmf_file,
        dim=3,
    )

    # --- 1. define mesh ---
    mesh = SeqMeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=100.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_weight',
        style='Volume',
        weight=1000.0,
        target_ratio=0.2
    )
    
    # def define_obstacle_func(xyzs: np.ndarray):
    #     obstacle_center = np.array([0.5, 0.5, 0.25])
    #     cell2obs_dist = np.linalg.norm(xyzs[:, :2] - obstacle_center[:2], axis=1)
    #     obs_dofs = np.where(cell2obs_dist < 0.1)[0]
    #     return obs_dofs
    
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_29',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        # topo_weight_init=regularization_item.kwargs['target_ratio'] * 1.05,
        topo_weight_init=0.5,  # TODO: it seems source term cause result wrong
        # obstacle_func=define_obstacle_func,
        obstacle_func=None,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.1,
        result_dir='/home/admin123456/Desktop/work/gmsh_test/t29/topo_opt_cache',
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
    
    recorder = TensorBoardRecorder(log_dir='/home/admin123456/Desktop/work/gmsh_test/t29/topo_opt_cache/tensorboardx_record')
    optim.run(
        line_search_steps=[0.2, 0.1, 0.05, 0.01],
        tensorboard_recorder=recorder
    )


def debug_3D_2():
    base_dir = '/home/admin123456/Desktop/work/gmsh_test/t31'
    xdmf_file = os.path.join(base_dir, 'model.xdmf')
    msh_file = os.path.join(base_dir, 'model.msh')
    MeshUtils.msh_to_XDMF(
        msh_file=msh_file,
        xdmf_file=xdmf_file,
        dim=3,
    )

    # --- 1. define mesh ---
    mesh = SeqMeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='obj_cost',
        style='MinDissipation',
        weight=10.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_weight',
        style='Volume',
        weight=100.0,
        target_ratio=0.2
    )
    
    # def define_obstacle_func(xyzs: np.ndarray):
    #     obstacle_center = np.array([0.5, 0.5, 0.25])
    #     cell2obs_dist = np.linalg.norm(xyzs[:, :2] - obstacle_center[:2], axis=1)
    #     obs_dofs = np.where(cell2obs_dist < 0.1)[0]
    #     return obs_dofs
    
    optim = TopoFluidOptimizerFipy(
        name='topo_fluid_31',
        mesh=mesh,
        msh_file=msh_file,
        nut_init=1.0 / 400.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        # topo_weight_init=regularization_item.kwargs['target_ratio'] * 1.05,
        topo_weight_init=0.5,  # TODO: it seems source term cause result wrong
        # obstacle_func=define_obstacle_func,
        obstacle_func=None,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.1,
        result_dir=os.path.join(base_dir, 'topo_opt_cache'),
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
        line_search_steps=[0.2, 0.1, 0.05, 0.01],
        tensorboard_recorder=recorder
    )


if __name__ == '__main__':
    debug_3D()
    # debug_2D()
    # debug_3D_2()
    
  