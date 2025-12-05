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


class TopoObstacleFluidOptimizerFipy(TopoFluidOptimizer):
    def __init__(
        self, 
        name: str,
        msh_file: str,
        mesh: SeqMeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        nut_init: float,
        cost_item: AutoFluidCost,
        regularization_item: AutoFluidRegularization,
        topo_weight_init: float,
        obstacle_func: Callable,
        obstacle_cost_weight: float,
        weight_filter_radius: float,
        weight_filter_is_cell_field: bool,
        result_dir: str,
        alpha_fun_max: float = 1000.0,
        alpha_fun_min: float = 0.01,
        alpha_fun_p: float = 3.0,
        velocity_degree = 1,
        pressure_degree = 1,
    ):
        super().__init__(
            name,
            mesh,
            fluid_type,
            nut_init,
            cost_item,
            regularization_item,
            topo_weight_init,
            weight_filter_radius,
            weight_filter_is_cell_field,
            result_dir,
            alpha_fun_max,
            alpha_fun_min,
            alpha_fun_p,
            velocity_degree=velocity_degree,
            pressure_degree=pressure_degree,
        )
        self.msh_file = msh_file
        
        self.obstacle_cost_weight = obstacle_cost_weight
        self.obstacle_source = DolfinxFunction(self.point_scalar_func_space, name="obstacle_source")
        obstacle_source = np.zeros(self.mesh.num_points)
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
                boundaries=[bry_info for bry_info in self.boundaries.values()],
                msh_file=self.msh_file,
                viscosity=self.nut_init,
                velocityRelaxation=0.5,
                pressureRelaxation=0.8,
                structure_mesh=self.mesh,  # type: ignore
                topo_weight_name=self.topo_weight.name,
                log_step=2,
                simulation_iterations=30,
                uh_type='point',
                V_to_W_dofs=self.V_to_W_dofs,
                Q_to_W_dofs=self.Q_to_W_dofs,
                
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
            
            self.obstacle_cost_form = self.obstacle_cost_weight * inner(self.obstacle_source, self.topo_weight) * ufl.dx  # type: ignore
            
            J_form = (
                self.cost_item.cost_form 
                # + self.regularization_item.cost_form
                + self.obstacle_cost_form
            )
            Jhat = assemble_scalar(J_form, domain=self.domain, tape=self.tape)  # type: ignore
            Jhat = Jhat + reg_cost
            
            self.control_phi = Control(self.topo_weight)
            self.opt_problem = ReducedFunctional(Jhat, [self.control_phi])
    
    def eval(self):
        return {
            self.cost_item.name: self.cost_item.eval(),
            # self.regularization_item.name: self.regularization_item.eval(),
            self.regularization_item.name: self.regularization_item.weight * (
                self.regularization_item.kwargs['target_ratio'] * self.volume - AssembleUtils.assemble_scalar(self.vol_ratio_form)
            ) ** 2,
            'obstacle_cost': AssembleUtils.assemble_scalar(self.obstacle_cost_form),
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
        self.recorder.update_point_data('topo_weight', self.topo_weight.x.array.copy())  # type: ignore
        self.recorder.update_point_data('obstacle_source', self.obstacle_source.x.array.copy())  # type: ignore
        if kwargs.get('gradient', None) is not None:
            self.recorder.update_point_data('gradient', kwargs['gradient'])
        
        self.recorder.save(step)
  
    def corase_step_run(self, step: float, iter: int = -1) -> dict:
        gradient_np = self.compute_gradient(adj_input=1.0, with_filter=True, with_norm=True)
        direction_np = np.sign(gradient_np)  # TODO: still unstable
        # direction_np = gradient_np
        
        # # ------ visulize obstacle and gradient
        # plotter = pyvista.Plotter()
        # plotter.add_mesh(self.mesh.grid, color='black', style='wireframe')
        
        # # pcd_mesh = pyvista.PointSet(self.mesh.points)
        # # pcd_mesh.point_data['gradient'] = gradient_np
        # # pcd_mesh.warp_by_scalar(scalars='gradient')
        # # plotter.add_mesh(pcd_mesh, point_size=5)
        
        # obs_pcd = self.mesh.points[self.obstacle_source.x.array > 0, :]  # type: ignore
        # if obs_pcd.shape[0] > 0:
        #     plotter.add_mesh(pyvista.PointSet(obs_pcd), color='black', point_size=10)
            
        # descend_pcd = self.mesh.points[direction_np > 0.0, :]
        # if descend_pcd.shape[0] > 0:
        #     plotter.add_mesh(pyvista.PointSet(descend_pcd), color='blue', point_size=5)
        # ascend_pcd = self.mesh.points[direction_np < 0.0, :]
        # if ascend_pcd.shape[0] > 0:
        #     plotter.add_mesh(pyvista.PointSet(ascend_pcd), color='red', point_size=5)
        # zero_pcd = self.mesh.points[direction_np == 0.0, :]
        # if zero_pcd.shape[0] > 0:
        #     plotter.add_mesh(pyvista.PointSet(zero_pcd), color='green', point_size=5)
            
        # plotter.show()
        # # ------ visulize obstacle and gradient
        
        topo_phi_origin: np.array = self.topo_phi_input.x.array.copy()  # type: ignore
        new_topo_phi = topo_phi_origin - direction_np * step
        new_topo_phi = np.maximum(np.minimum(new_topo_phi, 1.0 - 1e-16), 1e-16)
        self.topo_phi_input.x.array[:] = new_topo_phi  # type: ignore
        
        loss_new = self.opt_problem(values=[self.topo_phi_input])  # type: ignore
        if iter > 0:
            self.record_Upw(step=iter)
        
        detail_loss = self.eval()
        detail_loss['total_loss'] = loss_new
        
        return detail_loss


def debug_simulate():
    from scripts_py.auto_dolfinx.cbs_topo.fipy_topo_simulator import FipyTopoStokeSimulator
    
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t29/model.xdmf'
    msh_file = '/home/admin123456/Desktop/work/gmsh_test/t29/model.msh'
    MeshUtils.msh_to_XDMF(msh_file=msh_file, xdmf_file=xdmf_file, dim=3)
    structure_mesh = SeqMeshObj(xdmf_file=xdmf_file)
    obstacle_center = np.array([0.5, 0.5, 0.25])
    obstacle_radius = 0.2
    
    structure_topo_position = structure_mesh.grid.cell_centers().points
    structure_topo_weight = np.zeros(structure_topo_position.shape[0])
    obstacle_bool = np.linalg.norm(structure_topo_position[:, :2] - obstacle_center.reshape((1, -1))[:, :2], axis=1) < obstacle_radius
    structure_topo_weight[obstacle_bool] = 1000.0
    structure_mesh.grid.cell_data['obstacle'] = structure_topo_weight
    
    simulator = FipyTopoStokeSimulator.from_msh(
        source_msh_file=msh_file,
        dim=3,
        viscosity=1.0 / 400.0,
        velocityRelaxation=0.5,
        pressureRelaxation=0.8,
        structure_mesh=structure_mesh,
        structure_topo_weight=structure_topo_weight,
        structure_topo_position=structure_topo_position,
        topo_radius=0.03,
    )
        
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=20,
        face_center=[-0.05, 0.3, 0.25],
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
    
    simulator.add_boundary(inlet_bry)
    simulator.add_boundary(outlet_bry)
    simulator.add_boundary(wall_bry)
    simulator.define_boundary()
    
    simulator.run(iterations=30, log_step=2, save_dir='/home/admin123456/Desktop/work/gmsh_test/t29/simulate_cache')
    

def debug_opt():
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t29/model.xdmf'
    msh_file = '/home/admin123456/Desktop/work/gmsh_test/t29/model.msh'
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
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='reg_weight',
        style='Volume',
        weight=100.0,
        target_ratio=0.25
    )
    
    def define_obstacle_func(xyzs: np.ndarray):
        obstacle_center = np.array([0.5, 0.5, 0.25])
        cell2obs_dist = np.linalg.norm(xyzs[:, :2] - obstacle_center[:2], axis=1)
        obs_dofs = np.where(cell2obs_dist < 0.1)[0]
        return obs_dofs
    
    optim = TopoObstacleFluidOptimizerFipy(
        name='topo_fluid_29',
        mesh=mesh,
        msh_file=msh_file,
        fluid_type='stoke',
        nut_init=1.0 / 100.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        topo_weight_init=0.5,
        obstacle_func=define_obstacle_func,
        obstacle_cost_weight=100.0,
        weight_filter_radius=0.3,
        weight_filter_is_cell_field=False,
        result_dir='/home/admin123456/Desktop/work/gmsh_test/t29/cache',
    )
    
    tape = Tape()
    optim.setup_tape(tape)

    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.0, 0.0],
        marker=20,
        face_center=[-0.05, 0.3, 0.25],
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
    optim.define_problem()
    
    # step_list = np.array([0.1, 0.05, 0.025])
    # iter_list = np.array([10, 14, 18])
    # for iter in range(iter_list[-1]):
    #     step_idx = int(np.sum(iter >= iter_list))
    #     detail_loss = optim.corase_step_run(step_list[step_idx], iter=iter)
        
    #     log_str = f'[Log step:{step_list[step_idx]:.3f}]: loss:{detail_loss.pop('total_loss'):.5f}'
    #     for name, cost_value in detail_loss.items():
    #         log_str += f" {name}:{cost_value:.3f}"
    #     print(log_str)


if __name__ == '__main__':
    # debug_simulate()
    debug_opt()
    
  