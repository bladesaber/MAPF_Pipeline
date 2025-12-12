import os
import shutil
from typing import Literal

import dolfinx
import ufl
import ufl.tensors
import ufl.indexed
from ufl import grad, inner, div, dot

import numpy as np
from functools import partial

from scripts_py.simulator.dolfin_simulator import (
    FluidSimulator,
    FluidBoundary
)
from scripts_py.dolfinx_adjoint.topology_utils.topology_filter import DistanceBandFilter

from scripts_py.dolfinx_adjoint.base_type import (
    StateProblemDataBase,
    ControlProblemDataBase,
    BoundaryCell,
)
from scripts_py.dolfinx_adjoint.problem_state import StateProblem
from scripts_py.dolfinx_adjoint.cost_functions import (
    CostFunctional,
    IntegralFunctional,
    ScalarTrackingFunctional,
)
from scripts_py.dolfinx_adjoint.solver_optimize import ControlOptimizer
from scripts_py.dolfinx_adjoint.regularization_control import (
    ControlRegularizationPipe,
    ControlIntegralRegularization,
    ControlRegularization,
)
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.dolfinx_utils import AssembleUtils

from scripts_py.vtu_recorder import VtuRecorder


class FluidCost(object):
    def __init__(
        self, 
        name: str, 
        style: Literal['MinDissipation', 'ConstantFlux', 'MaxFlux'], 
        weight: float,
        **kwargs
    ):
        self.name = name
        self.style = style
        self.weight = weight
        self.kwargs = kwargs
        self.cost_form: ufl.Form
        self.cost_func: CostFunctional


class WeightRegularizationItem(object):
    def __init__(self, name: str, style: Literal['Volume'], weight: float, **kwargs):
        self.name = name
        self.style = style
        self.weight = weight
        self.kwargs = kwargs
        self.reg_form: ufl.Form
        self.reg_func: ControlRegularization


class FluidTopologyOptimizer(FluidSimulator):
    state_ksp_option = {
        'ksp_type': 'preonly', 
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", 
        "mat_mumps_icntl_24": 1
    }
    adjoint_ksp_option = {
        'ksp_type': 'preonly', 
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", 
        "mat_mumps_icntl_24": 1
    }
    gradient_ksp_option = {
        'ksp_type': 'preonly', 
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", 
        "mat_mumps_icntl_24": 1
    }
    
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        cost_item: FluidCost,
        regularization_item: WeightRegularizationItem,
        topo_weight_init: float,
        nut_init: float,
        weight_filter_radius: float,
        weight_filter_is_cell_field: bool,
        result_dir: str,
        alpha_fun_max: float = 1000.0,
        alpha_fun_min: float = 0.01,
        alpha_fun_p: float = 3.0,
        state_ksp_option: dict = state_ksp_option,
        adjoint_ksp_option: dict = adjoint_ksp_option,
        gradient_ksp_option: dict = gradient_ksp_option,
    ):
        super().__init__(name=name, mesh=mesh)
        
        self.cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(self.mesh.domain)
        self.cell_vector_func_space = DolfinUtils.cell_vector_function_space(self.mesh.domain)
        self.point_scalar_func_space = DolfinUtils.point_scalar_function_space(self.mesh.domain)
        self.point_vector_func_space = DolfinUtils.point_vector_function_space(self.mesh.domain)
        
        self.surface_area: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.ds  # type: ignore
        ))
        self.volume: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(  # type: ignore
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.dx  # type: ignore
        ))
        
        self.topo_weight_init = topo_weight_init
        self.topo_weight = dolfinx.fem.Function(self.cell_scalar_func_space, name=f'topo_weight')
        self.topo_weight.x.array[:] = topo_weight_init  # type: ignore    
        
        self.fluid_type = fluid_type
        self.cost_item = cost_item
        self.regularization_item = regularization_item
        self.nut_init = nut_init
        self.result_dir = result_dir
        self.recorder = VtuRecorder(self.mesh, save_dir=self.result_dir, is_binary=True)
        
        self.alpha_fun_max = alpha_fun_max
        self.alpha_fun_min = alpha_fun_min
        self.alpha_fun_p = alpha_fun_p
        self.state_ksp_option = state_ksp_option
        self.adjoint_ksp_option = adjoint_ksp_option
        self.gradient_ksp_option = gradient_ksp_option
        
        self.uh: dolfinx.fem.Function
        self.f: dolfinx.fem.Function
        self.nut: dolfinx.fem.Function
        self.weight_filter = DistanceBandFilter(self.mesh.grid, weight_filter_radius, weight_filter_is_cell_field, True)

    @staticmethod
    def alpha_fun(
        rho: np.ndarray|dolfinx.fem.Function,
        alpha_min=0.01, alpha_max=1000.0, p=3.0, pow_operator=ufl.elem_pow
    ):
        return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)  # type: ignore

    def set_topo_weight(self, dofs: np.ndarray, value: float):
        assert np.min(value) >= 0.0, f'topo_weight must be greater than 0, but got {np.min(value)}.'
        assert np.max(value) <= 1.0, f'topo_weight must be less than 1, but got {np.max(value)}.'
        self.topo_weight.x.array[:] = self.topo_weight_init  # type: ignore
        self.topo_weight.x.array[dofs] = value  # type: ignore

    def define_govern_equation(
        self, 
        u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed,
        v: ufl.tensors.ListTensor, q: ufl.indexed.Indexed,
        f: dolfinx.fem.Function,
        nut: dolfinx.fem.Function,
    ):
        '''
        Args:
            U: velocity field
            p: pressure field
            V: velocity test function
            q: pressure test function
            f: body force
            nut: viscosity
        '''
        if self.fluid_type == 'stoke':
            # govern_form = (
            #     nut * inner(grad(u), grad(v)) * ufl.dx  # type: ignore
            #     - inner(div(v), p) * ufl.dx
            #     - inner(q, div(u)) * ufl.dx
            #     - inner(f, v) * ufl.dx
            # )
            govern_form = (
                nut * inner(grad(u), grad(v)) * ufl.dx  # type: ignore
                - inner(p, div(v)) * ufl.dx 
                + inner(div(u), q) * ufl.dx
                - inner(f, v) * ufl.dx
            )
            
        elif self.fluid_type in ['naiver_stoke', 'rans']:
            govern_form = (
                nut * inner(grad(u), grad(v)) * ufl.dx  # type: ignore
                + inner(grad(u) * u, v) * ufl.dx  # type: ignore
                - inner(div(v), p) * ufl.dx
                + inner(q, div(u)) * ufl.dx
                - inner(f, v) * ufl.dx
            )
        else:
            raise ValueError(f'Invalid fluid type: {self.fluid_type}')
        return govern_form

    def define_boundary(self):
        for name, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'velocity':
                
                inflow_velocity_exp = partial(
                    FluidBoundary.inflow_velocity_exp,
                    velocity=np.array(bry_info.value),
                    tdim=self.mesh.tdim,
                    face_center=bry_info.face_center,  # type: ignore
                    face_norm_vector=bry_info.face_norm_vector,  # type: ignore
                    face_radius=bry_info.face_radius  # type: ignore
                )
                
                bry_info.bry_cell = BoundaryCell.define_bry_cell(
                    name=name,
                    bc_func_space=self.W0,
                    value_func_space=self.V,
                    dofs=MeshUtils.extract_entity_dofs(
                        (self.W0, self.V), self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    ),
                    bc_value=inflow_velocity_exp,
                    is_mixed_element=True,
                )

            elif bry_info.boundary_type == 'pressure':
                pressure_func = dolfinx.fem.Function(self.Q, name=name)
                pressure_func.x.array[:] = bry_info.value  # type: ignore
                bry_info.bry_cell = BoundaryCell.define_bry_cell(
                    name=name,
                    bc_func_space=self.W1,
                    value_func_space=self.Q,
                    dofs=MeshUtils.extract_entity_dofs(
                        (self.W1, self.Q), self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    ),
                    bc_value=pressure_func,
                    is_mixed_element=True,
                )

            elif bry_info.boundary_type == 'wall':
                wall_func = dolfinx.fem.Function(self.V, name=name)
                bry_info.bry_cell = BoundaryCell.define_bry_cell(
                    name=name,
                    bc_func_space=self.W0,
                    value_func_space=self.V,
                    dofs=MeshUtils.extract_entity_dofs(
                        (self.W0, self.V), self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    ),
                    bc_value=wall_func,
                    is_mixed_element=True,
                )

            else:
                raise ValueError('[Error]: Invalid Boundary Type.')

    def define_cost(
        self, 
        cost_item: FluidCost, 
        u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed, 
        alpha: ufl.Form, 
        **kwargs
    ):
        if cost_item.style == 'MinDissipation':
            cost_item.cost_form = (
                0.5 * alpha * inner(u, u) * ufl.dx + 
                kwargs['nut'] * inner(grad(u), grad(u)) * ufl.dx  # type: ignore
            )
            cost_item.cost_func = IntegralFunctional(
                name=f'{cost_item.name}_{cost_item.style}',
                domain=self.mesh.domain, 
                cost_form=cost_item.cost_form,
                weight=cost_item.weight
            )
        elif cost_item.style == 'MinPressure':
            assert 'marker' in cost_item.kwargs, 'Marker is required for MinPressure cost'
            area = AssembleUtils.assemble_scalar(
                dolfinx.fem.Constant(self.mesh.domain, 1.0) * self.ds(cost_item.kwargs['marker'])  # type: ignore
            )
            cost_item.cost_form = (1.0 / area) * p * self.ds(cost_item.kwargs['marker'])
            cost_item.cost_func = IntegralFunctional(
                name=f'{cost_item.name}_{cost_item.style}',
                domain=self.mesh.domain, 
                cost_form=cost_item.cost_form, 
                weight=cost_item.weight
            )
        elif cost_item.style == 'ConstantFlux':
            assert 'marker' in cost_item.kwargs, 'Marker is required for ConstantFlux cost'
            assert 'tracking_flux' in cost_item.kwargs, 'Tracking flux is required for ConstantFlux cost'
            cost_item.cost_form = ufl.dot(u, self.n_vec) * self.ds(cost_item.kwargs['marker'])  # type: ignore
            cost_item.cost_func = ScalarTrackingFunctional(
                name=f'{cost_item.name}_{cost_item.style}',
                domain=self.mesh.domain,
                form=cost_item.cost_form,
                tracking_goal=cost_item.kwargs['tracking_flux'],
                weight=cost_item.weight,
                method='pct_div'
            )
        elif cost_item.style == 'MaxFlux':
            # todo need to check direction of flux
            assert 'marker' in cost_item.kwargs, 'Marker is required for MaxFlux cost'
            cost_item.cost_form = ufl.dot(u, self.n_vec) * self.ds(cost_item.kwargs['marker'])  # type: ignore
            cost_item.cost_func = IntegralFunctional(
                name=f'{cost_item.name}_{cost_item.style}',
                domain=self.mesh.domain,
                cost_form=cost_item.cost_form,
                weight=cost_item.weight
            )
        else:
            raise ValueError(f'Invalid cost style: {cost_item.style}')

    def define_regularization(
        self, regularization_item: WeightRegularizationItem, u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed
    ):
        if regularization_item.style == 'Volume':
            assert 'target_ratio' in regularization_item.kwargs, 'Target ratio is required for Volume regularization'
            regularization_item.reg_form = self.topo_weight  # type: ignore
            regularization_item.reg_func = ControlIntegralRegularization(
                name=f'{regularization_item.name}_{regularization_item.style}',
                domain=self.mesh.domain,
                integral_form=regularization_item.reg_form,
                control=self.topo_weight,  # type: ignore
                weight=regularization_item.weight,
                target_value=self.volume * regularization_item.kwargs['target_ratio']
            )
        else:
            raise ValueError(f'Invalid regularization style: {regularization_item.style}')

    def define_problem(self):
        uh = dolfinx.fem.Function(self.W, name=f"{self.name}_state")
        u, p = ufl.split(uh)  # type: ignore # please don't use up.split()
        vq = dolfinx.fem.Function(self.W, name=f"{self.name}_adjoint")
        v, q = ufl.split(vq)  # type: ignore # please don't use up.split()

        f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
        nut = dolfinx.fem.Function(self.cell_scalar_func_space, name=f"{self.name}_nut")
        nut.x.array[:] = self.nut_init  # type: ignore

        govern_form = self.define_govern_equation(u, p, v, q, f, nut)  # type: ignore
        
        alpha = self.alpha_fun(
            rho=self.topo_weight,  # type: ignore
            alpha_min=self.alpha_fun_min, 
            alpha_max=self.alpha_fun_max, 
            p=self.alpha_fun_p, 
            pow_operator=np.power
        )
        govern_form += alpha * inner(u, v) * ufl.dx

        self.state_database = StateProblemDataBase(
            name=f"{self.name}_state_part",
            F_form=govern_form,
            state=uh,
            adjoint=vq,
            is_linear=True if self.fluid_type == 'stoke' else False,
            state_ksp_option=self.state_ksp_option,
            adjoint_ksp_option=self.adjoint_ksp_option,
        )
        
        self.define_boundary()
        for bry_info in self.boundaries.values():
            self.state_database.add_bc(bry_info.bry_cell)

        self.control_database = ControlProblemDataBase(
            controls=[self.topo_weight],  # type: ignore
            gradient_ksp_options=self.gradient_ksp_option,
        )
        
        self.define_cost(self.cost_item, u, p, alpha, nut=nut)  # type: ignore
        self.define_regularization(self.regularization_item, u, p)  # type: ignore
        
        self.optimizer = ControlOptimizer(
            state_problem=StateProblem([self.state_database]),
            control_database=self.control_database,
            cost_functional_list=[self.cost_item.cost_func],
            control_regularization_pipe=ControlRegularizationPipe([self.regularization_item.reg_func]),
            scalar_product_kwargs={},
        )
        
        self.uh = uh  # type: ignore
        self.f = f  # type: ignore
        self.nut = nut  # type: ignore
    
    def compute_gradient(
        self, 
        update_state: bool,
        update_adjoint: bool,
        state_kwargs: dict, 
        adjoint_kwargs: dict, 
        gradient_kwargs: dict, 
        with_filter=True,
        with_norm=True
    ):
        report_dict = self.optimizer.compute_gradient(
            self.mesh.domain.comm,
            state_kwargs=state_kwargs,
            adjoint_kwargs=adjoint_kwargs,
            gradient_kwargs=gradient_kwargs,
            update_state=update_state,
            update_adjoint=update_adjoint
        )
        is_success = report_dict['is_success']
        if not is_success:
            return {'is_success': False, 'log': report_dict['log']}
        
        grad: dolfinx.fem.Function = report_dict['gradients'][self.topo_weight.name]  # type: ignore
        grad_np: np.ndarray = grad.x.array.copy()
                
        if with_filter:
            grad_np = grad_np.reshape((-1, 1))
            grad_np = self.weight_filter.run(grad_np)
            grad_np = np.asarray(grad_np).reshape((-1))
        
        if with_norm:
            norm = np.max(np.abs(grad_np))
            grad_np = grad_np / norm
            # grad_np = grad_np / np.linalg.norm(grad_np)

        return {'is_success': True, 'gradient': grad_np}
    
    def record_Upw(self, step: int, **kwargs):
        vel_ref = dolfinx.fem.Function(self.point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(self.uh.sub(0).collapse())  # type: ignore
        p_ref = self.uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

        self.recorder.update_point_data('U', vel_ref_np)
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        self.recorder.update_cell_data('topo_weight', self.topo_weight.x.array.copy())  # type: ignore        
        if kwargs.get('shape_gradient', None) is not None:
            self.recorder.update_cell_data('shape_gradient', kwargs['shape_gradient'])
        
        self.recorder.save(step)

    def run(
        self,
        line_search_steps: list[float] = [0.1, 0.05, 0.01, 0.005, 0.001, 0.00025],
        state_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        adjoint_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        gradient_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        max_iteration=200,
    ):
        is_solved, step = False, 0
        loss_current = 0.0
        loss_last = 0.0
        loss_dict_current = {}
        line_search_steps_np: np.ndarray = np.array(line_search_steps)
        step_length_last = line_search_steps_np[0]
        
        loss_current = self.optimizer.evaluate_cost_functional(self.mesh.domain.comm, update_state=True)
        self.record_Upw(step)
        print(f"step/{step} loss:{loss_current}.")
        
        while True:
            step += 1
            if step > 1:
                loss_residual = loss_current - loss_last
                log = f"step/{step} loss:{loss_current} residual:{loss_residual:.6f} last_step_length:{step_length_last}."
                for name, loss_value in loss_dict_current.items():
                    log += f" {name}:{loss_value:.3f}"
                print(log)
                
                if np.abs(loss_residual) < 1e-6:
                    is_solved = True
                    break

            report_dict = self.compute_gradient(
                update_state=False,
                update_adjoint=True,
                state_kwargs=state_kwargs,
                adjoint_kwargs=adjoint_kwargs,
                gradient_kwargs=gradient_kwargs,
                with_filter=True,
                with_norm=True,
            )
            is_success = report_dict['is_success']
            if not is_success:
                print(f'[Error]: {report_dict["log"]}')
                is_solved = False
                break

            direction_np = report_dict['gradient']
            
            find_best_step = False
            topo_weight_origin: np.array = self.topo_weight.x.array.copy()  # type: ignore
                        
            last_step_index = np.where(line_search_steps_np == step_length_last)[0][0]
            line_search_start_index = np.maximum(last_step_index - 1, 0)
            for i in range(line_search_start_index, line_search_steps_np.shape[0]):
                step_length = line_search_steps_np[i]
                
                new_topo_weight = topo_weight_origin - direction_np * step_length
                new_topo_weight = np.maximum(np.minimum(new_topo_weight, 0.999), 0.001)
                self.topo_weight.x.array[:] = new_topo_weight  # type: ignore
                                
                loss_dict_current = self.optimizer.get_cost_info(
                    update_state=True,
                    comm=self.mesh.domain.comm,
                    **state_kwargs,
                )
                loss_new = np.sum(list(loss_dict_current.values()))
 
                if loss_new < loss_current:
                    find_best_step = True
                    loss_last = loss_current
                    loss_current = loss_new
                    step_length_last = step_length
                    self.record_Upw(step, shape_gradient=direction_np)
                    break
                
            if not find_best_step:
                print(f"[Warning]: line search fail.")
                is_solved = False
                break
                
            if step > max_iteration:
                print(f"[Warning]: max iteration reached.")
                is_solved = False
                break
        
        return {'is_success': is_solved, 'topo_weight': self.topo_weight.x.array.copy()}  # type: ignore


class ConstrainedTopologyOptimizer(FluidTopologyOptimizer):
    def __init__(
        self,
        name: str, 
        mesh: MeshObj,
        obstacle_cost_weight: float,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        cost_item: FluidCost,
        regularization_item: WeightRegularizationItem,
        topo_weight_init: float,
        nut_init: float,
        weight_filter_radius: float,
        weight_filter_is_cell_field: bool,
        result_dir: str,
        alpha_fun_max: float = 1000.0,
        alpha_fun_min: float = 0.01,
        alpha_fun_p: float = 3.0,
        state_ksp_option: dict = FluidTopologyOptimizer.state_ksp_option,
        adjoint_ksp_option: dict = FluidTopologyOptimizer.adjoint_ksp_option,
        gradient_ksp_option: dict = FluidTopologyOptimizer.gradient_ksp_option,
    ):
        super().__init__(
            name=name,
            mesh=mesh,
            fluid_type=fluid_type,
            cost_item=cost_item,
            regularization_item=regularization_item,
            topo_weight_init=topo_weight_init,
            nut_init=nut_init,
            weight_filter_radius=weight_filter_radius,
            weight_filter_is_cell_field=weight_filter_is_cell_field,
            result_dir=result_dir,
            alpha_fun_max=alpha_fun_max,
            alpha_fun_min=alpha_fun_min,
            alpha_fun_p=alpha_fun_p,
            state_ksp_option=state_ksp_option,
            adjoint_ksp_option=adjoint_ksp_option,
            gradient_ksp_option=gradient_ksp_option,
        )
        
        self.obstacle_cost_weight = obstacle_cost_weight
        self.obs_cost_weight_func = dolfinx.fem.Constant(self.mesh.domain, self.obstacle_cost_weight)  # type: ignore
        
        self.obstacle_source = dolfinx.fem.Function(self.cell_scalar_func_space, name='obstacle_source')
        self.obstacle_source.x.array[:] = 0.0  # type: ignore

    def set_obstacle_source(self, obstacle_dofs: np.ndarray):
        obstacle_source = np.zeros(self.mesh.num_cells)
        obstacle_source[obstacle_dofs] = 1.0
        self.obstacle_source.x.array[:] = obstacle_source  # type: ignore

    def define_cost(
        self, 
        cost_item: FluidCost, 
        u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed, 
        alpha: ufl.Form, 
        **kwargs
    ):
        if cost_item.style == 'MinDissipation':
            cost_item.cost_form = (
                0.5 * alpha * inner(u, u) * ufl.dx
                + kwargs['nut'] * inner(grad(u), grad(u)) * ufl.dx  # type: ignore
                + self.obs_cost_weight_func * inner(self.obstacle_source, self.topo_weight) * ufl.dx  # type: ignore
            )
            cost_item.cost_func = IntegralFunctional(
                name=f'{cost_item.name}_{cost_item.style}',
                domain=self.mesh.domain, 
                cost_form=cost_item.cost_form,
                weight=cost_item.weight
            )
        elif cost_item.style == 'MinPressure':
            assert 'marker' in cost_item.kwargs, 'Marker is required for MinPressure cost'
            area = AssembleUtils.assemble_scalar(
                dolfinx.fem.Constant(self.mesh.domain, 1.0) * self.ds(cost_item.kwargs['marker'])  # type: ignore
            )
            cost_item.cost_form = (
                (1.0 / area) * p * self.ds(cost_item.kwargs['marker'])
                + self.obs_cost_weight_func * inner(self.obstacle_source, self.topo_weight) * ufl.dx  # type: ignore
            )
            cost_item.cost_func = IntegralFunctional(
                name=f'{cost_item.name}_{cost_item.style}',
                domain=self.mesh.domain, 
                cost_form=cost_item.cost_form, 
                weight=cost_item.weight
            )
        elif cost_item.style == 'ConstantFlux':
            assert 'marker' in cost_item.kwargs, 'Marker is required for ConstantFlux cost'
            assert 'tracking_flux' in cost_item.kwargs, 'Tracking flux is required for ConstantFlux cost'
            cost_item.cost_form = (
                ufl.dot(u, self.n_vec) * self.ds(cost_item.kwargs['marker'])  # type: ignore
                + self.obs_cost_weight_func * inner(self.obstacle_source, self.topo_weight) * ufl.dx  # type: ignore
            )
            cost_item.cost_func = ScalarTrackingFunctional(
                name=f'{cost_item.name}_{cost_item.style}',
                domain=self.mesh.domain,
                form=cost_item.cost_form,
                tracking_goal=cost_item.kwargs['tracking_flux'],
                weight=cost_item.weight,
                method='pct_div'
            )
        elif cost_item.style == 'MaxFlux':
            # todo need to check direction of flux
            assert 'marker' in cost_item.kwargs, 'Marker is required for MaxFlux cost'
            cost_item.cost_form = (
                ufl.dot(u, self.n_vec) * self.ds(cost_item.kwargs['marker'])  # type: ignore
                + self.obs_cost_weight_func * inner(self.obstacle_source, self.topo_weight) * ufl.dx  # type: ignore
            )
            cost_item.cost_func = IntegralFunctional(
                name=f'{cost_item.name}_{cost_item.style}',
                domain=self.mesh.domain,
                cost_form=cost_item.cost_form,
                weight=cost_item.weight
            )
        else:
            raise ValueError(f'Invalid cost style: {cost_item.style}')

    def record_Upw(self, step: int, **kwargs):
        vel_ref = dolfinx.fem.Function(self.point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(self.uh.sub(0).collapse())  # type: ignore
        p_ref = self.uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

        self.recorder.update_point_data('U', vel_ref_np)
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        self.recorder.update_cell_data('topo_weight', self.topo_weight.x.array.copy())  # type: ignore
        self.recorder.update_cell_data('obstacle_source', self.obstacle_source.x.array.copy())  # type: ignore
        if kwargs.get('shape_gradient', None) is not None:
            self.recorder.update_cell_data('shape_gradient', kwargs['shape_gradient'])
        self.recorder.save(step)


if __name__ == '__main__':
    pass
    