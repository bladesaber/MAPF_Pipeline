import os
import shutil
from typing import Literal, Union

import dolfinx
import ufl
import ufl.tensors
import ufl.indexed
from ufl import grad, inner, div, dot
from basix.ufl import element, mixed_element

import numpy as np
from functools import partial

from scripts_py.simulator.dolfin_simulator import (
    FluidSimulator,
    FluidBoundary
)

from scripts_py.dolfinx_adjoint.base_type import (
    StateProblemDataBase,
    ShapeProblemDataBase,
    BoundaryCell,
)
from scripts_py.dolfinx_adjoint.problem_state import StateProblem
from scripts_py.dolfinx_adjoint.cost_functions import (
    CostFunctional,
    IntegralFunctional,
    ScalarTrackingFunctional,
)
from scripts_py.dolfinx_adjoint.solver_optimize import ShapeOptimizer
from scripts_py.dolfinx_adjoint.regularization_shape import (
    ShapeRegularizationPipe,
    ShapeRegularization,
    VolumeRegularization,
    SurfaceAreaRegularization,
    BarycenterRegularization,
)
from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.dolfinx_utils import AssembleUtils
from scripts_py.dolfinx_adjoint.topology_utils.topology_fluid import FluidCost

from scripts_py.vtu_recorder import VtuRecorder


class FluidPipeBoundary(FluidBoundary):
    def __init__(
        self,
        name: str,
        boundary_type: Literal['velocity', 'pressure', 'wall'],
        value: Union[float, list[float], np.ndarray],
        marker: int,
        is_free: bool,
        **kwargs
    ):
        super().__init__(name=name, boundary_type=boundary_type, value=value, marker=marker, **kwargs)
        self.is_free = is_free
        self.control_bry_cell: BoundaryCell


class ShapeRegularizationItem(object):
    def __init__(self, name: str, style: Literal['Volume', 'SurfaceArea', 'Barycenter'], weight: float, **kwargs):
        self.name = name
        self.style = style
        self.weight = weight
        self.kwargs = kwargs
        self.reg_form: ufl.Form
        self.reg_func: ShapeRegularization
    

class FluidShapeOptimizer(FluidSimulator):
    state_ksp_option = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    }
    adjoint_ksp_option = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
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
        nut_init: float,
        cost_item: FluidCost,
        regularization_item: ShapeRegularizationItem,
        with_vol_check: bool, vol_change: float,
        with_angle_check: bool, angle_tol_max: float, angle_tol_min: float,
        result_dir: str,
        state_ksp_option: dict = state_ksp_option,
        adjoint_ksp_option: dict = adjoint_ksp_option,
        gradient_ksp_option: dict = gradient_ksp_option,
        scalar_product_kwargs = {"method": "default"}
    ):
        super().__init__(name=name, mesh=mesh)
        
        self.boundaries: dict[str, FluidPipeBoundary] = {}
        
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
        
        self.fluid_type = fluid_type
        self.nut_init = nut_init
        self.cost_item = cost_item
        self.regularization_item = regularization_item
        self.scalar_product_kwargs = scalar_product_kwargs
               
        self.mesh.with_vol_check = with_vol_check
        self.mesh.vol_change = vol_change
        self.mesh.with_angle_check = with_angle_check
        self.mesh.angle_tol_max = angle_tol_max
        self.mesh.angle_tol_min = angle_tol_min
        
        self.result_dir = result_dir
        self.recorder = VtuRecorder(self.mesh, save_dir=self.result_dir, is_binary=True)
    
    def add_boundary(self, boundary: FluidPipeBoundary):
        self.boundaries[boundary.name] = boundary
    
    def define_cost(
        self, 
        cost_item: FluidCost, 
        u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed,
        **kwargs
    ):
        if cost_item.style == 'MinDissipation':
            cost_item.cost_form = kwargs['nut'] * inner(grad(u), grad(u)) * ufl.dx
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
        self, regularization_item: ShapeRegularizationItem, u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed
    ):
        if regularization_item.style == 'Volume':
            assert 'target_ratio' in regularization_item.kwargs, 'Target ratio is required for Volume regularization'
            regularization_item.reg_func = VolumeRegularization(
                name=f"{regularization_item.name}_{regularization_item.style}",
                shape_database=self.control_database,
                target_volume_rho=regularization_item.kwargs['target_ratio'],
                weight=regularization_item.weight
            )
        elif regularization_item.style == 'SurfaceArea':
            regularization_item.reg_func = SurfaceAreaRegularization(
                name=f"{regularization_item.name}_{regularization_item.style}",
                shape_database=self.control_database,
                target_surface_rho=regularization_item.kwargs['target_ratio'],
                weight=regularization_item.weight,
            )
        elif regularization_item.style == 'Barycenter':
            regularization_item.reg_func = BarycenterRegularization(
                name=f"{regularization_item.name}_{regularization_item.style}",
                shape_database=self.control_database,
                target_barycenter=regularization_item.kwargs['target_barycenter'],
                weight=regularization_item.weight,
            )
        else:
            raise ValueError(f'Invalid regularization style: {regularization_item.style}')
    
    def define_boundary(self):
        for name, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'velocity':
                
                inflow_velocity_exp = partial(
                    FluidBoundary.inflow_velocity_exp,
                    velocity=np.array(bry_info.value),
                    tdim=self.mesh.tdim,
                    face_center=bry_info.face_center,
                    face_norm_vector=bry_info.face_norm_vector,
                    face_radius=bry_info.face_radius
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
                    bc_value=dolfinx.fem.Function(self.V, name=name),
                    is_mixed_element=True,
                )

            else:
                raise ValueError('[Error]: Invalid Boundary Type.')

    def define_control_boundary(self):
        for name, bry_info in self.boundaries.items():
            if bry_info.is_free:
                continue
            
            bry_info.control_bry_cell = BoundaryCell.define_bry_cell(
                name=f'{name}_control',
                bc_func_space=self.point_vector_func_space,
                value_func_space=self.point_vector_func_space,
                dofs=MeshUtils.extract_entity_dofs(
                    self.point_vector_func_space, self.mesh.fdim,
                    MeshUtils.extract_entity_indices(
                        self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                    )
                ),
                bc_value=dolfinx.fem.Function(self.point_vector_func_space, name=f'{name}_control'),
                is_mixed_element=False,
            )

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

    def compute_gradient(
        self, 
        update_state: bool,
        update_adjoint: bool,
        state_kwargs: dict, 
        adjoint_kwargs: dict, 
        gradient_kwargs: dict, 
        scalar_product_kwargs: dict,
        with_norm=True
    ):
        report_dict = self.optimizer.compute_gradient(
            self.mesh.domain.comm,
            state_kwargs=state_kwargs,
            adjoint_kwargs=adjoint_kwargs,
            gradient_kwargs=gradient_kwargs,
            scalar_product_kwargs=scalar_product_kwargs,
            update_state=update_state,
            update_adjoint=update_adjoint
        )
        is_success = report_dict['is_success']
        if not is_success:
            return {'is_success': False, 'log': report_dict['log']}
        
        dire_np: np.ndarray = report_dict['shape_grad'].x.array.copy()
        dire_np = dire_np.reshape((-1, self.mesh.tdim))
        if with_norm:
            dire_np = dire_np / np.max(np.abs(np.linalg.norm(dire_np, ord=2, axis=1)))

        return {'is_success': True, 'direction': dire_np}
    
    def define_problem(self):
        uh = dolfinx.fem.Function(self.W, name=f"{self.name}_state")
        u, p = ufl.split(uh)  # type: ignore # please don't use up.split()
        vq = dolfinx.fem.Function(self.W, name=f"{self.name}_adjoint")
        v, q = ufl.split(vq)  # type: ignore # please don't use up.split()
        
        f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
        nut = dolfinx.fem.Function(self.point_scalar_func_space, name=f"{self.name}_nut")
        nut.x.array[:] = self.nut_init  # type: ignore
        
        govern_form = self.define_govern_equation(u, p, v, q, f, nut)  # type: ignore

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

        self.control_database = ShapeProblemDataBase(
            domain=self.mesh.domain,
            gradient_ksp_option=self.gradient_ksp_option,
        )

        self.define_control_boundary()
        for bry_info in self.boundaries.values():
            if bry_info.is_free:
                continue
            self.control_database.add_bc(bry_info.control_bry_cell)

        self.define_cost(self.cost_item, u, p, nut=nut)  # type: ignore
        self.define_regularization(self.regularization_item, u, p)  # type: ignore
        
        if self.scalar_product_kwargs['method'] == 'Poincare-Steklov operator':
            scalar_product_kwargs = {
                'method': 'Poincare-Steklov operator',
                'cell_tags': self.mesh.cell_tags,
                'facet_tags': self.mesh.facet_tags,
                'bry_free_markers': [bry_info.marker for bry_info in self.boundaries.values() if bry_info.is_free],
                'bry_fixed_markers': [bry_info.marker for bry_info in self.boundaries.values() if not bry_info.is_free],
                "lambda_lame": self.scalar_product_kwargs.get("lambda_lame", 1.0),
                "damping_factor": self.scalar_product_kwargs.get("damping_factor", 0.2),
                "mu_fix": self.scalar_product_kwargs.get("mu_fix", 1.0),
                "mu_free": self.scalar_product_kwargs.get("mu_free", 1.0),
                "use_inhomogeneous": self.scalar_product_kwargs.get("use_inhomogeneous", False),
                "inhomogeneous_exponent": self.scalar_product_kwargs.get("inhomogeneous_exponent", 1.0),
                "update_inhomogeneous": self.scalar_product_kwargs.get("update_inhomogeneous", False)
            }
        else:
            scalar_product_kwargs = self.scalar_product_kwargs
        
        self.optimizer = ShapeOptimizer(
            state_problem=StateProblem([self.state_database]),
            shape_database=self.control_database,
            cost_functional_list=[self.cost_item.cost_func],
            shape_regularization_list=ShapeRegularizationPipe([self.regularization_item.reg_func]),
            scalar_product_kwargs=scalar_product_kwargs
        )
        
        self.uh = uh  # type: ignore
        self.f = f  # type: ignore
        self.nut = nut  # type: ignore
    
    def run(
        self,
        line_search_steps: list[float] = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001],
        state_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        adjoint_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        gradient_kwargs: dict = {
            'mat_assemble_method': 'identity_row',
            'with_residual': False
        },
        scalar_product_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        max_iteration=100,
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
                scalar_product_kwargs=scalar_product_kwargs,
                with_norm=True,
            )
            is_success = report_dict['is_success']
            if not is_success:
                print(f'[Error]: {report_dict["log"]}')
                is_solved = False
                break

            direction_np = report_dict['direction']
            
            find_best_step = False
            last_step_index = np.where(line_search_steps_np == step_length_last)[0][0]
            line_search_start_index = np.maximum(last_step_index - 1, 0)
            for i in range(line_search_start_index, line_search_steps_np.shape[0]):
                step_length = line_search_steps_np[i]
                
                displacement_np = direction_np * step_length * -1.0             
                if displacement_np.shape[1] == 2:
                    displacement_np = np.concatenate([displacement_np, np.zeros((displacement_np.shape[0], 1))], axis=1)
                
                deform_check = self.mesh.deform_safe(displacement_np)
                if not deform_check['is_valid']:
                    print(deform_check['log'])
                    continue
                                
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
                    self.record_Upw(step)
                    break
                else:
                    self.mesh.deform(displacement_np * -1.0)  # recover mesh
                
            if not find_best_step:
                print(f"[Warning]: line search fail.")
                is_solved = False
                break
                
            if step > max_iteration:
                print(f"[Warning]: max iteration reached.")
                is_solved = False
                break
        
        return {'is_success': is_solved}
    
    def record_Upw(self, step: int):
        vel_ref = dolfinx.fem.Function(self.point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(self.uh.sub(0).collapse())  # type: ignore
        p_ref = self.uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

        self.recorder.update_point_data('U', vel_ref_np)
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        self.recorder.save(step)
    

def debug():
    msh_file = "/home/admin123456/Desktop/work/gmsh_test/test/t5/shape_fluid_05.msh"
    xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/test/t5/shape_fluid_05.xdmf"
    # MeshUtils.msh_to_XDMF(name='model', msh_file=msh_file, xdmf_file=xdmf_file, dim=2)

    # --- 1. define mesh ---
    mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")

    # --- 2. define cost and regularization ---
    cost_item = FluidCost(
        name='cost_1',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = ShapeRegularizationItem(
        name='regularization_1',
        style='Volume',
        weight=0.5,
        target_ratio=1.0
    )

    optim = FluidShapeOptimizer(
        name='shape_fluid_05',
        mesh=mesh,
        fluid_type='stoke',
        nut_init=1.0 / 400.0,
        cost_item=cost_item,
        regularization_item=regularization_item,
        with_vol_check=True,
        vol_change=0.25,
        with_angle_check=True,
        angle_tol_max=160.0,
        angle_tol_min=20.0,
        result_dir='/home/admin123456/Desktop/work/gmsh_test/test/t5/cache',
        scalar_product_kwargs={"method": "default"}
    )

    # --- 3. define boundary ---
    inlet_bry = FluidPipeBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.],
        marker=35,
        is_free=False,
        face_center=[0.0, 0.25],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.25
    )
    outlet_bry = FluidPipeBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=36,
        is_free=False
    )
    wall_bry = FluidPipeBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=37,
        is_free=True
    )

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)

    # --- 4. define problem ---
    optim.define_problem()

    # --- 5. run optimization ---
    optim.run(max_iteration=100)


if __name__ == '__main__':
    debug()

