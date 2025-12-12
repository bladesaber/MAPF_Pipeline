from typing import Literal, Union

import dolfinx
import ufl
from ufl import inner, grad, div
import ufl.tensors
import ufl.indexed

import numpy as np
from functools import partial
import pyvista

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.simulator.dolfin_simulator import FluidSimulator
from scripts_py.vtu_recorder import VtuRecorder
from scripts_py.eqn_solver import LinearProblemSolver

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape, stop_annotating

from scripts_py.mesh_utils import MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils, BoundaryUtils

from scripts_py.auto_dolfinx.derivative_regularization import SmoothDerivativeRegularization
from scripts_py.auto_dolfinx.derivative_regularization import StiffDerivativeRegularization

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.blocks.block_deform import deform, MeshDeformBlock
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidCost, AutoFluidRegularization, AutoFluidBoundary
from scripts_py.auto_dolfinx.derivative_regularization import draw_shape_derivative, draw_deform


class SimpleShapeFluidOptimizer(FluidSimulator):
    '''[Deprecated] gradient accumulate will cause sharp angle in mesh'''
    
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        nut_init: float,
        cost_item: AutoFluidCost,
        regularization_item: AutoFluidRegularization,
        result_dir: str,
        derivative_regularization_method: Literal['smooth', 'stiff'] = 'smooth',
    ):
        print("\033[91m [Warning]: SimpleShapeFluidOptimizer is deprecated, please use SmoothShapeFluidOptimizer instead. \033[0m")
        
        super().__init__(name, mesh)
        
        self.domain = DolfinxMesh(self.mesh.domain)
        self.mesh.domain = self.domain
        
        self.boundaries: dict[str, AutoFluidBoundary] = {}
        
        self.cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(self.mesh.domain)
        self.cell_vector_func_space = DolfinUtils.cell_vector_function_space(self.mesh.domain)
        self.point_scalar_func_space = DolfinUtils.point_scalar_function_space(self.mesh.domain)
        self.point_vector_func_space = DolfinUtils.point_vector_function_space(self.mesh.domain)
        
        coordinate_space = self.mesh.domain.ufl_domain().ufl_coordinate_element()
        self.deformation_space = dolfinx.fem.functionspace(self.domain, coordinate_space)
        
        self.surface_area: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.ds  # type: ignore
        ))
        self.volume: float = AssembleUtils.assemble_scalar(dolfinx.fem.form(  # type: ignore
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.dx  # type: ignore
        ))
        
        self.fluid_type = fluid_type
        self.nut_init = nut_init
        self.disp: DolfinxFunction = DolfinxFunction(self.deformation_space, name=f'disp_fun')  # type: ignore
        self.disp.x.array[:] = 0.0  # type: ignore
        
        self.cost_item = cost_item
        self.regularization_item = regularization_item
        
        self.result_dir = result_dir
        self.recorder = VtuRecorder(self.mesh, save_dir=self.result_dir, is_binary=True)
        
        self.control_phi: Control
        self.opt_problem: ReducedFunctional        
        self.deform_block: MeshDeformBlock
        
        # ------ define displacement boundary condition ------
        self.displacement_bcs: list[dolfinx.fem.DirichletBC] = []
        self.derivative_regularization_method = derivative_regularization_method
    
    def setup_tape(self, tape: Tape|None = None):
        if tape is None:
            self.tape = Tape()
        else:
            self.tape = tape
        set_working_tape(self.tape)

    def define_boundary(self):
        for name, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'velocity':
                
                inflow_velocity_exp = partial(
                    AutoFluidBoundary.inflow_velocity_exp,
                    velocity=np.array(bry_info.value),
                    tdim=self.mesh.tdim,
                    face_center=bry_info.face_center,
                    face_norm_vector=bry_info.face_norm_vector,
                    face_radius=bry_info.face_radius
                )
                
                bc_value_fun = dolfinx.fem.Function(self.V)
                bc_value_fun.interpolate(inflow_velocity_exp)  # type: ignore
                bry_info.bry_cell = DolfinxDirichletBC(
                    name=name,
                    bc_func_space=self.W0,
                    value_func_space=self.V,
                    dofs=MeshUtils.extract_entity_dofs(
                        (self.W0, self.V), self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    ),
                    bc_value=bc_value_fun,  # type: ignore
                    is_mixed_element=True
                )

            elif bry_info.boundary_type == 'pressure':
                pressure_func = dolfinx.fem.Function(self.Q, name=name)
                pressure_func.x.array[:] = bry_info.value  # type: ignore
                bry_info.bry_cell = DolfinxDirichletBC(
                    name=name,
                    bc_func_space=self.W1,
                    value_func_space=self.Q,
                    dofs=MeshUtils.extract_entity_dofs(
                        (self.W1, self.Q), self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    ),
                    bc_value=pressure_func,  # type: ignore
                    is_mixed_element=True
                )

            elif bry_info.boundary_type == 'wall':
                wall_func = dolfinx.fem.Function(self.V, name=name)
                
                bry_info.bry_cell = DolfinxDirichletBC(
                    name=name,
                    bc_func_space=self.W0,
                    value_func_space=self.V,
                    dofs=MeshUtils.extract_entity_dofs(
                        (self.W0, self.V), self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    ),
                    bc_value=wall_func,  # type: ignore
                    is_mixed_element=True
                )
            
            else:
                raise ValueError('[Error]: Invalid Boundary Type.')

        # ------ define displacement boundary condition ------
        for bry_info in self.boundaries.values():
            if not bry_info.is_free:
                fix_displacement = dolfinx.fem.Function(self.deformation_space, name=f'fix_displacement_{bry_info.name}')
                fix_displacement.x.array[:] = 0.0  # type: ignore
                
                self.displacement_bcs.append(
                    DolfinxDirichletBC(
                        name=f'fix_displacement_{bry_info.name}',
                        bc_func_space=self.deformation_space,
                        value_func_space=self.deformation_space,
                        dofs=MeshUtils.extract_entity_dofs(
                            self.deformation_space, self.mesh.fdim,
                            MeshUtils.extract_entity_indices(
                                self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                            )
                        ),
                        bc_value=fix_displacement,  # type: ignore
                        is_mixed_element=False
                    )
                )

    def define_govern_equation(
        self, 
        u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed,
        v: ufl.tensors.ListTensor, q: ufl.indexed.Indexed,
        f: dolfinx.fem.Function,
        nut: dolfinx.fem.Function,
    ):
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

    def define_cost(
        self, 
        cost_item: AutoFluidCost, 
        u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed, 
        **kwargs
    ):
        if cost_item.style == 'MinDissipation':
            assert 'nut' in kwargs, 'Viscosity is required for MinDissipation cost'
            cost_item.cost_form = cost_item.weight * kwargs['nut'] * inner(grad(u), grad(u)) * ufl.dx
            
        elif cost_item.style == 'MinPressure':
            assert 'marker' in cost_item.kwargs, 'Marker is required for MinPressure cost'
            area = AssembleUtils.assemble_scalar(
                dolfinx.fem.Constant(self.mesh.domain, 1.0) * self.ds(cost_item.kwargs['marker'])  # type: ignore
            )
            cost_item.cost_form = cost_item.weight * (1.0 / area) * p * self.ds(cost_item.kwargs['marker'])
            
        elif cost_item.style == 'MaxFlux':
            assert 'marker' in cost_item.kwargs, 'Marker is required for MaxFlux cost'
            cost_item.cost_form = cost_item.weight * ufl.dot(u, self.n_vec) * self.ds(cost_item.kwargs['marker'])  # type: ignore
            
        else:
            raise ValueError(f'Invalid cost style: {cost_item.style}')

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
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        forward_kwargs: dict = {
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
            
            self.define_boundary()
            
            _, self.deform_block = deform(self.domain, self.disp, tape=self.tape, with_block=True)  # type: ignore
            
            uh = equation_solve(
                domain=self.domain,
                F_form=govern_form,
                state=uh,  # type: ignore
                adjoint=vq,  # type: ignore
                bcs=[bry_info.bry_cell for bry_info in self.boundaries.values()],
                is_linear=True if self.fluid_type == 'stoke' else False,
                tlm_ksp_option=tlm_ksp_option,
                adj_ksp_option=adj_ksp_option,
                forward_ksp_option=forward_ksp_option,
                tlm_kwargs=tlm_kwargs,
                adj_kwargs=adj_kwargs,
                forward_kwargs=forward_kwargs,
                tape=self.tape,
            )
            
            self.uh = uh  # type: ignore
            self.f = f  # type: ignore
            self.nut = nut  # type: ignore
            
            u, p = ufl.split(self.uh)  # type: ignore
            self.define_cost(self.cost_item, u, p, nut=nut)  # type: ignore
            object_cost_ = assemble_scalar(self.cost_item.cost_form, domain=self.domain, tape=self.tape)  # type: ignore
            
            if self.regularization_item.style == 'Volume':
                if 'target_ratio' in self.regularization_item.kwargs:
                    target_volume = self.volume * self.regularization_item.kwargs['target_ratio']
                else:
                    assert 'target_volume' in self.regularization_item.kwargs, 'Target volume is required for Volume regularization'
                    target_volume = self.regularization_item.kwargs['target_volume']
                
                phi = dolfinx.fem.Constant(self.domain, 1.0)  # type: ignore    
                volume_ = assemble_scalar(phi * ufl.dx, self.domain, self.tape)
                regularization_cost_ = self.regularization_item.weight * (volume_ - target_volume) ** 2
                            
                def eval():
                    return self.regularization_item.weight * (AssembleUtils.assemble_scalar(phi * ufl.dx) - target_volume) ** 2

                self.regularization_item.eval = eval
            
            Jhat = object_cost_ + regularization_cost_
                        
            self.control_displacement = Control(
                self.disp,
                # riesz_map={'riesz_representation': 'L2', 'comm': self.domain.comm}
            )
            self.opt_problem = ReducedFunctional(Jhat, [self.control_displacement])
    
    def eval(self):
        return {
            self.cost_item.name: self.cost_item.eval(),
            self.regularization_item.name: self.regularization_item.eval(),
        }
    
    def compute_gradient(self, adj_input: float, with_norm=True) -> np.ndarray:
        dJ_dmesh = self.opt_problem.derivative(
            adj_input=float(adj_input),
            apply_riesz=False
        )[0]
        
        # draw_shape_derivative(self.mesh.grid, dJ_dmesh.reshape((-1, self.mesh.tdim)), scale=0.25, title='before_derivative')
        
        # apply control boundary to dJ_dmesh
        if self.derivative_regularization_method == 'smooth':
            derivative_regularization = SmoothDerivativeRegularization(
                name='derivative_regularization',
                mesh=self.mesh,
                boundaries=self.boundaries,
                mat_assemble_method='lift'
            )
            derivative_regularization.compute_mu()
            dJ_dmesh = derivative_regularization.solve(dJ_dmesh)
            
        elif self.derivative_regularization_method == 'stiff':
            derivative_regularization = StiffDerivativeRegularization(
                name='derivative_regularization',
                mesh=self.mesh,
                boundaries=self.boundaries,
                mat_assemble_method='lift'
            )
            derivative_regularization.compute_mu()
            dJ_dmesh = derivative_regularization.solve(dJ_dmesh)
        
        else:
            raise ValueError(f'Invalid derivative regularization method: {self.derivative_regularization_method}')
        
        # draw_shape_derivative(self.mesh.grid, dJ_dmesh.reshape((-1, self.mesh.tdim)), scale=0.25, title='after_derivative')
                
        dJ_dmesh = dJ_dmesh.reshape((-1, self.mesh.tdim))
        if with_norm:
            dJ_dmesh = dJ_dmesh / np.max(np.abs(np.linalg.norm(dJ_dmesh, ord=2, axis=1)))
        
        return dJ_dmesh

    def recover_deform_block_mesh(self, mesh_points: np.ndarray):
        # TODO: need to check whether it recover mesh points correctly
        for block_variable in self.deform_block.get_dependencies():
            if isinstance(block_variable.output, dolfinx.mesh.Mesh):
                block_variable.checkpoint = mesh_points
                c_ref = block_variable.saved_output
    
    def run(
        self,
        line_search_steps: list[float] = [0.1, 0.05, 0.01],
        max_iteration=200,
        with_line_search=True,
    ):
        # TODO OPTIMIZE: adding remesh tool may be better  
        
        is_solved, step = False, 0
        loss_current, loss_last = 0.0, 0.0
        line_search_steps_np: np.ndarray = np.array(line_search_steps)
        step_length_last = line_search_steps_np[0]
        
        displacement_fun = DolfinxFunction(self.deformation_space, name=f'displacement')  # type: ignore
        
        loss_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(step)
        print(f"step/{step} loss:{loss_current}.")
                        
        while True:
            step += 1
            
            if step > 1:
                loss_dict = self.eval()
                log = f"[step/{step} loss:{loss_current} last_step_length:{step_length_last:.3f}]"
                for name, cost_value in loss_dict.items():
                    log += f" {name}:{cost_value:.3f}"
                print(log)
                
                if with_line_search:
                    loss_residual = loss_current - loss_last
                    if np.abs(loss_residual) < 1e-6:
                        is_solved = True
                        break
            
            adj_input = np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2)
            direction_np = self.compute_gradient(adj_input=adj_input, with_norm=True)
            direction_np = direction_np.reshape(-1)
            
            if with_line_search:
                find_best_step = False
                displacement_origin: np.array = displacement_fun.x.array.copy()  # type: ignore
                mesh_points_origin = self.mesh.grid.points.copy()
                
                last_step_index = np.where(line_search_steps_np == step_length_last)[0][0]
                line_search_start_index = np.maximum(last_step_index - 1, 0)
                for i in range(line_search_start_index, line_search_steps_np.shape[0]):
                    step_length = line_search_steps_np[i]
                    
                    new_displacement = displacement_origin - direction_np * step_length
                    displacement_fun.x.array[:] = new_displacement  # type: ignore
                    
                    loss_new = self.opt_problem(values=[displacement_fun])  # type: ignore
                    if loss_new < loss_current:                    
                        find_best_step = True
                        loss_last = loss_current
                        loss_current = loss_new
                        step_length_last = step_length
                        self.record_Upw(step)
                        break
                    else:
                        # NOTE: if fail, please recover mesh points of deform block
                        self.recover_deform_block_mesh(mesh_points_origin)
                
                if not find_best_step:
                    print(f"[Warning]: line search fail.")
                    is_solved = False
                    break
                
            else:
                displacement_fun.x.array[:] = displacement_fun.x.array - direction_np * 0.01  # type: ignore
                loss_current = self.opt_problem(values=[displacement_fun])  # type: ignore
                self.record_Upw(step)
                
            if step > max_iteration:
                print(f"[Warning]: max iteration reached.")
                is_solved = False
                break
        
        return {'is_success': is_solved}
    
    def record_Upw(self, step: int, **kwargs):
        vel_ref = dolfinx.fem.Function(self.point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(self.uh.sub(0).collapse())  # type: ignore
        p_ref = self.uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

        self.recorder.update_point_data('U', vel_ref_np)
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        if kwargs.get('gradient', None) is not None:
            self.recorder.update_point_data('gradient', kwargs['gradient'])
        
        self.recorder.save(step)
    

def debug():
    xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t14/optimal_pipe_03.xdmf"
    mesh = MeshObj(xdmf_file=xdmf_file, mesh_name="model")
    
    cost_item = AutoFluidCost(
        name='cost_1',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='regularization_1',
        style='Volume',
        weight=1.0,
        target_ratio=1.0
    )
    
    optim = SimpleShapeFluidOptimizer(
        name='shape_fluid_13',
        mesh=mesh,
        fluid_type='stoke',
        nut_init=1.0 / 400.0,
        cost_item=cost_item,
        regularization_item=regularization_item,
        result_dir='/home/admin123456/Desktop/work/gmsh_test/t14/cache',
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[6.0, 0.],
        marker=14,
        is_free=False,
        face_center=[0.0, 0.5],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.5
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=15,
        is_free=False
    )
    wall_fix_bry = AutoFluidBoundary(
        name='wall_fix_bry',
        boundary_type='wall',
        value=0.0,
        marker=16,
        is_free=False
    )
    wall_free_bry = AutoFluidBoundary(
        name='wall_free_bry',
        boundary_type='wall',
        value=0.0,
        marker=17,
        is_free=True
    )
    
    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_fix_bry)
    optim.add_boundary(wall_free_bry)

    optim.define_problem()
    
    # optim.compute_gradient(adj_input=1.0, with_norm=True)
    optim.run(max_iteration=200, with_line_search=True, line_search_steps=[0.01])


if __name__ == '__main__':
    debug()
    