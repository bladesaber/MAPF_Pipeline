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

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve, EquationSolveBlock
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.blocks.block_deform import deform
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from scripts_py.auto_dolfinx.levelset_topo.topo_fluid import AutoFluidCost, AutoFluidRegularization, AutoFluidBoundary
from scripts_py.auto_dolfinx.derivative_regularization import draw_shape_derivative, draw_deform


class SmoothShapeFluidOptimizer(FluidSimulator):
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        nut_init: float,
        cost_item: AutoFluidCost,
        regularization_item: AutoFluidRegularization,
        result_dir: str,
        mu_max: float = 500.0,
        mu_min: float = 1.0,
    ):
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
        self.smooth_displacement_block: EquationSolveBlock
        
        # ------ define mu ------
        self.mu_max: float = mu_max
        self.mu_min: float = mu_min
        self.mu_bcs: list[dolfinx.fem.DirichletBC] = []
        
        # ------ define displacement boundary condition ------
        self.displacement_bcs: list[dolfinx.fem.DirichletBC] = []
    
    def setup_tape(self, tape: Tape|None = None):
        if tape is None:
            self.tape = Tape()
        else:
            self.tape = tape
        set_working_tape(self.tape)

    def update_smooth_displacement_block(self, update_mesh_points: np.ndarray):
        # NOTE: necessary for update mesh after each optimization step
        for block_variable in self.smooth_displacement_block.get_dependencies():
            coeff = block_variable.output
            if isinstance(coeff, dolfinx.mesh.Mesh):
                block_variable.checkpoint = update_mesh_points  # change checkpoint of dependency
                c_ref = block_variable.saved_output  # update block variable

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
        
        # ------ define mu boundary condition ------
        for bry_info in self.boundaries.values():
            if bry_info.is_free:
                self.mu_bcs.append(
                    BoundaryUtils.define_dirichlet_bc(
                        bc_func_space=self.point_scalar_func_space,
                        value_func_space=self.point_scalar_func_space,
                        dofs=MeshUtils.extract_entity_dofs(
                            self.point_scalar_func_space, self.mesh.fdim,
                            MeshUtils.extract_entity_indices(
                                self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                            )
                        ),
                        bc_value=self.mu_max,  # type: ignore
                        is_mixed_element=False
                    )
                )
                    
            else:
                self.mu_bcs.append(
                    BoundaryUtils.define_dirichlet_bc(
                        bc_func_space=self.point_scalar_func_space,
                        value_func_space=self.point_scalar_func_space,
                        dofs=MeshUtils.extract_entity_dofs(
                            self.point_scalar_func_space, self.mesh.fdim,
                            MeshUtils.extract_entity_indices(
                                self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                            )
                        ),
                        bc_value=self.mu_min,  # type: ignore
                        is_mixed_element=False
                    )
                )

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

    def smooth_deform(
        self, 
        disp_func: DolfinxFunction,
        disp_ksp_option: dict,
    ):
        mu = DolfinxFunction(self.point_scalar_func_space, name='mu')  # type: ignore
        
        # TODOL should i need mu as a block ?
        with stop_annotating():
            mu_trial = ufl.TrialFunction(self.point_scalar_func_space)
            mu_test = ufl.TestFunction(self.point_scalar_func_space)
            
            mu_a_form = -inner(grad(mu_trial), grad(mu_test)) * ufl.dx  # type: ignore
            mu_L_form = dolfinx.fem.Constant(self.domain, 0.0) * mu_test * ufl.dx  # type: ignore
                    
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=self.domain.comm,
                uh=mu,  # type: ignore
                a_form=mu_a_form,
                l_form=mu_L_form,
                bcs=self.mu_bcs,
                ksp_option={
                    'ksp_type': 'preonly',
                    'pc_type': 'lu',
                    "pc_factor_mat_solver_type": "mumps",
                    "mat_mumps_icntl_24": 1
                },
                mat_assemble_method='lift',
                with_residual=False,
            )
            if not res_dict['is_solved']:
                raise ValueError('Failed to solve mu problem.')
        
            # # ------ debug ------
            # self.mesh.grid.point_data['mu'] = mu.x.array
            # self.mesh.grid.save('/home/admin123456/Desktop/work/gmsh_test/t13/mu.vtu', binary=True)
            # # -------------------
        
        displacement = DolfinxFunction(self.deformation_space, name='displacement')
        displacement_v = DolfinxFunction(self.deformation_space, name='displacement_test')
        
        '''
        # TODO: why it fail ?
        # ------ Stiff elastic equation ------
        def epsilon(u):
            return ufl.sym(grad(u))

        def sigma(u, mu, lmb=0):
            return 2 * mu * epsilon(u) + lmb * ufl.tr(epsilon(u)) * ufl.Identity(2)  # type: ignore
        
        disp_a_form = inner(sigma(displacement, mu=mu), grad(displacement_v)) * ufl.dx
        disp_L_form = 0.0
        for bry_info in self.boundaries.values():
            if bry_info.is_free:
                disp_L_form += inner(disp_func, displacement_v) * self.ds(bry_info.marker)  # type: ignore
        # ----------------------------------------
        '''
        # ------ H1 riesz representation smooth ------
        disp_a_form = (
            mu * inner(grad(displacement), grad(displacement_v)) * ufl.dx  # type: ignore
            + mu * inner(displacement, displacement_v) * ufl.dx  # type: ignore
        )
        disp_L_form = 0.0
        for bry_info in self.boundaries.values():
            if bry_info.is_free:
                # TODO: do i need mu weight here ?
                disp_L_form += inner(disp_func, displacement_v) * self.ds(bry_info.marker)  # type: ignore
        # ----------------------------------------
             
        displacement, self.smooth_displacement_block = equation_solve(
            domain=self.domain,
            F_form=disp_a_form - disp_L_form,
            state=displacement,  # type: ignore
            adjoint=displacement_v,  # type: ignore
            bcs=self.displacement_bcs,
            is_linear=True,
            forward_ksp_option={
                'ksp_setting': disp_ksp_option,
                'snes_setting': {}
            },
            forward_kwargs={
                'mat_assemble_method': 'lift',
                'with_residual': False
            },
            tape=self.tape,
            with_block=True,
        )
        
        return displacement

    def define_problem(
        self,
        displacement_ksp_option: dict = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1
        },
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
            
            displacement = self.smooth_deform(self.disp, displacement_ksp_option)
            deform(self.domain, displacement, tape=self.tape)  # type: ignore
            
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
        dJ_dmesh = self.opt_problem.derivative(adj_input=float(adj_input))[0]
        dJ_dmesh = dJ_dmesh.reshape((-1, self.mesh.tdim))
        if with_norm:
            dJ_dmesh = dJ_dmesh / np.max(np.abs(np.linalg.norm(dJ_dmesh, ord=2, axis=1)))
        # draw_shape_derivative(self.mesh.grid, dJ_dmesh, scale=0.1, title='shape_derivative')
        return dJ_dmesh
    
    def run(
        self,
        line_search_steps: list[float] = [3.0, 1.5, 0.75],
        max_iteration=200,
    ):
        # TODO OPTIMIZE: adding remesh tool may be better  
        
        is_solved, step = False, 0
        loss_current, loss_last = 0.0, 0.0
        line_search_steps_np: np.ndarray = np.array(line_search_steps)
        step_length_last = line_search_steps_np[0]
        
        disp = DolfinxFunction(self.deformation_space, name=f'displacement')  # type: ignore
        
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
                
                loss_residual = loss_current - loss_last
                if np.abs(loss_residual) < 1e-6:
                    is_solved = True
                    break
            
            adj_input = np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2)
            disp_dire_np = self.compute_gradient(adj_input=adj_input, with_norm=True)
            disp_dire_np = disp_dire_np.reshape(-1)
            
            find_best_step = False            
            last_step_index = np.where(line_search_steps_np == step_length_last)[0][0]
            line_search_start_index = np.maximum(last_step_index - 1, 0)
            for i in range(line_search_start_index, line_search_steps_np.shape[0]):
                step_length = line_search_steps_np[i]
                                
                disp.x.array[:] = disp_dire_np * step_length * -1.0  # type: ignore
                loss_new = self.opt_problem(values=[disp])  # type: ignore
                if loss_new < loss_current:
                    # NOTE: be careful to update dependency mesh of smooth_displacement_block
                    self.update_smooth_displacement_block(self.mesh.grid.points.copy())
                    
                    find_best_step = True
                    loss_last = loss_current
                    loss_current = loss_new
                    step_length_last = step_length
                    self.record_Upw(step)
                    break
            
            if not find_best_step:
                print(f"[Warning]: line search fail.")
                is_solved = False
                break
                
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
    xdmf_file = "/home/admin123456/Desktop/work/gmsh_test/t13/optimal_pipe_03.xdmf"
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
    
    optim = SmoothShapeFluidOptimizer(
        name='shape_fluid_13',
        mesh=mesh,
        fluid_type='stoke',
        nut_init=1.0 / 400.0,
        cost_item=cost_item,
        regularization_item=regularization_item,
        result_dir='/home/admin123456/Desktop/work/gmsh_test/t13/cache',
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
    optim.run(max_iteration=200)


if __name__ == '__main__':
    debug()
    