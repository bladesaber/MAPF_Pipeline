from typing import Literal, Union
import os

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
from scripts_py.vtu_recorder import VtuRecorder, TensorBoardRecorder
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

from scripts_py.auto_dolfinx.opt_solver.topo_fluid import AutoFluidBoundary
from scripts_py.auto_dolfinx.derivative_regularization import draw_shape_derivative, draw_deform


class SimpleShapeFluidOptimizer(FluidSimulator):
    '''[Deprecated] gradient accumulate will cause sharp angle in mesh'''
    
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float,
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
        self.viscosity_init = viscosity_init
        self.disp: DolfinxFunction = DolfinxFunction(self.deformation_space, name=f'disp_fun')  # type: ignore
        self.disp.x.array[:] = 0.0  # type: ignore
        
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
            assert bry_info.boundary_type in ['velocity', 'pressure', 'wall'], 'Invalid boundary type'
               
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

    def get_bcs(self):
        """set wall boundary at last"""
        return [bry_info.bry_cell for bry_info in sorted(self.boundaries.values(), key=lambda x: x.priority)]

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

    def prepare_cost_args(self, u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed, obj_cost_args: dict, **kwargs):
        cost_parse_dict = {}
        
        if obj_cost_args['style'] == 'MinDissipation':
            with_viscosity = obj_cost_args['with_viscosity']
            if with_viscosity:
                min_dissipation_form = kwargs['viscosity'] * inner(grad(u), grad(u)) * ufl.dx
            else:
                min_dissipation_form = inner(grad(u), grad(u)) * ufl.dx
            cost_parse_dict.update({
                'min_dissipation_form': min_dissipation_form
            })
            
        elif obj_cost_args['style'] == 'MinPressure':
            inlet_marker: int = obj_cost_args['inlet_marker']
            face_area_form = dolfinx.fem.Constant(self.mesh.domain, 1.0) * self.ds(inlet_marker)  # type: ignore
            face_area = AssembleUtils.assemble_scalar(face_area_form)
            cost_parse_dict.update({
                'face_area_form': face_area_form,
                'face_area': face_area,
                'pressure_form': (1.0 / face_area) * p * self.ds(inlet_marker)
            })
            
        elif obj_cost_args['style'] == 'FixedFlux':
            inlet_markers: list[int] = obj_cost_args['inlet_markers']
            inlet_flux = 0.0
            for marker in inlet_markers:
                inlet_flux += AssembleUtils.assemble_scalar(ufl.dot(u, self.n_vec) * self.ds(marker))
            inlet_flux = inlet_flux * -1.0  # TODO: should I multiply -1.0?
            
            outlet_markers: list[int] = obj_cost_args['outlet_markers']
            assert len(outlet_markers) > 1, 'At least two outlet marker is required for FixedFlux cost'
            outlet_marker_ratios = obj_cost_args['outlet_marker_ratios']
            assert np.sum(outlet_marker_ratios) == 1.0, 'Outlet marker ratios must sum to 1.0'
            outlet_marker_weights = obj_cost_args['outlet_marker_weights']
            
            outlet_cost_forms = {}
            for marker, ratio, weight in zip(outlet_markers, outlet_marker_ratios, outlet_marker_weights):
                target_flux = inlet_flux * ratio
                outlet_flux_form = ufl.dot(u, self.n_vec) * self.ds(marker)
                outlet_cost_forms[marker] = {
                    'flux_form': outlet_flux_form,
                    'target_flux': target_flux,
                    'weight': weight,
                }
            cost_parse_dict.update({
                'inlet_flux': inlet_flux,
                'outlet_cost_forms': outlet_cost_forms,
            })
        
        else:
            raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')

        obj_cost_args.update(cost_parse_dict)
        
        return obj_cost_args

    def prepare_reg_args(self, reg_cost_args: dict):
        reg_parse_dict = {}

        if reg_cost_args['style'] == 'Volume':
            if 'target_ratio' in reg_cost_args.keys():
                target_volume = self.volume * reg_cost_args['target_ratio']
            else:
                target_volume = reg_cost_args['target_volume']
            
            reg_parse_dict.update({
                'target_volume': target_volume,
                'vol_form': dolfinx.fem.Constant(self.domain, 1.0) * ufl.dx  # type: ignore
            })
        
        reg_cost_args.update(reg_parse_dict)
        return reg_cost_args

    def define_problem(
        self,
        obj_cost_args: dict,
        reg_cost_args: dict,
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
        self.define_boundary()
        
        with start_annotation():
            uh = DolfinxFunction(self.W, name=f"{self.name}_state")
            u, p = ufl.split(uh)  # type: ignore # please don't use up.split()
            vq = DolfinxFunction(self.W, name=f"{self.name}_adjoint")
            v, q = ufl.split(vq)  # type: ignore # please don't use up.split()
            
            f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
            self.viscosity = DolfinxFunction(self.point_scalar_func_space, name=f"{self.name}_viscosity")
            self.viscosity.x.array[:] = self.viscosity_init  # type: ignore
                    
            govern_form = self.define_govern_equation(u, p, v, q, f, self.viscosity)  # type: ignore
            
            _, self.deform_block = deform(self.domain, self.disp, tape=self.tape, with_block=True)  # type: ignore
            
            uh = equation_solve(
                domain=self.domain,
                F_form=govern_form,
                state=uh,  # type: ignore
                adjoint=vq,  # type: ignore
                bcs=self.get_bcs(),  # type: ignore
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
            u, p = ufl.split(self.uh)  # type: ignore
            
            # ------ define object cost ------
            obj_cost_args = self.prepare_cost_args(
                u=u, p=p, obj_cost_args=obj_cost_args,  # type: ignore
                viscosity=self.viscosity,
            )
            if obj_cost_args['style'] == 'MinDissipation':
                obj_cost = assemble_scalar(
                    form=obj_cost_args['weight'] * obj_cost_args['min_dissipation_form'], 
                    domain=self.domain, tape=self.tape
                )
            elif obj_cost_args['style'] == 'MinPressure':
                obj_cost = assemble_scalar(
                    form=obj_cost_args['weight'] * obj_cost_args['pressure_form'],
                    domain=self.domain, tape=self.tape
                )
            elif obj_cost_args['style'] == 'FixedFlux':
                for i, (_, outlet_item) in enumerate(obj_cost_args['outlet_cost_forms'].items()):
                    outlet_flux = assemble_scalar(
                        form=outlet_item['flux_form'],
                        domain=self.domain, tape=self.tape
                    )
                    target_flux = outlet_item['target_flux']
                    weight = outlet_item['weight']
                    if i == 0:
                        obj_cost = weight * (outlet_flux - target_flux) ** 2
                    else:
                        obj_cost += weight * (outlet_flux - target_flux) ** 2
            else:
                raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')
            self.obj_cost_args = obj_cost_args
            
            # ------ define regularization cost ------
            reg_cost_args = self.prepare_reg_args(reg_cost_args)  # type: ignore
            if reg_cost_args['style'] == 'Volume':
                target_volume = reg_cost_args['target_volume']
                topo_volume = assemble_scalar(reg_cost_args['vol_form'], domain=self.domain, tape=self.tape)
                volume_dif = target_volume - topo_volume
                if reg_cost_args['reg_rescale']:
                    volume_dif = volume_dif / target_volume
                reg_cost = reg_cost_args['weight'] * (volume_dif ** 2)
            else:
                raise ValueError(f'Invalid regularization style: {reg_cost_args['style']}')
            self.reg_cost_args = reg_cost_args
            
            loss_all = obj_cost + reg_cost
            
            # ------ define optimization problem ------
            self.control_displacement = Control(
                self.disp,
                # riesz_map={'riesz_representation': 'L2', 'comm': self.domain.comm}
            )
            self.opt_problem = ReducedFunctional(loss_all, [self.control_displacement])
    
    def eval(self):
        eval_dict = {}
        log_dict = {}
        
        # ------ compute object cost ------
        if self.obj_cost_args['style'] == 'MinDissipation':
            eval_dict['obj_cost'] = AssembleUtils.assemble_scalar(
                form=self.obj_cost_args['weight'] * self.obj_cost_args['min_dissipation_form']
            )
            log_dict['obj_cost'] = eval_dict['obj_cost']
                        
        elif self.obj_cost_args['style'] == 'MinPressure':
            eval_dict['obj_cost'] = AssembleUtils.assemble_scalar(
                form=self.obj_cost_args['weight'] * self.obj_cost_args['pressure_form']
            )
            log_dict['obj_cost'] = eval_dict['obj_cost']
            
        elif self.obj_cost_args['style'] == 'FixedFlux':
            flux_dict = {}
            for i, (_, outlet_item) in enumerate(self.obj_cost_args['outlet_cost_forms'].items()):
                outlet_flux = AssembleUtils.assemble_scalar(form=outlet_item['flux_form'])
                eval_dict[f'{i}_cost'] = outlet_item['weight'] * (outlet_flux - outlet_item['target_flux']) ** 2
                flux_dict[f'{i}_flux'] = np.abs(outlet_flux)
            log_dict['flux'] = flux_dict
        else:
            raise ValueError(f'Invalid cost style: {self.obj_cost_args['style']}')
        
        # ------ compute regularization cost ------
        if self.reg_cost_args['style'] == 'Volume':
            target_volume = self.reg_cost_args['target_volume']
            topo_volume = AssembleUtils.assemble_scalar(self.reg_cost_args['vol_form'])
            volume_dif = target_volume - topo_volume
            if self.reg_cost_args['reg_rescale']:
                volume_dif = volume_dif / target_volume
            eval_dict['reg_cost'] = self.reg_cost_args['weight'] * (volume_dif ** 2)
            
            log_dict['reg_cost'] = eval_dict['reg_cost']
            log_dict['volume'] = topo_volume
            
        else:
            raise ValueError(f'Invalid regularization style: {self.reg_cost_args['style']}')
        
        log_dict['loss_all'] = np.sum(list(eval_dict.values()))
                        
        return eval_dict, log_dict
    
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
        for block_variable in self.deform_block.get_dependencies():
            if isinstance(block_variable.output, dolfinx.mesh.Mesh):
                block_variable.checkpoint = mesh_points
                c_ref = block_variable.saved_output
    
    def run(
        self,
        line_search_steps: list[float] = [0.1, 0.05, 0.01],
        max_iteration=200,
        full_run: bool = False,
        recorder: TensorBoardRecorder|None = None,
    ):
        with_line_search = len(line_search_steps) > 1
        line_search_steps_np: np.ndarray = np.array(line_search_steps)
        step_length_last = line_search_steps_np[0]
        if full_run:
            assert not with_line_search, 'Full run and line search cannot be used together'
        
        is_solved, step = False, 0
        loss_current, loss_last = 0.0, 0.0
        
        displacement_fun = DolfinxFunction(self.deformation_space, name=f'displacement')  # type: ignore
        
        loss_dict, log_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(step)
        print(f"step/{step} loss:{loss_current}.")
                        
        while True:
            step += 1
            if step > 1:
                loss_dict, log_dict = self.eval()
                log = f"[step/{step} loss:{loss_current:.5f} opt_length:{step_length_last:.5f}]"
                for name, cost_value in loss_dict.items():
                    log += f" {name}:{cost_value:.5f}"
                print(log)
                
                if recorder is not None:
                    for name, value in log_dict.items():
                        if isinstance(value, dict):
                            recorder.write_scalars(tag=name, scalar_dict=value, step=step)
                        else:
                            recorder.write_scalar(tag=name, scalar_value=value, step=step)
                
                if np.abs(loss_current - loss_last) < 1e-8 and (not full_run):
                    is_solved = True
                    break
            
            direction_np = self.compute_gradient(
                adj_input=np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2), 
                with_norm=True
            )
            direction_np = direction_np.reshape(-1)
            
            find_best_step = False
            if with_line_search:
                displacement_origin: np.ndarray = displacement_fun.x.array.copy()  # type: ignore
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
                
            else:
                displacement_fun.x.array[:] = displacement_fun.x.array - direction_np * step_length_last  # type: ignore
                loss_new = self.opt_problem(values=[displacement_fun])  # type: ignore
                if loss_new < loss_current or full_run:
                    find_best_step = True
                    loss_last = loss_current
                    loss_current = loss_new
                    self.record_Upw(step)
            
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
        
        self.recorder.clear(clear_point_data=True, clear_cell_data=True)
        self.recorder.update_point_data('vel', vel_ref_np)
        self.recorder.update_point_data('vel_mag', np.linalg.norm(vel_ref_np, axis=1))
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        
        self.recorder.save(step)


def debug():
    base_dir = "/home/admin123456/Desktop/work/gmsh_test/t8"
    msh_file = os.path.join(base_dir, 'model.msh')

    domain, cell_tags, facet_tags = MeshUtils.read_msh(msh_file, dim=2)
    mesh = MeshObj(xdmf_file=None, domain=domain, cell_tags=cell_tags, facet_tags=facet_tags)
    
    optim = SimpleShapeFluidOptimizer(
        name='shape_fluid_13',
        mesh=mesh,
        fluid_type='stoke',
        viscosity_init=1.0 / 400.0,
        result_dir=os.path.join(base_dir, 'dolfinx_shape'),
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

    optim.define_problem(
        obj_cost_args={
            'style': 'MinDissipation',
            'with_viscosity': True,
            'weight': 1.0,
        },
        reg_cost_args={
            'style': 'Volume',
            'weight': 1.0,
            'target_ratio': 0.9,
            'reg_rescale': False,
        },
    )
    optim.run(max_iteration=200, line_search_steps=[0.05])


if __name__ == '__main__':
    debug()
    