from typing import Literal, Union

import dolfinx
import ufl
from ufl import inner, grad, div
import ufl.tensors
import ufl.indexed

import numpy as np
from functools import partial

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.simulator.dolfin_simulator import FluidBoundary, FluidSimulator
from scripts_py.vtu_recorder import VtuRecorder, TensorBoardRecorder

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control, compute_gradient
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape

from scripts_py.mesh_utils import MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from scripts_py.dolfinx_adjoint.topology_utils.topology_filter import DistanceBandFilter


class AutoFluidCost(object):
    def __init__(
        self, 
        name: str, 
        style: Literal['MinDissipation', 'MinPressure', 'MaxFlux'], 
        weight: float,
        **kwargs
    ):
        self.name = name
        self.style = style
        self.weight = weight
        self.kwargs = kwargs
        self.cost_form: ufl.Form
    
    def eval(self) -> float:
        return AssembleUtils.assemble_scalar(self.cost_form)


class AutoFluidRegularization(object):
    def __init__(
        self, 
        name: str, 
        style: str, 
        weight: float,
        **kwargs
    ):
        self.name = name
        self.style = style
        self.weight = weight
        self.kwargs = kwargs
        self.cost_form: ufl.Form
    
    def eval(self) -> float:
        return AssembleUtils.assemble_scalar(self.cost_form)


class AutoFluidBoundary(FluidBoundary):
    def __init__(
        self, 
        name: str, 
        boundary_type: Literal['velocity', 'pressure', 'wall'], 
        value: Union[float, list[float], np.ndarray], 
        marker: int, 
        is_free: bool|None = None,
        **kwargs
    ):
        super().__init__(name, boundary_type, value, marker, **kwargs)
        self.bry_cell: DolfinxDirichletBC
        self.is_free = is_free
        # self.control_bry_cell: DolfinxDirichletBC


class TopoFluidOptimizer(FluidSimulator):
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        nut_init: float,
        cost_item: AutoFluidCost,
        regularization_item: AutoFluidRegularization,
        topo_weight_init: float|np.ndarray,
        weight_filter_radius: float,
        weight_filter_is_cell_field: bool,
        result_dir: str,
        alpha_fun_max: float = 1000.0,
        alpha_fun_min: float = 0.01,
        alpha_fun_p: float = 3.0,
        velocity_degree = 2,
        pressure_degree = 1,
        velocity_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        pressure_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
    ):
        super().__init__(
            name, 
            mesh, 
            velocity_degree=velocity_degree, 
            pressure_degree=pressure_degree,
            velocity_space=velocity_space, 
            pressure_space=pressure_space,
        )
        
        self.domain = DolfinxMesh(self.mesh.domain)
        self.mesh.domain = self.domain
        
        self.boundaries: dict[str, AutoFluidBoundary] = {}
        
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
        self.topo_weight_init = topo_weight_init
        self.alpha_fun_max = alpha_fun_max
        self.alpha_fun_min = alpha_fun_min
        self.alpha_fun_p = alpha_fun_p
        self.topo_weight: DolfinxFunction
        
        self.cost_item = cost_item
        self.regularization_item = regularization_item
        
        self.result_dir = result_dir
        self.recorder = VtuRecorder(self.mesh, save_dir=self.result_dir, is_binary=True)
        
        self.weight_filter_radius = weight_filter_radius
        self.weight_filter_is_cell_field = weight_filter_is_cell_field
        self.weight_filter = DistanceBandFilter(self.mesh.grid, weight_filter_radius, weight_filter_is_cell_field, True)
        
        self.control_phi: Control
        self.opt_problem: ReducedFunctional
        
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
                    face_center=bry_info.face_center, # type: ignore
                    face_norm_vector=bry_info.face_norm_vector, # type: ignore
                    face_radius=bry_info.face_radius, # type: ignore
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

    @staticmethod
    def alpha_fun(
        rho: np.ndarray|dolfinx.fem.Function, alpha_min=0.01, alpha_max=1000.0, p=3.0, pow_operator=ufl.elem_pow
    ):
        return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)  # type: ignore

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

    def define_cost(
        self, 
        cost_item: AutoFluidCost, 
        u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed, 
        alpha: ufl.Form, 
        **kwargs
    ):
        if cost_item.style == 'MinDissipation':
            assert 'nut' in kwargs, 'Viscosity is required for MinDissipation cost'
            cost_item.cost_form = cost_item.weight * (
                0.5 * alpha * inner(u, u) * ufl.dx + 
                kwargs['nut'] * inner(grad(u), grad(u)) * ufl.dx  # type: ignore
            )
            
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

    def define_regularization(self, regularization_item: AutoFluidRegularization):
        if regularization_item.style == 'Volume':
            assert 'target_ratio' in regularization_item.kwargs, 'Target ratio is required for Volume regularization'
            regularization_item.cost_form = regularization_item.weight * (
                (regularization_item.kwargs['target_ratio'] - self.topo_weight) ** 2
            ) * ufl.dx

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
            
            self.topo_weight = DolfinxFunction(self.point_scalar_func_space, name=f'topo_weight')  # type: ignore
            self.topo_weight.x.array[:] = self.topo_weight_init
            
            alpha = self.alpha_fun(
                rho=self.topo_weight,  # type: ignore
                alpha_min=self.alpha_fun_min, 
                alpha_max=self.alpha_fun_max, 
                p=self.alpha_fun_p, 
                pow_operator=np.power
            )
            govern_form += alpha * inner(u, v) * ufl.dx
            
            self.define_boundary()
            
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
            self.record_Upw(0)
            
            u, p = ufl.split(self.uh)  # type: ignore
            self.define_cost(self.cost_item, u, p, alpha, nut=nut)  # type: ignore
            self.define_regularization(self.regularization_item)
            J_form = self.cost_item.cost_form + self.regularization_item.cost_form  # type: ignore
            Jhat = assemble_scalar(J_form, domain=self.domain, tape=self.tape)  # type: ignore
            
            self.control_phi = Control(self.topo_weight)
            self.opt_problem = ReducedFunctional(Jhat, [self.control_phi])
    
    def compute_gradient(self, adj_input: float, with_filter=True, with_norm=True) -> np.ndarray:
        dJdm = self.opt_problem.derivative(adj_input=float(adj_input))[0]
        if with_filter:
            dJdm = dJdm.reshape((-1, 1))
            dJdm = self.weight_filter.run(dJdm)
            dJdm = np.asarray(dJdm).reshape((-1))
        if with_norm:
            norm = np.max(np.abs(dJdm))
            dJdm = dJdm / norm
        return dJdm
    
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
        return {
            self.cost_item.name: self.cost_item.eval(),
            self.regularization_item.name: self.regularization_item.eval(),
        }
    
    def record_Upw(self, step: int, **kwargs):
        vel_ref = dolfinx.fem.Function(self.point_vector_func_space, name='vel_ref')
        vel_ref.interpolate(self.uh.sub(0).collapse())  # type: ignore
        p_ref = self.uh.sub(1).collapse()  # type: ignore

        vel_ref_np = vel_ref.x.array.copy().reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)

        self.recorder.clear(clear_point_data=True, clear_cell_data=True)
        self.recorder.update_point_data('U', vel_ref_np)
        self.recorder.update_point_data('p', p_ref.x.array.copy())
        self.recorder.update_point_data('topo_weight', self.topo_weight.x.array.copy())  # type: ignore        
        if kwargs.get('gradient', None) is not None:
            self.recorder.update_point_data('gradient', kwargs['gradient'])
        
        self.recorder.save(step)
    

def debug():
    mesh_file = '/home/admin123456/Desktop/work/gmsh_test/t12/topo_fluid_04.msh'
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t12/topo_fluid_04.xdmf'
    # MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)

    # --- 1. define mesh ---
    mesh = MeshObj(xdmf_file=xdmf_file)
    
    cost_item = AutoFluidCost(
        name='cost_1',
        style='MinDissipation',
        weight=1.0,
    )
    regularization_item = AutoFluidRegularization(
        name='weight_1',
        style='Volume',
        weight=10.0,
        target_ratio=0.25
    )
    
    optim = TopoFluidOptimizer(
        name='topo_fluid_12',
        mesh=mesh,
        fluid_type='stoke',
        nut_init=1.0 / 100.,
        cost_item=cost_item,
        regularization_item=regularization_item,
        topo_weight_init=0.5,
        weight_filter_radius=0.2,
        weight_filter_is_cell_field=False,
        result_dir='/home/admin123456/Desktop/work/gmsh_test/t12/cache'
    )
    
    tape = Tape()
    optim.setup_tape(tape)
    
    inlet_bry = AutoFluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[1.0, 0.],
        marker=18,
        face_center=[0.0, 0.5],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.5,
    )
    outlet_bry = AutoFluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=19
    )
    wall_bry = AutoFluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=20
    )

    optim.add_boundary(inlet_bry)
    optim.add_boundary(outlet_bry)
    optim.add_boundary(wall_bry)
    optim.define_problem()
    
    result = optim.run()
    print(result)


if __name__ == '__main__':
    debug()
