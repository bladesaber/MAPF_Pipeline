import shutil
from typing import Literal, Union, Callable
import os

import dolfinx
import ufl
from ufl import inner, grad, div
import ufl.tensors
import ufl.indexed

import numpy as np
from functools import partial
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.simulator.dolfin_simulator import FluidBoundary, FluidSimulator
from scripts_py.vtu_recorder import VtuRecorder, TensorBoardRecorder

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control, compute_gradient
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape
from thirdparty.pyadjoint.pyadjoint.adjfloat import max, min  # NOTE: please use max and min from pyadjoint
from thirdparty.pyadjoint.pyadjoint.adjfloat import AdjFloat

from scripts_py.mesh_utils import MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils, AssembleUtils
from scripts_py.dolfinx_adjoint.topology_utils.topology_filter import DistanceBandFilter
from scripts_py.eqn_solver import LinearProblemSolver
from scripts_py.auto_dolfinx.opt_solver.grad_optimizer import MomentumSGDOptimizer, BaseGradOptimizer

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.types.type_dirichletBC import DolfinxDirichletBC
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.tape_utils import start_annotation

from scripts_py.auto_dolfinx.opt_solver.topo_fluid import AutoFluidBoundary, AlphaFunction


"""
应该通过设置区域boundary值来约束渗透问题,而不是通过设置alpha函数来约束. 该方法只能在逐步收缩体积约束的这种优化路径下使用.
通过区域边界(Area Boundary)的使用也可以控制流通域不消失

TODO:
1. 求解器能设置区域边界(Area Boundary)
2. 假设网格挖去部分cell后能生成msh文件,方便体积剪除后仿真
3. 拓扑优化时体积约束必须逐步减少,而不是预先设定
4. 体积剪除必须保留联通域,方法是如果剪除后不联通,则在剪除后的不联通的块的边界进行搜索,搜索区域必须在剪除前的区域内,然后保留最短搜索路径长度下一定阀值的路径
"""


class TopoWeightConstraintTerm(object):
    def __init__(
        self,
        name: str,
        mesh: MeshObj,
        init_method: Literal['boundary_raw', 'boundary_radius', 'obstacle'],
        constraint_method: Literal['none', 'zero', 'fixed'] | None = None,
        set_idxs: np.ndarray|None = None,
        set_value: np.ndarray|None = None,
        **kwargs
    ):
        self.name = name
        self.init_priority: int
        self.init_method = init_method
        self.constraint_priority: int
        self.constraint_method: str = constraint_method  # type: ignore
        
        self.set_idxs: np.ndarray|None = set_idxs
        self.set_value: np.ndarray|None = set_value
        self.define_init_weight(mesh, **kwargs)
    
    def define_init_weight(self, mesh: MeshObj, **kwargs):
        if self.init_method == 'boundary_raw':
            bry_info = kwargs['bry_info']
            assert bry_info.boundary_type in ['velocity', 'pressure', 'wall']
            self.set_idxs = mesh.extract_surface_indices(bry_info.marker)
            if bry_info.boundary_type in ['velocity', 'pressure']:
                self.set_value = np.ones(shape=self.set_idxs.shape[0])
            elif bry_info.boundary_type == 'wall':
                self.set_value = np.zeros(shape=self.set_idxs.shape[0])
                
            if bry_info.boundary_type in ['velocity', 'pressure']:
                self.init_priority = 1
            elif bry_info.boundary_type == 'wall':
                self.init_priority = 2
            
            # if bry_info.boundary_type in ['velocity', 'wall']:
            #     self.constraint_method = 'zero'
            # elif bry_info.boundary_type == 'pressure':
            #     self.constraint_method = 'none'
            self.constraint_method = 'zero'  # TODO: should i use zero for all boundary?
        
        elif self.init_method == 'boundary_radius':
            bry_info = kwargs['bry_info']
            assert bry_info.boundary_type in ['velocity', 'pressure']
            assert bry_info.face_center is not None
            assert bry_info.face_radius is not None
            face_center = bry_info.face_center
            points = mesh.points[:, :mesh.tdim]
            face_radius = bry_info.face_radius
            dists = np.linalg.norm(points - face_center.reshape((1, -1)), axis=1)
            set_idxs = np.arange(0, dists.shape[0], 1)
            # adding 1e-4 to guarantee the boundary is set to 1.0 and symmetry
            self.set_idxs = set_idxs[dists < (face_radius + 1e-4)]
            self.set_value = np.ones(shape=self.set_idxs.shape[0])  # type: ignore
            self.init_priority = 1
            
            # if bry_info.boundary_type in ['velocity', 'wall']:
            #     self.constraint_method = 'zero'
            # elif bry_info.boundary_type == 'pressure':
            #     self.constraint_method = 'none'
            self.constraint_method = 'zero'  # TODO: should i use zero for all boundary?
        
        elif self.init_method == 'obstacle':
            obs_func: Callable = kwargs['obstacle_func']
            if obs_func is not None:
                scalar_func_space = DolfinUtils.point_scalar_function_space(mesh.domain)
                self.set_idxs = obs_func(DolfinUtils.tabulate_dof_coordinates(scalar_func_space))
                if self.set_idxs.shape[0] > 0:  # type: ignore
                    self.set_value = np.zeros(shape=self.set_idxs.shape[0])  # type: ignore
            self.init_priority = 2
            self.constraint_method = 'zero'
        
        else:
            raise ValueError(f"Invalid init method: {self.init_method}")
    
    def apply_constraint_gradient(self, gradient_np: np.ndarray):
        if self.constraint_method == 'fixed':
            gradient_np[self.set_idxs] = self.set_value
        elif self.constraint_method == 'zero':
            gradient_np[self.set_idxs] = 0.0
        return gradient_np


class AlphaExponentialFunc(AlphaFunction):
    def __init__(
        self,
        name: str,
        func_args: dict,
    ):
        super().__init__(name, func_args)
    
    def __call__(self, rho: np.ndarray|dolfinx.fem.Function):
        return self.alpha_fun(rho, **self.func_args)
    
    @staticmethod
    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_min=1e-12, alpha_max=100.0, p=3.0):
        pow_operator = ufl.elem_pow if isinstance(rho, dolfinx.fem.Function) else np.power
        return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)  # type: ignore


class AlphaSigmoidFunc(AlphaFunction):
    def __init__(
        self,
        name: str,
        func_args: dict,
    ):
        super().__init__(name, func_args)
    
    def __call__(self, rho: np.ndarray|dolfinx.fem.Function):
        return self.alpha_fun(rho, **self.func_args)
    
    @staticmethod
    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_min=1e-12, alpha_max=100.0, x0=0.5, k=20.0, inverse=False):
        exp_operator = ufl.exp if isinstance(rho, dolfinx.fem.Function) else np.exp
        if inverse:
            return alpha_min + (alpha_max - alpha_min) * (1.0 - 1.0 / (1.0 + exp_operator(-k * (rho - x0))))  # type: ignore
        else:
            return alpha_min + (alpha_max - alpha_min) * (1.0 / (1.0 + exp_operator(-k * (rho - x0))))  # type: ignore


class TopoFluidOptimizer(FluidSimulator):
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float|np.ndarray,
        topo_weight_init: float|np.ndarray,
        init_topo_weight_method: Literal['poisson', 'filter', 'none', 'raw', 'random'],
        weight_filter_radius: float,
        result_dir: str,
        alpha_fun: dict = {
            'name': 'exponential',
            'args': {
                'alpha_min': 1e-12,
                'alpha_max': 100.0,
                'p': 3.0,
            }
        },
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
        
        self.fluid_type: Literal['stoke', 'naiver_stoke', 'rans'] = fluid_type
        self.viscosity_init = viscosity_init
        
        if alpha_fun['name'] == 'exponential':
            self.alpha_func = AlphaExponentialFunc(alpha_fun['name'], alpha_fun['args'])
        elif alpha_fun['name'] == 'sigmoid':
            self.alpha_func = AlphaSigmoidFunc(alpha_fun['name'], alpha_fun['args'])
        else:
            raise ValueError(f"Invalid alpha function: {alpha_fun['name']}")
        
        self.topo_weight_init = 0.9
        self.init_topo_weight_method = init_topo_weight_method
        self.topo_weight_constraint_terms: dict[str, TopoWeightConstraintTerm] = {}
        
        self.result_dir = result_dir
        self.recorder = VtuRecorder(self.mesh, save_dir=self.result_dir, is_binary=True)
        
        self.weight_filter_radius = weight_filter_radius
        self.is_topo_cell = False
        self.weight_filter = DistanceBandFilter(self.mesh.grid, weight_filter_radius, self.is_topo_cell, True)
        
        self.control_phi: Control
        self.opt_problem: ReducedFunctional
    
    def init_topo_weight(self, topo_weight_fun: DolfinxFunction, method: Literal['poisson', 'filter', 'raw', 'random']):
        """当整个场的topo_weight都很少时,速度与压力会汇聚在出口的边角区域,造成崩角的效果.
        崩角效果会造成优化畸形(因为拓扑优化易于陷入局部最优点),因此构造一个球形区域,使出口区域比边缘更流畅,减缓崩角的影响.
        当然也可以通过减少reg_cost的权重来减缓崩角的影响.
        """
        init_weight = dolfinx.fem.Function(self.point_scalar_func_space)
        init_weight_value = topo_weight_fun.x.array.copy()  # type: ignore
        
        sorted_terms = [term for term in sorted(self.topo_weight_constraint_terms.values(), key=lambda x: x.init_priority)]
        if method in ['filter', 'raw']:
            for _ in range(3):
                init_weight_value = np.asarray(self.weight_filter.run(init_weight_value.reshape((-1, 1)))).reshape((-1))
                for term in sorted_terms:
                    init_weight_value[term.set_idxs] = term.set_value
            init_weight.x.array[:] = init_weight_value  # type: ignore
            
        elif method == 'poisson':
            u = ufl.TrialFunction(self.point_scalar_func_space)
            v = ufl.TestFunction(self.point_scalar_func_space)
            f = dolfinx.fem.Function(self.point_scalar_func_space)
            
            a_form = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
            L_form = f * v * ufl.dx  # type: ignore
            
            bcs = []
            for _, term in self.topo_weight_constraint_terms.items():
                if term.init_method == 'boundary_raw':
                    bc_value = 1.0 if term.init_priority == 1 else 0.0
                    bcs.append(dolfinx.fem.dirichletbc(float(bc_value), term.set_idxs, V=self.point_scalar_func_space))  # type: ignore
            
            res_dict = LinearProblemSolver.solve_fea_petsc(
                comm=self.mesh.domain.comm,
                uh=init_weight,  # type: ignore
                a_form=a_form,
                l_form=L_form,
                bcs=bcs,
                mat_assemble_method='lift',
                ksp_option={
                    'ksp_type': 'preonly',
                    'pc_type': 'lu',
                    "pc_factor_mat_solver_type": "mumps",
                    "mat_mumps_icntl_24": 1
                },
                with_residual=False
            )
            if not res_dict['is_solved']:
                raise ValueError(f"Failed to solve the Poisson equation for topo weight initialization.")
    
        elif method == 'random':
            rnd_array = np.random.uniform(low=0.0, high=1.0, size=init_weight.x.array.shape)  # type: ignore
            rnd_array[rnd_array >= 0.5] = 1.0
            rnd_array[rnd_array < 0.5] = 0.0
            for term in sorted_terms:
                rnd_array[term.set_idxs] = term.set_value
            init_weight.x.array[:] = rnd_array  # type: ignore
    
        else:
            raise ValueError(f"Invalid init method: {method}")
        
        topo_weight_fun.interpolate(init_weight)
        
    def compute_gradient(self, adj_input: float, with_filter=True, with_norm=True) -> np.ndarray:
        dJdm = self.opt_problem.derivative(adj_input=float(adj_input))[0]
        
        # ------ apply constraint to gradient ------
        if self.init_topo_weight_method != 'none':
            for _, term in self.topo_weight_constraint_terms.items():
                dJdm = term.apply_constraint_gradient(dJdm)
        
        if with_filter:
            dJdm = dJdm.reshape((-1, 1))
            dJdm = self.weight_filter.run(dJdm)
            dJdm = np.asarray(dJdm).reshape((-1))
        
        # ------ apply constraint to gradient again ------
        if self.init_topo_weight_method != 'none' and with_filter:
            for _, term in self.topo_weight_constraint_terms.items():
                dJdm = term.apply_constraint_gradient(dJdm)
        
        if with_norm:
            norm = np.max(np.abs(dJdm))
            dJdm = dJdm / norm
        
        return dJdm
    
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
                    face_center=bry_info.face_center, # type: ignore
                    face_norm_vector=bry_info.face_norm_vector, # type: ignore
                    face_radius=bry_info.face_radius, # type: ignore
                    method=bry_info.interpolate_method,
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
        """Args:
            U: velocity field
            p: pressure field
            V: velocity test function
            q: pressure test function
            f: body force
            nut: viscosity
        """
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
            """NOTE: Not suitable for topology optimization"""
            with_viscosity = obj_cost_args['with_viscosity']
            if with_viscosity:
                min_dissipation_form = kwargs['viscosity'] * inner(grad(u), grad(u)) * ufl.dx
            else:
                min_dissipation_form = inner(grad(u), grad(u)) * ufl.dx
            cost_parse_dict.update({
                'min_dissipation_form': min_dissipation_form
            })
            
        elif obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
            with_viscosity = obj_cost_args['with_viscosity']
            if with_viscosity:
                min_dissipation_form = kwargs['viscosity'] * inner(grad(u), grad(u)) * ufl.dx
            else:
                min_dissipation_form = inner(grad(u), grad(u)) * ufl.dx
            penalized_term = 0.5 * AlphaSigmoidFunc.alpha_fun(
                kwargs['rho'], 
                alpha_min=obj_cost_args.get('penalized_alpha_min', 1e-12), 
                alpha_max=obj_cost_args.get('penalized_alpha_max', 300.0), 
                x0=0.5, 
                k=obj_cost_args.get('penalized_alpha_k', 20.0),
                inverse=True
            ) * inner(u, u) * ufl.dx  # type: ignore

            cost_parse_dict.update({
                'min_dissipation_form': min_dissipation_form,
                'penalized_term': penalized_term,
            })
        
        elif obj_cost_args['style'] == 'MinDissipation+GradBrinkmanPenalized':
            with_viscosity = obj_cost_args['with_viscosity']
            if with_viscosity:
                min_dissipation_form = kwargs['viscosity'] * inner(grad(u), grad(u)) * ufl.dx
            else:
                min_dissipation_form = inner(grad(u), grad(u)) * ufl.dx
            penalized_term = 0.5 * AlphaSigmoidFunc.alpha_fun(
                kwargs['rho'], 
                alpha_min=obj_cost_args.get('penalized_alpha_min', 1e-12), 
                alpha_max=obj_cost_args.get('penalized_alpha_max', 300.0), 
                x0=0.5, 
                k=obj_cost_args.get('penalized_alpha_k', 20.0),
                inverse=True
            ) * inner(grad(u), grad(u)) * ufl.dx  # type: ignore
            
            cost_parse_dict.update({
                'min_dissipation_form': min_dissipation_form,
                'grad_penalized_term': penalized_term,
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
            assert np.isclose(np.sum(outlet_marker_ratios), 1.0, atol=1e-2), 'Outlet marker ratios must sum to 1.0'
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
                'num_outlets': len(outlet_markers),
            })
        
        elif obj_cost_args['style'] == 'BalanceFlux':
            outlet_forms = {}
            for marker in obj_cost_args['outlet_markers']:
                outlet_flux_form = ufl.dot(u, self.n_vec) * self.ds(marker)
                outlet_forms[marker] = outlet_flux_form
            cost_parse_dict.update({
                'outlet_forms': outlet_forms,
                'num_outlets': len(obj_cost_args['outlet_markers']),
            })
        
        else:
            raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')

        obj_cost_args.update(cost_parse_dict)
        
        return obj_cost_args

    def define_problem(
        self,
        obj_cost_args_list: list[dict],
        reg_cost_args: dict,
        obstacle_cost_args: dict|None = None,
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
            'critical_method': 'residual',
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        forward_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        }
    ):
        # ------ init topo weight boundary constraint terms ------
        if self.init_topo_weight_method != 'none':
            for name, bry_info in self.boundaries.items():
                assert bry_info.boundary_type in ['velocity', 'pressure', 'wall']
                if bry_info.boundary_type in ['velocity', 'pressure']:
                    init_method = 'boundary_raw'
                    if self.init_topo_weight_method in ['filter']:
                        init_method='boundary_radius'
                    term = TopoWeightConstraintTerm(name=name, mesh=self.mesh, bry_info=bry_info, init_method=init_method)
                elif bry_info.boundary_type == 'wall':
                    term = TopoWeightConstraintTerm(name=name, mesh=self.mesh, init_method='boundary_raw', bry_info=bry_info)
                self.topo_weight_constraint_terms[name] = term
        
        # ------ obstacle part ------
        if obstacle_cost_args is not None:
            obstacle_weight = np.zeros(self.mesh.num_points)
            obstacle_dofs = obstacle_cost_args['obstacle_func'](DolfinUtils.tabulate_dof_coordinates(self.point_scalar_func_space))
            if obstacle_dofs.shape[0] > 0:
                obstacle_weight[obstacle_dofs] = 1.0
                
            obs_point_mask = DolfinxFunction(self.point_scalar_func_space, name="obstacle_source")
            obs_point_mask.x.array[:] = obstacle_weight  # type: ignore
            obstacle_cost_args['obs_point_mask'] = obs_point_mask
            obstacle_cost_args['is_cell_field'] = False
            
            if obstacle_dofs.shape[0] > 0 and self.init_topo_weight_method != 'none':
                obs_term = TopoWeightConstraintTerm(
                    name='obstacle', mesh=self.mesh, 
                    init_method='obstacle', 
                    obstacle_func=obstacle_cost_args['obstacle_func']
                )
                self.topo_weight_constraint_terms[obs_term.name] = obs_term 
        self.obstacle_cost_args = obstacle_cost_args
        
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
            
            if self.is_topo_cell:
                self.topo_weight = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_weight')  # type: ignore
            else:
                self.topo_weight = DolfinxFunction(self.point_scalar_func_space, name=f'topo_weight')  # type: ignore
            self.topo_weight.x.array[:] = self.topo_weight_init  # type: ignore
            if self.init_topo_weight_method != 'none':
                self.init_topo_weight(self.topo_weight, method=self.init_topo_weight_method)  # type: ignore
            
            alpha = self.alpha_func(self.topo_weight)  # type: ignore
            govern_form += alpha * inner(u, v) * ufl.dx  # type: ignore
            
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
            self.record_Upw(0)
            u, p = ufl.split(self.uh)  # type: ignore
            
            # ------ define object cost ------
            for obj_idx, obj_cost_args in enumerate(obj_cost_args_list):
                obj_cost_args = self.prepare_cost_args(
                    u=u, p=p, obj_cost_args=obj_cost_args,  # type: ignore
                    viscosity=self.viscosity, alpha=alpha, rho=self.topo_weight,
                )
                if obj_cost_args['style'] == 'MinDissipation':
                    obj_cost = assemble_scalar(
                        form=obj_cost_args['weight'] * obj_cost_args['min_dissipation_form'], 
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
                    obj_cost = assemble_scalar(
                        form=(
                            obj_cost_args['dissipation_weight'] * obj_cost_args['min_dissipation_form'] 
                            + obj_cost_args['penalized_weight'] * obj_cost_args['penalized_term']
                        ),
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'MinDissipation+GradBrinkmanPenalized':
                    obj_cost = assemble_scalar(
                        form=(
                            obj_cost_args['dissipation_weight'] * obj_cost_args['min_dissipation_form'] 
                            + obj_cost_args['penalized_weight'] * obj_cost_args['grad_penalized_term']
                        ),
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'MinPressure':
                    obj_cost = assemble_scalar(
                        form=obj_cost_args['weight'] * obj_cost_args['pressure_form'],
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'FixedFlux':
                    # NOTE: 该lost的波动无法通过扩大volume来降低
                    with_rescale = obj_cost_args['with_rescale']
                    with_max_dif_cost = obj_cost_args['with_max_dif_cost']
                    
                    if with_max_dif_cost:
                        max_dif_cost, max_dif_weight = 0.0, 0.0
                        
                    for i, (marker, outlet_item) in enumerate(obj_cost_args['outlet_cost_forms'].items()):
                        outlet_flux = assemble_scalar(
                            form=outlet_item['flux_form'],
                            domain=self.domain, tape=self.tape
                        )
                        target_flux = outlet_item['target_flux']
                        print(f"{marker} flux: {outlet_flux}, target: {target_flux}, inlet_flux: {obj_cost_args['inlet_flux']}")
                                                
                        flux_dif = outlet_flux - target_flux
                        if with_rescale:
                            flux_dif = flux_dif / target_flux
                        flux_dif_cost = flux_dif ** 2
                        
                        if with_max_dif_cost:
                            max_dif_cost = max(max_dif_cost, flux_dif_cost)
                            max_dif_weight = outlet_item['weight']
                        
                        weight = outlet_item['weight']
                        if i == 0:
                            obj_cost = weight * flux_dif_cost
                        else:
                            obj_cost += weight * flux_dif_cost
                            
                    obj_cost = obj_cost / obj_cost_args['num_outlets']
                    if with_max_dif_cost:
                        obj_cost += max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)
                elif obj_cost_args['style'] == 'BalanceFlux':
                    weight = obj_cost_args['weight']
                    with_rescale = obj_cost_args['with_rescale']
                    with_max_dif_cost = obj_cost_args['with_max_dif_cost']
                    
                    outlet_flux_list = []
                    for marker, outlet_flux_form in obj_cost_args['outlet_forms'].items():
                        outlet_flux = assemble_scalar(form=outlet_flux_form, domain=self.domain, tape=self.tape)
                        outlet_flux_list.append(outlet_flux)
                    
                    mean_flux = 0.0
                    for outlet_flux in outlet_flux_list:
                        mean_flux += outlet_flux
                    mean_flux = mean_flux / obj_cost_args['num_outlets']
                    
                    if with_max_dif_cost:
                        max_dif_cost = 0.0
                        
                    for i, outlet_flux in enumerate(outlet_flux_list):
                        flux_dif = outlet_flux - mean_flux
                        if with_rescale:
                            flux_dif = flux_dif / mean_flux
                        flux_dif_cost = flux_dif ** 2
                        
                        if with_max_dif_cost:
                            max_dif_cost = max(max_dif_cost, flux_dif_cost)
                        
                        if i == 0:
                            obj_cost = weight * flux_dif_cost
                        else:
                            obj_cost += weight * flux_dif_cost
                        
                    obj_cost = obj_cost / obj_cost_args['num_outlets']
                    if with_max_dif_cost:
                        obj_cost += max_dif_cost * obj_cost_args.get('max_dif_weight', weight)
                else:
                    raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')
                                
                if obj_idx == 0:
                    all_obj_cost = obj_cost
                else:
                    all_obj_cost += obj_cost  # type: ignore
                # print(f'{obj_cost_args["style"]}: {obj_cost}')
                
            self.obj_cost_args_list = obj_cost_args_list
            
            # ------ define regularization cost ------
            if reg_cost_args['style'] == 'Volume':
                if 'target_ratio' in reg_cost_args.keys():
                    target_volume = self.volume * reg_cost_args['target_ratio']
                else:
                    target_volume = reg_cost_args['target_volume']
                reg_cost_args.update({
                    'target_volume': target_volume,
                    'vol_form': AlphaSigmoidFunc.alpha_fun(
                        self.topo_weight, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=10.0, inverse=False   # type: ignore
                    ) * ufl.dx
                })
                
                align_volume = AdjFloat(self.volume * self.topo_weight_init)
                
            topo_volume = assemble_scalar(reg_cost_args['vol_form'], domain=self.domain, tape=self.tape)

            if reg_cost_args['reg_rescale']:
                topo_volume_rescale = topo_volume / align_volume  # type: ignore               
                volume_dif = topo_volume_rescale - 1.0
                reg_cost = reg_cost_args['weight'] * (volume_dif ** 2)
                # reg_cost += reg_cost_args['weight'] * (max(volume_dif, 0.0) + -1.0 * min(volume_dif, 0.0))
            else:
                reg_cost = reg_cost_args['weight'] * ((topo_volume - align_volume) ** 2)  # type: ignore
                
            self.reg_cost_args = reg_cost_args
            loss_all = all_obj_cost + reg_cost
            
            # ------ define obstacle cost ------
            if self.obstacle_cost_args is not None:
                if self.obstacle_cost_args['is_cell_field']:
                    obs_cost_form = inner(self.obstacle_cost_args['obs_cell_mask'], self.topo_weight) * ufl.dx
                else:
                    obs_cost_form = inner(self.obstacle_cost_args['obs_point_mask'], self.topo_weight) * ufl.dx
                self.obstacle_cost_args['cost_form'] = obs_cost_form
                
                obs_cost = assemble_scalar(obs_cost_form, domain=self.domain, tape=self.tape)
                obs_cost = obs_cost * self.obstacle_cost_args['weight']
                
                loss_all += obs_cost
            
            # ------ define optimization problem ------
            self.control_phi = Control(self.topo_weight)  # type: ignore
            self.control_align_volume = Control(align_volume)  # type: ignore
            self.opt_problem = ReducedFunctional(loss_all, [self.control_phi, self.control_align_volume])

        loss_dict, log_dict = self.eval()
        print('[Log]: Initial loss:', loss_dict)
        
        # for block in self.tape.get_blocks():
        #     print(block, ' type: ', type(block))
        
    
    def run_with_lr_scheduler(
        self,
        min_lr: float = 1e-2,
        max_lr: float = 1e-1,
        max_iteration=200,
        hot_step: int = 20,
        momentum_weight: float = 0.8,
        full_run: bool = False,
        recorder: TensorBoardRecorder|None = None,
        result_vtu_file: str|None = None,
        loss_decay_weight: float = 0.0,
    ) -> dict:
        topo_phi = DolfinxFunction(self.topo_weight.function_space, name=f'topo_phi')  # type: ignore
        topo_phi.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        
        loss_current, loss_last = 0.0, 0.0
        loss_dict, log_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(0)
        
        with_momentum = momentum_weight > 0.0
        if with_momentum:
            grad_optimizer = MomentumSGDOptimizer(shape=self.topo_weight.x.array.shape, momentum=momentum_weight)  # type: ignore
        else:
            grad_optimizer = BaseGradOptimizer(shape=self.topo_weight.x.array.shape)  # type: ignore
        
        topo_phi_tensor = torch.from_numpy(topo_phi.x.array.copy())  # type: ignore
        topo_phi_tensor.requires_grad = True
        optimizer = torch.optim.Adam([topo_phi_tensor], lr=max_lr)  # type: ignore
        # optimizer = torch.optim.SGD([topo_phi_tensor], lr=max_lr, momentum=momentum_weight)  # type: ignore
        
        scheduler = ReduceLROnPlateau(optimizer, min_lr=min_lr, factor=0.5)  # type: ignore
        last_lr = scheduler.get_last_lr()[0]
        volume_interpolate = np.linspace(self.control_align_volume.data(), self.reg_cost_args['target_volume'], max_iteration)  # type: ignore
        
        for step, volume_stage in zip(range(max_iteration), volume_interpolate):
            step += 1
            if step > 1:
                loss_dict, log_dict = self.eval()
                log = f"[step/{step} loss:{loss_current:.6f} lr:{last_lr:.3f} volume:{volume_stage:.6f}]"
                for name, cost_value in loss_dict.items():
                    log += f" {name}:{cost_value:.6f}"
                print(log)
                
                if recorder is not None:
                    for name, value in log_dict.items():
                        if isinstance(value, dict):
                            recorder.write_scalars(tag=name, scalar_dict=value, step=step)
                        else:
                            recorder.write_scalar(tag=name, scalar_value=value, step=step)
                
            direction_np = self.compute_gradient(
                adj_input=np.minimum(np.maximum(loss_current * 0.1, 0.01), 1e2), 
                with_filter=True, with_norm=True
            )
            
            # # ------ custome momentum optimizer ------
            # direction_np = grad_optimizer.update(direction_np)
            # new_topo_phi = topo_phi.x.array - direction_np * last_lr  # type: ignore
            # new_topo_phi = np.maximum(np.minimum(new_topo_phi, 1.0 - 1e-16), 1e-16)
            
            # ------ torch optimizer ------ not stable enough
            topo_phi_tensor.grad = torch.from_numpy(direction_np)
            optimizer.step()
            new_topo_phi = np.maximum(np.minimum(topo_phi_tensor.detach().numpy(), 1.0 - 1e-16), 1e-16)
            topo_phi.x.array[:] = new_topo_phi  # type: ignore
            
            self.control_align_volume.update(volume_stage)  # TODO how to update align_volume?
            loss_new = self.opt_problem(values=[topo_phi, self.control_align_volume.data()])  # type: ignore
            
            scheduler.step(loss_new)
            last_lr = scheduler.get_last_lr()[0]
            
            loss_last = loss_current
            loss_current = loss_new * (1.0 - loss_decay_weight) + loss_last * loss_decay_weight
            self.record_Upw(step, gradient=direction_np)
        
        if result_vtu_file is not None:
            last_step_record_file = self.recorder.get_step_file_path(step)
            if not os.path.exists(last_step_record_file):
                last_step_record_file = self.recorder.get_step_file_path(step - 1)
            shutil.copy(last_step_record_file, result_vtu_file)

        return {'is_success': True}  # type: ignore
    
    def eval(self):
        eval_dict = {}
        log_dict = {'cost': {}}
        
        # ------ compute object cost ------
        for obj_cost_args in self.obj_cost_args_list:
            if obj_cost_args['style'] == 'MinDissipation':
                eval_dict['MinDiss_cost'] = AssembleUtils.assemble_scalar(
                    form=obj_cost_args['weight'] * obj_cost_args['min_dissipation_form']
                )
                log_dict['cost']['MinDiss_cost'] = eval_dict['MinDiss_cost']
                
            elif obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
                min_dissipation_cost = obj_cost_args['dissipation_weight'] * AssembleUtils.assemble_scalar(obj_cost_args['min_dissipation_form'])
                penalized_cost = obj_cost_args['penalized_weight'] * AssembleUtils.assemble_scalar(obj_cost_args['penalized_term'])
                eval_dict['MinDiss_cost'] = min_dissipation_cost
                eval_dict['Penalized_cost'] = penalized_cost
                log_dict['cost']['MinDiss_cost'] = min_dissipation_cost
                log_dict['cost']['Penalized_cost'] = penalized_cost
            
            elif obj_cost_args['style'] == 'MinDissipation+GradBrinkmanPenalized':
                min_dissipation_cost = obj_cost_args['dissipation_weight'] * AssembleUtils.assemble_scalar(obj_cost_args['min_dissipation_form'])
                grad_penalized_cost = obj_cost_args['penalized_weight'] * AssembleUtils.assemble_scalar(obj_cost_args['grad_penalized_term'])
                eval_dict['MinDiss_cost'] = min_dissipation_cost
                eval_dict['GradPenalized_cost'] = grad_penalized_cost
                log_dict['cost']['MinDiss_cost'] = min_dissipation_cost
                log_dict['cost']['GradPenalized_cost'] = grad_penalized_cost
            
            elif obj_cost_args['style'] == 'MinPressure':
                eval_dict['MinPressure_cost'] = AssembleUtils.assemble_scalar(
                    form=obj_cost_args['weight'] * obj_cost_args['pressure_form']
                )
                log_dict['cost']['MinPressure_cost'] = eval_dict['MinPressure_cost']
                
            elif obj_cost_args['style'] == 'FixedFlux':
                if 'flux' not in log_dict.keys():
                    log_dict['flux'] = {}
                if 'flux_cost' not in log_dict.keys():
                    log_dict['flux_cost'] = {}
                
                outlet_cost_list = []
                if obj_cost_args['with_max_dif_cost']:
                    max_dif_cost, max_dif_weight = 0.0, 0.0
                
                for (marker, outlet_item) in zip(obj_cost_args['outlet_markers'], obj_cost_args['outlet_cost_forms'].values()):
                    outlet_flux = AssembleUtils.assemble_scalar(form=outlet_item['flux_form'])
                    log_dict['flux'][f'{marker}_flux'] = outlet_flux
                    
                    flux_dif = outlet_flux - outlet_item['target_flux']
                    if obj_cost_args['with_rescale']:
                        flux_dif = flux_dif / outlet_item['target_flux']
                    flux_cost = flux_dif ** 2
                    outlet_cost_list.append(outlet_item['weight'] * flux_cost)
                    log_dict['flux_cost'][f'{marker}_flux_cost'] = outlet_item['weight'] * flux_cost
                    
                    if obj_cost_args['with_max_dif_cost']:
                        max_dif_cost = max(max_dif_cost, flux_cost)
                        max_dif_weight = outlet_item['weight']
                
                outlet_cost_list = np.array(outlet_cost_list)
                eval_dict['FixedFlux_cost'] = float(np.mean(outlet_cost_list))
                log_dict['cost']['FixedFlux_cost'] = eval_dict['FixedFlux_cost']
                if obj_cost_args['with_max_dif_cost']:
                    eval_dict[f'FixedFlux_cost'] += max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)
                    log_dict['cost'][f'FixedFlux_cost_max_dif'] = max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)
            
            elif obj_cost_args['style'] == 'BalanceFlux':
                if 'flux' not in log_dict.keys():
                    log_dict['flux'] = {}
                
                outlet_flux_list = []
                if obj_cost_args['with_max_dif_cost']:
                    max_dif_cost = 0.0
                
                for marker, outlet_flux_form in obj_cost_args['outlet_forms'].items():
                    outlet_flux = AssembleUtils.assemble_scalar(form=outlet_flux_form)
                    log_dict['flux'][f'{marker}_flux'] = outlet_flux
                    outlet_flux_list.append(outlet_flux)
                outlet_flux_list = np.array(outlet_flux_list)                
                mean_flux = np.mean(outlet_flux_list)                
                flux_dif = outlet_flux_list - mean_flux
                if obj_cost_args['with_rescale']:
                    flux_dif = flux_dif / mean_flux
                flux_dif_cost = flux_dif ** 2
                balance_cost = float(np.mean(obj_cost_args['weight'] * flux_dif_cost))
                if obj_cost_args['with_max_dif_cost']:
                    balance_cost += float(np.max(flux_dif_cost) * obj_cost_args.get('max_dif_weight', obj_cost_args['weight']))
                
                log_dict['cost']['balance_cost'] = balance_cost
                eval_dict['BalanceFlux_cost'] = balance_cost            
            else:
                raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')
            
        # ------ compute regularization cost ------
        if self.reg_cost_args['style'] == 'Volume':
            align_volume = self.control_align_volume.data()  # type: ignore
            topo_volume = AssembleUtils.assemble_scalar(self.reg_cost_args['vol_form'])
            if self.reg_cost_args['reg_rescale']:
                topo_volume_rescale = topo_volume / align_volume                
                volume_dif = topo_volume_rescale - 1.0
                eval_dict['reg_cost'] = self.reg_cost_args['weight'] * float(volume_dif ** 2)
            else:
                eval_dict['reg_cost'] = self.reg_cost_args['weight'] * (float(align_volume - topo_volume) ** 2)
                            
            log_dict['cost']['reg_cost'] = eval_dict['reg_cost']
            log_dict['volume'] = topo_volume
                
        else:
            raise ValueError(f'Invalid regularization style: {self.reg_cost_args['style']}')
        
        # ------ compute obstacle cost ------
        if self.obstacle_cost_args is not None:
            eval_dict['obs_cost'] = AssembleUtils.assemble_scalar(self.obstacle_cost_args['cost_form']) * self.obstacle_cost_args['weight']
            log_dict['cost']['obs_cost'] = eval_dict['obs_cost']
        
        log_dict['loss_all'] = np.sum(list(eval_dict.values()))
                        
        return eval_dict, log_dict
    
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
        
        if self.is_topo_cell:
            self.recorder.update_cell_data('topo_weight', self.topo_weight.x.array)    # type: ignore
            point_scalar_fun = dolfinx.fem.Function(self.point_scalar_func_space)
            point_scalar_fun.interpolate(self.topo_weight)  # type: ignore
            self.recorder.update_point_data('topo_weight', point_scalar_fun.x.array)  # type: ignore
            
            
            self.recorder.update_cell_data(
                'alpha', AlphaSigmoidFunc.alpha_fun(self.topo_weight.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
            )
            self.recorder.update_point_data(
                'alpha', AlphaSigmoidFunc.alpha_fun(point_scalar_fun.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
            )
            
        else:
            self.recorder.update_point_data('topo_weight', self.topo_weight.x.array)  # type: ignore
            cell_scalar_fun = dolfinx.fem.Function(self.cell_scalar_func_space)
            cell_scalar_fun.interpolate(self.topo_weight)  # type: ignore
            self.recorder.update_cell_data('topo_weight', cell_scalar_fun.x.array)  # type: ignore
            
            self.recorder.update_point_data(
                'alpha', AlphaSigmoidFunc.alpha_fun(self.topo_weight.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
            )
            self.recorder.update_cell_data(
                'alpha', AlphaSigmoidFunc.alpha_fun(cell_scalar_fun.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
            )
            
        if self.obstacle_cost_args is not None:
            if self.obstacle_cost_args['is_cell_field']:
                self.recorder.update_cell_data('obs_source', self.obstacle_cost_args['obs_cell_mask'].x.array.copy())  # type: ignore
            else:
                self.recorder.update_point_data('obs_source', self.obstacle_cost_args['obs_point_mask'].x.array.copy())  # type: ignore
        
        if kwargs.get('gradient', None) is not None:
            if self.is_topo_cell:
                self.recorder.update_cell_data('gradient', kwargs['gradient'])
            else:
                self.recorder.update_point_data('gradient', kwargs['gradient'])
        
        self.recorder.save(step)
    

if __name__ == '__main__':
    pass
