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
import pyvista
from sklearn.neighbors import KDTree

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.auto_dolfinx.opt_solver.seq_mesh import SequenceMeshObj
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


# TODO: 关于最优化路径控制问题的研究


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

# -------------------------------------
class AlphaFunction(object):
    def __init__(
        self,
        name: str,
        func_args: dict,
    ):
        self.name = name
        self.func_args = func_args
    
    def __call__(self, rho: np.ndarray|dolfinx.fem.Function):
        raise NotImplementedError(f"AlphaFunction {self.name} is not implemented")


class AlphaExponentialFunc(AlphaFunction):
    def __call__(self, rho: np.ndarray|dolfinx.fem.Function):
        return self.alpha_fun(rho, **self.func_args)
    
    @staticmethod
    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_min=1e-12, alpha_max=100.0, p=3.0, inverse=True):
        pow_operator = ufl.elem_pow if isinstance(rho, dolfinx.fem.Function) else np.power
        if inverse:
            return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)  # type: ignore
        else:
            return alpha_min + pow_operator(rho, p) * (alpha_max - alpha_min)  # type: ignore


class AlphaSigmoidFunc(AlphaFunction):
    def __call__(self, rho: np.ndarray|dolfinx.fem.Function):
        return self.alpha_fun(rho, **self.func_args)
    
    @staticmethod
    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_min=1e-12, alpha_max=100.0, x0=0.5, k=20.0, inverse=True):
        exp_operator = ufl.exp if isinstance(rho, dolfinx.fem.Function) else np.exp
        if inverse:
            return alpha_min + (alpha_max - alpha_min) * (1.0 - 1.0 / (1.0 + exp_operator(-k * (rho - x0))))  # type: ignore
        else:
            return alpha_min + (alpha_max - alpha_min) * (1.0 / (1.0 + exp_operator(-k * (rho - x0))))  # type: ignore


class AlphaHeavisideFunc(AlphaFunction):
    def __call__(self, rho: np.ndarray|dolfinx.fem.Function):
        return self.alpha_fun(rho, **self.func_args)
    
    @staticmethod
    def alpha_fun(rho: np.ndarray|dolfinx.fem.Function, alpha_min=1e-12, alpha_max=100.0, eta=0.5, beta=20.0, inverse=True):
        tanh_operator = ufl.tanh if isinstance(rho, dolfinx.fem.Function) else np.tanh
        numerator = tanh_operator(beta * eta) + tanh_operator(beta * (rho - eta))  # type: ignore
        denominator = tanh_operator(beta * eta) + tanh_operator(beta * (1.0 - eta))  # type: ignore
        H_smooth = numerator / denominator
        if inverse:
            return alpha_min + (alpha_max - alpha_min) * (1.0 - H_smooth)  # type: ignore
        else:
            return alpha_min + (alpha_max - alpha_min) * H_smooth  # type: ignore
    

# -------------------------------------
class TopoGradientConstraintTerm(object):
    def __init__(
        self, name: str, init_priority: int, 
        set_idxs: np.ndarray, set_value: np.ndarray, 
        init_method: str, constraint_method: str
    ):
        self.name: str = name
        self.init_priority: int = init_priority
        self.init_method: str = init_method
        self.set_idxs: np.ndarray = set_idxs
        self.set_value: np.ndarray = set_value
        self.constraint_method: str = constraint_method
    
    def apply_constraint_gradient(self, gradient_np: np.ndarray):
        if self.constraint_method == 'fixed':
            gradient_np[self.set_idxs] = self.set_value
        elif self.constraint_method == 'zero':
            gradient_np[self.set_idxs] = 0.0
        elif self.constraint_method == 'additional':
            gradient_np[self.set_idxs] = gradient_np[self.set_idxs] + self.set_value
        return gradient_np

    @staticmethod
    def from_boundary(mesh: MeshObj, boundary_info: FluidBoundary):
        if boundary_info.boundary_dim == 'facet':
            set_idxs = mesh.extract_surface_indices(boundary_info.marker)
            if boundary_info.boundary_type in ['velocity', 'pressure']:
                init_method = 'raw'
                set_value = np.ones(shape=set_idxs.shape[0])
                init_priority = 1
                
            elif boundary_info.boundary_type == 'wall':
                init_method = 'obstacle'
                set_value = np.zeros(shape=set_idxs.shape[0])
                init_priority = 2
                
            else:
                raise ValueError(f"Invalid boundary type: {boundary_info.boundary_type}")
            constraint_method = 'zero'
            
        elif boundary_info.boundary_dim == 'cell':
            set_idxs = mesh.extract_cell_indices(boundary_info.marker, point_space=True)
            if boundary_info.boundary_type != 'wall':
                raise ValueError(f"Invalid boundary type: {boundary_info.boundary_type}")
            init_method = 'obstacle'
            set_value = np.zeros(shape=set_idxs.shape[0])
            constraint_method = 'zero'
            init_priority = 2
            
        else:
            raise ValueError(f"Invalid boundary dimension: {boundary_info.boundary_dim}")
        
        return TopoGradientConstraintTerm(
            name=boundary_info.name, 
            init_priority=init_priority, 
            set_idxs=set_idxs, 
            set_value=set_value, 
            init_method=init_method, 
            constraint_method=constraint_method
        )


class TopoWeightProjector(object):
    def __init__(
        self, 
        mesh: SequenceMeshObj, 
        contain_positions: np.ndarray,
        additional_value: float,
        boundaries: list[AutoFluidBoundary],
        relax_factor: int, filter_threshold: float = 0.5
    ):
        self.mesh = mesh
        self.additional_value = additional_value
        self.relax_factor = relax_factor
        self.filter_threshold = filter_threshold
        self.boundaries = boundaries
        self.contain_idxs: np.ndarray = self.init_contain_idxs_from_positions(contain_positions)
        
        self.point_tree = KDTree(self.mesh.points[:, :self.mesh.tdim])
    
    def init_contain_idxs_from_positions(self, positions: np.ndarray):
        point_tree = KDTree(self.mesh.points[:, :self.mesh.tdim])
        contain_idxs = point_tree.query(positions, k=1, return_distance=False)
        contain_idxs = np.concatenate(contain_idxs, axis=-1)
        contain_idxs = np.unique(contain_idxs)
        return contain_idxs
    
    def project(self, topo_map_np: np.ndarray):        
        current_conn_map = np.zeros(shape=(self.mesh.num_points,))
        current_conn_map[topo_map_np >= self.filter_threshold] = 1.0
        clusters = self.mesh.extract_clusters(current_conn_map, cell_mode=False, contain_idxs=self.contain_idxs.copy())
                
        if len(clusters) > 1:
            print(f'[DEBUG]: Projecting topo weight with multiple clusters')
            for cluster in clusters:
                self.mesh.extract_cluster_feature(cluster)
            
            reachable_map = np.ones((self.mesh.num_points,))
            for boundary in self.boundaries:
                if boundary.boundary_type == 'wall':
                    if boundary.boundary_dim == 'facet':
                        bry_indices = self.mesh.extract_surface_indices(boundary.marker)
                        reachable_map[bry_indices] = 0.0
                    elif boundary.boundary_dim == 'cell':
                        bry_indices = self.mesh.extract_cell_indices(boundary.marker, point_space=True)
                        reachable_map[bry_indices] = 0.0
            
            conn_set_dict = self.mesh.compute_distance_to_two_closest_clusters(clusters, reachable_map)
            
            new_topo_map_np = topo_map_np.copy()
            nearest_idxs_list = []
            for _, conn_set in conn_set_dict.items():
                conn_idxs = conn_set.get_conn_idxs(relax_factor=self.relax_factor)
                new_topo_map_np[conn_idxs] += self.additional_value
                
            #     nearest_idxs = self.point_tree.query_radius(self.mesh.points[conn_idxs, :self.mesh.tdim], r=0.03, return_distance=False)
            #     nearest_idxs_list.append(np.unique(np.concatenate(nearest_idxs, axis=-1)))
            # nearest_idxs_list = np.concatenate(nearest_idxs_list, axis=-1)
            # nearest_idxs_list = np.unique(nearest_idxs_list)
            # new_topo_map_np[nearest_idxs_list] += self.additional_value
            
            # ------ debug visualize ------
            plotter = pyvista.Plotter()
            plotter.add_mesh(self.mesh.grid, style='wireframe', color='black')
            self.mesh.draw_cluster_feature(clusters, plotter=plotter)
            for _, conn_set in conn_set_dict.items():
                point_mesh = pyvista.PointSet(self.mesh.points[conn_set.get_conn_idxs(relax_factor=self.relax_factor), :])
                plotter.add_mesh(point_mesh, color='green', point_size=10)
            plotter.show()
            # ------ debug visualize ------
            
            """
            # ------ debug visualize ------
            plotter = pyvista.Plotter()
            # plotter.add_mesh(self.mesh.grid, style='wireframe', color='black')
            
            xlength = np.max(self.mesh.points[:, 0])
            y_length = np.max(self.mesh.points[:, 1])
            
            before_grid = self.mesh.grid.copy()
            before_grid.points[:, 1] += y_length + 0.1  # type: ignore
            before_grid.point_data["reachable_map"] = (topo_map_np > self.filter_threshold).astype(np.float32)
            before_grid.warp_by_scalar("reachable_map", inplace=True)
            plotter.add_mesh(before_grid, style='surface')
            
            after_grid = self.mesh.grid.copy()
            after_grid.points[:, 0] += xlength + 0.1  # type: ignore
            after_grid.point_data["reachable_map"] = (new_topo_map_np > self.filter_threshold).astype(np.float32)
            after_grid.warp_by_scalar("reachable_map", inplace=True)
            plotter.add_mesh(after_grid, style='surface')
            
            self.mesh.draw_cluster_feature(clusters, plotter=plotter)
            for _, conn_set in conn_set_dict.items():
                point_mesh = pyvista.PointSet(self.mesh.points[conn_set.get_conn_idxs(relax_factor=self.relax_factor)])
                plotter.add_mesh(point_mesh, color='green', point_size=8)
            plotter.show()
            # ------ debug visualize ------
            """
            
            return new_topo_map_np
        
        else:
            return topo_map_np


# -------------------------------------
class TopoFluidOptimizer(FluidSimulator):
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
        viscosity_init: float|np.ndarray,
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
        self.heat_boundaries: dict[str, DolfinxDirichletBC] = {}
        
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
        
        self.use_alpha_sigmoid = True
        # if alpha_fun['name'] == 'exponential':
        #     self.alpha_func = AlphaExponentialFunc(alpha_fun['name'], alpha_fun['args'])
        if alpha_fun['name'] == 'sigmoid':
            self.alpha_func = AlphaSigmoidFunc(alpha_fun['name'], alpha_fun['args'])
        elif alpha_fun['name'] == 'heaviside':
            self.alpha_func = AlphaHeavisideFunc(alpha_fun['name'], alpha_fun['args'])
            self.use_alpha_sigmoid = False
        else:
            raise ValueError(f"Invalid alpha function: {alpha_fun['name']}")
        
        self.topo_weight_init = 0.9
        self.topo_weight_constraint_terms: dict[str, TopoGradientConstraintTerm] = {}
        self.topo_weight: DolfinxFunction
        
        self.result_dir = result_dir
        self.recorder = VtuRecorder(self.mesh, save_dir=self.result_dir, is_binary=True)
        
        self.weight_filter_radius = weight_filter_radius
        self.is_topo_cell = False
        self.weight_filter = DistanceBandFilter(self.mesh.grid, weight_filter_radius, self.is_topo_cell, True)
        
        self.control_phi: Control
        self.opt_problem: ReducedFunctional
    
    def setup_tape(self, tape: Tape|None = None):
        if tape is None:
            self.tape = Tape()
        else:
            self.tape = tape
        set_working_tape(self.tape)
    
    def define_fluid_govern_equation(
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
    
    def compute_gradient(self, adj_input: float, with_filter=True, with_norm=True) -> np.ndarray:
        dJdm = self.opt_problem.derivative(adj_input=float(adj_input))[0]
        
        # ------ apply constraint to gradient ------
        for _, term in self.topo_weight_constraint_terms.items():
            dJdm = term.apply_constraint_gradient(dJdm)
        
        if with_filter:
            dJdm = dJdm.reshape((-1, 1))
            dJdm = self.weight_filter.run(dJdm)
            dJdm = np.asarray(dJdm).reshape((-1))
        
        # ------ apply constraint to gradient again ------
        for _, term in self.topo_weight_constraint_terms.items():
            dJdm = term.apply_constraint_gradient(dJdm)
        
        if with_norm:
            norm = np.max(np.abs(dJdm))
            dJdm = dJdm / norm
        
        return dJdm
    
    def init_topo_weight(self, topo_weight_fun: DolfinxFunction):
        init_weight = dolfinx.fem.Function(self.point_scalar_func_space)
        init_weight_value = topo_weight_fun.x.array.copy()  # type: ignore
        
        sorted_terms = [term for term in sorted(self.topo_weight_constraint_terms.values(), key=lambda x: x.init_priority)]
        for term in sorted_terms:
            init_weight_value[term.set_idxs] = term.set_value
        init_weight.x.array[:] = init_weight_value  # type: ignore

        topo_weight_fun.interpolate(init_weight)
    
    def define_fluid_boundary(self):
        for name, bry_info in self.boundaries.items():
            assert bry_info.boundary_type in ['velocity', 'pressure', 'wall'], 'Invalid boundary type'
            
            if bry_info.boundary_type == 'velocity':
                if bry_info.boundary_dim != 'facet':
                    raise ValueError(f"Invalid boundary dimension: {bry_info.boundary_dim}")
                
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
                if bry_info.boundary_dim != 'facet':
                    raise ValueError(f"Invalid boundary dimension: {bry_info.boundary_dim}")
                
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
                
                if bry_info.boundary_dim == 'facet':
                    dofs = MeshUtils.extract_entity_dofs(
                        (self.W0, self.V), self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    )
                elif bry_info.boundary_dim == 'cell':
                    entity_indices = self.mesh.cell_tags.find(bry_info.marker)
                    dofs = MeshUtils.extract_entity_dofs((self.W0, self.V), self.mesh.tdim, entity_indices)
                else:
                    raise ValueError(f"[Error]: {bry_info.boundary_dim} boundary is not supported for wall boundary.")
                
                bry_info.bry_cell = DolfinxDirichletBC(
                    name=name,
                    bc_func_space=self.W0,
                    value_func_space=self.V,
                    dofs=dofs,
                    bc_value=wall_func,  # type: ignore
                    is_mixed_element=True
                )
            
            else:
                raise ValueError(f"Invalid boundary type: {bry_info.boundary_type}")
        
        for name, bry_info in self.boundaries.items():
            self.topo_weight_constraint_terms[name] = TopoGradientConstraintTerm.from_boundary(self.mesh, bry_info)

    def add_topo_weight_constraint_term(
        self, name: str, init_priority: int, 
        set_idxs: np.ndarray, set_value: np.ndarray, 
        init_method: str, constraint_method: str
    ):
        self.topo_weight_constraint_terms[name] = TopoGradientConstraintTerm(
            name=name, 
            init_priority=init_priority, 
            set_idxs=set_idxs, 
            set_value=set_value, 
            init_method=init_method, 
            constraint_method=constraint_method
        )
    
    def get_bcs(self):
        """set wall boundary at last"""        
        return [bry_info.bry_cell for bry_info in sorted(self.boundaries.values(), key=lambda x: x.priority)]

    def prepare_cost_args(self, u: ufl.tensors.ListTensor, p: ufl.indexed.Indexed, obj_cost_args: dict, **kwargs):
        cost_parse_dict = {}
        
        if obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
            with_viscosity = obj_cost_args['with_viscosity']
            if with_viscosity:
                min_dissipation_form = kwargs['viscosity'] * inner(grad(u), grad(u)) * ufl.dx
            else:
                min_dissipation_form = inner(grad(u), grad(u)) * ufl.dx
            penalized_term = 0.5 * AlphaSigmoidFunc.alpha_fun(
                kwargs['rho'], 
                alpha_min=obj_cost_args.get('penalized_alpha_min', 1e-12), 
                alpha_max=obj_cost_args.get('penalized_alpha_max', 1000.0), 
                x0=0.5, 
                k=obj_cost_args.get('penalized_alpha_k', 20.0),
                inverse=True
            ) * inner(u, u) * ufl.dx  # type: ignore
            
            cost_parse_dict.update({
                'min_dissipation_form': min_dissipation_form,
                'penalized_term': penalized_term,
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
        
        else:
            raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')

        obj_cost_args.update(cost_parse_dict)
        
        return obj_cost_args
    
    def prepare_reg_args(self, reg_cost_args: dict):
        if 'target_ratio' in reg_cost_args.keys():
            target_volume = self.volume * reg_cost_args['target_ratio']
        else:
            target_volume = reg_cost_args['target_volume']
        if self.use_alpha_sigmoid:
            reg_cost_args.update({
                'target_volume': target_volume,
                'vol_form': AlphaSigmoidFunc.alpha_fun(
                    self.topo_weight, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=10.0, inverse=False
                ) * ufl.dx
            })
        else:
            reg_cost_args.update({
                'target_volume': target_volume,
                'vol_form': AlphaHeavisideFunc.alpha_fun(
                    self.topo_weight, alpha_min=1e-12, alpha_max=1.0, eta=0.5, beta=20.0, inverse=False
                ) * ufl.dx
            })
        return reg_cost_args

    def define_fluid_problem(
        self,
        obj_cost_args_list: list[dict],
        reg_cost_args: dict,
        heat_problem_args: dict = {
            'use_heat_problem': False,
            'weight': 1.0,
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
            'critical_method': 'residual',
            'mat_assemble_method': 'lift',
            'with_residual': False
        },
        forward_kwargs: dict = {
            'mat_assemble_method': 'lift',
            'with_residual': False
        }
    ):
        self.define_fluid_boundary()
        with start_annotation():
            uh = DolfinxFunction(self.W, name=f"{self.name}_state")
            u, p = ufl.split(uh)  # type: ignore # please don't use up.split()
            vq = DolfinxFunction(self.W, name=f"{self.name}_adjoint")
            v, q = ufl.split(vq)  # type: ignore # please don't use up.split()
            
            f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
            self.viscosity = DolfinxFunction(self.point_scalar_func_space, name=f"{self.name}_viscosity")
            self.viscosity.x.array[:] = self.viscosity_init  # type: ignore
                    
            govern_form = self.define_fluid_govern_equation(u, p, v, q, f, self.viscosity)  # type: ignore
            
            if self.is_topo_cell:
                self.topo_weight = DolfinxFunction(self.cell_scalar_func_space, name=f'topo_weight')  # type: ignore
            else:
                self.topo_weight = DolfinxFunction(self.point_scalar_func_space, name=f'topo_weight')  # type: ignore
            self.topo_weight.x.array[:] = self.topo_weight_init  # type: ignore
            self.init_topo_weight(self.topo_weight)
            
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
            u, p = ufl.split(self.uh)  # type: ignore
            
            # ------ define object cost ------
            self.obj_cost_args_list = obj_cost_args_list
            for obj_idx, obj_cost_args in enumerate(self.obj_cost_args_list):
                obj_cost_args = self.prepare_cost_args(
                    u=u, p=p, obj_cost_args=obj_cost_args,  # type: ignore
                    viscosity=self.viscosity, alpha=alpha, rho=self.topo_weight,
                )
                if obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
                    obj_cost = assemble_scalar(
                        form=(
                            obj_cost_args['dissipation_weight'] * obj_cost_args['min_dissipation_form'] 
                            + obj_cost_args['penalized_weight'] * obj_cost_args['penalized_term']
                        ),
                        domain=self.domain, tape=self.tape
                    )
                elif obj_cost_args['style'] == 'FixedFlux':
                    with_rescale = obj_cost_args['with_rescale']
                    with_max_dif_cost = obj_cost_args['with_max_dif_cost']
                    
                    if with_max_dif_cost:
                        max_dif_cost, max_dif_weight = 0.0, 0.0
                        
                    for i, (marker, outlet_item) in enumerate(obj_cost_args['outlet_cost_forms'].items()):
                        target_flux = outlet_item['target_flux']
                        outlet_flux = assemble_scalar(form=outlet_item['flux_form'], domain=self.domain, tape=self.tape)
                        print(f"{marker} flux: {outlet_flux}, target: {target_flux}, inlet_flux: {obj_cost_args['inlet_flux']}")
                                                
                        flux_dif = outlet_flux - target_flux
                        if with_rescale:
                            flux_dif = flux_dif / target_flux
                        flux_dif_cost = flux_dif ** 2
                        
                        if with_max_dif_cost:
                            max_dif_cost = max(max_dif_cost, flux_dif_cost)
                            max_dif_weight = outlet_item['weight']
                        
                        if i == 0:
                            obj_cost = outlet_item['weight'] * flux_dif_cost
                        else:
                            obj_cost += outlet_item['weight'] * flux_dif_cost
                            
                    obj_cost = obj_cost / obj_cost_args['num_outlets']
                    if with_max_dif_cost:
                        obj_cost += max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)
                else:
                    raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')
            
                if obj_idx == 0:
                    all_obj_cost = obj_cost
                else:
                    all_obj_cost += obj_cost  # type: ignore

            # ------ define regularization cost ------
            self.reg_cost_args = self.prepare_reg_args(reg_cost_args)
            topo_volume = assemble_scalar(self.reg_cost_args['vol_form'], domain=self.domain, tape=self.tape)
            
            align_volume = AssembleUtils.assemble_scalar(self.reg_cost_args['vol_form'])
            if self.reg_cost_args['target_volume'] < align_volume:
                align_volume = align_volume * 0.99
            else:
                align_volume = align_volume * 1.01
            align_volume = AdjFloat(align_volume)
            
            if self.reg_cost_args['reg_rescale']:
                volume_dif = topo_volume / align_volume - 1.0  # type: ignore
                reg_cost = self.reg_cost_args['weight'] * (volume_dif ** 2)
            else:
                reg_cost = self.reg_cost_args['weight'] * ((topo_volume - align_volume) ** 2)  # type: ignore
            
            loss_all = all_obj_cost + reg_cost
            
            """
            # TODO: Minimum length scale in topology optimization by geometric constraints.pdf
            # ------ define Helmholtz-type filter ------
            filter_topo = DolfinxFunction(self.point_scalar_func_space, name=f'filter_topo')
            filter_topo_test = DolfinxFunction(self.point_scalar_func_space, name=f'filter_topo_test')
            
            min_length_scale = 0.03
            a_filter_form = (
                min_length_scale**2 * ufl.inner(ufl.grad(filter_topo), ufl.grad(filter_topo_test)) * ufl.dx  # type: ignore
                + ufl.inner(filter_topo, filter_topo_test) * ufl.dx
            )
            L_filter_form = ufl.inner(self.topo_weight, filter_topo_test) * ufl.dx
            F_filter_form = a_filter_form - L_filter_form

            filter_topo = equation_solve(
                domain=self.domain,
                F_form=F_filter_form,
                state=filter_topo,  # type: ignore
                adjoint=filter_topo_test,  # type: ignore
                bcs=[],
                is_linear=True,
                tlm_ksp_option=tlm_ksp_option,
                adj_ksp_option=adj_ksp_option,
                forward_ksp_option=forward_ksp_option,
                tlm_kwargs=tlm_kwargs,
                adj_kwargs=adj_kwargs,
                forward_kwargs=forward_kwargs,
                tape=self.tape,
            )
            
            # ------ define geometric constraints ------
            c = 200.0
            if self.use_alpha_sigmoid:
                physical_topo = AlphaSigmoidFunc.alpha_fun(filter_topo, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
            else:
                physical_topo = AlphaHeavisideFunc.alpha_fun(filter_topo, alpha_min=1e-12, alpha_max=1.0, eta=0.5, beta=20.0, inverse=False)  # type: ignore
                
            I_solid = physical_topo * ufl.exp(-c * ufl.dot(ufl.grad(filter_topo), ufl.grad(filter_topo)))  # type: ignore
            I_volid = (1.0 - physical_topo) * ufl.exp(-c * ufl.dot(ufl.grad(filter_topo), ufl.grad(filter_topo)))  # type: ignore
            
            g_solid_form = I_solid * (ufl.min_value(filter_topo - 0.75, 0.0)) ** 2 * ufl.dx  # type: ignore
            g_volid_form = I_volid * (ufl.min_value(0.25 - filter_topo, 0.0)) ** 2 * ufl.dx  # type: ignore
            
            loss_solid = assemble_scalar(g_solid_form, domain=self.domain, tape=self.tape)
            loss_volid = assemble_scalar(g_volid_form, domain=self.domain, tape=self.tape)
            
            self.filter_topo = filter_topo
            self.physical_topo = physical_topo
                
            self.I_solid_expr = DolfinUtils.create_expression(I_solid, self.point_scalar_func_space)  # type: ignore
            self.I_volid_expr = DolfinUtils.create_expression(I_volid, self.point_scalar_func_space)  # type: ignore
            self.g_solid_form = g_solid_form
            self.g_volid_form = g_volid_form
            
            self.g_solid_weight = 1.0
            self.g_volid_weight = 1.0
            
            loss_all = loss_all + self.g_solid_weight * loss_solid + self.g_volid_weight * loss_volid  # type: ignore
            """
            
            self.heat_problem_args = heat_problem_args
            if self.heat_problem_args['use_heat_problem']:
                """
                # ------ define Helmholtz-type filter ------
                filter_topo = DolfinxFunction(self.point_scalar_func_space, name=f'filter_topo')
                filter_topo_test = DolfinxFunction(self.point_scalar_func_space, name=f'filter_topo_test')
                
                min_length_scale = 0.02
                a_filter_form = (
                    min_length_scale**2 * ufl.inner(ufl.grad(filter_topo), ufl.grad(filter_topo_test)) * ufl.dx  # type: ignore
                    + ufl.inner(filter_topo, filter_topo_test) * ufl.dx
                )
                L_filter_form = ufl.inner(self.topo_weight, filter_topo_test) * ufl.dx
                F_filter_form = a_filter_form - L_filter_form

                filter_topo = equation_solve(
                    domain=self.domain,
                    F_form=F_filter_form,
                    state=filter_topo,  # type: ignore
                    adjoint=filter_topo_test,  # type: ignore
                    bcs=[],
                    is_linear=True,
                    tlm_ksp_option=tlm_ksp_option,
                    adj_ksp_option=adj_ksp_option,
                    forward_ksp_option=forward_ksp_option,
                    tlm_kwargs=tlm_kwargs,
                    adj_kwargs=adj_kwargs,
                    forward_kwargs=forward_kwargs,
                    tape=self.tape,
                )
                """
                
                # ------ define heat problem ------
                self.define_heat_boundary()
                
                tempature = DolfinxFunction(self.point_scalar_func_space, name='tempature')
                tempature_test = DolfinxFunction(self.point_scalar_func_space, name='tempature_test')
                
                heat_source = DolfinxFunction(self.point_scalar_func_space, name='heat_source')
                for name, bry_info in self.boundaries.items():
                    if bry_info.boundary_type == 'pressure':
                        outlet_dofs = MeshUtils.extract_entity_dofs(
                            self.point_scalar_func_space, self.mesh.fdim,
                            MeshUtils.extract_entity_indices(self.mesh.domain, style='facet', marker=bry_info.marker, mesh_tags=self.mesh.facet_tags)
                        )
                        heat_source.x.array[outlet_dofs] = -1.0  # type: ignore
                
                # heat_conn = AlphaExponentialFunc.alpha_fun(filter_topo, alpha_min=1e-8, alpha_max=0.1, p=3.0, inverse=False)  # type: ignore
                # heat_conn = AlphaSigmoidFunc.alpha_fun(filter_topo, alpha_min=1e-8, alpha_max=0.1, x0=0.55, k=20.0, inverse=False)  # type: ignore
                heat_conn = AlphaExponentialFunc.alpha_fun(self.topo_weight, alpha_min=1e-8, alpha_max=0.1, p=3.0, inverse=False)  # type: ignore
                
                F_heat_form = (
                    heat_conn * ufl.dot(ufl.grad(tempature), ufl.grad(tempature_test)) * ufl.dx  # type: ignore
                    + heat_source * tempature_test * ufl.dx  # type: ignore
                )
                
                tempature = equation_solve(
                    domain=self.domain,
                    F_form=F_heat_form,
                    state=tempature,  # type: ignore
                    adjoint=tempature_test,  # type: ignore
                    bcs=[self.heat_boundaries[name] for name in self.heat_boundaries.keys()],
                    is_linear=True,
                    tape=self.tape,
                )
                                                
                self.heat_problem_args['boundary_info'] = {}
                heat_problem_loss = 0.0
                for name, bry_info in self.boundaries.items():
                    if bry_info.boundary_type == 'pressure':
                        face_area = AssembleUtils.assemble_scalar(dolfinx.fem.Constant(self.mesh.domain, 1.0) * self.ds(bry_info.marker))  # type: ignore
                        tempature_avg_form = tempature / face_area * self.ds(bry_info.marker)
                        tempature_avg: float = assemble_scalar(tempature_avg_form, domain=self.domain, tape=self.tape)  # type: ignore
                        heat_problem_loss = heat_problem_loss + tempature_avg
                        self.heat_problem_args['boundary_info'][name] = {
                            'face_area': face_area,
                            'tempature_avg_form': tempature_avg_form,
                        }
                
                heat_weight = AdjFloat(self.heat_problem_args['weight'])
                loss_all = loss_all + heat_weight * heat_problem_loss
                
                # self.filter_topo = filter_topo
                self.heat_conn_expr = DolfinUtils.create_expression(heat_conn, self.point_scalar_func_space)  # type: ignore
                self.tempature = tempature
            
            # ------ define optimization problem ------
            self.control_phi = Control(self.topo_weight)  # type: ignore
            self.control_align_volume = Control(align_volume)  # type: ignore
            if self.heat_problem_args['use_heat_problem']:
                self.control_heat_weight = Control(heat_weight)
                self.opt_problem = ReducedFunctional(loss_all, [self.control_phi, self.control_align_volume, self.control_heat_weight])
            else:
                self.opt_problem = ReducedFunctional(loss_all, [self.control_phi, self.control_align_volume])
        
        self.record_Upw(0)
        loss_dict, log_dict = self.eval()
        print('[Log]: Initial loss:', loss_dict)
    
    def eval(self):
        eval_dict = {}
        log_dict = {'cost': {}}
        
        # ------ compute object cost ------
        for obj_cost_args in self.obj_cost_args_list:
            if obj_cost_args['style'] == 'MinDissipation+BrinkmanPenalized':
                min_dissipation_cost = obj_cost_args['dissipation_weight'] * AssembleUtils.assemble_scalar(obj_cost_args['min_dissipation_form'])
                penalized_cost = obj_cost_args['penalized_weight'] * AssembleUtils.assemble_scalar(obj_cost_args['penalized_term'])
                eval_dict['MinDissCost'] = min_dissipation_cost
                eval_dict['PenalizedCost'] = penalized_cost
                log_dict['cost']['MinDissCost'] = eval_dict['MinDissCost']
                log_dict['cost']['PenalizedCost'] = eval_dict['PenalizedCost']
            
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
                eval_dict['FixedFluxCost'] = float(np.mean(outlet_cost_list))
                log_dict['cost']['FixedFluxCost'] = eval_dict['FixedFluxCost']
                if obj_cost_args['with_max_dif_cost']:
                    eval_dict[f'FixedFluxCost'] += max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)
                    log_dict['cost'][f'FixedFluxCost_max_dif'] = max_dif_cost * obj_cost_args.get('max_dif_weight', max_dif_weight)      
            
            else:
                raise ValueError(f'Invalid cost style: {obj_cost_args['style']}')
            
        # ------ compute regularization cost ------
        align_volume = self.control_align_volume.data()  # type: ignore
        topo_volume = AssembleUtils.assemble_scalar(self.reg_cost_args['vol_form'])
        if self.reg_cost_args['reg_rescale']:
            volume_dif = topo_volume / align_volume - 1.0
            eval_dict['RegCost'] = self.reg_cost_args['weight'] * float(volume_dif ** 2)
        else:
            eval_dict['RegCost'] = self.reg_cost_args['weight'] * (float(align_volume - topo_volume) ** 2)
        log_dict['cost']['RegCost'] = eval_dict['RegCost']
        log_dict['volume'] = topo_volume
        
        """
        # ------ compute geometric constraints ------
        loss_solid = AssembleUtils.assemble_scalar(self.g_solid_form)
        loss_volid = AssembleUtils.assemble_scalar(self.g_volid_form)
        eval_dict['solidCost'] = self.g_solid_weight * loss_solid
        eval_dict['volidCost'] = self.g_volid_weight * loss_volid
        log_dict['cost']['solidCost'] = eval_dict['solidCost']
        log_dict['cost']['volidCost'] = eval_dict['volidCost']
        """
        
        if self.heat_problem_args['use_heat_problem']:
            if 'heat_cost' not in log_dict.keys():
                log_dict['heat_cost'] = {}
                
            heat_problem_loss = 0.0
            for name, bry_info in self.heat_problem_args['boundary_info'].items():
                tempature_avg = AssembleUtils.assemble_scalar(bry_info['tempature_avg_form'])
                log_dict['heat_cost'][name] = tempature_avg
                heat_problem_loss = heat_problem_loss + tempature_avg
            eval_dict['HeatProblemCost'] = self.control_heat_weight.data() * heat_problem_loss  # type: ignore
            log_dict['cost']['HeatProblemCost'] = eval_dict['HeatProblemCost']
        
        # ------ aggregate cost ------        
        log_dict['loss_all'] = np.sum(list(eval_dict.values()))
        
        return eval_dict, log_dict
    
    def record_Upw(self, step: int, gradient: np.ndarray|None=None):
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
            
            self.recorder.update_point_data('topo_binary', (self.topo_weight.x.array >= 0.5).astype(np.float32))
            self.recorder.update_cell_data('topo_binary', (cell_scalar_fun.x.array >= 0.5).astype(np.float32))  # type: ignore
            
            if self.use_alpha_sigmoid:
                self.recorder.update_point_data(
                    'alpha', AlphaSigmoidFunc.alpha_fun(self.topo_weight.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
                )
                self.recorder.update_cell_data(
                    'alpha', AlphaSigmoidFunc.alpha_fun(cell_scalar_fun.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False)  # type: ignore
                )
            else:
                self.recorder.update_point_data(
                    'alpha', AlphaHeavisideFunc.alpha_fun(self.topo_weight.x.array, alpha_min=1e-12, alpha_max=1.0, eta=0.5, beta=20.0, inverse=False)  # type: ignore
                )
                self.recorder.update_cell_data(
                    'alpha', AlphaHeavisideFunc.alpha_fun(cell_scalar_fun.x.array, alpha_min=1e-12, alpha_max=1.0, eta=0.5, beta=20.0, inverse=False)  # type: ignore
                )
            
        # ------ record obstacle source ------
        point_obs = np.zeros(shape=(self.mesh.num_points))
        cell_obs = np.zeros(shape=(self.mesh.num_cells))
        for bry_info in self.boundaries.values():
            if bry_info.boundary_type == 'wall':
                if bry_info.boundary_dim == 'cell':
                    point_insides =MeshUtils.extract_entity_dofs(
                        DolfinUtils.point_scalar_function_space(self.mesh.domain), 
                        self.mesh.tdim, self.mesh.cell_tags.find(bry_info.marker)
                    )
                    point_obs[point_insides] = 1
                    
                    cell_insides =MeshUtils.extract_entity_dofs(
                        DolfinUtils.cell_scalar_function_space(self.mesh.domain), 
                        self.mesh.tdim, self.mesh.cell_tags.find(bry_info.marker)
                    )
                    cell_obs[cell_insides] = 1
        
        if point_obs.sum() > 0:
            self.recorder.update_point_data('obstacle', point_obs)
            self.recorder.update_cell_data('obstacle', cell_obs)
        
        """
        # ------ record geometric constraints ------
        self.recorder.update_point_data('filter_topo', self.filter_topo.x.array)  # type: ignore
        if self.use_alpha_sigmoid:
            self.recorder.update_point_data('physical_topo', AlphaSigmoidFunc.alpha_fun(
                self.filter_topo.x.array, alpha_min=1e-12, alpha_max=1.0, x0=0.5, k=20.0, inverse=False  # type: ignore
            ))
        else:
            self.recorder.update_point_data('physical_topo', AlphaHeavisideFunc.alpha_fun(
                self.filter_topo.x.array, alpha_min=1e-12, alpha_max=1.0, eta=0.5, beta=20.0, inverse=False  # type: ignore
            ))

        I_solid_func = dolfinx.fem.Function(self.point_scalar_func_space)
        I_solid_func.interpolate(self.I_solid_expr)  # type: ignore
        self.recorder.update_point_data('I_solid', I_solid_func.x.array)  # type: ignore
        I_volid_func = dolfinx.fem.Function(self.point_scalar_func_space)
        I_volid_func.interpolate(self.I_volid_expr)  # type: ignore
        self.recorder.update_point_data('I_volid', I_volid_func.x.array)  # type: ignore
        """
        
        # ------ record heat problem ------
        if self.heat_problem_args['use_heat_problem']:
            # self.recorder.update_point_data('filter_topo', self.filter_topo.x.array)  # type: ignore
            self.recorder.update_point_data('tempature', self.tempature.x.array)  # type: ignore
            
            heat_conn = dolfinx.fem.Function(self.point_scalar_func_space)
            heat_conn.interpolate(self.heat_conn_expr)  # type: ignore
            self.recorder.update_point_data('heat_conn', heat_conn.x.array)  # type: ignore
        
        if gradient is not None:
            self.recorder.update_point_data('gradient', gradient)
        
        self.recorder.save(step)
    
    def run_along_volume(
        self,
        min_lr: float = 1e-2,
        max_lr: float = 1e-1,
        clip_lr=False,
        max_iteration=200,
        recorder: TensorBoardRecorder|None = None,
        result_vtu_file: str|None = None,
        feedback_func: Callable|None = None,
    ) -> dict:
        topo_phi = DolfinxFunction(self.topo_weight.function_space, name=f'topo_phi')  # type: ignore
        topo_phi.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        
        loss_dict, log_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(0)
        
        topo_phi_tensor = torch.from_numpy(topo_phi.x.array.copy())  # type: ignore
        topo_phi_tensor.requires_grad = True
        optimizer = torch.optim.Adam([topo_phi_tensor], lr=max_lr)  # type: ignore
        
        scheduler = ReduceLROnPlateau(optimizer, min_lr=min_lr, factor=0.5, patience=10)  # type: ignore
        last_lr = scheduler.get_last_lr()[0]
        volume_stages = np.linspace(self.control_align_volume.data(), self.reg_cost_args['target_volume'], max_iteration)  # type: ignore
        
        stop_process = False
        for step, stage_vol in zip(range(max_iteration), volume_stages):
            step += 1
            if step > 1:
                loss_dict, log_dict = self.eval()
                log = f"[step/{step} loss:{loss_current:.6f} lr:{last_lr:.3f} vol:{stage_vol:.3f}]"
                
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
            
            # ------ update parameters ------
            if clip_lr:
                old_topo_np = topo_phi.x.array.copy()  # type: ignore
                topo_phi_tensor.grad = torch.from_numpy(direction_np)
                optimizer.step()
                
                new_topo_np = topo_phi_tensor.detach().numpy()
                max_update_grad = np.max(np.abs(new_topo_np - old_topo_np))
                lr_limit = last_lr * 2.0
                if max_update_grad > lr_limit:
                    # print(f'[DEBUG]: Reset gradient to {lr_limit}')
                    update_grad = new_topo_np - old_topo_np
                    update_grad = update_grad / np.max(np.abs(update_grad)) * lr_limit
                    new_topo_np = old_topo_np + update_grad
            else:
                topo_phi_tensor.grad = torch.from_numpy(direction_np)
                optimizer.step()
                new_topo_np = topo_phi_tensor.detach().numpy()
            
            new_topo_np = np.maximum(np.minimum(new_topo_np, 1.0 - 1e-16), 1e-16) 
            topo_phi_tensor.data = torch.from_numpy(new_topo_np).to(topo_phi_tensor.dtype)
            topo_phi.x.array[:] = new_topo_np  # type: ignore
            
            # ------ update loss weight ------
            self.control_align_volume.update(stage_vol)
            
            # ------ recompute loss
            if self.heat_problem_args['use_heat_problem']:
                loss_current = self.opt_problem(values=[topo_phi, self.control_align_volume.data(), self.control_heat_weight.data()])    # type: ignore
            else:
                loss_current = self.opt_problem(values=[topo_phi, self.control_align_volume.data()])    # type: ignore
            self.record_Upw(step, gradient=direction_np)
            
            scheduler.step(loss_current)
            last_lr = scheduler.get_last_lr()[0]
            
            if feedback_func is not None:
                stop_process = feedback_func(self)
            
            if stop_process:
                break
                    
        if result_vtu_file is not None:
            last_step_record_file = self.recorder.get_step_file_path(step)
            if not os.path.exists(last_step_record_file):
                last_step_record_file = self.recorder.get_step_file_path(step - 1)
            print(f'[DEBUG]: copy {last_step_record_file} to {result_vtu_file}')
            shutil.copy(last_step_record_file, result_vtu_file)

        return {'is_success': True if not stop_process else False}
    
    def run_raw(
        self,
        min_lr: float = 1e-2,
        max_lr: float = 1e-1,
        max_iteration=200,
        recorder: TensorBoardRecorder|None = None,
        result_vtu_file: str|None = None,
        feedback_func: Callable|None = None,
    ) -> dict:
        topo_phi = DolfinxFunction(self.topo_weight.function_space, name=f'topo_phi')  # type: ignore
        topo_phi.x.array[:] = self.topo_weight.x.array.copy()  # type: ignore
        
        loss_dict, log_dict = self.eval()
        loss_current = np.sum(list(loss_dict.values()))
        self.record_Upw(0)
        
        topo_phi_tensor = torch.from_numpy(topo_phi.x.array.copy())  # type: ignore
        topo_phi_tensor.requires_grad = True
        optimizer = torch.optim.Adam([topo_phi_tensor], lr=max_lr)  # type: ignore
        
        scheduler = ReduceLROnPlateau(optimizer, min_lr=min_lr, factor=0.5, patience=10)  # type: ignore
        last_lr = scheduler.get_last_lr()[0]
        self.control_align_volume.update(self.reg_cost_args['target_volume'])
        
        stop_process = False
        for step in range(max_iteration):
            step += 1
            if step > 1:
                loss_dict, log_dict = self.eval()
                log = f"[step/{step} loss:{loss_current:.6f} lr:{last_lr:.3f} vol:{self.control_align_volume.data():.3f}]"
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
            
            # ------ update parameters ------
            topo_phi_tensor.grad = torch.from_numpy(direction_np)
            optimizer.step()
            new_topo_phi = np.maximum(np.minimum(topo_phi_tensor.detach().numpy(), 1.0 - 1e-16), 1e-16)
            topo_phi.x.array[:] = new_topo_phi  # type: ignore
            
            # ------ recompute loss
            loss_current = self.opt_problem(values=[topo_phi, self.control_align_volume.data()])  # type: ignore
            scheduler.step(loss_current)
            last_lr = scheduler.get_last_lr()[0]
            self.record_Upw(step)
            
            if feedback_func is not None:
                stop_process = feedback_func(self)
            
            if stop_process:
                break
        
        if result_vtu_file is not None:
            last_step_record_file = self.recorder.get_step_file_path(step)
            if not os.path.exists(last_step_record_file):
                last_step_record_file = self.recorder.get_step_file_path(step - 1)
            print(f'[DEBUG]: copy {last_step_record_file} to {result_vtu_file}')
            shutil.copy(last_step_record_file, result_vtu_file)

        return {'is_success': True if not stop_process else False}
    
    def define_heat_boundary(self):
        for name, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'velocity':
                bc_inlet_func = dolfinx.fem.Function(self.point_scalar_func_space, name='heat_inlet_func')
                bc_inlet_func.x.array[:] = 0.0  # type: ignore
                self.heat_boundaries[name] = DolfinxDirichletBC(
                    name=f'heat_{name}',
                    bc_func_space=self.point_scalar_func_space,
                    value_func_space=self.point_scalar_func_space,
                    dofs=MeshUtils.extract_entity_dofs(
                        self.point_scalar_func_space, self.mesh.fdim,
                        MeshUtils.extract_entity_indices(self.mesh.domain, style='facet', marker=bry_info.marker, mesh_tags=self.mesh.facet_tags)
                    ),
                    bc_value=bc_inlet_func,  # type: ignore
                    is_mixed_element=False
                )

   