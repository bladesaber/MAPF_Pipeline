import os
import shutil
from typing import Union, Callable, Literal, Any

import dolfinx
import ufl
from ufl import inner, grad, div
from basix.ufl import element, mixed_element

# strange import order, but it works
import numpy as np
import pandas as pd
import pyvista
from functools import partial

from scripts_py.mesh_utils import MeshUtils, MeshObj
from scripts_py.dolfinx_utils import DolfinUtils, BoundaryUtils
from scripts_py.eqn_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.dolfinx_adjoint.base_type import BoundaryCell


class FluidBoundary(object):
    def __init__(
        self,
        name: str, 
        boundary_type: Literal['velocity', 'pressure', 'wall'],
        value: Union[float, list[float], np.ndarray],
        marker: int,
        boundary_dim: Literal['facet', 'cell'] = 'facet',
        **kwargs
    ):
        if boundary_type == 'velocity':
            assert isinstance(value, (list, np.ndarray)), "[Error]: velocity value must be more than one dimension"
            if isinstance(value, list):
                value = np.array(value)
            assert value.shape[0] == 2 or value.shape[0] == 3
        elif boundary_type == 'pressure':
            assert isinstance(value, float), "[Error]: pressure value must be a scalar"
        elif boundary_type == 'wall':
            assert isinstance(value, float), "[Error]: wall value must be a scalar"
        else:
            raise ValueError("[Error]: Invalid boundary type.")
        
        self.name: str = name
        self.boundary_type = boundary_type
        self.value: np.ndarray|float = value
        self.marker = marker
        self.boundary_dim = boundary_dim
        self.bry_cell: BoundaryCell
        
        self.priority = -1
        if boundary_type in ['velocity', 'pressure']:
            self.priority = 1
        elif boundary_type == 'wall':
            self.priority = 2
        
        if boundary_type == 'velocity':
            assert 'face_center' in kwargs, "[Error]: face_center is required for velocity boundary"
            assert 'face_norm_vector' in kwargs, "[Error]: face_norm_vector is required for velocity boundary"
            assert 'face_radius' in kwargs, "[Error]: face_radius is required for velocity boundary"
        self.face_center = kwargs.pop('face_center', None)
        self.face_norm_vector = kwargs.pop('face_norm_vector', None)
        self.face_radius = kwargs.pop('face_radius', None)
        self.interpolate_method = kwargs.pop('interpolate_method', 'circle')
        self.kwargs = kwargs
    
    @staticmethod
    def inflow_velocity_exp(
        pos: np.ndarray, 
        velocity: np.ndarray, 
        tdim: int, 
        face_center: np.ndarray, 
        face_norm_vector: np.ndarray,
        face_radius: float,
        method: Literal['quadratic', 'circle', 'constant'] = 'circle'
    ):
        bry_values = np.zeros_like(pos.T)
        
        face_norm_vector = face_norm_vector / np.linalg.norm(face_norm_vector)
        face_norm_velocity = np.dot(velocity, face_norm_vector) * face_norm_vector
        bry_values[:, :tdim] = face_norm_velocity
        
        pos = pos.T[:, :tdim]
        dist2center = np.linalg.norm(pos - face_center, ord=2, axis=1)
        dist2plane = np.sum((pos - face_center) * face_norm_vector, axis=1)
       
        if method == 'quadratic':
            # 使用二次函数平滑权重，使得权重在边界处逐渐减小
            weight = np.maximum((face_radius - dist2center) / face_radius, 0.0)
            weight[np.abs(dist2plane) > 1e-6] = 0.0
            weight = -np.power(weight - 1.0, 2.0) + 1.0
        
        elif method == 'circle':
            # 使用圆函数平滑
            positive_bool = dist2center < face_radius
            positive_bool[np.abs(dist2plane) > 1e-6] = False
            
            weight = np.zeros(shape=positive_bool.shape)
            weight[positive_bool] = np.sqrt(1.0 ** 2 - (dist2center[positive_bool] / face_radius) ** 2)
        
        elif method == 'constant':
            positive_bool = dist2center < face_radius
            positive_bool[np.abs(dist2plane) > 1e-6] = False

            weight = np.zeros(shape=positive_bool.shape)
            weight[positive_bool] = 1.0
        else:
            raise ValueError(f'[Error]: Invalid method: {method}')
                
        bry_values = bry_values * weight.reshape((-1, 1))
        bry_values = bry_values.T[:tdim, :]

        # FluidBoundary.visulize_bry_values(pos, bry_values, tdim)
        
        return bry_values

    @staticmethod
    def visulize_bry_values(pos: np.ndarray, bry_values: np.ndarray, tdim: int):
        pos_vis = pos
        if tdim == 2:
            pos_vis = np.concatenate([pos_vis, np.zeros((pos_vis.shape[0], 1))], axis=1)
        
        bry_vis = bry_values.T
        if tdim == 2:
            bry_vis = np.concatenate([bry_vis, np.zeros((bry_vis.shape[0], 1))], axis=1)
        
        pl = pyvista.Plotter()
        pl.add_mesh(pyvista.PolyData(pos_vis), color='red', style='wireframe')
        pl.add_arrows(pos_vis, bry_vis, color='blue', mag=0.1)
        pl.show()

    @staticmethod
    def locate_indices(
        x: np.ndarray, tdim: int, face_center: np.ndarray, face_norm_vector: np.ndarray, face_radius: float, tol: float = 1e-6):
        if x.shape[0] in [2, 3]:
            pos = x.T[:, :tdim]
        else:
            pos = x[:, :tdim]
        face_norm_vector = face_norm_vector / np.linalg.norm(face_norm_vector)
        center2pos = pos - face_center
        dist2plane = np.sum(center2pos * face_norm_vector, axis=1)
        dist2center = np.linalg.norm(center2pos, ord=2, axis=1)
        positive_bool = np.bitwise_and(dist2center <=face_radius, np.abs(dist2plane) > tol)
        return positive_bool


class FluidSimulator(object):
    '''
    based on taylor-hood element
    '''
    def __init__(
        self, 
        name: str, 
        mesh: MeshObj,
        velocity_degree: int = 2,
        pressure_degree: int = 1,
        velocity_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
        pressure_space: Literal['CG', 'DG', 'Lagrange'] = 'Lagrange',
    ):
        self.name = name
        self.mesh = mesh

        self.velocity_element = element(velocity_space, self.mesh.cell_types(), degree=velocity_degree, shape=(self.mesh.tdim,))
        # can't use shape=(1, ), dolfinx will see it as vector of shape 1
        # self.pressure_element = element("Lagrange", self.mesh.cell_types(), degree=1, shape=(1,))  # error
        self.pressure_element = element(pressure_space, self.mesh.cell_types(), degree=pressure_degree)
        self.mix_element = mixed_element([self.velocity_element, self.pressure_element])
        
        self.W = dolfinx.fem.functionspace(self.mesh.domain, self.mix_element)
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()
        
        self.vector_element = element("CG", self.mesh.cell_types(), degree=1, shape=(self.mesh.tdim,))  # point vector
        self.scalar_element = element("CG", self.mesh.cell_types(), degree=1)                           # point scalar
        self.V_map_space = dolfinx.fem.functionspace(self.mesh.domain, self.vector_element)
        self.Q_map_space = dolfinx.fem.functionspace(self.mesh.domain, self.scalar_element)
        self.nu_function_space = dolfinx.fem.functionspace(self.mesh.domain, self.scalar_element)

        self.n_vec = MeshUtils.define_facet_norm(self.mesh.domain)
        self.ds = MeshUtils.define_ds(self.mesh.domain, self.mesh.facet_tags)
        self.dx = MeshUtils.define_dx(self.mesh.domain)
        
        self.is_linear: bool
        self.boundaries: dict[str, FluidBoundary] = {}

    def add_boundary(self, boundary: FluidBoundary):
        self.boundaries[boundary.name] = boundary

    def define_equation(self, **args):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError

    def eval_point_by_dolfin(self, uh: dolfinx.fem.Function, points: np.ndarray):
        cell_indices: list[int] = MeshUtils.cell_indices_of_points(self.mesh.grid, points)  # type: ignore
        return DolfinUtils.function_eval(uh, pcd_xyz=points, cell_indices=cell_indices)


class StokeSimulator(FluidSimulator):
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }

    def __init__(self, name: str, mesh: MeshObj, velocity_degree: int = 2, pressure_degree: int = 1):
        super().__init__(name=name, mesh=mesh, velocity_degree=velocity_degree, pressure_degree=pressure_degree)
        self.is_linear = True
        self.eqn_map: dict[str, Any] = {}

    def define_equation(self, nu_value: float|np.ndarray, body_force=(0., 0., 0.)):
        u, p = ufl.split(ufl.TrialFunction(self.W))  # type: ignore
        v, q = ufl.split(ufl.TestFunction(self.W))  # type: ignore

        nu = dolfinx.fem.Function(self.nu_function_space)
        nu.x.array[:] = nu_value  # type: ignore
        # f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.tdim))
        # f = dolfinx.fem.Constant(self.mesh.domain, body_force[:self.mesh.tdim])
        f = dolfinx.fem.Function(self.V_map_space, name='body_force')
        f.x.array[:] = np.tile(np.array(body_force[:self.mesh.tdim]), (self.mesh.num_points, 1)).reshape(-1)  # type: ignore

        stoke_form = (
            nu * inner(grad(u), grad(v)) * self.dx  # type: ignore
            - inner(p, div(v)) * self.dx
            - inner(q, div(u)) * self.dx
            - inner(f, v) * self.dx
        )
        # print(p.ufl_shape, nu.ufl_shape)
        
        lhs_form = ufl.lhs(stoke_form)  # type: ignore
        rhs_form = ufl.rhs(stoke_form)  # type: ignore
        lhs_form_dolfin = dolfinx.fem.form(lhs_form)  # type: ignore
        rhs_form_dolfin = dolfinx.fem.form(rhs_form)  # type: ignore
        self.eqn_map.update({
            'lhs_form': lhs_form,
            'rhs_form': rhs_form,
            'lhs_form_dolfin': lhs_form_dolfin,
            'rhs_form_dolfin': rhs_form_dolfin,
            'u': u, 'p': p, 'uh': dolfinx.fem.Function(self.W, name='state'), 'nu': nu, 'f': f,
        })

    def define_boundary(self):
        for name, bry_info in self.boundaries.items():
            if bry_info.boundary_type == 'velocity':
                if bry_info.boundary_dim != 'facet':
                    raise NotImplementedError(f"[Error]: {bry_info.boundary_dim} boundary is not supported for velocity boundary.")
                                
                inflow_velocity_exp = partial(
                    FluidBoundary.inflow_velocity_exp,
                    velocity=np.array(bry_info.value),
                    tdim=self.mesh.tdim,
                    face_center=bry_info.face_center, # type: ignore
                    face_norm_vector=bry_info.face_norm_vector, # type: ignore
                    face_radius=bry_info.face_radius, # type: ignore
                    method=bry_info.interpolate_method,
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
                if bry_info.boundary_dim != 'facet':
                    raise NotImplementedError(f"[Error]: {bry_info.boundary_dim} boundary is not supported for pressure boundary.")
                
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
                
                if bry_info.boundary_dim == 'facet':
                    dofs = MeshUtils.extract_entity_dofs(
                        (self.W0, self.V), self.mesh.fdim,
                        MeshUtils.extract_entity_indices(
                            self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=bry_info.marker
                        )
                    )
                elif bry_info.boundary_dim == 'cell':
                    cell_bool = bry_info.kwargs['cell_bool']
                    entity_indices = self.mesh.cell_tags.indices[cell_bool]
                    dofs = MeshUtils.extract_entity_dofs((self.W0, self.V), self.mesh.tdim, entity_indices)
                else:
                    raise ValueError(f"[Error]: {bry_info.boundary_dim} boundary is not supported for wall boundary.")
                
                bry_info.bry_cell = BoundaryCell.define_bry_cell(
                    name=name,
                    bc_func_space=self.W0,
                    value_func_space=self.V,
                    dofs=dofs,
                    bc_value=wall_func,
                    is_mixed_element=True,
                )

            else:
                raise ValueError('[Error]: Invalid Boundary Type.')                    
        
    def get_bcs(self):
        """set wall boundary at last"""        
        return [bry_info.bry_cell.bc for bry_info in sorted(self.boundaries.values(), key=lambda x: x.priority)]

    def run(self, ksp_option=None, mat_assemble_method: Literal['lift', 'identity_row'] = 'lift', with_residual=False) -> dict:
        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.mesh.domain.comm,
            uh=self.eqn_map['uh'],
            a_form=self.eqn_map['lhs_form_dolfin'],
            l_form=self.eqn_map['rhs_form_dolfin'],
            bcs=self.get_bcs(),
            mat_assemble_method=mat_assemble_method,
            ksp_option=ksp_option if (ksp_option is not None) else self.ksp_option,
            with_residual=with_residual
        )
        if with_residual:
            print(f"[Log] Stoke Simulator isSolved:{res_dict['is_solved']}, maxResidual:{res_dict['max_residual']}.")
        return res_dict

    def save(self, file: str):
        assert file.endswith('.vtu')
        uh: dolfinx.fem.Function = self.eqn_map['uh']
        
        vel_ref = dolfinx.fem.Function(self.V_map_space, name='vel_ref')
        vel_ref.interpolate(uh.sub(0).collapse())  # type: ignore
        p_ref = uh.sub(1).collapse()

        vel_ref_np = vel_ref.x.array.reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)
        
        self.mesh.record_point_value('velocity', vel_ref_np)
        self.mesh.record_point_value('pressure', p_ref.x.array)
        self.mesh.record_point_value('velocity_mag', np.linalg.norm(vel_ref_np, axis=1))
        
        cell_vector_func_space = DolfinUtils.cell_vector_function_space(self.mesh.domain)
        vel_cell_func = dolfinx.fem.Function(cell_vector_func_space)
        vel_cell_func.interpolate(vel_ref)  # type: ignore
        
        vel_cell = vel_cell_func.x.array.reshape((-1, self.mesh.tdim))  # type: ignore
        if vel_cell.shape[1] == 2:
            vel_cell = np.concatenate([vel_cell, np.zeros((vel_cell.shape[0], 1))], axis=1)
        
        cell_scalar_func_space = DolfinUtils.cell_scalar_function_space(self.mesh.domain)
        p_cell_func = dolfinx.fem.Function(cell_scalar_func_space)
        p_cell_func.interpolate(p_ref)  # type: ignore
        
        self.mesh.record_cell_value('velocity', vel_cell)
        self.mesh.record_cell_value('pressure', p_cell_func.x.array)  # type: ignore
        self.mesh.record_cell_value('velocity_mag', np.linalg.norm(vel_cell, axis=1))  # type: ignore
        
        self.mesh.save_grid(file)


class NavierStokeSimulator(StokeSimulator):
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    snes_option = {}

    def __init__(self, name: str, mesh: MeshObj, velocity_degree: int = 2, pressure_degree: int = 1):
        super().__init__(name=name, mesh=mesh, velocity_degree=velocity_degree, pressure_degree=pressure_degree)
        self.is_linear = False
        self.eqn_map: dict[str, Any] = {}

    def define_equation(self, nu_value: float, constant_nu=True, body_force=(0., 0., 0.)):
        up = dolfinx.fem.Function(self.W, name='state')
        u, p = ufl.split(up)  # type: ignore
        v, q = ufl.split(ufl.TestFunction(self.W))  # type: ignore

        nu = dolfinx.fem.Function(self.nu_function_space)
        nu.x.array[:] = nu_value  # type: ignore
        # f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.topology_dim))
        # f = dolfinx.fem.Constant(self.mesh.domain, body_force[:self.mesh.topology_dim])
        f = dolfinx.fem.Function(self.V_map_space, name='body_force')
        f.x.array[:] = np.tile(np.array(body_force[:self.mesh.tdim]), (self.mesh.num_points, 1)).reshape(-1)  # type: ignore

        lhs_form = (
            nu * inner(grad(u), grad(v)) * ufl.dx  # type: ignore
            + inner(grad(u) * u, v) * ufl.dx  # type: ignore
            - inner(p, div(v)) * ufl.dx
            + inner(div(u), q) * ufl.dx
            - inner(f, v) * ufl.dx
        )
        lhs_form_dolfin = dolfinx.fem.form(lhs_form)

        self.eqn_map.update({
            'lhs_form': lhs_form,
            'lhs_form_dolfin': lhs_form_dolfin,
            'u': u, 'p': p, 'uh': up, 'nu': nu, 'f': f,
        })
    
    def run(
            self,
            snes_option: dict|None = None, ksp_option: dict|None = None,
            criterion={}, init_uh: dolfinx.fem.Function|None = None,
            mat_assemble_method: Literal['lift', 'identity_row'] = 'lift',
            with_residual=False, with_monitor=False
    ):
        uh: dolfinx.fem.Function = self.eqn_map['uh']
        if init_uh is not None:
            uh.x.array[:] = init_uh.x.array[:]

        jacobi_ufl = ufl.derivative(self.eqn_map['lhs_form'], uh, ufl.TrialFunction(uh.function_space))
        jacobi_form = dolfinx.fem.form(jacobi_ufl)

        res_dict = NonLinearProblemSolver.solve_fea_petsc(
            comm=self.mesh.domain.comm,
            f_form=self.eqn_map['lhs_form'],
            uh=uh,
            jacobi_form=jacobi_form,  # type: ignore
            bcs=[bry_info.bry_cell.bc for bry_info in self.boundaries.values()],
            mat_assemble_method=mat_assemble_method,
            snes_setting=snes_option if (snes_option is not None) else self.snes_option,
            ksp_setting=ksp_option if (ksp_option is not None) else self.ksp_option,
            criterion=criterion,
            with_residual=with_residual,
            with_monitor=with_monitor
        )
        if with_residual:
            print(f"[Info] Navier Stoke Simulator isSolved:{res_dict['is_solved']}, maxResidual:{res_dict['max_residual']}.")
        return res_dict


def debug_stoke():
    mesh_file = '/home/admin123456/Desktop/work/gmsh_test/t8/model.msh'
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/t8/model.xdmf'
    MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)
    
    mesh = MeshObj(xdmf_file=xdmf_file)
    simulator = StokeSimulator('stoke', mesh)
    simulator.define_equation(nu_value=1e-4)
    
    inlet_bry = FluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[3.0, 0.],
        marker=1,
        face_center=[-0.05, 0.2],
        face_norm_vector=[1.0, 0.0],
        face_radius=0.1
    )
    outlet_bry = FluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=2
    )
    wall_bry = FluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=3
    )
    
    simulator.add_boundary(inlet_bry)
    simulator.add_boundary(outlet_bry)
    simulator.add_boundary(wall_bry)
    
    simulator.define_boundary()
    simulator.run(with_residual=True)
    simulator.save('/home/admin123456/Desktop/work/gmsh_test/t8/stoke_res.vtu')


def debug_navier_stoke():
    mesh_file = '/home/admin123456/Desktop/work/gmsh_test/test3.msh'
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/test3.xdmf'
    # MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)
    
    mesh = MeshObj(xdmf_file=xdmf_file)
    simulator = NavierStokeSimulator('navier_stoke', mesh)
    simulator.define_equation(nu_value=1.0 / 400.0)
    
    inlet_bry = FluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[2.0, 0.0],
        marker=13
    )
    outlet_bry = FluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=14
    )
    wall_bry = FluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=15
    )
    
    simulator.add_boundary(inlet_bry)
    simulator.add_boundary(outlet_bry)
    simulator.add_boundary(wall_bry)
    
    simulator.define_boundary()
    simulator.run(with_residual=True, with_monitor=True)
    simulator.save('/home/admin123456/Desktop/work/gmsh_test/navier_stoke_res.vtu')


def debug_navier_stoke_2():
    mesh_file = '/home/admin123456/Desktop/work/gmsh_test/test2.msh'
    xdmf_file = '/home/admin123456/Desktop/work/gmsh_test/test2.xdmf'
    # MeshUtils.msh_to_XDMF(mesh_file, xdmf_file, dim=2)
    
    mesh = MeshObj(xdmf_file=xdmf_file)
    simulator = NavierStokeSimulator('navier_stoke', mesh)
    simulator.define_equation(nu_value=1.0 / 400.0)
    
    inlet_bry = FluidBoundary(
        name='inlet',
        boundary_type='velocity',
        value=[3.5, 0.],
        marker=22
    )
    outlet_bry = FluidBoundary(
        name='outlet',
        boundary_type='pressure',
        value=0.0,
        marker=23
    )
    wall_bry = FluidBoundary(
        name='wall',
        boundary_type='wall',
        value=0.0,
        marker=24
    )
    
    simulator.add_boundary(inlet_bry)
    simulator.add_boundary(outlet_bry)
    simulator.add_boundary(wall_bry)
    
    simulator.define_boundary()
    simulator.run(
        snes_option={
            'snes_type': 'newtonls',
            'snes_linesearch_type': 'bt',
            'snes_linesearch_damping': 1e-1,    # initial step length
            # 'snes_linesearch_maxstep': 1.0
        },
        with_residual=True, with_monitor=True
    )
    simulator.save('/home/admin123456/Desktop/work/gmsh_test/navier_stoke_res.vtu')


if __name__ == '__main__':
    debug_stoke()
    # debug_navier_stoke()
    # debug_navier_stoke_2()
    
    pass