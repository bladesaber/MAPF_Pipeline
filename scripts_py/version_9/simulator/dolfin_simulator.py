import os
import shutil
import numpy as np
import pandas as pd
import dolfinx
from typing import Dict, List, Union, Callable, Literal, Tuple

import ufl
from ufl import inner, grad, div

from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj
from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils, DolfinUtils
from scripts_py.version_9.dolfinx_grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver


class FluidSimulator(object):
    def __init__(self, name: str, mesh: MeshObj):
        self.name = name
        self.mesh = mesh
        self.W = dolfinx.fem.FunctionSpace(
            self.mesh.domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", self.mesh.domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", self.mesh.domain.ufl_cell(), 1)
            ])
        )
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()

        self.V_map_space = dolfinx.fem.VectorFunctionSpace(self.mesh.domain, ("CG", 1))
        self.Q_map_space = dolfinx.fem.FunctionSpace(self.mesh.domain, ("CG", 1))
        self.nu_function_space = dolfinx.fem.FunctionSpace(self.mesh.domain, ("CG", 1))

        self.n_vec = MeshUtils.define_facet_norm(self.mesh.domain)
        self.ds = MeshUtils.define_ds(self.mesh.domain, self.mesh.facet_tags)
        self.dx = MeshUtils.define_dx(self.mesh.domain)

    @staticmethod
    def set_boundary_value(name: str, value: Union[float, int, Callable], function_space):
        fun = dolfinx.fem.Function(function_space, name=name)
        if isinstance(value, (float, int)):
            fun.x.array[:] = value
        elif isinstance(value, Callable):
            fun.interpolate(value)
        else:
            raise ValueError("[Error]: Non-valid Value.")
        return fun

    def add_boundary(self, name: str, value: Union[float, int, Callable], marker: int, is_velocity: bool):
        raise NotImplementedError

    def define_equation(self, **args):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError

    def eval_point_by_dolfin(self, uh: dolfinx.fem.Function, points: np.ndarray):
        cell_indices: List[int] = MeshUtils.cell_indices_of_points(self.mesh.grid, points)
        return DolfinUtils.function_eval(uh, pcd_xyz=points, cell_indices=cell_indices)


class StokeSimulator(FluidSimulator):
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }

    def __init__(self, name: str, mesh: MeshObj):
        super().__init__(name=name, mesh=mesh)
        self.equation_map = {'bcs': {}}

    def define_equation(self, nu_value: float, body_force=(0., 0., 0.)):
        u, p = ufl.split(ufl.TrialFunction(self.W))
        v, q = ufl.split(ufl.TestFunction(self.W))

        nu = dolfinx.fem.Function(self.nu_function_space)
        nu.x.array[:] = nu_value
        # f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.topology_dim))
        # f = dolfinx.fem.Constant(self.mesh.domain, body_force[:self.mesh.topology_dim])
        f = dolfinx.fem.Function(self.V_map_space, name='body_force')
        f.x.array[:] = np.tile(np.array(body_force[:self.mesh.topology_dim]), (self.mesh.num_points, 1)).reshape(-1)

        stoke_form = (
                nu * inner(grad(u), grad(v)) * self.dx
                - p * div(v) * self.dx
                - q * div(u) * self.dx
                - inner(f, v) * self.dx
        )
        self.equation_map.update({
            'lhs': ufl.lhs(stoke_form),
            'rhs': ufl.rhs(stoke_form),
            'lhs_form': dolfinx.fem.form(ufl.lhs(stoke_form)),
            'rhs_form': dolfinx.fem.form(ufl.rhs(stoke_form)),
            'u': u, 'p': p, 'uh': dolfinx.fem.Function(self.W)
        })

    def add_boundary(
            self, name: str, value: Union[float, int, List[float]],
            marker: int, boundary_type: Literal['velocity', 'pressure', 'wall']
    ):
        if boundary_type == 'velocity':
            assert len(value) == 2 or len(value) == 3

            def vel_exp(x):
                num = x.shape[1]
                bry_values = np.zeros((self.mesh.topology_dim, num))
                bry_values[0] = value[0]
                bry_values[1] = value[1]
                if self.mesh.topology_dim == 3:
                    bry_values[2] = value[2]
                return bry_values

            bc_value = self.set_boundary_value(name, vel_exp, self.V)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=marker
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)

        elif boundary_type == 'pressure':
            bc_value = self.set_boundary_value(name, value, self.Q)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=marker
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W1)

        elif boundary_type == 'wall':
            bc_value = dolfinx.fem.Function(self.V, name=name)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=marker
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)

        else:
            raise ValueError('[Error]: Non-valid Boundary Type.')

        self.equation_map['bcs'][name] = bc

    def run(
            self, ksp_option=None, mat_assemble_method: Literal['lift', 'Identity_row'] = 'lift', with_debug=False
    ):
        res_dict = LinearProblemSolver.solve_fea_petsc(
            comm=self.mesh.domain.comm,
            uh=self.equation_map['uh'],
            a_form=self.equation_map['lhs_form'],
            l_form=self.equation_map['rhs_form'],
            bcs=list(self.equation_map['bcs'].values()),
            mat_assemble_method=mat_assemble_method,
            ksp_option=ksp_option if (ksp_option is not None) else self.ksp_option,
            with_debug=with_debug
        )
        if with_debug:
            print(f"[Info] Stoke Simulator isSolved:{res_dict['is_solved']}, maxError:{res_dict['max_error']}.")
        return res_dict['is_solved'], res_dict

    def save(self, file: str):
        assert file.endswith('.vtu')

        uh: dolfinx.fem.Function = self.equation_map['uh']
        vel_ref = dolfinx.fem.Function(self.V_map_space, name='vel_ref')
        vel_ref.interpolate(uh.sub(0).collapse())
        p_ref = uh.sub(1).collapse()

        vel_ref_np = vel_ref.x.array.reshape((-1, self.mesh.topology_dim))
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)
        self.mesh.grid.point_data['velocity'] = vel_ref_np
        self.mesh.grid.point_data['pressure'] = p_ref.x.array

        self.mesh.save_mesh(file)


class NavierStokeSimulator(FluidSimulator):
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    snes_option = {}

    def __init__(self, name: str, mesh: MeshObj):
        super().__init__(name=name, mesh=mesh)
        self.equation_map = {'bcs': {}}

    def define_equation(self, nu_value: float, constant_nu=True, body_force=(0., 0., 0.)):
        up = dolfinx.fem.Function(self.W, name='state')
        u, p = ufl.split(up)
        v, q = ufl.split(ufl.TestFunction(self.W))

        nu = dolfinx.fem.Function(self.nu_function_space)
        nu.x.array[:] = nu_value
        # f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.topology_dim))
        # f = dolfinx.fem.Constant(self.mesh.domain, body_force[:self.mesh.topology_dim])
        f = dolfinx.fem.Function(self.V_map_space, name='body_force')
        f.x.array[:] = np.tile(np.array(body_force[:self.mesh.topology_dim]), (self.mesh.num_points, 1)).reshape(-1)

        navier_stoke_form = (
                nu * inner(grad(u), grad(v)) * ufl.dx
                + inner(grad(u) * u, v) * ufl.dx
                - inner(p, div(v)) * ufl.dx
                + inner(div(u), q) * ufl.dx
                - inner(f, v) * ufl.dx
        )

        self.equation_map.update({
            'lhs': navier_stoke_form,
            'lhs_form': dolfinx.fem.form(navier_stoke_form),
            'u': u, 'p': p, 'uh': up, 'nu': nu,
        })

    def add_boundary(
            self, name: str, value: Union[float, int, List[float]],
            marker: int, boundary_type: Literal['velocity', 'pressure', 'wall']
    ):
        if boundary_type == 'velocity':
            assert len(value) == 2 or len(value) == 3

            def vel_exp(x):
                num = x.shape[1]
                bry_values = np.zeros((self.mesh.topology_dim, num))
                bry_values[0] = value[0]
                bry_values[1] = value[1]
                if self.mesh.topology_dim == 3:
                    bry_values[2] = value[2]
                return bry_values

            bc_value = self.set_boundary_value(name, vel_exp, self.V)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=marker
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)

        elif boundary_type == 'pressure':
            bc_value = self.set_boundary_value(name, value, self.Q)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=marker
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W1)

        elif boundary_type == 'wall':
            bc_value = dolfinx.fem.Function(self.V, name=name)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=marker
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)

        else:
            raise ValueError('[Error]: Non-valid Boundary Type.')

        self.equation_map['bcs'][name] = bc

    def run(
            self,
            snes_option: Dict = None, ksp_option: Dict = None,
            criterion={}, init_uh: dolfinx.fem.Function = None,
            mat_assemble_method: Literal['lift', 'Identity_row'] = 'lift',
            with_debug=False, with_monitor=False
    ):
        uh: dolfinx.fem.Function = self.equation_map['uh']
        if init_uh is not None:
            uh.vector.aypx(0.0, init_uh.vector)

        jacobi_ufl = ufl.derivative(self.equation_map['lhs'], uh, ufl.TrialFunction(uh.function_space))
        jacobi_form = dolfinx.fem.form(jacobi_ufl)

        res_dict = NonLinearProblemSolver.solve_fea_petsc(
            comm=self.mesh.domain.comm,
            f_form=self.equation_map['lhs_form'],
            uh=uh,
            jacobi_form=jacobi_form,
            bcs=list(self.equation_map['bcs'].values()),
            mat_assemble_method=mat_assemble_method,
            snes_setting=snes_option if (snes_option is not None) else self.snes_option,
            ksp_setting=ksp_option if (ksp_option is not None) else self.ksp_option,
            criterion=criterion,
            with_debug=with_debug,
            with_monitor=with_monitor
        )
        if with_debug:
            print(f"[Info] Navier Stoke Simulator isSolved:{res_dict['is_solved']}, maxError:{res_dict['max_error']}.")
        return res_dict

    def save(self, file: str):
        assert file.endswith('.vtu')

        uh: dolfinx.fem.Function = self.equation_map['uh']
        vel_ref = dolfinx.fem.Function(self.V_map_space, name='vel_ref')
        vel_ref.interpolate(uh.sub(0).collapse())
        p_ref = uh.sub(1).collapse()

        vel_ref_np = vel_ref.x.array.reshape((-1, self.mesh.topology_dim))
        if vel_ref_np.shape[1] == 2:
            vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)
        self.mesh.grid.point_data['velocity'] = vel_ref_np
        self.mesh.grid.point_data['pressure'] = p_ref.x.array

        self.mesh.save_mesh(file)


def debug_stoke():
    mesh = MeshObj(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example01/model.msh', dim=2
    )
    simulator = StokeSimulator('stoke', mesh)
    simulator.define_equation(nu_value=1. / 400.)

    simulator.add_boundary('inlet', value=[1., 0., 0.], marker=13, boundary_type='velocity')
    simulator.add_boundary('outlet', value=0.0, marker=14, boundary_type='pressure')
    simulator.add_boundary('wall', value=0.0, marker=15, boundary_type='wall')

    simulator.run(with_debug=True)
    simulator.save('/home/admin123456/Desktop/work/topopt_exps/new_test/example01/simulate_res.vtu')


def debug_navier_stoke():
    mesh = MeshObj(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example01/model.msh', dim=2
    )
    simulator = NavierStokeSimulator('navier_stoke', mesh)
    simulator.define_equation(nu_value=1. / 400.)

    simulator.add_boundary('inlet', value=[1., 0., 0.], marker=13, boundary_type='velocity')
    simulator.add_boundary('outlet', value=0.0, marker=14, boundary_type='pressure')
    simulator.add_boundary('wall', value=0.0, marker=15, boundary_type='wall')

    simulator.run(with_debug=True, with_monitor=True)
    simulator.save('/home/admin123456/Desktop/work/topopt_exps/new_test/example01/simulate_res02.vtu')


if __name__ == '__main__':
    # debug_stoke()
    debug_navier_stoke()
