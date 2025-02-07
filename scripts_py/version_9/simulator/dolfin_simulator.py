import numpy as np
import dolfinx
import os
import shutil
import ufl
from ufl import sym, grad, nabla_grad, dot, inner, div, Identity
from ufl import algebra
from typing import List, Union, Dict, Callable
from petsc4py import PETSc
import json
import pickle
import pandas as pd
from dolfinx import geometry
import pyvista

from ..dolfinx_Grad.recorder_utils import PVDRecorder
from ..dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from ..dolfinx_Grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver
from ..dolfinx_Grad.vis_mesh_utils import VisUtils


class FluidSimulator(object):
    def __init__(
            self,
            name,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            velocity_order: int = 2,
            pressure_order: int = 1,
            nu_order: int = 1
    ):
        self.name = name
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1
        self.velocity_order = velocity_order
        self.pressure_order = pressure_order

        self.W = dolfinx.fem.FunctionSpace(
            domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", domain.ufl_cell(), velocity_order),
                ufl.FiniteElement("Lagrange", domain.ufl_cell(), pressure_order)
            ])
        )
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()
        self.V_mapping_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        self.Q_mapping_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))
        self.nu_function_space = dolfinx.fem.FunctionSpace(domain, ("CG", nu_order))
        self.n_vec = MeshUtils.define_facet_norm(domain)
        self.ds = MeshUtils.define_ds(domain, facet_tags)
        self.dx = MeshUtils.define_dx(domain)

        self.equation_map = {}
        self.define_equation()

    @staticmethod
    def set_boundary_value(name: str, value: Union[float, int, Callable], function_space):
        fun = dolfinx.fem.Function(function_space, name=name)
        if isinstance(value, (float, int)):
            fun.x.array[:] = value
        elif isinstance(value, Callable):
            fun.interpolate(value)
        else:
            raise ValueError("[ERROR]: Wrong value.")
        return fun

    def add_boundary(self, name: str, value: Union[float, int, Callable], marker: int, is_velocity: bool):
        raise NotImplementedError

    def define_equation(self, **args):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def log2json(logger_dicts: Dict[str, Dict[str, dolfinx.fem.Form]]):
        record_json = {}
        for tag_name in logger_dicts.keys():
            inspectors_dict = logger_dicts[tag_name]
            if len(inspectors_dict) == 1:
                marker_name = list(inspectors_dict.keys())[0]
                value = AssembleUtils.assemble_scalar(inspectors_dict[marker_name])
                record_json[marker_name] = value
            else:
                data_cell = {}
                for marker_name in inspectors_dict.keys():
                    data_cell[marker_name] = AssembleUtils.assemble_scalar(inspectors_dict[marker_name])
                record_json[tag_name] = data_cell
        return record_json

    @staticmethod
    def eval_point_by_dolfin(
            uh: dolfinx.fem.Function, points: np.ndarray,
            domain: dolfinx.mesh.Mesh, bb_tree: geometry.BoundingBoxTree = None
    ):
        if bb_tree is None:
            bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
        cell_idxs = []
        for i in range(points.shape[0]):
            cell_idxs.append(colliding_cells.links(i)[0])
        res = uh.eval(points, cell_idxs)
        return res

    @staticmethod
    def eval_point_by_pyvista(
            uh: dolfinx.fem.Function, points: np.ndarray,
            domain: dolfinx.mesh.Mesh, grid: pyvista.UnstructuredGrid = None
    ):
        if grid is None:
            grid = VisUtils.convert_to_grid(domain)
        cell_idxs = grid.find_containing_cell(points)
        res = uh.eval(points, cell_idxs)
        return res


class StokeSimulator(FluidSimulator):
    def define_equation(self):
        u, p = ufl.split(ufl.TrialFunction(self.W))
        v, q = ufl.split(ufl.TestFunction(self.W))
        f = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))
        stoke_form = (
                inner(grad(u), grad(v)) * self.dx
                - p * div(v) * self.dx
                - q * div(u) * self.dx
                - inner(f, v) * self.dx
        )
        self.equation_map.update({
            'lhs': ufl.lhs(stoke_form),
            'rhs': ufl.rhs(stoke_form),
            'lhs_form': dolfinx.fem.form(ufl.lhs(stoke_form)),
            'rhs_form': dolfinx.fem.form(ufl.rhs(stoke_form)),
            'up': dolfinx.fem.Function(self.W)
        })

    def add_boundary(self, name: str, value: Union[float, int, Callable], marker: int, is_velocity: bool):
        if is_velocity:
            bc_value = self.set_boundary_value(name, value, self.V)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)
        else:
            bc_value = self.set_boundary_value(name, value, self.Q)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W1)

        if 'bcs' in self.equation_map.keys():
            self.equation_map['bcs'].append(bc)
        else:
            self.equation_map['bcs'] = [bc]

    def simulate(self, ksp_option, **kwargs):
        up: dolfinx.fem.Function = self.equation_map['up']
        res_dict = LinearProblemSolver.solve_by_petsc_form(
            comm=self.domain.comm,
            uh=up,
            # a_form=self.equation_map['lhs_form'],
            # L_form=self.equation_map['rhs_form'],
            a_form=self.equation_map['lhs'],
            L_form=self.equation_map['rhs'],
            bcs=self.equation_map['bcs'],
            ksp_option=ksp_option,
            **kwargs
        )
        if kwargs.get('with_debug', False):
            print(f"[DEBUG FluidSimulator Stoke]: "
                  f"max_error:{res_dict['max_error']:.8f} cost_time:{res_dict['cost_time']:.2f}")
        return res_dict

    def run(
            self, result_dir, name,
            ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            logger_dicts={}, **kwargs
    ):
        simulator_dir = os.path.join(result_dir, f"simulate_{name}")
        if os.path.exists(simulator_dir):
            shutil.rmtree(simulator_dir)
        os.mkdir(simulator_dir)
        u_recorder = PVDRecorder(file=os.path.join(simulator_dir, 'velocity.pvd'))
        p_recorder = PVDRecorder(file=os.path.join(simulator_dir, 'pressure.pvd'))

        res_dict = self.simulate(ksp_option, **kwargs)
        up: dolfinx.fem.Function = self.equation_map['up']
        u_recorder.write_function(up.sub(0).collapse(), step=0)
        p_recorder.write_function(up.sub(1).collapse(), step=0)

        data_json = self.log2json(logger_dicts)
        with open(os.path.join(simulator_dir, 'log_res.json'), 'w') as f:
            json.dump(data_json, f, indent=4)

        return res_dict

    def merge_data(self, velocity: dolfinx.fem.Function, pressure: dolfinx.fem.Function):
        assert (velocity.function_space == self.V) and (pressure.function_space == self.Q)
        up = self.equation_map['up']
        up.x.array[self.V_to_W_dofs] = velocity.x.array
        up.x.array[self.Q_to_W_dofs] = pressure.x.array

    def split_data(self, velocity: dolfinx.fem.Function = None, pressure: dolfinx.fem.Function = None):
        up = self.equation_map['up']

        if velocity is not None:
            assert velocity.function_space == self.V
            velocity.vector.aypx(0.0, up.sub(0).collapse().vector)
        else:
            velocity = up.sub(0).collapse()

        if pressure is not None:
            assert pressure.function_space == self.Q
            pressure.vector.aypx(0.0, up.sub(1).collapse().vector)
        else:
            pressure = up.sub(1).collapse()

    def get_up_form(self):
        up = self.equation_map['up']
        u_n, p_n = ufl.split(up)
        return u_n, p_n


class NavierStokeSimulator(StokeSimulator):
    def define_equation(self, nu_value, constant_nu=True):
        up = dolfinx.fem.Function(self.W, name='state')
        u, p = ufl.split(up)
        v, q = ufl.split(ufl.TestFunction(self.W))
        f = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))
        if constant_nu:
            nu = dolfinx.fem.Constant(self.domain, nu_value)
        else:
            nu = dolfinx.fem.Function(self.nu_function_space)
            nu.x.array[:] = nu_value

        navier_stoke_form = (
                nu * inner(grad(u), grad(v)) * ufl.dx
                + inner(grad(u) * u, v) * ufl.dx
                - inner(p, div(v)) * ufl.dx
                + inner(div(u), q) * ufl.dx
                - inner(f, v) * ufl.dx
        )

        self.equation_map.update({
            'lhs': navier_stoke_form,
            'up': up,
            'nu': nu,
            'nu_constant': nu_value
        })

    def add_boundary(self, name: str, value: Union[float, int, Callable], marker: int, is_velocity: bool):
        if is_velocity:
            bc_value = self.set_boundary_value(name, value, self.V)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)
        else:
            bc_value = self.set_boundary_value(name, value, self.Q)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W1)

        if 'bcs' in self.equation_map.keys():
            self.equation_map['bcs'].append(bc)
        else:
            self.equation_map['bcs'] = [bc]

    def simulate(
            self,
            snes_option: dict, ksp_option: dict,
            criterion, guess_up: dolfinx.fem.Function = None, **kwargs
    ):
        nstoke_dict = self.equation_map['navier_stoke']
        up: dolfinx.fem.Function = nstoke_dict['up']
        if guess_up is not None:
            up.vector.aypx(0.0, guess_up.vector)

        jacobi_form = ufl.derivative(nstoke_dict['lhs'], up, ufl.TrialFunction(up.function_space))
        res_dict = NonLinearProblemSolver.solve_by_petsc(
            comm=self.domain.comm,
            F_form=nstoke_dict['lhs'], uh=up, jacobi_form=jacobi_form, bcs=nstoke_dict['bcs'],
            snes_setting=snes_option, ksp_option=ksp_option, criterion=criterion,
            **kwargs
        )

        if kwargs.get('with_debug', False):
            print(f"[DEBUG FluidSimulator Navier Stoke]: max_error:{res_dict['max_error']:.8f}"
                  f" residual:{res_dict['last_error']}"
                  f" cost_time:{res_dict['cost_time']:.2f}")
        return res_dict

    def run(
            self, result_dir, name,
            guess_up: dolfinx.fem.Function = None,
            snes_option: dict = {},
            snes_criterion={"rtol": 1e-06, "atol": 1e-06, "max_it": 1000.0},
            ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            logger_dicts={},
            **kwargs
    ):
        simulator_dir = os.path.join(result_dir, f"simulate_{name}")
        if os.path.exists(simulator_dir):
            shutil.rmtree(simulator_dir)
        os.mkdir(simulator_dir)
        u_recorder = PVDRecorder(file=os.path.join(simulator_dir, 'velocity.pvd'))
        p_recorder = PVDRecorder(file=os.path.join(simulator_dir, 'pressure.pvd'))

        res_dict = self.simulate(snes_option, ksp_option, snes_criterion, guess_up=guess_up, **kwargs)

        nstoke_dict = self.equation_map['navier_stoke']
        up: dolfinx.fem.Function = nstoke_dict['up']
        u_recorder.write_function(up.sub(0).collapse(), step=0)
        p_recorder.write_function(up.sub(1).collapse(), step=0)

        data_json = self.log2json(logger_dicts)
        with open(os.path.join(simulator_dir, 'log_res.json'), 'w') as f:
            json.dump(data_json, f, indent=4)

        return res_dict

