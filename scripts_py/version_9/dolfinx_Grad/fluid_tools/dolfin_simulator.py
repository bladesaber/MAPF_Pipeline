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

from ..recorder_utils import VTKRecorder, TensorBoardRecorder
from ..dolfinx_utils import MeshUtils, AssembleUtils, BoundaryUtils
from ..equation_solver import LinearProblemSolver, NonLinearProblemSolver
from ..vis_mesh_utils import VisUtils
from ..simulator_convert import CrossSimulatorUtil


def epsilon(u: Union[dolfinx.fem.Function, ufl.Argument]):
    """
    Define strain-rate tensor: 0.5 * (grad(U) + grad(U).T)
    """
    return sym(nabla_grad(u))


def sigma(u, p, mu: dolfinx.fem.Constant):
    """
    Define stress tensor:
        mu: Dynamic viscosity
    """
    return 2.0 * mu * epsilon(u) - p * Identity(len(u))


class FluidSimulator(object):
    def __init__(
            self,
            name,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            velocity_order: int = 2,
            pressure_order: int = 1,
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
        self.n_vec = MeshUtils.define_facet_norm(domain)
        self.ds = MeshUtils.define_ds(domain, facet_tags)
        self.dx = MeshUtils.define_dx(domain)

        self.equation_map = {}

    def define_stoke_equation(self):
        u, p = ufl.split(ufl.TrialFunction(self.W))
        v, q = ufl.split(ufl.TestFunction(self.W))
        f = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))
        stoke_form = (
                inner(grad(u), grad(v)) * self.dx
                - p * div(v) * self.dx
                - q * div(u) * self.dx
                - inner(f, v) * self.dx
        )
        self.equation_map['stoke'] = {
            'lhs': ufl.lhs(stoke_form),
            'rhs': ufl.rhs(stoke_form),
            'lhs_form': dolfinx.fem.form(ufl.lhs(stoke_form)),
            'rhs_form': dolfinx.fem.form(ufl.rhs(stoke_form)),
            'up': dolfinx.fem.Function(self.W)
        }

    def define_navier_stoke_equation(self, Re):
        up = dolfinx.fem.Function(self.W, name='state')
        u, p = ufl.split(up)
        v, q = ufl.split(ufl.TestFunction(self.W))
        f = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))
        nu = dolfinx.fem.Constant(self.domain, 1. / Re)

        conservation_form = (
                nu * inner(grad(u), grad(v)) * ufl.dx
                + inner(grad(u) * u, v) * ufl.dx
                - inner(p, div(v)) * ufl.dx
                - inner(f, v) * ufl.dx
        )
        continue_form = inner(div(u), q) * ufl.dx
        navier_stoke_form = conservation_form + continue_form

        # navier_stoke_form = (
        #         nu * inner(grad(u), grad(v)) * ufl.dx
        #         + inner(grad(u) * u, v) * ufl.dx
        #         - inner(p, div(v)) * ufl.dx
        #         + inner(div(u), q) * ufl.dx
        #         - inner(f, v) * ufl.dx
        # )

        self.equation_map['navier_stoke'] = {
            'lhs': navier_stoke_form,
            'up': up,
            'conservation_form': conservation_form,
            'continue_form': continue_form
        }

    def define_ipcs_equation(self, dt: float, dynamic_viscosity: float, density: float, is_channel_fluid: bool):
        dt = dt
        k = dolfinx.fem.Constant(self.domain, dt)  # time step
        mu = dolfinx.fem.Constant(self.domain, dynamic_viscosity)
        rho = dolfinx.fem.Constant(self.domain, density)
        f = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))

        # ------ Define the variational problem for the first step
        u, p = ufl.TrialFunction(self.V), ufl.TrialFunction(self.Q)
        v, q = ufl.TestFunction(self.V), ufl.TestFunction(self.Q)

        u_n = dolfinx.fem.Function(self.V, name='velocity_n_step')
        p_n = dolfinx.fem.Function(self.Q, name='pressure_n_step')
        U = 0.5 * (u_n + u)

        F1 = rho * ufl.dot((u - u_n) / k, v) * ufl.dx
        F1 += rho * ufl.dot(ufl.dot(u_n, ufl.nabla_grad(u_n)), v) * ufl.dx
        F1 += ufl.inner(sigma(U, p_n, mu), epsilon(v)) * ufl.dx
        F1 += ufl.dot(p_n * self.n_vec, v) * ufl.ds - ufl.dot(mu * ufl.nabla_grad(U) * self.n_vec, v) * ufl.ds
        F1 -= ufl.dot(f, v) * ufl.dx
        lhs1 = ufl.lhs(F1)
        rhs1 = ufl.rhs(F1)
        a1_form = dolfinx.fem.form(lhs1)
        L1_form = dolfinx.fem.form(rhs1)

        # ------ Define variational problem for step 2
        u_ = dolfinx.fem.Function(self.V, name='u_')
        lhs2 = ufl.dot(ufl.nabla_grad(p), ufl.nabla_grad(q)) * ufl.dx  # compute correction pressure
        rhs2 = ufl.dot(ufl.nabla_grad(p_n), ufl.nabla_grad(q)) * ufl.dx - (rho / k) * ufl.div(u_) * q * ufl.dx
        a2_form = dolfinx.fem.form(lhs2)
        L2_form = dolfinx.fem.form(rhs2)

        # ------ Define variational problem for step 3
        p_ = dolfinx.fem.Function(self.Q, name='pressure_n_plus_1_step')
        lhs3 = rho * ufl.dot(u, v) * ufl.dx  # compute correction velocity
        rhs3 = rho * ufl.dot(u_, v) * ufl.dx - k * ufl.dot(ufl.nabla_grad(p_ - p_n), v) * ufl.dx
        a3_form = dolfinx.fem.form(lhs3)
        L3_form = dolfinx.fem.form(rhs3)

        self.equation_map['ipcs'] = {
            'dt': dt,
            'is_channel_fluid': is_channel_fluid,
            'u_n': u_n,
            'p_n': p_n,
            'u_': u_,
            'p_': p_,
            'a1_form': a1_form,
            'L1_form': L1_form,
            'a2_form': a2_form,
            'L2_form': L2_form,
            'a3_form': a3_form,
            'L3_form': L3_form,
        }

    @staticmethod
    def get_boundary_function(name: str, value: Union[float, int, Callable], function_space):
        fun = dolfinx.fem.Function(function_space, name=name)
        if isinstance(value, (float, int)):
            fun.x.array[:] = value
        else:
            fun.interpolate(value)
        return fun

    def add_boundary(self, name: str, value: Union[float, int, Callable], marker: int, is_velocity: bool):
        if self.equation_map.get('ipcs', False):
            if is_velocity:
                bc_value = self.get_boundary_function(name, value, self.V)
                bc_dof = MeshUtils.extract_entity_dofs(
                    self.V, self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
                )
            else:
                bc_value = self.get_boundary_function(name, value, self.Q)
                bc_dof = MeshUtils.extract_entity_dofs(
                    self.Q, self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
                )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof)

            if is_velocity:
                if 'bcs_velocity' in self.equation_map['ipcs'].keys():
                    self.equation_map['ipcs']['bcs_velocity'].append(bc)
                else:
                    self.equation_map['ipcs']['bcs_velocity'] = [bc]

            else:
                if 'bcs_pressure' in self.equation_map['ipcs'].keys():
                    self.equation_map['ipcs']['bcs_pressure'].append(bc)
                else:
                    self.equation_map['ipcs']['bcs_pressure'] = [bc]

        if self.equation_map.get('navier_stoke', False):
            if is_velocity:
                bc_value = self.get_boundary_function(name, value, self.V)
                bc_dof = MeshUtils.extract_entity_dofs(
                    (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
                )
                bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)
            else:
                bc_value = self.get_boundary_function(name, value, self.Q)
                bc_dof = MeshUtils.extract_entity_dofs(
                    (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
                )
                bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W1)

            if 'bcs' in self.equation_map['navier_stoke'].keys():
                self.equation_map['navier_stoke']['bcs'].append(bc)
            else:
                self.equation_map['navier_stoke']['bcs'] = [bc]

        if self.equation_map.get('stoke', False):
            if is_velocity:
                bc_value = self.get_boundary_function(name, value, self.V)
                bc_dof = MeshUtils.extract_entity_dofs(
                    (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
                )
                bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)
            else:
                bc_value = self.get_boundary_function(name, value, self.Q)
                bc_dof = MeshUtils.extract_entity_dofs(
                    (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
                )
                bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W1)

            if 'bcs' in self.equation_map['stoke'].keys():
                self.equation_map['stoke']['bcs'].append(bc)
            else:
                self.equation_map['stoke']['bcs'] = [bc]

    def simulate_stoke_equation(self, ksp_option, **kwargs):
        stoke_dict = self.equation_map['stoke']
        up: dolfinx.fem.Function = stoke_dict['up']

        res_dict = LinearProblemSolver.solve_by_petsc_form(
            comm=self.domain.comm,
            uh=up,
            # a_form=stoke_dict['lhs_form'],
            # L_form=stoke_dict['rhs_form'],
            a_form=stoke_dict['lhs'],
            L_form=stoke_dict['rhs'],
            bcs=stoke_dict['bcs'],
            ksp_option=ksp_option,
            **kwargs
        )
        if kwargs.get('with_debug', False):
            print(f"[DEBUG FluidSimulator Stoke]: "
                  f"max_error:{res_dict['max_error']:.8f} cost_time:{res_dict['cost_time']:.2f}")

        return res_dict

    def run_stoke_process(
            self,
            proj_dir, name,
            ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            logger_dicts={},
            **kwargs
    ):
        if not self.equation_map.get('stoke', False):
            return

        simulator_dir = os.path.join(proj_dir, f"simulate_{name}")
        if os.path.exists(simulator_dir):
            shutil.rmtree(simulator_dir)
        os.mkdir(simulator_dir)

        u_recorder = VTKRecorder(file=os.path.join(simulator_dir, 'velocity.pvd'))
        p_recorder = VTKRecorder(file=os.path.join(simulator_dir, 'pressure.pvd'))
        data_json = {}

        res_dict = self.simulate_stoke_equation(ksp_option, **kwargs)

        stoke_dict = self.equation_map['stoke']
        up: dolfinx.fem.Function = stoke_dict['up']
        u_recorder.write_function(up.sub(0).collapse(), step=0)
        p_recorder.write_function(up.sub(1).collapse(), step=0)

        for tag_name in logger_dicts.keys():
            inspectors_dict = logger_dicts[tag_name]
            if len(inspectors_dict) == 1:
                marker_name = list(inspectors_dict.keys())[0]
                value = AssembleUtils.assemble_scalar(inspectors_dict[marker_name])
                data_json[marker_name] = value
            else:
                data_cell = {}
                for marker_name in inspectors_dict.keys():
                    data_cell[marker_name] = AssembleUtils.assemble_scalar(inspectors_dict[marker_name])
                data_json[tag_name] = data_cell

        with open(os.path.join(simulator_dir, 'log_res.json'), 'w') as f:
            json.dump(data_json, f, indent=4)

        res_dict['simulator_dir'] = simulator_dir

        return res_dict

    def simulate_navier_stoke_equation(
            self, snes_option: dict, ksp_option: dict, criterion: dict,
            guess_up: dolfinx.fem.Function = None,
            **kwargs
    ):
        nstoke_dict = self.equation_map['navier_stoke']
        up: dolfinx.fem.Function = nstoke_dict['up']

        if guess_up is not None:
            up.vector.aypx(0.0, guess_up.vector)

        jacobi_form = ufl.derivative(
            nstoke_dict['lhs'], up, ufl.TrialFunction(up.function_space)
        )
        res_dict = NonLinearProblemSolver.solve_by_petsc(
            comm=self.domain.comm,
            F_form=nstoke_dict['lhs'], uh=up, jacobi_form=jacobi_form, bcs=nstoke_dict['bcs'],
            snes_setting=snes_option, ksp_option=ksp_option, criterion=criterion,
            **kwargs
        )

        # res_dict = NonLinearProblemSolver.solve_by_dolfinx(
        #     F_form=nstoke_dict['lhs'], uh=up, bcs=nstoke_dict['bcs'],
        #     comm=self.domain.comm, ksp_option=ksp_option,
        #     with_debug=True
        # )

        if kwargs.get('with_debug', False):
            print(f"[DEBUG FluidSimulator Navier Stoke]: max_error:{res_dict['max_error']:.8f}"
                  f" residual:{res_dict['last_error']}"
                  f" cost_time:{res_dict['cost_time']:.2f}")

        return res_dict

    def run_navier_stoke_process(
            self,
            proj_dir, name,
            guess_up: dolfinx.fem.Function = None,
            snes_option: dict = {}, snes_criterion={},
            ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            logger_dicts={},
            **kwargs
    ):
        if not self.equation_map.get('navier_stoke', False):
            return

        simulator_dir = os.path.join(proj_dir, f"simulate_{name}")
        if os.path.exists(simulator_dir):
            shutil.rmtree(simulator_dir)
        os.mkdir(simulator_dir)

        u_recorder = VTKRecorder(file=os.path.join(simulator_dir, 'velocity.pvd'))
        p_recorder = VTKRecorder(file=os.path.join(simulator_dir, 'pressure.pvd'))
        data_json = {}

        res_dict = self.simulate_navier_stoke_equation(
            snes_option, ksp_option, snes_criterion, guess_up=guess_up, **kwargs
        )

        nstoke_dict = self.equation_map['navier_stoke']
        up: dolfinx.fem.Function = nstoke_dict['up']
        u_recorder.write_function(up.sub(0).collapse(), step=0)
        p_recorder.write_function(up.sub(1).collapse(), step=0)

        for tag_name in logger_dicts.keys():
            inspectors_dict = logger_dicts[tag_name]
            if len(inspectors_dict) == 1:
                marker_name = list(inspectors_dict.keys())[0]
                value = AssembleUtils.assemble_scalar(inspectors_dict[marker_name])
                data_json[marker_name] = value
            else:
                data_cell = {}
                for marker_name in inspectors_dict.keys():
                    data_cell[marker_name] = AssembleUtils.assemble_scalar(inspectors_dict[marker_name])
                data_json[tag_name] = data_cell

        with open(os.path.join(simulator_dir, 'log_res.json'), 'w') as f:
            json.dump(data_json, f, indent=4)

        res_dict['simulator_dir'] = simulator_dir
        return res_dict

    def run_ipcs_process(
            self,
            proj_dir, name,
            max_iter, log_iter, tol,
            data_convergence,
            update_inflow_condition: Callable = None,
            logger_dicts={},
            pre_function: Callable = None,
            post_function: Callable = None,
            with_debug=False
    ):
        if not self.equation_map.get('ipcs', False):
            return

        simulator_dir = os.path.join(proj_dir, f"simulate_{name}")
        if os.path.exists(simulator_dir):
            shutil.rmtree(simulator_dir)
        os.mkdir(simulator_dir)

        u_record_dir = os.path.join(simulator_dir, 'velocity')
        os.mkdir(u_record_dir)
        u_recorder = VTKRecorder(file=os.path.join(u_record_dir, 'velocity.pvd'))

        p_record_dir = os.path.join(simulator_dir, 'pressure')
        os.mkdir(p_record_dir)
        p_recorder = VTKRecorder(file=os.path.join(p_record_dir, 'pressure.pvd'))

        tensorBoard_dir = os.path.join(simulator_dir, 'log')
        os.mkdir(tensorBoard_dir)
        log_recorder = TensorBoardRecorder(tensorBoard_dir)

        # ------ Step 1: init environment
        ipcs_dict: dict = self.equation_map['ipcs']
        a1_form, L1_form = ipcs_dict['a1_form'], ipcs_dict['L1_form']
        a2_form, L2_form = ipcs_dict['a2_form'], ipcs_dict['L2_form']
        a3_form, L3_form = ipcs_dict['a3_form'], ipcs_dict['L3_form']
        bcs_velocity, bcs_pressure = ipcs_dict['bcs_velocity'], ipcs_dict['bcs_pressure']
        u_n, p_n, u_, p_ = ipcs_dict['u_n'], ipcs_dict['p_n'], ipcs_dict['u_'], ipcs_dict['p_']

        A1 = AssembleUtils.assemble_mat(a1_form, bcs=bcs_velocity)
        b1 = AssembleUtils.create_vector(L1_form)
        solver1 = LinearProblemSolver.create_petsc_solver(
            comm=self.domain.comm, A_mat=A1,
            solver_setting={
                'ksp_type': PETSc.KSP.Type.BCGS,
                'pc_type': PETSc.PC.Type.HYPRE,
                'pc_hypre_mat_solver_type': 'boomeramg'
            }
        )

        A2 = AssembleUtils.assemble_mat(a2_form, bcs=bcs_pressure)
        b2 = AssembleUtils.create_vector(L2_form)
        solver2 = LinearProblemSolver.create_petsc_solver(
            comm=self.domain.comm, A_mat=A2,
            solver_setting={
                'ksp_type': PETSc.KSP.Type.BCGS,
                'pc_type': PETSc.PC.Type.HYPRE,
                'pc_hypre_mat_solver_type': 'boomeramg'
            }
        )

        A3 = AssembleUtils.assemble_mat(a3_form, bcs=[])
        b3 = AssembleUtils.create_vector(L3_form)
        solver3 = LinearProblemSolver.create_petsc_solver(
            comm=self.domain.comm, A_mat=A3,
            solver_setting={
                'ksp_type': PETSc.KSP.Type.CG,
                'pc_type': PETSc.PC.Type.SOR
            }
        )

        # ------ Step 2: compuatation begin
        t, step = 0, 0
        while True:
            step += 1
            t += ipcs_dict['dt']

            if pre_function is not None:
                pre_function(self, step)

            if update_inflow_condition is not None:
                update_inflow_condition(t)

            # ------ Step 1: Tentative velocity step
            if not ipcs_dict['is_channel_fluid']:
                A1.zeroEntries()
                AssembleUtils.assemble_mat(a1_form, bcs=bcs_velocity, A_mat=A1)
            AssembleUtils.assemble_vec(L1_form, b1, clear_vec=True)
            BoundaryUtils.apply_boundary_to_vec(b1, bcs=bcs_velocity, a_form=a1_form, clean_vec=False)
            res1_dict = LinearProblemSolver.solve_by_petsc(
                b_vec=b1, solver=solver1, A_mat=A1, setOperators=False, with_debug=True
            )
            u_.vector.aypx(0.0, res1_dict['res'])

            # ------ Step 2: Pressure step
            AssembleUtils.assemble_vec(L2_form, b2, clear_vec=True)
            BoundaryUtils.apply_boundary_to_vec(b2, bcs=bcs_pressure, a_form=a2_form, clean_vec=False)
            res2_dict = LinearProblemSolver.solve_by_petsc(
                b_vec=b2, solver=solver2, A_mat=A2, setOperators=False, with_debug=True
            )
            p_.vector.aypx(0.0, res2_dict['res'])

            # ------ Step 3: Velocity correction step
            AssembleUtils.assemble_vec(L3_form, b3, clear_vec=True)
            res3_dict = LinearProblemSolver.solve_by_petsc(
                b_vec=b3, solver=solver3, A_mat=A3, setOperators=False, with_debug=True
            )
            u_.vector.aypx(0.0, res3_dict['res'])

            # ------ Step 4 Update solution to current time step
            # u_n.x.array[:] = u_.x.array[:]
            # p_n.x.array[:] = p_.x.array[:]
            u_n.vector.aypx(0.0, u_.vector)
            p_n.vector.aypx(0.0, p_.vector)

            if post_function is not None:
                post_res_dict = post_function(self, step, simulator_dir)
                sub_stop = post_res_dict['state']
            else:
                sub_stop = False

            # ------ Step 5 record result
            if step % log_iter == 0:
                u_recorder.write_function(u_n, t)
                p_recorder.write_function(p_n, t)

                for tag_name in logger_dicts.keys():
                    inspectors_dict = logger_dicts[tag_name]
                    if len(inspectors_dict) == 1:
                        marker_name = list(inspectors_dict.keys())[0]
                        value = AssembleUtils.assemble_scalar(inspectors_dict[marker_name])
                        log_recorder.write_scalar(marker_name, value, step)
                    else:
                        data_cell = {}
                        for marker_name in inspectors_dict.keys():
                            data_cell[marker_name] = AssembleUtils.assemble_scalar(inspectors_dict[marker_name])
                        log_recorder.write_scalars(tag_name, data_cell, step)

            if with_debug:
                computation_max_errors = {
                    'step1': res1_dict['max_error'], 'step2': res2_dict['max_error'], 'step3': res3_dict['max_error']
                }
                log_recorder.write_scalars('computation_errors', computation_max_errors, step=step)
                print(f"[Info iter:{step}] Step1 Error:{computation_max_errors['step1']:.8f}, "
                      f"Step2 Error:{computation_max_errors['step2']:.8f}, "
                      f"Step3 Error:{computation_max_errors['step3']:.8f}")
            else:
                print(f"[Info iter:{step}] Complete.")

            # ------ Step 6 check convergence
            if len(data_convergence) > 0:
                is_converge = True
                convergence_cells = {}
                for name in data_convergence.keys():
                    old_value = data_convergence[name]['cur_value']
                    new_value = AssembleUtils.assemble_scalar(data_convergence[name]['form'])
                    ratio = np.abs(new_value / (old_value + 1e-6) - 1.0)
                    is_converge = is_converge and (ratio < tol)

                    data_convergence[name]['old_value'] = old_value
                    data_convergence[name]['cur_value'] = new_value
                    convergence_cells[name] = ratio
                log_recorder.write_scalars('data_convergence', convergence_cells, step)
            else:
                is_converge = False

            if step < 3:  # Warmup
                is_converge = False

            if step > max_iter:
                print('[Debug] Fail Converge, Max Iter Reach')
                break

            if is_converge or sub_stop:
                if step > 10:
                    print('[Debug] Successful Converge')
                else:
                    print('[Debug] Time Step May Be Too Small')
                break

        return {'is_converge': is_converge, 'step': step, 'simulator_dir': simulator_dir}

    def merge_up(self, fun_V: dolfinx.fem.Function, fun_Q: dolfinx.fem.Function, fun_W: dolfinx.fem.Function = None):
        assert (fun_V.function_space == self.V) and (fun_Q.function_space == self.Q)
        if fun_W is not None:
            assert fun_W.function_space == self.W
        else:
            fun_W = dolfinx.fem.Function(self.W)
        fun_W.x.array[self.V_to_W_dofs] = fun_V.x.array
        fun_W.x.array[self.Q_to_W_dofs] = fun_Q.x.array
        return fun_W

    def split_up(
            self, fun_W: dolfinx.fem.Function, fun_V: dolfinx.fem.Function = None, fun_Q: dolfinx.fem.Function = None
    ):
        assert fun_W.function_space == self.W

        if fun_V is not None:
            assert fun_V.function_space == self.V
            fun_V.vector.aypx(0.0, fun_W.sub(0).collapse().vector)
        else:
            fun_V = fun_W.sub(0).collapse()

        if fun_Q is not None:
            assert fun_Q.function_space == self.Q
            fun_Q.vector.aypx(0.0, fun_W.sub(1).collapse().vector)
        else:
            fun_Q = fun_W.sub(1).collapse()

        return fun_V, fun_Q

    def get_up(self, method):
        if not self.equation_map.get(method, False):
            raise ValueError("[ERROR]: Non-Valid Parameters")
        if 'up' in self.equation_map[method]:
            up: dolfinx.fem.Function = self.equation_map[method]['up']
            u_n, p_n = ufl.split(up)
        else:
            u_n, p_n = self.equation_map[method]['u_n'], self.equation_map[method]['p_n']
        return u_n, p_n

    def save_result(self, save_dir: str, methods: List[str], save_type='pkl'):
        for method in methods:
            if method == 'ipcs' and self.equation_map.get('ipcs', False):
                if save_type == 'pkl':
                    with open(os.path.join(save_dir, 'ipcs_u.pkl'), 'wb') as f:
                        pickle.dump(self.equation_map['ipcs']['u_n'].x.array, f)
                    with open(os.path.join(save_dir, 'ipcs_p.pkl'), 'wb') as f:
                        pickle.dump(self.equation_map['ipcs']['p_n'].x.array, f)

            if method == 'navier_stoke' and self.equation_map.get('navier_stoke', False):
                if save_type == 'pkl':
                    with open(os.path.join(save_dir, 'navier_stoke_u.pkl'), 'wb') as f:
                        pickle.dump(self.equation_map['navier_stoke']['up'].sub(0).collapse().x.array, f)
                    with open(os.path.join(save_dir, 'navier_stoke_p.pkl'), 'wb') as f:
                        pickle.dump(self.equation_map['navier_stoke']['up'].sub(1).collapse().x.array, f)

            if method == 'stoke' and self.equation_map.get('stoke', False):
                if save_type == 'pkl':
                    with open(os.path.join(save_dir, 'stoke_u.pkl'), 'wb') as f:
                        pickle.dump(self.equation_map['stoke']['up'].sub(0).collapse().x.array, f)
                    with open(os.path.join(save_dir, 'stoke_p.pkl'), 'wb') as f:
                        pickle.dump(self.equation_map['stoke']['up'].sub(1).collapse().x.array, f)

    def load_result(self, file_info: dict, load_type='pkl'):
        """
        only support navier stoke
        """
        if load_type == 'pkl':
            with open(file_info['u_pkl_file'], 'rb') as f:
                u_array = pickle.load(f)
            with open(file_info['p_pkl_file'], 'rb') as f:
                p_array = pickle.load(f)

            up: dolfinx.fem.Function = self.equation_map['navier_stoke']['up']
            up.x.array[self.V_to_W_dofs] = u_array
            up.x.array[self.Q_to_W_dofs] = p_array

        elif load_type == 'csv':
            df = pd.read_csv(file_info['csv_file'], index_col=0)

            if self.tdim == 2:
                coords = df[['coord_x', 'coord_y']].values
                vel = df[['vel_x', 'vel_y']].values
            elif self.tdim == 3:
                coords = df[['coord_x', 'coord_y', 'coord_z']].values
                vel = df[['vel_x', 'vel_y', 'vel_z']].values
            else:
                raise ValueError("[ERROR]: Non-Valid Mesh")
            pressure = df['pressure'].values

            vel_fun, pressure_fun = CrossSimulatorUtil.convert_to_simple_function(
                self.domain, coords=coords, value_list=[vel, pressure],
            )

            if self.equation_map.get('ipcs', False):
                self.equation_map['ipcs']['u_n'].interpolate(vel_fun)
                self.equation_map['ipcs']['p_n'].interpolate(pressure_fun)

            if self.equation_map.get('stoke', False):
                u_n_temp = dolfinx.fem.Function(self.V)
                u_n_temp.interpolate(vel_fun)
                p_n_temp = dolfinx.fem.Function(self.Q)
                p_n_temp.interpolate(pressure_fun)

                up: dolfinx.fem.Function = self.equation_map['stoke']['up']
                up.x.array[self.V_to_W_dofs] = u_n_temp.x.array
                up.x.array[self.Q_to_W_dofs] = p_n_temp.x.array

            if self.equation_map.get('navier_stoke', False):
                u_n_temp = dolfinx.fem.Function(self.V)
                u_n_temp.interpolate(vel_fun)
                p_n_temp = dolfinx.fem.Function(self.Q)
                p_n_temp.interpolate(pressure_fun)

                up: dolfinx.fem.Function = self.equation_map['navier_stoke']['up']
                up.x.array[self.V_to_W_dofs] = u_n_temp.x.array
                up.x.array[self.Q_to_W_dofs] = p_n_temp.x.array

    def convert_result_to_csv(self, u_fun: dolfinx.fem.Function, p_fun: dolfinx.fem.Function):
        u_map = dolfinx.fem.Function(self.V_mapping_space)
        u_map.interpolate(u_fun)
        p_map = dolfinx.fem.Function(self.Q_mapping_space)
        p_map.interpolate(p_fun)

        df = pd.DataFrame()
        if self.tdim == 2:
            df[['coord_x', 'coord_y']] = self.domain.geometry.x[:, :2].copy()
            df[['vel_x', 'vel_y']] = u_map.x.array.reshape((-1, 2))
        else:
            df[['coord_x', 'coord_y', 'coord_z']] = self.domain.geometry.x[:, :3].copy()
            df[['vel_x', 'vel_y', 'vel_z']] = u_map.x.array.reshape((-1, 3))
        df['pressure'] = p_map.x.array

        return df

    def estimate_equation_lost(self, method):
        if method == 'stoke':
            res_dict = LinearProblemSolver.equation_investigation(
                a_form=dolfinx.fem.form(self.equation_map['stoke']['lhs_form']),
                L_form=dolfinx.fem.form(self.equation_map['stoke']['rhs_form']),
                bcs=self.equation_map['stoke']['bcs'],
                uh=self.equation_map['stoke']['up']
            )
        elif method == 'navier_stoke':
            res_dict = NonLinearProblemSolver.equation_investigation(
                lhs_form=dolfinx.fem.form(self.equation_map['navier_stoke']['lhs']),
                bcs=self.equation_map['navier_stoke']['bcs'],
                uh=self.equation_map['navier_stoke']['up']
            )
        else:
            raise ValueError('[ERROR]: Non-Valid method')
        return res_dict

    def find_navier_stoke_initiation(
            self,
            proj_dir, max_iter, log_iter, trial_iter,
            snes_option, ksp_option, snes_criterion, with_debug=False, **kwargs
    ):
        def post_function(sim: FluidSimulator, step: int, simulator_dir):
            if step % trial_iter == 0:
                guess_up = self.merge_up(self.equation_map['ipcs']['u_n'], self.equation_map['ipcs']['p_n'])
                res_dict = self.run_navier_stoke_process(
                    proj_dir=proj_dir, name='NS',
                    guess_up=guess_up,
                    snes_option=snes_option,
                    snes_criterion=snes_criterion,
                    ksp_option=ksp_option,
                    logger_dicts=kwargs.get('ns_logger_dicts', {}),
                    with_debug=True,  # Must Be True
                    with_monitor=False,
                )

                converged_code = res_dict['converged_reason']
                converged_error = res_dict['max_error']
                if (
                        NonLinearProblemSolver.is_converged(converged_code)
                        and (converged_error < 1e-6)
                ):
                    step_dir = os.path.join(proj_dir, f"init_step_{step}")
                    if os.path.exists(step_dir):
                        shutil.rmtree(step_dir)
                    os.mkdir(step_dir)
                    self.save_result(step_dir, methods=['navier_stoke'])
                    print(f"[Info FluidSimulator] found SNES_code:{converged_code} error:{converged_error}")
                    return {'state': True}
                else:
                    print(f"[Info FluidSimulator] fail SNES_code:{converged_code} error:{converged_error}")
                    return {'state': False}

            return {'state': False}

        res_dict = self.run_ipcs_process(
            proj_dir=proj_dir, name='ipcs', max_iter=max_iter, log_iter=log_iter,
            tol=kwargs.get('tol', None),
            data_convergence=kwargs.get('data_convergence', {}),
            pre_function=None,
            post_function=post_function,
            with_debug=with_debug,
            logger_dicts=kwargs.get('logger_dicts', {}),
        )

        save_dir = os.path.join(res_dict['simulator_dir'], 'res_pkl')
        os.mkdir(save_dir)
        self.save_result(save_dir, methods=['ipcs'])

        # if res_dict['is_converge']:
        #     post_function(self, step=res_dict['step'], simulator_dir=None)

    @staticmethod
    def eval_point(uh: dolfinx.fem.Function, points: np.ndarray, domain: dolfinx.mesh.Mesh):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
        cells = []
        for i in range(points.shape[0]):
            cells.append(colliding_cells.links(i)[0])
        res = uh.eval(points, cells).reshape(-1)
        return res
