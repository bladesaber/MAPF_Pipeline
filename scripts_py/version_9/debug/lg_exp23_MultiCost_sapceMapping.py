"""
Fail 2024/03/06
"""

import numpy as np
import ufl
from ufl import grad, inner, dot, div
import dolfinx
import os
import ctypes
from typing import List, Callable, Dict
from functools import partial
import pyvista
import shutil
import pickle

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.lagrange_method.space_mapping_algo import FineModel, CoarseModel, \
    ParameterExtraction, SpaceMappingProblem
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_state_problem, create_shape_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional, IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.remesh_helper import ReMesher
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.remesh_helper import MeshDeformationRunner
from scripts_py.version_9.dolfinx_Grad.optimizer_utils import CostConvergeHandler

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape7'
model_xdmf = os.path.join(proj_dir, 'model.xdmf')
msh_file = os.path.join(proj_dir, 'model.msh')

# ------ create xdmf
MeshUtils.msh_to_XDMF(name='model', msh_file=msh_file, output_file=model_xdmf, dim=2)

# ------ mutual parameters
input_marker = 1
output_markers = [5, 6, 7]
bry_markers = [2, 3, 4]

bry_fixed_markers = [1, 4, 5, 6, 7]
bry_free_markers = [2, 3]

Re = 100
nuValue = 1. / Re


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
    return values


def extract_parameter_func(domain: dolfinx.mesh.Mesh, tdim=2):
    return np.copy(domain.geometry.x[:, :tdim])


def define_state_boundary(
        domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
        W0: dolfinx.fem.functionspace, V: dolfinx.fem.functionspace,
        W1: dolfinx.fem.functionspace, Q: dolfinx.fem.functionspace
):
    tdim = domain.topology.dim
    fdim = tdim - 1

    bcs_info = []
    for marker in bry_markers:
        bc_value = dolfinx.fem.Function(V, name=f"bry_u_{marker}")
        bc_dofs = MeshUtils.extract_entity_dofs(
            (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
        )
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, W0)
        bcs_info.append((bc, W0, bc_dofs, bc_value))

    bc_in1_value = dolfinx.fem.Function(V, name='inflow_u')
    bc_in1_value.interpolate(partial(inflow_velocity_exp, tdim=tdim))
    bc_in1_dofs = MeshUtils.extract_entity_dofs(
        (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
    )
    bc_in1 = dolfinx.fem.dirichletbc(bc_in1_value, bc_in1_dofs, W0)
    bcs_info.append((bc_in1, W0, bc_in1_dofs, bc_in1_value))

    for marker in output_markers:
        bc_out_value = dolfinx.fem.Function(Q, name=f"outflow_p_{marker}")
        bc_out_dofs = MeshUtils.extract_entity_dofs(
            (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
        )
        bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs, W1)
        bcs_info.append((bc_out, W1, bc_out_dofs, bc_out_value))

    return bcs_info


def define_shape_boundary(
        domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
        V_S: dolfinx.fem.functionspace
):
    tdim = domain.topology.dim
    fdim = tdim - 1
    bcs_info = []
    for marker in bry_fixed_markers:
        bc_value = dolfinx.fem.Function(V_S, name=f"fix_bry_shape_{marker}")
        bc_dofs = MeshUtils.extract_entity_dofs(V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker))
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)
        bcs_info.append((bc, V_S, bc_dofs, bc_value))
    return bcs_info


# --------------------------------------------------------------
class LowReFluidFineModel(FineModel):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
            msh_file: str, stoke_ksp_option: Dict, nstoke_ksp_option: Dict,
    ):
        self.msh_file = msh_file
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.stoke_ksp_option = stoke_ksp_option
        self.nstoke_ksp_option = nstoke_ksp_option

        # ------ init workspace
        self.vertex_indices = ReMesher.reconstruct_vertex_indices(self.msh_file, domain=self.domain, check=True)
        self.tdim = self.domain.topology.dim

        W = dolfinx.fem.FunctionSpace(
            self.domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", self.domain.ufl_cell(), 1)
            ])
        )
        W0, W1 = W.sub(0), W.sub(1)
        V, _ = W0.collapse()
        Q, _ = W1.collapse()
        self.V_mapping_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        self.Q_mapping_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))

        ds = MeshUtils.define_ds(self.domain, self.facet_tags)
        n_vec = MeshUtils.define_facet_norm(self.domain)

        self.deformation_handler = MeshDeformationRunner(
            self.domain,
            # volume_change=-1.0,
            # quality_measures={
            #     'max_angle': {
            #         'measure_type': 'max',
            #         'tol_upper': 165.0,
            #         'tol_lower': 0.0
            #     },
            #     'min_angle': {
            #         'measure_type': 'min',
            #         'tol_upper': 180.0,
            #         'tol_lower': 15.0
            #     }
            # }
        )

        # ------ define boundary
        self.bcs_info = define_state_boundary(self.domain, self.cell_tags, self.facet_tags, W0=W0, V=V, W1=W1, Q=Q)
        self.bcs = [info[0] for info in self.bcs_info]

        # ------ define problem
        nu = dolfinx.fem.Constant(self.domain, nuValue)
        u1, p1 = ufl.split(ufl.TrialFunction(W))
        v1, q1 = ufl.split(ufl.TestFunction(W))
        f1 = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))
        stoke_form = (
                inner(grad(u1), grad(v1)) * ufl.dx
                - p1 * div(v1) * ufl.dx
                - q1 * div(u1) * ufl.dx
                - inner(f1, v1) * ufl.dx
        )
        self.stoke_lhs = dolfinx.fem.form(ufl.lhs(stoke_form))
        self.stoke_rhs = dolfinx.fem.form(ufl.rhs(stoke_form))

        self.uh_fine = dolfinx.fem.Function(W, name='fine_state')  # state of FineModel
        u2, p2 = ufl.split(self.uh_fine)
        v2, q2 = ufl.split(ufl.TestFunction(W))
        f2 = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))
        self.navier_stoke_lhs = (
                nu * inner(grad(u2), grad(v2)) * ufl.dx
                + inner(grad(u2) * u2, v2) * ufl.dx
                - inner(p2, div(v2)) * ufl.dx
                + inner(div(u2), q2) * ufl.dx
                - inner(f2, v2) * ufl.dx
        )

        # ------ define target
        inflow_fun = dolfinx.fem.Function(V, name='inflow_u')
        inflow_fun.interpolate(partial(inflow_velocity_exp, tdim=self.tdim))
        self.tracking_goal = -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(
            ufl.dot(inflow_fun, n_vec) * ds(input_marker)
        )) / 3.0

        self.outflow_data = {}
        for marker in output_markers:
            self.outflow_data[f"marker_{marker}"] = {
                'form': dolfinx.fem.form(ufl.dot(u2, n_vec) * ds(marker)),
                'value': ctypes.c_double(0.0),
                'target': self.tracking_goal
            }

        self.energy_loss_form = dolfinx.fem.form(inner(grad(u2), grad(u2)) * ufl.dx)
        self.energy_form = dolfinx.fem.form(inner(u2, u2) * ufl.dx)
        self.energy_loss_value = 0.0
        self.energy_value = 0.0

    def solve_and_evaluate(
            self, guass_uh: dolfinx.fem.Function = None, use_guass_init=False, **kwargs
    ):
        if use_guass_init:
            if guass_uh is None:
                res_dict = LinearProblemSolver.solve_by_petsc_form(
                    comm=self.domain.comm, uh=self.uh_fine, a_form=self.stoke_lhs, L_form=self.stoke_rhs, bcs=self.bcs,
                    ksp_option=self.stoke_ksp_option,
                    **kwargs
                )
                if kwargs.get('with_debug', False):
                    print(f"[DEBUG FineModel Stoke]: max_error:{res_dict['max_error']:.8f} "
                          f"cost_time:{res_dict['cost_time']:.2f}")

            else:
                self.uh_fine.vector.aypx(0.0, guass_uh.vector)

        jacobi_form = ufl.derivative(
            self.navier_stoke_lhs, self.uh_fine, ufl.TrialFunction(self.uh_fine.function_space)
        )
        res_dict = NonLinearProblemSolver.solve_by_petsc(
            F_form=self.navier_stoke_lhs, uh=self.uh_fine, jacobi_form=jacobi_form, bcs=self.bcs,
            comm=self.domain.comm, ksp_option=self.nstoke_ksp_option,
            **kwargs
        )
        if kwargs.get('with_debug', False):
            print(f"[DEBUG FineModel Navier Stoke]: max_error:{res_dict['max_error']:.8f} "
                  f"cost_time:{res_dict['cost_time']:.2f}")

        self.evaluate_loss()

    def evaluate_loss(self):
        self.cost_functional_value = 0.0
        for key in self.outflow_data.keys():
            outflow_value = AssembleUtils.assemble_scalar(self.outflow_data[key]['form'])
            self.outflow_data[key]['value'].value = outflow_value
            self.cost_functional_value = np.maximum(
                self.cost_functional_value, np.abs(self.tracking_goal - outflow_value)
            )

        self.energy_loss_value = AssembleUtils.assemble_scalar(self.energy_loss_form)
        self.energy_value = AssembleUtils.assemble_scalar(self.energy_form)

    def update_geometry(self, grad: np.ndarray):
        displacement_np = np.zeros(self.domain.geometry.x.shape)
        displacement_np[:, :self.tdim] = grad.reshape((-1, self.tdim))

        # stepSize = 1.0
        # success_flag, info = self.deformation_handler.move_mesh(displacement_np)
        # assert success_flag

        success_flag, stepSize = self.deformation_handler.move_mesh_by_line_search(
            displacement_np, max_iter=10, init_stepSize=1.0, stepSize_lower=1e-4,
            detect_cost_valid_func=None, with_debug_info=False
        )

        return success_flag, stepSize

    def save_anchor(
            self, record_dir, u_vtk_recorder: VTKRecorder, p_vtk_recorder: VTKRecorder, step, name: str = 'fine_model'
    ):
        u_res = dolfinx.fem.Function(self.V_mapping_space)
        u_res.interpolate(self.uh_fine.sub(0).collapse())

        u_grid = np.zeros(self.domain.geometry.x.shape)
        u_grid[:, :self.tdim] = u_res.x.array.reshape((-1, self.tdim))
        p_grid = self.uh_fine.sub(1).collapse().x.array

        grid = VisUtils.convert_to_grid(self.domain)
        grid['velocity'] = u_grid
        grid['pressure'] = p_grid
        pyvista.save_meshio(os.path.join(record_dir, f"{name}.vtk"), grid)

        u_vtk_recorder.write_function(u_res, step=step)
        p_vtk_recorder.write_function(self.uh_fine.sub(1).collapse(), step=step)


class FluidCoarseModel(CoarseModel):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
            fine_model: LowReFluidFineModel, record_dir: str,
    ):
        self.fine_model = fine_model
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.record_dir = record_dir

        # ------ init workspace
        self.tdim = self.domain.topology.dim

        W = dolfinx.fem.FunctionSpace(
            self.domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", self.domain.ufl_cell(), 1)
            ])
        )
        W0, W1 = W.sub(0), W.sub(1)
        V, _ = W0.collapse()
        Q, _ = W1.collapse()
        self.V_mapping_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        self.Q_mapping_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))

        self.deformation_handler = MeshDeformationRunner(
            self.domain,
            volume_change=0.15,
            quality_measures={
                'max_angle': {
                    'measure_type': 'max',
                    'tol_upper': 165.0,
                    'tol_lower': 0.0
                },
                'min_angle': {
                    'measure_type': 'min',
                    'tol_upper': 180.0,
                    'tol_lower': 15.0
                }
            }
        )

        # ------ define boundary
        bcs_info = define_state_boundary(self.domain, self.cell_tags, self.facet_tags, W0=W0, V=V, W1=W1, Q=Q)

        # ------ define state system
        self.up = dolfinx.fem.Function(W, name='coarse_state')
        u, p = ufl.split(self.up)
        vq = dolfinx.fem.Function(W, name='coarse_adjoint')
        v, q = ufl.split(vq)
        f = dolfinx.fem.Constant(domain, np.zeros(self.tdim))

        F_form = inner(grad(u), grad(v)) * ufl.dx - p * div(v) * ufl.dx - q * div(u) * ufl.dx - inner(f, v) * ufl.dx
        state_problem = create_state_problem(
            name='coarse_1', F_form=F_form, state=self.up, adjoint=vq, is_linear=True,
            bcs_info=bcs_info,
            state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        )
        state_system = StateProblem([state_problem])

        # ------ define control system
        coordinate_space = self.domain.ufl_domain().ufl_coordinate_element()
        V_S = dolfinx.fem.FunctionSpace(self.domain, coordinate_space)

        bcs_info = define_shape_boundary(self.domain, self.cell_tags, self.facet_tags, V_S=V_S)
        control_problem = create_shape_problem(
            domain=self.domain, bcs_info=bcs_info,
            gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        )

        # ------ define cost function
        ds = MeshUtils.define_ds(self.domain, facet_tags)
        n_vec = MeshUtils.define_facet_norm(self.domain)
        self.outflow_data = {}

        cost_functional_list = []
        for marker in output_markers:
            integrand_form = ufl.dot(u, n_vec) * ds(marker)
            target_value = self.fine_model.outflow_data[f"marker_{marker}"]['target']
            cost_functional_list.append(ScalarTrackingFunctional(
                domain, integrand_form, target_value, name=f"track_{marker}"
            ))

            self.outflow_data[f"marker_{marker}"] = {
                'form': dolfinx.fem.form(ufl.dot(u, n_vec) * ds(marker)),
                'target': target_value
            }

        energy_loss = inner(grad(u), grad(u)) * ufl.dx
        energy_loss_fun = IntegralFunction(domain=domain, form=energy_loss, name=f"energy_loss")
        cost_functional_list.append(energy_loss_fun)

        self.energy_loss_form = dolfinx.fem.form(energy_loss)
        self.energy_form = dolfinx.fem.form(inner(u, u) * ufl.dx)

        # ------ define opt problem
        opt_problem = OptimalShapeProblem(
            state_system=state_system,
            shape_problem=control_problem,
            shape_regulariztions=ShapeRegularization([
                VolumeRegularization(control_problem, mu=1.0, target_volume_rho=1.0)
            ]),
            cost_functional_list=cost_functional_list,
            scalar_product=None,
            scalar_product_method={
                'method': 'Poincare-Steklov operator',
                'lambda_lame': 0.0,
                'damping_factor': 0.0,
                'cell_tags': cell_tags,
                'facet_tags': facet_tags,
                'bry_free_markers': bry_free_markers,
                'bry_fixed_markers': bry_fixed_markers,
                'mu_fix': 1.0,
                'mu_free': 1.0,
                'use_inhomogeneous': True,
                'inhomogeneous_exponent': 1.0,
                'update_inhomogeneous': False
            }
        )

        opt_problem.state_system.solve(domain.comm, with_debug=False)
        for cost_func in cost_functional_list:
            cost = cost_func.evaluate()
            if cost_func.name == 'energy_loss':
                weight = 2.0 / cost
            else:
                weight = (1.0 / 3.0) / cost
            cost_func.update_scale(weight)

        # ------
        super().__init__(
            opt_problem=opt_problem,
            control=self.domain,
            extract_parameter_func=extract_parameter_func
        )

    def _optimize(self, **kwargs):
        if os.path.exists(self.record_dir):
            shutil.rmtree(self.record_dir)
        os.mkdir(self.record_dir)

        simulate_dir = os.path.join(self.record_dir, 'simulate')
        os.mkdir(simulate_dir)
        vtk_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))

        tensorBoard_dir = os.path.join(self.record_dir, 'log')
        os.mkdir(tensorBoard_dir)
        log_recorder = TensorBoardRecorder(tensorBoard_dir)

        init_loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=True)
        loss_storge_ctype = ctypes.c_double(init_loss)
        cost_converger = CostConvergeHandler(stat_num=20, warm_up_num=20, tol=5e-3, scale=1.0 / init_loss)

        def detect_cost_valid_func(tol_rho=0.05):
            loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=True)
            is_valid = loss < loss_storge_ctype.value + np.abs(loss_storge_ctype.value) * tol_rho
            return is_valid

        step = 0
        while True:
            step += 1

            debug_log = f"CoarseModel Step {step}: "
            shape_grad: dolfinx.fem.Function = self.opt_problem.compute_gradient(
                self.domain.comm,
                state_kwargs={'with_debug': kwargs.get('with_debug', False)},
                adjoint_kwargs={'with_debug': kwargs.get('with_debug', False)},
                gradient_kwargs={
                    'with_debug': kwargs.get('with_debug', False), 'A_assemble_method': 'Identity_row'
                },
            )

            shape_grad_np = shape_grad.x.array
            # shapr_grade_scale = np.linalg.norm(shape_grad_np, ord=2)
            shape_grad_np = shape_grad_np * -1.0

            displacement_np = np.zeros(self.domain.geometry.x.shape)
            displacement_np[:, :self.tdim] = shape_grad_np.reshape((-1, self.tdim))

            # grid['grad_test'] = displacement_np
            # VisUtils.show_arrow_from_grid(grid, 'grad_test', scale=30.0).show()

            # success_flag, info = deformation_handler.move_mesh(displacement_np)
            success_flag, stepSize = self.deformation_handler.move_mesh_by_line_search(
                displacement_np, max_iter=10, init_stepSize=2.0, stepSize_lower=1e-3,
                detect_cost_valid_func=detect_cost_valid_func
            )

            if success_flag:
                loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=False)
                loss_storge_ctype.value = loss

                is_converge = cost_converger.is_converge(loss)

                # ------ record
                scale_loss = cost_converger.compute_scale_loss(loss)
                log_recorder.write_scalar('scale_loss', scale_loss, step=step)
                log_recorder.write_scalar('scale_loss_var', cost_converger.scale_cost_variation, step=step)

                outflow_cells = {}
                for key in self.outflow_data.keys():
                    outflow_cells[key] = AssembleUtils.assemble_scalar(self.outflow_data[key]['form'])
                log_recorder.write_scalars('outflow', outflow_cells, step=step)

                energy_loss_value = AssembleUtils.assemble_scalar(self.energy_loss_form)
                log_recorder.write_scalar('energy_loss', energy_loss_value, step)

                energy_value = AssembleUtils.assemble_scalar(self.energy_form)
                log_recorder.write_scalar('energy_value', energy_value, step)

                vtk_recorder.write_function(self.up.sub(0).collapse(), step=step)

                # ------ debug output
                if kwargs.get('output_info', False):
                    for key in self.outflow_data.keys():
                        target_flow, out_flow = self.outflow_data[key]['target'], outflow_cells[key]
                        ratio = out_flow / target_flow
                        debug_log += f"[{key}: {ratio:.2f}| {out_flow:.3f}/{target_flow: .3f}] "
                    debug_log += f"loss:{loss:.8f}, energy:{energy_value:.4f}, stepSize:{stepSize}"
                    print(debug_log)
                # ------

                if is_converge:
                    break

                if step > 150:
                    break

            else:
                break

        self.save_anchor()
        print(f"[Debug] CoarseModel End")

    def set_best_parameter(self, coordinate_pkl: str):
        with open(coordinate_pkl, 'rb') as f:
            coordinate_xyzs = pickle.load(f)
        self.domain.geometry.x[:] = coordinate_xyzs
        self.parameter_optimal = extract_parameter_func(self.domain)

    def save_anchor(self, name: str = 'opt_model'):
        u_res = dolfinx.fem.Function(self.V_mapping_space)
        u_res.interpolate(self.up.sub(0).collapse())

        u_grid = np.zeros(self.control.geometry.x.shape)
        u_grid[:, :self.tdim] = u_res.x.array.reshape((-1, self.tdim))
        p_grid = self.up.sub(1).collapse().x.array

        grid = VisUtils.convert_to_grid(self.control)
        grid['velocity'] = u_grid
        grid['pressure'] = p_grid
        pyvista.save_meshio(os.path.join(self.record_dir, f"{name}.vtk"), grid)

        with open(os.path.join(self.record_dir, 'coordinate.pkl'), 'wb') as f:
            pickle.dump(self.domain.geometry.x, f)


class FluidParameterExtraction(ParameterExtraction):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
            fine_model: LowReFluidFineModel
    ):
        self.fine_model = fine_model
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags

        # ------ init workspace
        self.tdim = self.domain.topology.dim

        W = dolfinx.fem.FunctionSpace(
            self.domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", self.domain.ufl_cell(), 1)
            ])
        )
        W0, W1 = W.sub(0), W.sub(1)
        V, _ = W0.collapse()
        Q, _ = W1.collapse()
        self.V_mapping_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        self.Q_mapping_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))

        self.deformation_handler = MeshDeformationRunner(
            self.domain,
            volume_change=0.15,
            quality_measures={
                'max_angle': {
                    'measure_type': 'max',
                    'tol_upper': 165.0,
                    'tol_lower': 0.0
                },
                'min_angle': {
                    'measure_type': 'min',
                    'tol_upper': 180.0,
                    'tol_lower': 15.0
                }
            }
        )

        # ------ define boundary
        bcs_info = define_state_boundary(self.domain, self.cell_tags, self.facet_tags, W0=W0, V=V, W1=W1, Q=Q)

        # ------ define state system
        self.up = dolfinx.fem.Function(W, name='paraExt_state')
        u, p = ufl.split(self.up)
        vq = dolfinx.fem.Function(W, name='paraExt_adjoint')
        v, q = ufl.split(vq)
        f = dolfinx.fem.Constant(domain, np.zeros(self.tdim))

        F_form = inner(grad(u), grad(v)) * ufl.dx - p * div(v) * ufl.dx - q * div(u) * ufl.dx - inner(f, v) * ufl.dx
        state_problem = create_state_problem(
            name='coarse_1', F_form=F_form, state=self.up, adjoint=vq, is_linear=True,
            bcs_info=bcs_info,
            state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        )
        state_system = StateProblem([state_problem])

        # ------ define control system
        coordinate_space = self.domain.ufl_domain().ufl_coordinate_element()
        V_S = dolfinx.fem.FunctionSpace(self.domain, coordinate_space)

        bcs_info = define_shape_boundary(self.domain, self.cell_tags, self.facet_tags, V_S=V_S)
        control_problem = create_shape_problem(
            domain=self.domain, bcs_info=bcs_info,
            gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        )

        # ------ define cost function
        ds = MeshUtils.define_ds(self.domain, facet_tags)
        n_vec = MeshUtils.define_facet_norm(self.domain)
        self.outflow_data = {}

        cost_functional_list = []
        for marker in output_markers:
            integrand_form = ufl.dot(u, n_vec) * ds(marker)
            cost_functional_list.append(ScalarTrackingFunctional(
                domain, integrand_form, fine_model.outflow_data[f"marker_{marker}"]['value'], name=f"track_{marker}"
            ))

            self.outflow_data[f"marker_{marker}"] = {
                'form': dolfinx.fem.form(ufl.dot(u, n_vec) * ds(marker)),
                'target': fine_model.outflow_data[f"marker_{marker}"]['value']
            }

        energy_loss = inner(grad(u), grad(u)) * ufl.dx
        self.energy_loss_form = dolfinx.fem.form(energy_loss)
        self.energy_form = dolfinx.fem.form(inner(u, u) * ufl.dx)

        # ------ define opt problem
        opt_problem = OptimalShapeProblem(
            state_system=state_system,
            shape_problem=control_problem,
            shape_regulariztions=ShapeRegularization([]),
            cost_functional_list=cost_functional_list,
            scalar_product=None,
            scalar_product_method={
                'method': 'Poincare-Steklov operator',
                'lambda_lame': 0.0,
                'damping_factor': 0.0,
                'cell_tags': cell_tags,
                'facet_tags': facet_tags,
                'bry_free_markers': bry_free_markers,
                'bry_fixed_markers': bry_fixed_markers,
                'mu_fix': 1.0,
                'mu_free': 1.0,
                'use_inhomogeneous': True,
                'inhomogeneous_exponent': 1.0,
                'update_inhomogeneous': False
            }
        )

        super().__init__(
            opt_problem=opt_problem,
            control=self.domain,
            extract_parameter_func=extract_parameter_func
        )

    def _optimize(self, record_dir, **kwargs):
        if os.path.exists(record_dir):
            shutil.rmtree(record_dir)
        os.mkdir(record_dir)

        simulate_dir = os.path.join(record_dir, 'simulate')
        os.mkdir(simulate_dir)
        vtk_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))

        tensorBoard_dir = os.path.join(record_dir, 'log')
        os.mkdir(tensorBoard_dir)
        log_recorder = TensorBoardRecorder(tensorBoard_dir)

        init_loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=False)
        loss_storge_ctype = ctypes.c_double(init_loss)
        cost_converger = CostConvergeHandler(stat_num=10, warm_up_num=10, tol=5e-3, scale=1.0 / init_loss)

        def detect_cost_valid_func(tol_rho=0.05):
            loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=True)
            is_valid = loss < loss_storge_ctype.value + np.abs(loss_storge_ctype.value) * tol_rho
            return is_valid

        step = 0
        while True:
            step += 1

            debug_log = f"ParameterExtraction Step {step}:"
            shape_grad: dolfinx.fem.Function = self.opt_problem.compute_gradient(
                self.domain.comm,
                state_kwargs={'with_debug': kwargs.get('with_debug', False)},
                adjoint_kwargs={'with_debug': kwargs.get('with_debug', False)},
                gradient_kwargs={'with_debug': kwargs.get('with_debug', False), 'A_assemble_method': 'Identity_row'},
            )

            shape_grad_np = shape_grad.x.array
            # shapr_grade_scale = np.linalg.norm(shape_grad_np, ord=2)
            shape_grad_np = shape_grad_np * -1.0

            displacement_np = np.zeros(self.domain.geometry.x.shape)
            displacement_np[:, :self.tdim] = shape_grad_np.reshape((-1, self.tdim))

            # grid['grad_test'] = displacement_np
            # VisUtils.show_arrow_from_grid(grid, 'grad_test', scale=30.0).show()

            # success_flag, info = deformation_handler.move_mesh(displacement_np)
            success_flag, stepSize = self.deformation_handler.move_mesh_by_line_search(
                displacement_np, max_iter=10, init_stepSize=2.0, stepSize_lower=1e-3,
                detect_cost_valid_func=detect_cost_valid_func
            )

            if success_flag:
                loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=False)
                loss_storge_ctype.value = loss

                is_converge = cost_converger.is_converge(loss)

                # ------ record
                scale_loss = cost_converger.compute_scale_loss(loss)
                log_recorder.write_scalar('scale_loss', scale_loss, step=step)
                log_recorder.write_scalar('scale_loss_var', cost_converger.scale_cost_variation, step=step)

                outflow_cells = {}
                for key in self.outflow_data.keys():
                    outflow_cells[key] = AssembleUtils.assemble_scalar(self.outflow_data[key]['form'])
                log_recorder.write_scalars('outflow', outflow_cells, step=step)

                energy_loss = AssembleUtils.assemble_scalar(self.energy_loss_form)
                log_recorder.write_scalar('energy_loss', energy_loss, step)

                vtk_recorder.write_function(self.up.sub(0).collapse(), step=step)

                # ------ debug output
                if kwargs.get('output_info', False):
                    for key in self.outflow_data.keys():
                        target_flow, out_flow = self.outflow_data[key]['target'].value, outflow_cells[key]
                        ratio = out_flow / target_flow
                        debug_log += f"[{key}: {ratio:.2f}| {out_flow:.3f}/{target_flow: .3f}] "
                    debug_log += f"loss:{loss:.8f}, stepSize:{stepSize}"
                    print(debug_log)
                # ------

                if is_converge:
                    break

                if step > 100:
                    break

            else:
                break

        print(f"[ParameterExtraction] Optimize Step:{step}")

    def save_anchor(self, record_dir, name: str = 'para_model'):
        u_res = dolfinx.fem.Function(self.V_mapping_space)
        u_res.interpolate(self.up.sub(0).collapse())

        u_grid = np.zeros(self.control.geometry.x.shape)
        u_grid[:, :self.tdim] = u_res.x.array.reshape((-1, self.tdim))
        p_grid = self.up.sub(1).collapse().x.array

        grid = VisUtils.convert_to_grid(self.control)
        grid['velocity'] = u_grid
        grid['pressure'] = p_grid
        pyvista.save_meshio(os.path.join(record_dir, f"{name}.vtk"), grid)


class CustomProblem(SpaceMappingProblem):
    def _compute_eps(self, updated_info: Dict) -> float:
        return np.max(np.abs(updated_info['displacement']))

    def _compute_direction_step(self, para_s2z: np.ndarray, para_best_z: np.ndarray, **kwargs) -> np.ndarray:
        grad_np = self.invert_scale * (para_s2z - para_best_z)
        return grad_np

    def _update(self, grad: np.ndarray, fine_model: LowReFluidFineModel, **kwargs) -> Dict:
        success_flag, step_size = fine_model.update_geometry(grad)
        return {
            'displacement': step_size * grad,
            'stepSize': step_size,
        }

    def pre_log(self, step: int):
        pass

    def post_log(self, step: int, eps: float):
        pass

    def solve(
            self,
            record_dir,
            coarseModel_kwargs: Dict = {},
            fineModel_kwargs: Dict = {},
            paraExtract_kwargs: Dict = {},
            calculate_coarse_best=True,
            **kwargs
    ):
        if os.path.exists(record_dir):
            shutil.rmtree(record_dir)
        os.mkdir(record_dir)

        simulate_dir = os.path.join(record_dir, 'simulate')
        if os.path.exists(simulate_dir):
            shutil.rmtree(simulate_dir)
        os.mkdir(simulate_dir)
        u_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))
        p_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_p.pvd'))

        tensorBoard_dir = os.path.join(record_dir, 'log')
        if os.path.exists(tensorBoard_dir):
            shutil.rmtree(tensorBoard_dir)
        os.mkdir(tensorBoard_dir)
        log_recorder = TensorBoardRecorder(tensorBoard_dir)

        # ----------------------------
        if calculate_coarse_best:
            self.coarse_model.solve(**coarseModel_kwargs)
        best_para_of_z_space = self.coarse_model.parameter_optimal

        # ----------------------------
        step = 0
        converged = False
        while True:
            step += 1

            # ------ step 1: create record directory
            sub_record_dir = os.path.join(record_dir, f"step_{step}")
            os.mkdir(sub_record_dir)

            # ------ step 2: solve fine model
            self.fine_model.solve_and_evaluate(**fineModel_kwargs)
            print(f"[Info] SpaceMapping {step}, "
                  f"FineModel loss:{self.fine_model.cost_functional_value}, "
                  f"energy_value:{self.fine_model.energy_value}")

            self.fine_model.save_anchor(
                record_dir=sub_record_dir,
                u_vtk_recorder=u_recorder,
                p_vtk_recorder=p_recorder,
                step=step
            )

            data_cells = {}
            for key in self.fine_model.outflow_data.keys():
                data_cells[key] = self.fine_model.outflow_data[key]['value'].value
            log_recorder.write_scalars('outflow_u', data_cells, step)
            log_recorder.write_scalar('energy_loss', self.fine_model.energy_loss_value, step)
            log_recorder.write_scalar('energy', self.fine_model.energy_value, step)

            # ------ step 3: solve para extraction model
            para_solving_record_dir = os.path.join(sub_record_dir, "para_extract")
            os.mkdir(para_solving_record_dir)
            paraExtract_kwargs['record_dir'] = para_solving_record_dir

            self.parameter_extraction.solve(**paraExtract_kwargs)
            self.parameter_extraction.save_anchor(record_dir=sub_record_dir)

            # ------ step 4: compute grad and update model
            grad = self._compute_direction_step(
                para_s2z=self.parameter_extraction.parameter_extract,
                para_best_z=best_para_of_z_space,
                **kwargs
            )

            updated_info = self._update(grad, self.fine_model, **kwargs)

            # ------ step 5: solve para extraction model
            eps = self._compute_eps(updated_info)
            print(f"[Info] SpaceMapping {step}: eps:{eps} stepSize:{updated_info['stepSize']} \n")

            # ------ step 6: check convergence
            if eps <= self.tol:
                converged = True
                break

            if step >= self.max_iter:
                break

        return converged


# ---------------------------------------------------------------------------------------
# MeshUtils.msh_to_XDMF(name='model', msh_file=msh_file, output_file=model_xdmf, dim=2)

domain0, cell_tags0, facet_tags0 = MeshUtils.read_XDMF(
    file=model_xdmf, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)
fine_model = LowReFluidFineModel(
    domain=domain0, cell_tags=cell_tags0, facet_tags=facet_tags0, msh_file=msh_file,
    stoke_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    nstoke_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
)

# fine_model.solve_and_evaluate(with_debug=True)
# # print(fine_model.outflow_data)
# u_recorder = VTKRecorder('/home/admin123456/Desktop/work/topopt_exps/fluid_shape4/tst/simulate_u.pvd')
# u_res = dolfinx.fem.Function(fine_model.V_mapping_space)
# u_res.interpolate(fine_model.uh_fine.sub(0).collapse())
# u_recorder.write_function(u_res, step=0)

domain1, cell_tags1, facet_tags1 = MeshUtils.read_XDMF(
    file=model_xdmf, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)
coarse_model = FluidCoarseModel(
    domain1, cell_tags1, facet_tags1, fine_model,
    record_dir=os.path.join(proj_dir, 'coarse_model')
)
# coarse_model.solve(with_debug=False, output_info=True)
coarse_model.set_best_parameter(os.path.join(coarse_model.record_dir, 'coordinate.pkl'))

domain2, cell_tags2, facet_tags2 = MeshUtils.read_XDMF(
    file=model_xdmf, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)
para_extraction = FluidParameterExtraction(domain2, cell_tags2, facet_tags2, fine_model)

problem = CustomProblem(
    coarse_model=coarse_model,
    fine_model=fine_model,
    parameter_extraction=para_extraction,
    tol=1e-3, max_iter=20,
)
problem.solve(
    record_dir=os.path.join(proj_dir, 'spaceMapping'),
    fineModel_kwargs={'with_debug': True},
    paraExtract_kwargs={'output_info': True},
    calculate_coarse_best=False,
)

# ReMesher.convert_domain_to_new_msh(
#     orig_msh_file=msh_file,
#     new_msh_file=os.path.join(proj_dir, 'last_model.msh'),
#     domain=fine_model.domain,
#     dim=fine_model.tdim,
#     vertex_indices=fine_model.vertex_indices
# )
# pyvista.save_meshio(
#     os.path.join(proj_dir, 'last_model.vtk'),
#     VisUtils.convert_to_grid(fine_model.domain)
# )
