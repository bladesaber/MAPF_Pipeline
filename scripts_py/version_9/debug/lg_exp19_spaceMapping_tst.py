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

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.lagrange_method.space_mapping_algo import FineModel, CoarseModel, \
    ParameterExtraction, SpaceMappingProblem
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_state_problem, create_shape_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization
from scripts_py.version_9.dolfinx_Grad.remesh_helper import ReMesher, MeshQuality
from scripts_py.version_9.dolfinx_Grad.equation_solver import LinearProblemSolver, NonLinearProblemSolver
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder, TensorBoardRecorder

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape4'
model_xdmf = os.path.join(proj_dir, 'model.xdmf')
msh_file = os.path.join(proj_dir, 'model.msh')

# ------ create xdmf
MeshUtils.msh_to_XDMF(name='model', msh_file=msh_file, output_file=model_xdmf, dim=2)

# ------ mutual parameters
input_marker = 1
output_markers = [5, 6, 7]
bry_markers = [2, 3, 4]

bry_fixed_markers = [1, 4, 5, 6, 7]
bry_free_marker = [2, 3]

Re = 100
nuValue = 1. / Re


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    # values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
    values[0] = 2.0
    return values


def extract_parameter_func(domain: dolfinx.mesh.Mesh, tdim=2):
    return np.copy(domain.geometry.x[:, :tdim])


# --------------------------------------------------------------
class LowReFluidFineModel(FineModel):
    def __init__(
            self,
            control_s: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            msh_file: str,
            stoke_ksp_option: Dict,
            nstoke_ksp_option: Dict,
    ):
        self.control_s = control_s
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.tdim = self.control_s.topology.dim
        self.fdim = self.tdim - 1

        self.input_marker = input_marker
        self.output_markers = output_markers
        self.bry_markers = bry_markers

        self.stoke_ksp_option = stoke_ksp_option
        self.nstoke_ksp_option = nstoke_ksp_option

        self.W = dolfinx.fem.FunctionSpace(
            self.control_s,
            ufl.MixedElement([
                ufl.VectorElement("Lagrange", self.control_s.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", self.control_s.ufl_cell(), 1)
            ])
        )
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, _ = self.W0.collapse()
        self.Q, _ = self.W1.collapse()
        self.ds = MeshUtils.define_ds(self.control_s, self.facet_tags)
        self.n_vec = MeshUtils.define_facet_norm(self.control_s)
        self.grid = VisUtils.convert_to_grid(self.control_s)
        self.collide_counts = MeshQuality.compute_collide_counts(self.control_s)
        self.vertex_indices = ReMesher.reconstruct_vertex_indices(
            orig_msh_file=msh_file, domain=self.control_s, check=True
        )

        self.define_boundarys()
        self.define_form()

    def define_boundarys(self):
        self.bcs_info = []

        for marker in self.bry_markers:
            bc_value = dolfinx.fem.Function(self.V, name=f"bry_u_{marker}")
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.control_s, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, self.W0)
            self.bcs_info.append((bc, self.W0, bc_dofs, bc_value))

        bc_in1_value = dolfinx.fem.Function(self.V, name='inflow_u')
        bc_in1_value.interpolate(partial(inflow_velocity_exp, tdim=self.tdim))
        bc_in1_dofs = MeshUtils.extract_entity_dofs(
            (self.W0, self.V), self.fdim,
            MeshUtils.extract_facet_entities(self.control_s, self.facet_tags, input_marker)
        )
        bc_in1 = dolfinx.fem.dirichletbc(bc_in1_value, bc_in1_dofs, self.W0)
        self.bcs_info.append((bc_in1, self.W0, bc_in1_dofs, bc_in1_value))

        for marker in self.output_markers:
            bc_out_value = dolfinx.fem.Function(self.Q, name=f"outflow_p_{marker}")
            bc_out_dofs = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(self.control_s, self.facet_tags, marker)
            )
            bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs, self.W1)
            self.bcs_info.append((bc_out, self.W1, bc_out_dofs, bc_out_value))

        self.bcs = [info[0] for info in self.bcs_info]

        # define target
        self.tracking_goal = -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(
            ufl.dot(bc_in1_value, self.n_vec) * self.ds(input_marker)
        )) / 3.0

    def define_form(self):
        self.uh_fine = dolfinx.fem.Function(self.W, name='fine_state')  # state of FineModel
        nu = dolfinx.fem.Constant(self.control_s, nuValue)

        u1, p1 = ufl.split(ufl.TrialFunction(self.W))
        v1, q1 = ufl.split(ufl.TestFunction(self.W))
        f1 = dolfinx.fem.Constant(self.control_s, np.zeros(self.tdim))
        stoke_form = nu * inner(grad(u1), grad(v1)) * ufl.dx - p1 * div(v1) * ufl.dx - div(u1) * q1 * ufl.dx - \
                     inner(f1, v1) * ufl.dx
        self.stoke_lhs = dolfinx.fem.form(ufl.lhs(stoke_form))
        self.stoke_rhs = dolfinx.fem.form(ufl.rhs(stoke_form))

        u2, p2 = ufl.split(self.uh_fine)
        v2, q2 = ufl.split(ufl.TestFunction(self.W))
        f2 = dolfinx.fem.Constant(self.control_s, np.zeros(self.tdim))
        self.nstoke_lhs = nu * inner(grad(u2), grad(v2)) * ufl.dx + \
                          inner(grad(u2) * u2, v2) * ufl.dx - \
                          inner(p2, div(v2)) * ufl.dx + \
                          inner(div(u2), q2) * ufl.dx - \
                          inner(f2, v2) * ufl.dx

        self.outflow_u = {}
        self.outflow_u_forms = {}
        for marker in output_markers:
            self.outflow_u[f"marker_{marker}"] = ctypes.c_double(0.0)
            self.outflow_u_forms[f"marker_{marker}"] = dolfinx.fem.form(ufl.dot(u2, self.n_vec) * self.ds(marker))

    def solve_and_evaluate(self, guass_uh: dolfinx.fem.Function = None, use_stoke_guass=False, **kwargs):
        if use_stoke_guass and (guass_uh is None):
            res_dict = LinearProblemSolver.solve_by_petsc_form(
                comm=self.control_s.comm,
                uh=self.uh_fine,
                a_form=self.stoke_lhs,
                L_form=self.stoke_rhs,
                bcs=self.bcs,
                ksp_option=self.stoke_ksp_option,
                **kwargs
            )
            if kwargs.get('with_debug', False):
                print(f"[DEBUG Stoke]: max_error:{res_dict['max_error']:.8f} cost_time:{res_dict['cost_time']:.2f}")

        if guass_uh is not None:
            self.uh_fine.vector.aypx(0.0, guass_uh.vector)

        jacobi_form = ufl.derivative(
            self.nstoke_lhs, self.uh_fine, ufl.TrialFunction(self.uh_fine.function_space)
        )
        res_dict = NonLinearProblemSolver.solve_by_petsc(
            F_form=self.nstoke_lhs,
            uh=self.uh_fine,
            jacobi_form=jacobi_form,
            bcs=self.bcs,
            comm=self.control_s.comm,
            ksp_option=self.nstoke_ksp_option,
            **kwargs
        )
        if kwargs.get('with_debug', False):
            print(f"[DEBUG Navier Stoke]: max_error:{res_dict['max_error']:.8f} cost_time:{res_dict['cost_time']:.2f}")

        self.evaluate_loss()

    def evaluate_loss(self):
        self.cost_functional_value = 0.0
        for marker in output_markers:
            outflow_u = AssembleUtils.assemble_scalar(self.outflow_u_forms[f"marker_{marker}"])
            self.outflow_u[f"marker_{marker}"].value = outflow_u
            self.cost_functional_value = np.maximum(
                self.cost_functional_value,
                np.abs(self.tracking_goal - outflow_u)
            )

    def update_geometry(self, grad: np.ndarray):
        # coordinate_space = self.control_s.ufl_domain().ufl_coordinate_element()
        # deformation_space = dolfinx.fem.FunctionSpace(self.control_s, coordinate_space)
        # dJ = dolfinx.fem.Function(deformation_space)
        # dJ.x.array[:] = grad.reshape(-1)
        # VisUtils.show_vector_res_vtk(self.grid, dJ, dim=2, with_wrap=True).show()

        displacement_np = np.zeros(self.control_s.geometry.x.shape)
        displacement_np[:, :self.tdim] = grad.reshape((-1, self.tdim))

        MeshUtils.move(self.control_s, displacement_np)
        # is_intersection = MeshQuality.detect_collision(self.control_s, self.collide_counts)
        # assert is_intersection


domain_S, cell_tags_S, facet_tags_S = MeshUtils.read_XDMF(
    file=model_xdmf, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)  # control parameter of FineModel FUnctionSpace

fine_model = LowReFluidFineModel(
    control_s=domain_S,
    cell_tags=cell_tags_S,
    facet_tags=facet_tags_S,
    msh_file=msh_file,
    stoke_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    nstoke_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
)


# fine_model.solve_and_evaluate(with_debug=True)

# # ------ debug vis fine model
# print(fine_model.outflow_u)
# grid = VisUtils.convert_to_grid(fine_model.control_s)
# u_res = fine_model.uh_fine.sub(0).collapse()
# VisUtils.show_arrow_res_vtk(grid, u_res, fine_model.V, scale=0.1).show()
# -------


# --------------------------------------------------------------
class FluidCoarseModel(CoarseModel):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            fine_model,
    ):
        opt_problem, self.collide_counts, para_dict = self.define_problem(
            domain_Z, cell_tags_Z, facet_tags_Z, fine_model
        )
        super().__init__(
            opt_problem=opt_problem,
            control=domain,
            extract_parameter_func=extract_parameter_func
        )

        self.up = para_dict['up']
        self.u, self.p = ufl.split(self.up)
        self.ds = para_dict['ds']
        self.n_vec = para_dict['n_vec']
        self.V, self.Q = para_dict['V'], para_dict['Q']
        self.target_goal_dict = para_dict['target_goal_dict']
        self.loss_scale = np.mean([self.target_goal_dict[key] for key in self.target_goal_dict]) * 0.8
        self.grid = VisUtils.convert_to_grid(self.control)

    def _optimize(self, **kwargs):
        tdim = self.control.topology.dim

        best_loss = np.inf
        step = 0
        while True:
            step += 1

            shape_grad: dolfinx.fem.Function = self.opt_problem.compute_gradient(self.control.comm)

            shape_grad_np = shape_grad.x.array
            # shape_grad_np = shape_grad_np / np.linalg.norm(shape_grad_np, ord=2)
            shape_grad_np = shape_grad_np * -0.2

            displacement_np = np.zeros(self.control.geometry.x.shape)
            displacement_np[:, :tdim] = shape_grad_np.reshape((-1, tdim))

            # dJ = dolfinx.fem.Function(shape_grad.function_space)
            # dJ.x.array[:] = shape_grad_np.reshape(-1)
            # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True).show()

            MeshUtils.move(self.control, displacement_np)
            # is_intersection = MeshQuality.detect_collision(self.control, self.collide_counts)
            # assert is_intersection

            loss = np.sqrt(self.opt_problem.evaluate_cost_functional(self.control.comm, update_state=False))
            loss = loss / self.loss_scale

            # # ------ debug output
            # debug_log = f"[###Step {step}] "
            # for marker in output_markers:
            #     out_vel_form = ufl.dot(self.u, self.n_vec) * self.ds(marker)
            #     out_flow = AssembleUtils.assemble_scalar(dolfinx.fem.form(out_vel_form))
            #     target_flow = self.target_goal_dict[f"marker_{marker}"]
            #     ratio = out_flow / target_flow
            #     debug_log += f"[out_{marker}: {ratio:.2f}| {out_flow:.3f}/{target_flow: .3f}] "
            # best_loss = np.minimum(loss, best_loss)
            # debug_log += f"loss:{loss:.8f} / best_loss:{best_loss:.8f}"
            # print(debug_log)
            # # ------

            if loss < 0.01:
                break

            if step > 100:
                raise ValueError("[DEBUG] ParameterExtraction Can't Converge")

        # u_res = self.up.sub(0).collapse()
        # VisUtils.show_arrow_res_vtk(self.grid, u_res, self.V, scale=0.1).show()
        print(f"[LOG] CoarseModel loss:{loss}")

    @staticmethod
    def define_problem(
            domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
            fine_model: LowReFluidFineModel
    ):
        tdim = domain.topology.dim
        fdim = tdim - 1

        collide_counts = MeshQuality.compute_collide_counts(domain)

        W = dolfinx.fem.FunctionSpace(
            domain,
            ufl.MixedElement([
                ufl.VectorElement("Lagrange", domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
            ])
        )
        W0, W1 = W.sub(0), W.sub(1)
        V, V_to_W = W0.collapse()
        Q, Q_to_W = W1.collapse()

        # ---------------------
        bcs_info = []
        for marker in bry_markers:
            bc_value = dolfinx.fem.Function(V, name=f"bry_u{marker}")
            bc_dofs = MeshUtils.extract_entity_dofs(
                (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, W0)
            bcs_info.append((bc, W0, bc_dofs, bc_value))

        bc_in_value = dolfinx.fem.Function(V, name='inflow_u')
        bc_in_value.interpolate(partial(inflow_velocity_exp, tdim=tdim))
        bc_in1_dofs = MeshUtils.extract_entity_dofs(
            (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
        )
        bc_in1 = dolfinx.fem.dirichletbc(bc_in_value, bc_in1_dofs, W0)
        bcs_info.append((bc_in1, W0, bc_in1_dofs, bc_in_value))

        for marker in output_markers:
            bc_out_value = dolfinx.fem.Function(Q, name=f"outflow_p_{marker}")
            bc_out_dofs = MeshUtils.extract_entity_dofs(
                (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
            )
            bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs, W1)
            bcs_info.append((bc_out, W1, bc_out_dofs, bc_out_value))

        # ---------------------
        up = dolfinx.fem.Function(W, name='coarse_state')
        u, p = ufl.split(up)
        vq = dolfinx.fem.Function(W, name='coarse_adjoint')
        v, q = ufl.split(vq)
        f = dolfinx.fem.Constant(domain, np.zeros(tdim))
        nu = dolfinx.fem.Constant(domain, nuValue)

        F_form = nu * inner(grad(u), grad(v)) * ufl.dx - p * div(v) * ufl.dx - div(u) * q * ufl.dx - \
                 inner(f, v) * ufl.dx

        state_problem = create_state_problem(
            name='coarse_1', F_form=F_form, state=up, adjoint=vq, is_linear=True,
            bcs_info=bcs_info,
            state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        )
        state_system = StateProblem([state_problem])

        # --------------------------
        coordinate_space = domain.ufl_domain().ufl_coordinate_element()
        V_S = dolfinx.fem.FunctionSpace(domain, coordinate_space)

        bcs_info = []
        for marker in bry_fixed_markers:
            bc_value = dolfinx.fem.Function(V_S, name=f"fix_bry_shape_{marker}")
            bc_dofs = MeshUtils.extract_entity_dofs(
                V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)
            bcs_info.append((bc, V_S, bc_dofs, bc_value))

        control_problem = create_shape_problem(
            domain=domain, bcs_info=bcs_info,
            gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        )

        ds = MeshUtils.define_ds(domain, facet_tags)
        n_vec = MeshUtils.define_facet_norm(domain)

        target_goal_dict = {}
        for marker in output_markers:
            target_goal_dict[f"marker_{marker}"] = fine_model.tracking_goal

        # target_goal_dict = {
        #     f"marker_{5}": 0.47,
        #     f"marker_{6}": 0.74,
        #     f"marker_{7}": 0.79,
        # }

        cost_functional_list = []
        for marker in output_markers:
            integrand_form = ufl.dot(u, n_vec) * ds(marker)
            cost_functional_list.append(ScalarTrackingFunctional(
                domain, integrand_form, target_goal_dict[f"marker_{marker}"])
            )

        opt_problem = OptimalShapeProblem(
            state_system=state_system,
            shape_problem=control_problem,
            shape_regulariztions=ShapeRegularization([]),
            cost_functional_list=cost_functional_list,
            scalar_product=None,
            # scalar_product_method='surface_metrics'
        )

        return opt_problem, collide_counts, {
            'up': up, 'ds': ds, 'n_vec': n_vec, 'V': V, 'Q': Q, 'target_goal_dict': target_goal_dict
        }


domain_Z, cell_tags_Z, facet_tags_Z = MeshUtils.read_XDMF(
    file=model_xdmf, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)  # control parameter of CoarseModel FUnctionSpace

coarse_model = FluidCoarseModel(
    domain=domain_Z,
    facet_tags=facet_tags_Z,
    cell_tags=cell_tags_Z,
    fine_model=fine_model,
)
# coarse_model.solve()


# --------------------------------------------------------------
class FluidParameterExtraction(ParameterExtraction):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            fine_model,
    ):
        opt_problem, self.collide_counts, para_dict = self.define_problem(domain, cell_tags, facet_tags, fine_model)
        super().__init__(opt_problem=opt_problem, control_z=domain, extract_parameter_func=extract_parameter_func)

        self.tdim = para_dict['tdim']
        self.up = para_dict['up']
        self.u, self.p = ufl.split(self.up)
        self.ds = para_dict['ds']
        self.n_vec = para_dict['n_vec']
        self.V, self.Q = para_dict['V'], para_dict['Q']
        self.track_goals_dict = para_dict['track_goal_dict']
        self.track_goals = [self.track_goals_dict[key] for key in self.track_goals_dict]
        self.track_goal_scale = 1.0
        self.grid = VisUtils.convert_to_grid(self.control_z)

    @staticmethod
    def define_problem(
            domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
            fine_model: LowReFluidFineModel
    ):
        tdim = domain.topology.dim
        fdim = tdim - 1

        collide_counts = MeshQuality.compute_collide_counts(domain)

        W = dolfinx.fem.FunctionSpace(
            domain,
            ufl.MixedElement([
                ufl.VectorElement("Lagrange", domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
            ])
        )
        W0, W1 = W.sub(0), W.sub(1)
        V, V_to_W = W0.collapse()
        Q, Q_to_W = W1.collapse()

        # ---------------------
        bcs_info = []
        for marker in bry_markers:
            bc_value = dolfinx.fem.Function(V, name=f"bry_u{marker}")
            bc_dofs = MeshUtils.extract_entity_dofs(
                (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, W0)
            bcs_info.append((bc, W0, bc_dofs, bc_value))

        bc_in_value = dolfinx.fem.Function(V, name='inflow_u')
        bc_in_value.interpolate(partial(inflow_velocity_exp, tdim=tdim))
        bc_in1_dofs = MeshUtils.extract_entity_dofs(
            (W0, V), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
        )
        bc_in1 = dolfinx.fem.dirichletbc(bc_in_value, bc_in1_dofs, W0)
        bcs_info.append((bc_in1, W0, bc_in1_dofs, bc_in_value))

        for marker in output_markers:
            bc_out_value = dolfinx.fem.Function(Q, name=f"outflow_p_{marker}")
            bc_out_dofs = MeshUtils.extract_entity_dofs(
                (W1, Q), fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
            )
            bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs, W1)
            bcs_info.append((bc_out, W1, bc_out_dofs, bc_out_value))

        # ---------------------
        up = dolfinx.fem.Function(W, name='coarse_state')
        u, p = ufl.split(up)
        vq = dolfinx.fem.Function(W, name='coarse_adjoint')
        v, q = ufl.split(vq)
        f = dolfinx.fem.Constant(domain, np.zeros(tdim))
        nu = dolfinx.fem.Constant(domain, nuValue)

        F_form = nu * inner(grad(u), grad(v)) * ufl.dx - p * div(v) * ufl.dx - div(u) * q * ufl.dx - \
                 inner(f, v) * ufl.dx

        state_problem = create_state_problem(
            name='coarse_1', F_form=F_form, state=up, adjoint=vq, is_linear=True,
            bcs_info=bcs_info,
            state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        )
        state_system = StateProblem([state_problem])

        # --------------------------
        coordinate_space = domain.ufl_domain().ufl_coordinate_element()
        V_S = dolfinx.fem.FunctionSpace(domain, coordinate_space)

        bcs_info = []
        for marker in bry_fixed_markers:
            bc_value = dolfinx.fem.Function(V_S, name=f"fix_bry_shape_{marker}")
            bc_dofs = MeshUtils.extract_entity_dofs(
                V_S, fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)
            bcs_info.append((bc, V_S, bc_dofs, bc_value))

        control_problem = create_shape_problem(
            domain=domain, bcs_info=bcs_info,
            gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        )

        ds = MeshUtils.define_ds(domain, facet_tags)
        n_vec = MeshUtils.define_facet_norm(domain)

        cost_functional_list = []
        for marker in output_markers:
            integrand_form = ufl.dot(u, n_vec) * ds(marker)
            cost_functional_list.append(ScalarTrackingFunctional(
                domain, integrand_form, fine_model.outflow_u[f"marker_{marker}"]
            ))

        opt_problem = OptimalShapeProblem(
            state_system=state_system,
            shape_problem=control_problem,
            shape_regulariztions=ShapeRegularization([]),
            cost_functional_list=cost_functional_list,
            scalar_product=None,
        )

        return opt_problem, collide_counts, {
            'up': up, 'ds': ds, 'n_vec': n_vec, 'V': V, 'Q': Q, 'tdim': tdim,
            'track_goal_dict': fine_model.outflow_u
        }

    def compute_loss(self, update_scale=False, update_state=False):
        if update_scale:
            self.track_goal_scale = np.mean([g.value for g in self.track_goals])
        loss = np.sqrt(self.opt_problem.evaluate_cost_functional(self.control_z.comm, update_state=update_state))
        loss = loss / self.track_goal_scale
        return loss

    def _optimize(self, **kwargs):
        step = 0
        orig_loss = self.compute_loss(update_scale=True, update_state=True)  # necessary
        best_loss = orig_loss
        while True:
            step += 1

            shape_grad: dolfinx.fem.Function = self.opt_problem.compute_gradient(self.control_z.comm)

            shape_grad_np = shape_grad.x.array
            # shape_grad_np = shape_grad_np / np.linalg.norm(shape_grad_np, ord=2)
            shape_grad_np = shape_grad_np * -0.2

            displacement_np = np.zeros(self.control_z.geometry.x.shape)
            displacement_np[:, :self.tdim] = shape_grad_np.reshape((-1, self.tdim))

            # dJ = dolfinx.fem.Function(shape_grad.function_space)
            # dJ.x.array[:] = shape_grad_np.reshape(-1)
            # VisUtils.show_vector_res_vtk(grid, dJ, dim=2, with_wrap=True)

            MeshUtils.move(self.control_z, displacement_np)
            # is_intersection = MeshQuality.detect_collision(self.control_z, self.collide_counts)
            # assert is_intersection

            loss = self.compute_loss(update_scale=False, update_state=False)  # necessary
            best_loss = np.minimum(loss, best_loss)

            # # ------ debug output
            # debug_log = f"[###Step {step}] "
            # for marker in output_markers:
            #     out_vel_form = ufl.dot(self.u, self.n_vec) * self.ds(marker)
            #     out_flow = AssembleUtils.assemble_scalar(dolfinx.fem.form(out_vel_form))
            #     target_flow = self.track_goals_dict[f"marker_{marker}"].value
            #     ratio = out_flow / target_flow
            #     debug_log += f"[out_{marker}: {ratio:.2f}| {out_flow:.3f}/{target_flow: .3f}] "
            # debug_log += f"loss:{loss:.8f} / best_loss:{best_loss:.8f}"
            # print(debug_log)
            # # ------

            if loss < 0.01:
                break

            if step > 100:
                raise ValueError("[DEBUG] ParameterExtraction Can't Converge")

        # u_res = self.up.sub(0).collapse()
        # VisUtils.show_arrow_res_vtk(self.grid, u_res, self.V, scale=0.1).show()
        # print(f"[LOG] ParameterExtraction loss_decrease:{orig_loss - best_loss}")


domain_extpar, cell_tags_extpar, facet_tags_extpar = MeshUtils.read_XDMF(
    file=model_xdmf, mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

pararmeter_extraction = FluidParameterExtraction(
    domain=domain_extpar,
    cell_tags=cell_tags_extpar,
    facet_tags=facet_tags_extpar,
    fine_model=fine_model,
)
# fine_model.solve_and_evaluate(with_debug=True)
# pararmeter_extraction.solve()


# ------ define Space Mapping
opt_dir = os.path.join(proj_dir, 'opt')
if os.path.exists(opt_dir):
    shutil.rmtree(opt_dir)
os.mkdir(opt_dir)
fine_recorder = VTKRecorder(file=os.path.join(opt_dir, 'u_res.pvd'))
fine_recorder.write_mesh(fine_model.control_s, step=0)

log_dir = os.path.join(proj_dir, 'log')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)
tensor_recorder = TensorBoardRecorder(log_dir=log_dir)


class FluidSpaceMappingProblem(SpaceMappingProblem):
    def _update(self, grad: np.ndarray, fine_model: LowReFluidFineModel):
        fine_model.update_geometry(grad)

    def _compute_direction_step(self, z_star, z_cur, **kwargs) -> np.ndarray:
        grad = self.invert_scale * (z_cur - z_star) * 0.5
        return grad

    def _compute_eps(self, z_cur, z_star) -> float:
        return np.max(np.abs(z_cur - z_star))

    def pre_log(self, step: int):
        # plotter = pyvista.Plotter()
        # plotter.add_mesh(self.coarse_model.grid, color='r', style='wireframe')
        # plotter.add_mesh(self.parameter_extraction.grid, color='b', style='wireframe')
        # plotter.show()

        tensor_recorder.write_scalar('loss', self.fine_model.cost_functional_value, step)
        flow_dict = {}
        for key in self.fine_model.outflow_u.keys():
            flow_dict[key] = self.fine_model.outflow_u[key].value
        tensor_recorder.write_scalars('flow', flow_dict, step)

    def post_log(self, step: int, eps: float):
        print(f"[LOG] step:{step} eps:{eps} SpaceMapping loss: {self.fine_model.cost_functional_value}")

        u_res = self.fine_model.uh_fine.sub(0).collapse()
        fine_recorder.write_function(u_res, step)


problem = FluidSpaceMappingProblem(
    coarse_model=coarse_model,
    fine_model=fine_model,
    parameter_extraction=pararmeter_extraction,
    tol=1e-4,
    max_iter=10,
    is_coarse_fine_collinear=False
)
problem.solve(
    coarseModel_kwargs={},
    fineModel_kwargs={'with_debug': True},
    paraExtract_kwargs={}
)

fine_model.solve_and_evaluate(with_debug=True)
print(fine_model.outflow_u)

new_msh_file = os.path.join(proj_dir, 'opt_model.msh')
ReMesher.convert_domain_to_new_msh(
    orig_msh_file=msh_file,
    new_msh_file=new_msh_file,
    domain=fine_model.control_s,
    dim=fine_model.tdim,
    vertex_indices=fine_model.vertex_indices
)
