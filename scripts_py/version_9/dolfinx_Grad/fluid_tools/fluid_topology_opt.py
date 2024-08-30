import shutil
import numpy as np
import ufl
import dolfinx
from typing import Dict, Union, List, Callable
import os
from ufl import div, inner, grad
from mpi4py import MPI
import scipy

from scripts_py.version_9.dolfinx_Grad.lagrange_method.type_database import create_control_problem, create_state_problem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.solver_optimize import OptimalControlProblem
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction, ScalarTrackingFunctional
from scripts_py.version_9.dolfinx_Grad.lagrange_method.control_regularization import ControlRegularization, \
    ControlIntegralRegularization
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_finder.topology_optimize_utils import DistanceBandFilter, DensityBandFilter


class StopLearner(object):
    def __init__(self, stat_num=100):
        self.losses = []
        self.stat_num = stat_num

    def is_stop(self, loss: float) -> bool:
        self.losses.append(loss)

        num = len(self.losses)
        if num < (self.stat_num * 2):
            return False

        if num % self.stat_num == 0:
            loss_seq0 = np.array(self.losses[-self.stat_num:])
            loss_seq1 = np.array(self.losses[-self.stat_num * 2: -self.stat_num])
            p_value = scipy.stats.ttest_ind(loss_seq0, loss_seq1, alternative='less').pvalue
            # print(f"[Debug] pvalue: {p_value}")
            if p_value > 0.5:
                return True
        return False


class FluidTopoLogyModel(object):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh, cell_tags: dolfinx.mesh.MeshTags, facet_tags: dolfinx.mesh.MeshTags,
            grad_filter: Union[DensityBandFilter, DistanceBandFilter],
            alpha_function_info: Dict[str, float], learn_info: Dict[str, float],
            res_dir: str
    ):
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags

        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

        P2 = ufl.VectorElement("Lagrange", self.domain.ufl_cell(), 2)
        P1 = ufl.FiniteElement("Lagrange", self.domain.ufl_cell(), 1)
        TH = ufl.MixedElement([P2, P1])
        self.W = dolfinx.fem.FunctionSpace(domain, TH)
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()
        self.V_control = dolfinx.fem.FunctionSpace(domain, ("DG", 0))
        self.n_vec = MeshUtils.define_facet_norm(domain)
        self.ds = MeshUtils.define_ds(domain, facet_tags)
        self.dx = MeshUtils.define_dx(domain)
        self.volume = AssembleUtils.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(self.domain, 1.0) * ufl.dx))

        self.alpha_function_info = alpha_function_info
        self.learn_info = learn_info
        self.grad_filter = grad_filter

        self.res_dir = res_dir
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
        self.rho_dir = os.path.join(self.res_dir, 'rho')
        if os.path.exists(self.rho_dir):
            shutil.rmtree(self.rho_dir)
        os.mkdir(self.rho_dir)

        self.recorder_dict = {}
        self.state_problems = []
        self.regularization_list = []
        self.controls = []
        self.control_bcs_info = {}
        self.gradient_ksp_options = {}
        self.cost_functional_list = []
        self.state_system: StateProblem = None
        self.control_problem = None
        self.opt_problem: OptimalControlProblem = None

    def generate_control(self, name: str, init_value=0.9):
        rho = dolfinx.fem.Function(self.V_control, name=f"control_{name}")
        rho.x.array[:] = init_value

        rho_recorder = VTKRecorder(file=os.path.join(self.rho_dir, f"control_{name}.pvd"))
        rho_recorder.write_function(rho, step=0)
        self.recorder_dict.update({rho.name: rho_recorder})
        self.controls.append(rho)
        self.control_bcs_info.update({rho.name: []})
        self.gradient_ksp_options.update({
            rho.name: {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        })
        return rho

    def add_condition(
            self,
            name: str,
            input_dict: Dict[int, Callable],
            output_dict: Dict[int, Callable],
            boundary_markers: List[int],
            rho: dolfinx.fem.Function, other_rhos: List[dolfinx.fem.Function],
            cost_dict: Dict,
            regularization_volume_ratio: float = -1.0, regularization_volume_weight: float = 1000.0,
    ):
        up = dolfinx.fem.Function(self.W, name=f"state_{name}")
        u, p = ufl.split(up)  # please don't use up.split()
        vq = dolfinx.fem.Function(self.W, name=f"adjoint_{name}")
        v, q = ufl.split(vq)  # please don't use vq.split()
        f = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))

        eq_form = self.generate_stoke_equation_form(u, p, v, q, f)

        rho_aggregate = 1.0 - rho
        for other_rho in other_rhos:
            assert other_rho != rho
            rho_aggregate += other_rho
        alpha = self.alpha_fun(
            rho_aggregate,
            alpha_min=self.alpha_function_info.get('alpha_min', 0.01),
            alpha_max=self.alpha_function_info.get('alpha_max', 1000.0),
            p=self.alpha_function_info.get('p', 3.0)
        )
        eq_form += alpha * inner(u, v) * ufl.dx

        bcs_info_state = []
        for marker in boundary_markers:
            bc_value = dolfinx.fem.Function(self.V, name=f"boundary_{marker}")
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, self.W0)
            bcs_info_state.append((bc, self.W0, bc_dofs, bc_value))

        inlet_flux = 0.0
        for inlet_marker, inlet_function in input_dict.items():
            inlet_value = dolfinx.fem.Function(self.V, name=f"inlet_{inlet_marker}")
            inlet_value.interpolate(inlet_function)
            inlet_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(
                    self.domain, self.facet_tags, inlet_marker
                )
            )
            inlet_bc = dolfinx.fem.dirichletbc(inlet_value, inlet_dofs, self.W0)
            bcs_info_state.append((inlet_bc, self.W0, inlet_dofs, inlet_value))
            inlet_flux += self.compute_flux(inlet_value, inlet_marker, self.n_vec, self.ds)

        outlet_num = 0.0
        for outlet_marker, _ in output_dict.items():
            outlet_value = dolfinx.fem.Function(self.Q, name=f"outlet_{outlet_marker}")
            outlet_dofs = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(
                    self.domain, self.facet_tags, outlet_marker
                )
            )
            outlet_bc = dolfinx.fem.dirichletbc(outlet_value, outlet_dofs, self.W1)
            bcs_info_state.append((outlet_bc, self.W1, outlet_dofs, outlet_value))
            outlet_num += 1.0

        state_problem = create_state_problem(
            name=f"problem_{name}",
            F_form=eq_form, state=up, adjoint=vq,
            is_linear=True,  # todo
            bcs_info=bcs_info_state,
            state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        )
        self.state_problems.append(state_problem)

        if regularization_volume_ratio >= 0.0:
            reg_volume = ControlIntegralRegularization(
                self.domain, control=rho,
                integral_form=rho,
                mu=regularization_volume_weight,
                target_control_volume=self.volume * regularization_volume_ratio, name=f"regVolume_{name}",
                # method='percentage_div'
            )
            self.regularization_list.append(reg_volume)

        for cost_name, cost_info in cost_dict.items():
            if cost_info['type'] == 'minimum_dissipation':
                cost_form = cost_info['weight'] * self.generate_minium_energy_form(alpha=alpha, u=u, p=p)
                cost_fun = IntegralFunction(domain=self.domain, form=cost_form, name=f"integral_{name}")
                self.cost_functional_list.append(cost_fun)

            elif cost_info['type'] == 'average_flux':
                average_flux = inlet_flux / outlet_num
                for outlet_marker, _ in output_dict.items():
                    outlet_flux_form = cost_info['weight'] * ufl.dot(u, self.n_vec) * self.ds(outlet_marker)
                    cost_fun = ScalarTrackingFunctional(
                        self.domain, outlet_flux_form, average_flux, name=f"track_scalar_{outlet_marker}"
                    )
                    self.cost_functional_list.append(cost_fun)
            else:
                raise ValueError('[ERROR]: Non valid cost type.')

        sub_res_dir = os.path.join(self.res_dir, f"res_problem_{name}")
        if os.path.exists(sub_res_dir):
            shutil.rmtree(sub_res_dir)
        os.mkdir(sub_res_dir)
        u_recorder = VTKRecorder(file=os.path.join(sub_res_dir, 'u.pvd'))
        self.recorder_dict.update({f"{up.name}_u": u_recorder})

    def generate_problem(self):
        self.state_system = StateProblem(self.state_problems)
        self.control_problem = create_control_problem(
            controls=self.controls,
            bcs_info=self.control_bcs_info,
            gradient_ksp_options=self.gradient_ksp_options
        )

        self.opt_problem = OptimalControlProblem(
            state_system=self.state_system,
            control_problem=self.control_problem,
            cost_functional_list=self.cost_functional_list,
            control_regularization=ControlRegularization(self.regularization_list)
        )

    def update_grad(
            self, control: dolfinx.fem.Function, grad_dict: Dict[str, dolfinx.fem.Function]
    ):
        grad_np: np.array = grad_dict[control.name].x.array
        grad_np = grad_np.reshape((-1, 1))
        control_np: np.array = control.x.array
        control_np = control_np.reshape(-1)

        if isinstance(self.grad_filter, DistanceBandFilter):
            grad_np = self.grad_filter.run(grad_np)
        elif isinstance(self.grad_filter, DensityBandFilter):
            grad_np = self.grad_filter.run(grad_np, control_np)
        else:
            raise ValueError("[ERROR]: Non-valid filter")

        grad_np = np.asarray(grad_np).reshape(-1)

        norm = np.linalg.norm(grad_np, ord=2)
        max_step = self.learn_info.get('max_step', 0.5)
        min_step = self.learn_info.get('min_step', 0.1)
        if norm > max_step:
            grad_np = grad_np / norm * max_step
        elif norm < min_step:
            grad_np = grad_np / norm * min_step

        control_np = control_np - grad_np
        control_np = np.maximum(np.minimum(control_np, 0.99), 0.01)

        control.x.array[:] = control_np
        return control

    def run_optimize(self):
        stop_learner = StopLearner(stat_num=self.learn_info.get('stat_num', 100))
        max_iteration = self.learn_info.get('max_iteration', 500)
        log_step = self.learn_info.get('log_step', 10)
        record_step = self.learn_info.get('record_step', 5)

        # ------ for debug
        last_loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=True, with_debug=True)
        print(f"[Debug]: Init loss:{last_loss}")
        cost_info = self.opt_problem.get_cost_info(comm=MPI.COMM_WORLD, update_state=False)
        print(cost_info)
        # ------

        step = 0
        while True:
            step += 1

            grads_dict: Dict[str, dolfinx.fem.Function] = self.opt_problem.compute_gradient(self.domain.comm)
            for control in self.controls:
                self.update_grad(control, grads_dict)

            loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=False)
            print(f"[Debug]: Step_{step} loss:{loss:.6f}")

            if step % record_step == 0:
                for control in self.controls:
                    self.recorder_dict[control.name].write_function(control, step=step)
                for problem in self.state_system.state_problems:
                    self.recorder_dict[f"{problem.state.name}_u"].write_function(problem.state.sub(0).collapse(),
                                                                                 step=step)

            if step % log_step == 0:
                cost_info = self.opt_problem.get_cost_info(comm=MPI.COMM_WORLD, update_state=False)
                print(cost_info)

            if stop_learner.is_stop(loss) or (step > max_iteration):
                break

    @staticmethod
    def generate_stoke_equation_form(
            u: dolfinx.fem.Function, p: dolfinx.fem.Function,
            v: dolfinx.fem.Function, q: dolfinx.fem.Function,
            f: Union[dolfinx.fem.Function, dolfinx.fem.Constant]
    ):
        eq_form = (inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * ufl.dx - inner(f, v) * ufl.dx
        return eq_form

    @staticmethod
    def alpha_fun(
            rho: Union[np.ndarray, dolfinx.fem.Function], alpha_min=0.01, alpha_max=10000.0, p=2.0,
            pow_operator=ufl.elem_pow, is_inverse=False
    ):
        # alpha_max与p对边界锐利程度有很大影响
        if is_inverse:
            return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)
        else:
            return alpha_min + pow_operator(rho, p) * (alpha_max - alpha_min)

    @staticmethod
    def generate_minium_energy_form(alpha: ufl.Form, u: dolfinx.fem.Function, p: dolfinx.fem.Function):
        # todo some problem here
        return 0.5 * alpha * inner(u, u) * ufl.dx + inner(grad(u), grad(u)) * ufl.dx

    @staticmethod
    def compute_flux(inlet_function: dolfinx.fem.Function, marker: int, n_vec: ufl.FacetNormal, ds: ufl.Measure):
        flux = -1.0 * AssembleUtils.assemble_scalar(
            dolfinx.fem.form(ufl.dot(inlet_function, n_vec) * ds(marker))
        )
        return flux
