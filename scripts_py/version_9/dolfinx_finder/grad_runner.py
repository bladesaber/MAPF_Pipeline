import os
import shutil
import numpy as np
import dolfinx
import ufl
from typing import Dict, Union, List
from ufl import grad, inner, div, dot
from functools import partial
import pyvista
import time

from .topology_optimize_utils import DistanceBandFilter
from ..dolfinx_Grad.vis_mesh_utils import VisUtils
from ..dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from .simulator_wrapper import FoamSimulator
from ..dolfinx_Grad.lagrange_method.problem_state import StateProblem
from ..dolfinx_Grad.lagrange_method.solver_optimize import OptimalControlProblem
from ..dolfinx_Grad.lagrange_method.type_database import ControlDataBase
from ..dolfinx_Grad.recorder_utils import PVDRecorder
from ..dolfinx_Grad.lagrange_method.type_database import create_control_problem, create_state_problem
from ..dolfinx_Grad.lagrange_method.control_regularization import ControlRegularization, ControlIntegralRegularization
from ..dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction, ScalarTrackingFunctional


def inlet_velocity_expression(x, tdim: int, velocity_vector: np.ndarray):
    assert velocity_vector.shape[0] == tdim
    num = x.shape[1]
    values = np.tile(velocity_vector[np.newaxis, :], (num, 1)).T
    return values


class GradRunner(object):
    def __init__(
            self,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            grad_cfg: Dict,  # equ_info, grad_filter_radius, max_step_size, min_step_size
    ):
        self.grad_cfg = grad_cfg

        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

        self.grid = VisUtils.convert_to_grid(self.domain)
        self.grad_filter = DistanceBandFilter(
            self.grid, radius_filter=grad_cfg['grad_filter_radius'], is_cell_field=True
        )

        self.W = dolfinx.fem.FunctionSpace(
            domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", domain.ufl_cell(), grad_cfg['equ_info'].get('velocity_order', 2)),
                ufl.FiniteElement("Lagrange", domain.ufl_cell(), grad_cfg['equ_info'].get('pressure_order', 1))
            ])
        )
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()
        self.vector_cell_functionSpace = dolfinx.fem.VectorFunctionSpace(domain, ("DG", 0))
        self.scalar_cell_functionSpace = dolfinx.fem.FunctionSpace(domain, ("DG", 0))
        self.vector_point_functionSpace = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
        self.scalar_point_functionSpace = dolfinx.fem.FunctionSpace(domain, ("CG", 1))

        self.control_functionSpace = self.scalar_cell_functionSpace
        self.nut_functionSpace = self.scalar_cell_functionSpace

        self.n_vec = MeshUtils.define_facet_norm(domain)
        self.ds = MeshUtils.define_ds(domain, facet_tags)
        self.dx = MeshUtils.define_dx(domain)
        self.volume = AssembleUtils.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(domain, 1.0) * ufl.dx))

        # ------
        self.recorder_dict = {}
        self.state_problems = []
        self.regularization_list = []
        self.cost_functional_list = []

        self.state_system: StateProblem = None
        self.control_problem: ControlDataBase = None
        self.opt_problem: OptimalControlProblem = None

        # ------
        self.condition_dicts = {}
        self.controls_dicts: Dict[str, dolfinx.fem.Function] = {}

    def generate_control(self, name: str, result_dir: str, init_value=0.1):
        rho = dolfinx.fem.Function(self.control_functionSpace, name=name)
        rho.x.array[:] = init_value

        if name in self.controls_dicts.keys():
            return None
        self.controls_dicts[rho.name] = rho

        rho_dir = os.path.join(result_dir, name)
        if os.path.exists(rho_dir):
            shutil.rmtree(rho_dir)
        os.mkdir(rho_dir)
        rho_recorder = PVDRecorder(file=os.path.join(rho_dir, f"{name}.pvd"))
        rho_recorder.write_function(rho, step=0)
        self.recorder_dict.update({rho.name: rho_recorder})

        return rho

    @staticmethod
    def generate_equation_form(
            u: dolfinx.fem.Function, p: dolfinx.fem.Function,
            v: dolfinx.fem.Function, q: dolfinx.fem.Function,
            nut: dolfinx.fem.Function,
            f: Union[dolfinx.fem.Function, dolfinx.fem.Constant],
            is_stoke
    ):
        if is_stoke:
            # 虽然梯度分布合理，但原目标有差异
            eq_form = (
                    inner(grad(u), grad(v)) * ufl.dx
                    - div(v) * p * ufl.dx
                    - q * div(u) * ufl.dx
                    - inner(f, v) * ufl.dx
            )
        else:
            # todo 收敛梯度衰减很快 Fail
            eq_form = (
                    nut * inner(grad(u), grad(v)) * ufl.dx
                    + inner(grad(u) * u, v) * ufl.dx
                    - div(v) * p * ufl.dx
                    + q * div(u) * ufl.dx
                    - inner(f, v) * ufl.dx
            )
        return eq_form

    @staticmethod
    def alpha_fun(
            rho: Union[np.ndarray, dolfinx.fem.Function], alpha_min=0.01, alpha_max=1000.0, p=3.0,
            pow_operator=ufl.elem_pow, is_inverse=False
    ):
        # alpha_max与p对边界锐利程度有很大影响
        if is_inverse:
            return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)
        else:
            return alpha_min + pow_operator(rho, p) * (alpha_max - alpha_min)

    def compute_alpha(
            self,
            rho: Union[dolfinx.fem.Function, np.ndarray],
            other_rhos: List[Union[dolfinx.fem.Function, np.ndarray]],
            pow_operator=ufl.elem_pow
    ):
        rho_aggregate = 1.0 - rho
        for other_rho in other_rhos:
            rho_aggregate += other_rho
        alpha = self.alpha_fun(
            rho_aggregate,
            alpha_min=self.grad_cfg['alpha_info'].get('alpha_min', 0.01),
            alpha_max=self.grad_cfg['alpha_info'].get('alpha_max', 1000.0),
            p=self.grad_cfg['alpha_info'].get('p', 3.0),
            pow_operator=pow_operator,
            is_inverse=False
        )
        return alpha

    def add_condition(
            self,
            cond_name: str,
            input_dict: Dict[int, np.ndarray],
            output_dict: Dict[int, float],
            boundary_markers: List[int],
            rho: dolfinx.fem.Function,
            rho_openfoam_name: str,
            other_rhos: List[dolfinx.fem.Function],
            cost_cfgs: List[Dict],
            regularization_cfgs: Dict,
            openfoam_cfg: Dict,
            result_dir: str
    ):
        up = dolfinx.fem.Function(self.W, name=f"{cond_name}_state")
        u, p = ufl.split(up)  # please don't use up.split()
        vq = dolfinx.fem.Function(self.W, name=f"{cond_name}_adjoint")
        v, q = ufl.split(vq)  # please don't use vq.split()
        f = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))
        nut = dolfinx.fem.Function(self.nut_functionSpace, name=f"{cond_name}_nut")
        nut.x.array[:] = openfoam_cfg['kinematic_viscosity']

        simulator = FoamSimulator(
            cond_name, self.domain, self.cell_tags, self.facet_tags, openfoam_cfg,
            dolfinFun2openfoamFun_nameMap={rho.name: rho_openfoam_name}
        )
        simulator.init_env()

        eq_form = self.generate_equation_form(u, p, v, q, nut, f, is_stoke=self.grad_cfg['equ_info']['is_stoke'])
        alpha = self.compute_alpha(rho, other_rhos, pow_operator=ufl.elem_pow)
        eq_form += alpha * inner(u, v) * ufl.dx

        bcs_info_state = []
        for marker in boundary_markers:
            bc_value = dolfinx.fem.Function(self.V, name=f"boundary_{marker}")
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, self.W0)
            bcs_info_state.append((bc, self.W0, bc_dofs, bc_value))

            simulator.add_boundary_condition(marker)

        inlet_flux = 0.0
        for inlet_marker, inlet_vector in input_dict.items():
            inlet_value = dolfinx.fem.Function(self.V, name=f"inlet_{inlet_marker}")
            inlet_value.interpolate(partial(inlet_velocity_expression, tdim=self.tdim, velocity_vector=inlet_vector))
            inlet_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(
                    self.domain, self.facet_tags, inlet_marker
                )
            )
            inlet_bc = dolfinx.fem.dirichletbc(inlet_value, inlet_dofs, self.W0)
            bcs_info_state.append((inlet_bc, self.W0, inlet_dofs, inlet_value))

            simulator.add_inlet_condition(inlet_marker, inlet_vector)
            inlet_flux += self.compute_flux(inlet_value, inlet_marker, self.n_vec, self.ds)

        for outlet_marker, outlet_pressure in output_dict.items():
            outlet_value = dolfinx.fem.Function(self.Q, name=f"outlet_{outlet_marker}")
            outlet_value.x.array[:] = outlet_pressure
            outlet_dofs = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(
                    self.domain, self.facet_tags, outlet_marker
                )
            )
            outlet_bc = dolfinx.fem.dirichletbc(outlet_value, outlet_dofs, self.W1)
            bcs_info_state.append((outlet_bc, self.W1, outlet_dofs, outlet_value))

            simulator.add_outlet_condition(outlet_marker, outlet_pressure)

        state_problem = create_state_problem(
            name=f"{cond_name}_StateProblem",
            F_form=eq_form, state=up, adjoint=vq,
            is_linear=self.grad_cfg['equ_info']['is_stoke'],
            bcs_info=bcs_info_state,
            state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
        )
        self.state_problems.append(state_problem)

        for reg_name, reg_cfg in regularization_cfgs.items():
            if reg_name == 'volume_regularization':
                reg_volume = ControlIntegralRegularization(
                    self.domain,
                    control=rho,
                    integral_form=rho,
                    mu=reg_cfg['weight'],
                    target_control_volume=self.volume * reg_cfg['ratio'],
                    name=f"{cond_name}_regVolume",
                    method='percentage_div'
                )
                self.regularization_list.append(reg_volume)

        for cost_cfg in cost_cfgs:
            if cost_cfg['type'] == 'minimum_dissipation':
                cost_form = self.generate_minium_energy_form(alpha=alpha, u=u)
                cost_fun = IntegralFunction(
                    domain=self.domain, form=cost_form, name=f"{cond_name}_minDissipation",
                    weight=cost_cfg['weight']
                )
                self.cost_functional_list.append(cost_fun)

            elif cost_cfg['type'] == 'average_flux':
                average_flux = inlet_flux / float(len(output_dict))
                for outlet_marker, _ in output_dict.items():
                    outlet_flux_form = ufl.dot(u, self.n_vec) * self.ds(outlet_marker)
                    cost_fun = ScalarTrackingFunctional(
                        self.domain, outlet_flux_form, average_flux,
                        name=f"{cond_name}_{outlet_marker}_avgFlux",
                        weight=cost_cfg['weight'],
                        method=cost_cfg['method']
                    )
                    self.cost_functional_list.append(cost_fun)

            elif cost_cfg['type'] == 'minimum_pressure_loss':
                cost_form = self.generate_minium_pressure_loss_form(
                    domain=self.domain,
                    alpha=alpha,
                    p=p,
                    marker=cost_cfg['marker'],
                    ds=self.ds
                )
                cost_fun = IntegralFunction(
                    domain=self.domain, form=cost_form, name=f"{cond_name}_minPressure",
                    weight=cost_cfg['weight']
                )
                self.cost_functional_list.append(cost_fun)

            elif cost_cfg['type'] == 'maximum_outlet_flux':
                cost_form = self.generate_maximum_outlet_flux_form(
                    domain=self.domain,
                    alpha=alpha,
                    marker=cost_cfg['marker'],
                    u=u,
                    ds=self.ds,
                    n_vec=self.n_vec
                )
                cost_fun = IntegralFunction(
                    domain=self.domain, form=cost_form, name=f"{cond_name}_maxFlux",
                    weight=cost_cfg['weight']
                )
                self.cost_functional_list.append(cost_fun)

            else:
                raise ValueError('[ERROR]: Non valid cost type.')

        self.condition_dicts.update({
            cond_name: {
                'simulator': simulator,
                'control': rho,
                'other_controls': other_rhos,
                'state_vars': {'up': up, 'nut': nut},
                'kinematic_viscosity': openfoam_cfg['kinematic_viscosity']
            }
        })

        cond_dir = os.path.join(result_dir, cond_name)
        if os.path.exists(cond_dir):
            shutil.rmtree(cond_dir)
        os.mkdir(cond_dir)
        u_recorder = PVDRecorder(file=os.path.join(cond_dir, 'u.pvd'))
        self.recorder_dict.update({f"{up.name}_U": u_recorder})
        p_recorder = PVDRecorder(file=os.path.join(cond_dir, 'p.pvd'))
        self.recorder_dict.update({f"{up.name}_p": p_recorder})

    def generate_problem(self):
        controls = []
        control_bcs_info = {}
        gradient_ksp_options = {}
        for name, fun in self.controls_dicts.items():
            controls.append(fun)
            control_bcs_info.update({name: []})
            gradient_ksp_options.update({
                name: {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
            })

        self.state_system = StateProblem(self.state_problems)
        self.control_problem = create_control_problem(
            controls=controls,
            bcs_info=control_bcs_info,
            gradient_ksp_options=gradient_ksp_options
        )

        self.opt_problem = OptimalControlProblem(
            state_system=self.state_system,
            control_problem=self.control_problem,
            cost_functional_list=self.cost_functional_list,
            control_regularization=ControlRegularization(self.regularization_list)
        )

    def simulate(self):
        for cond_name, cond_cfg in self.condition_dicts.items():
            simulator: FoamSimulator = cond_cfg['simulator']

            rho = cond_cfg['control']
            rho_np = rho.x.array.copy()
            other_rhos = cond_cfg['other_controls']
            other_rhos_np = [i.x.array.copy() for i in other_rhos]
            alpha_np = self.compute_alpha(rho_np, other_rhos_np, pow_operator=np.power)

            U_tmp, p_tmp, nut_tmp = simulator.simulate(
                modify_data={rho.name: alpha_np},
                return_point_field=True,
                apply_boundary=True
            )

            nut: dolfinx.fem.Function = cond_cfg['state_vars']['nut']
            nut.interpolate(nut_tmp)
            nut.x.array[:] = nut.x.array + cond_cfg['kinematic_viscosity']

            up: dolfinx.fem.Function = cond_cfg['state_vars']['up']
            up.x.array[self.Q_to_W_dofs] = p_tmp.x.array.copy()

            vel_tmp = dolfinx.fem.Function(self.V, name='vel_tmp')
            vel_tmp.interpolate(U_tmp)
            up.x.array[self.V_to_W_dofs] = vel_tmp.x.array.copy()

    def compute_gradient(self):
        t1 = time.time()
        self.simulate()
        simulate_time_cost = time.time() - t1

        t1 = time.time()
        grad_dict = self.opt_problem.compute_gradient(
            self.domain.comm,
            update_state=False,
            update_adjoint=True,
            # state_kwargs={'with_debug': True},
            # adjoint_kwargs={'with_debug': True},
            gradient_kwargs={
                'with_debug': False,
                'A_assemble_method': 'Identity_row'
            },
        )
        grad_time_cost = time.time() - t1

        # print(f"Simulate timeCost:{simulate_time_cost}, Grad timeCost:{grad_time_cost}.")

        return grad_dict

    def update_control(self, grad_dict: Dict[str, dolfinx.fem.Function]):
        for name, control in self.controls_dicts.items():
            grad_np: np.array = grad_dict[name].x.array.copy().reshape((-1, 1))

            grad_np = self.grad_filter.run(grad_np)
            grad_np = np.asarray(grad_np).reshape(-1)

            max_step = self.grad_cfg.get('max_step_size', 0.2)
            min_step = self.grad_cfg.get('min_step_size', 0.01)

            # ------ method 1 too slow, how to speed it
            # norm = np.linalg.norm(grad_np, ord=2)
            norm = np.max(np.abs(grad_np))
            if norm > max_step:
                grad_np = grad_np / norm * max_step
            elif norm < min_step:
                grad_np = grad_np / norm * min_step
            # ------

            # # ------ debug visual for gradient
            # # np.save('/home/admin123456/Desktop/work/path_examples/topo_path_02/data.npy', grad_np)
            # # tmp_grid: pyvista.UnstructuredGrid = self.grid.copy()
            # # tmp_grid.cell_data['tmp'] = grad_np
            # # pyvista.save_meshio('/home/admin123456/Desktop/work/path_examples/topo_path_02/tmp_grid.vtu', tmp_grid)
            #
            # tmp_mesh = pyvista.PointSet(self.grid.cell_centers().points)
            # tmp_mesh.point_data['tmp'] = grad_np
            # tmp_mesh.set_active_scalars('tmp')
            # tmp_mesh.plot()
            # # ------

            control_np: np.array = control.x.array.copy().reshape(-1)
            control_np = control_np - grad_np
            control_np = np.maximum(np.minimum(control_np, 0.99), 0.01)
            control.x.array[:] = control_np

    @staticmethod
    def compute_flux(inlet_function: dolfinx.fem.Function, marker: int, n_vec: ufl.FacetNormal, ds: ufl.Measure):
        flux = -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(
            ufl.dot(inlet_function, n_vec) * ds(marker)
        ))
        return flux

    @staticmethod
    def generate_minium_energy_form(alpha: ufl.Form, u: dolfinx.fem.Function):
        # todo some problem here
        return 0.5 * alpha * inner(u, u) * ufl.dx + inner(grad(u), grad(u)) * ufl.dx

    @staticmethod
    def generate_minium_pressure_loss_form(
            domain: dolfinx.mesh.Mesh, alpha: ufl.Form, p: dolfinx.fem.Function, marker: int, ds: ufl.Measure
    ):
        # outlet pressure is always constant, so we just optimize the inlet pressure
        area1 = AssembleUtils.assemble_scalar(dolfinx.fem.form(dolfinx.fem.Constant(domain, 1.0) * ds(marker)))
        # return (1.0 / area1) * ufl.elem_pow(p, 2.0) * ds(marker)
        return (1.0 / area1) * p * ds(marker)

    @staticmethod
    def generate_maximum_outlet_flux_form(
            domain: dolfinx.mesh.Mesh, alpha: ufl.Form, u: dolfinx.fem.Function,
            marker: int, ds: ufl.Measure, n_vec: ufl.FacetNormal
    ):
        # todo why fail
        return dot(u, n_vec) * ds(marker)

    def log_info(self, step):
        for name, fun in self.controls_dicts.items():
            self.recorder_dict[name].write_function(fun, step=step)
        for problem in self.state_system.state_problems:
            self.recorder_dict[f"{problem.state.name}_U"].write_function(problem.state.sub(0).collapse(), step=step)
            self.recorder_dict[f"{problem.state.name}_p"].write_function(problem.state.sub(1).collapse(), step=step)
