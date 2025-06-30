import os
import shutil
import numpy as np
import dolfinx
import ufl
from typing import Dict, Union, List, Literal, Tuple
from ufl import grad, inner, div, dot
import pyvista

from scripts_py.version_9.dolfinx_grad.topo_utils import DistanceBandFilter
from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils, MeshObj
from scripts_py.version_9.dolfinx_grad.dynamic_recorder import VtuRecorder

from scripts_py.version_9.dolfinx_grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.solver_optimize import OptimalControlProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import create_control_problem, create_state_problem
from scripts_py.version_9.dolfinx_grad.lagrange_method.control_regularization import ControlRegularizationPipe, \
    ControlIntegralRegularization
from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import IntegralFunctional, \
    ScalarTrackingFunctional


class TopoLogyFluidOptimizer(object):
    state_ksp_option = {
        'ksp_type': 'preonly', 'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": 1
    }
    adjoint_ksp_option = {
        'ksp_type': 'preonly', 'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": 1
    }
    gradient_ksp_option = {
        'ksp_type': 'preonly', 'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_24": 1
    }

    def __init__(
            self,
            mesh: MeshObj,
            filter_radius: float,
            control_init: float,
            result_dir,
    ):
        self.mesh: MeshObj = mesh
        self.band_filter = DistanceBandFilter(
            self.mesh.grid, filter_radius=filter_radius, is_cell_field=True, is_center_important=True
        )

        self.W = dolfinx.fem.FunctionSpace(
            self.mesh.domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", self.mesh.domain.ufl_cell(), 2),
                ufl.FiniteElement("Lagrange", self.mesh.domain.ufl_cell(), 1)
            ])
        )
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()

        self.vector_cell_functionSpace = dolfinx.fem.VectorFunctionSpace(self.mesh.domain, ("DG", 0))
        self.scalar_cell_functionSpace = dolfinx.fem.FunctionSpace(self.mesh.domain, ("DG", 0))
        self.vector_point_functionSpace = dolfinx.fem.VectorFunctionSpace(self.mesh.domain, ("CG", 1))
        self.scalar_point_functionSpace = dolfinx.fem.FunctionSpace(self.mesh.domain, ("CG", 1))

        self.n_vec = MeshUtils.define_facet_norm(self.mesh.domain)
        self.ds = MeshUtils.define_ds(self.mesh.domain, self.mesh.facet_tags)
        self.dx = MeshUtils.define_dx(self.mesh.domain)

        self.control_rho = dolfinx.fem.Function(self.scalar_cell_functionSpace, name=f'control_rho')
        self.control_rho.x.array[:] = control_init

        self.surface_area = AssembleUtils.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.ds
        ))
        self.volume = AssembleUtils.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.dx
        ))

        self.regularization_list = []
        self.problem_cfgs = {}

        self.recorder = VtuRecorder(self.mesh, save_dir=result_dir, is_binary=True)

    def add_regularization(self, reg_type, **kwargs):
        if reg_type == 'volume_regularization':
            reg = ControlIntegralRegularization(
                name='reg_vol',
                domain=self.mesh.domain,
                integral_form=self.control_rho,
                control=self.control_rho,
                weight=kwargs['weight'],
                target_value=self.volume * kwargs['target_ratio'],
                method='percentage_div'
            )
        else:
            raise ValueError('[Error]: Non-valid regularization type.')
        self.regularization_list.append(reg)

    def add_problem(
            self,
            problem_name: str,
            inlet_cfg: Dict[str, Dict[str, Union[int, np.ndarray, List]]],
            outlet_cfg: Dict[str, Dict[str, Union[int, float]]],
            wall_cfg: Dict[str, Dict],
            constant_nu: float,
            fluid_type: Literal['stoke', 'naiver_stoke', 'rans'],
            loss_cfgs: List[Dict[str, Union[str, int, float]]],
            state_ksp_option: Dict = None,
            adjoint_ksp_option: Dict = None,
            gradient_ksp_option: Dict = None,
            alpha_min=0.01, alpha_max=1000.0, alpha_p=3.0,
    ):
        uh = dolfinx.fem.Function(self.W, name=f"{problem_name}_state")
        u, p = ufl.split(uh)  # please don't use up.split()
        vq = dolfinx.fem.Function(self.W, name=f"{problem_name}_adjoint")
        v, q = ufl.split(vq)  # please don't use vq.split()

        f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.topology_dim))
        nut = dolfinx.fem.Function(self.scalar_cell_functionSpace, name=f"{problem_name}_nut")
        nut.x.array[:] = constant_nu

        # ------ govern equation
        if fluid_type == 'stoke':
            govern_form = (
                    inner(grad(u), grad(v)) * ufl.dx
                    - div(v) * p * ufl.dx
                    - q * div(u) * ufl.dx
                    - inner(f, v) * ufl.dx
            )
        elif fluid_type == 'naiver_stoke':
            govern_form = (
                    nut * inner(grad(u), grad(v)) * ufl.dx
                    + inner(grad(u) * u, v) * ufl.dx
                    - div(v) * p * ufl.dx
                    + q * div(u) * ufl.dx
                    - inner(f, v) * ufl.dx
            )
        elif fluid_type == 'rans':
            raise NotImplementedError('[Error]: SU2 is too complex to change the governing equation.')
        else:
            raise ValueError('[Error]: Non-valid simulator_type')

        alpha = self.alpha_fun(
            rho=self.control_rho, alpha_min=alpha_min, alpha_max=alpha_max, p=alpha_p, pow_operator=ufl.elem_pow
        )
        govern_form += alpha * inner(u, v) * ufl.dx

        # ------ boundary
        bcs_info = []
        for name, cfg in inlet_cfg.items():
            def vel_exp(x):
                num = x.shape[1]
                bry_values = np.zeros((self.mesh.topology_dim, num))
                bry_values[0] = cfg['value'][0]
                bry_values[1] = cfg['value'][1]
                if self.mesh.topology_dim == 3:
                    bry_values[2] = cfg['value'][2]
                return bry_values

            bc_value = dolfinx.fem.Function(self.V, name=f'bry_{name}')
            bc_value.interpolate(vel_exp)
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=cfg['marker']
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, self.W0)
            bcs_info.append((bc, self.W0, bc_dofs, bc_value))

        for name, cfg in outlet_cfg.items():
            bc_value = dolfinx.fem.Function(self.Q, name=f'bry_{name}')
            bc_value.x.array[:] = cfg['value']
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=cfg['marker']
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, self.W1)
            bcs_info.append((bc, self.W1, bc_dofs, bc_value))

        for name, cfg in wall_cfg.items():
            bc_value = dolfinx.fem.Function(self.V, name=f'bry_{name}')
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=cfg['marker']
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, self.W0)
            bcs_info.append((bc, self.W0, bc_dofs, bc_value))

        state_problem = create_state_problem(
            name=f"{problem_name}_state_part",
            F_form=govern_form, state=uh, adjoint=vq,
            is_linear=True if fluid_type == 'stoke' else False,
            bcs_info=bcs_info,
            state_ksp_option=state_ksp_option if (state_ksp_option is not None) else self.state_ksp_option,
            adjoint_ksp_option=adjoint_ksp_option if (adjoint_ksp_option is not None) else self.adjoint_ksp_option,
        )
        state_system = StateProblem([state_problem])

        # ------ control problem
        control_problem = create_control_problem(
            controls=[self.control_rho],
            bcs_info={self.control_rho.name: []},
            gradient_ksp_options={
                self.control_rho.name: gradient_ksp_option if (
                        gradient_ksp_option is not None) else self.gradient_ksp_option
            }
        )

        # ------ loss type
        cost_functional_list = []
        for cfg in loss_cfgs:
            if cfg['style'] == 'minimum_dissipation':
                cost_form = 0.5 * alpha * inner(u, u) * ufl.dx + inner(grad(u), grad(u)) * ufl.dx
                cost_fun = IntegralFunctional(
                    name=f'{problem_name}_loss_minimum_dissipation',
                    domain=self.mesh.domain, form=cost_form, weight=cfg['weight']
                )
                cost_functional_list.append(cost_fun)

            elif cfg['style'] == 'minimum_stress':
                area = AssembleUtils.assemble_scalar(dolfinx.fem.form(
                    dolfinx.fem.Constant(self.mesh.domain, 1.0) * self.ds(cfg['marker'])
                ))
                cost_form = (1.0 / area) * p * self.ds(cfg['marker'])
                cost_fun = IntegralFunctional(
                    name=f'{problem_name}_loss_minimum_stress_{cfg["marker"]}',
                    domain=self.mesh.domain, form=cost_form, weight=cfg['weight']
                )
                cost_functional_list.append(cost_fun)

            elif cfg['style'] == 'constant_flux':
                flux_form = ufl.dot(u, self.n_vec) * self.ds(cfg['marker'])
                cost_fun = ScalarTrackingFunctional(
                    name=f'{problem_name}_loss_constant_flux_{cfg["marker"]}',
                    domain=self.mesh.domain,
                    form=flux_form,
                    tracking_goal=cfg['tracking_flux'],
                    weight=cfg['weight'],
                    method='percentage_div'
                )
                cost_functional_list.append(cost_fun)

            elif cfg['style'] == 'maximum_flux':
                flux_form = ufl.dot(u, self.n_vec) * self.ds(cfg['marker'])  # todo is positive or negative
                cost_fun = IntegralFunctional(
                    name=f'{problem_name}_loss_maximum_flux_{cfg["marker"]}',
                    domain=self.mesh.domain, form=flux_form, weight=cfg['weight']
                )
                cost_functional_list.append(cost_fun)

            else:
                raise ValueError('[Error]: Non-valid loss type.')

        opt_problem = OptimalControlProblem(
            state_system=state_system,
            control_problem=control_problem,
            cost_functional_list=cost_functional_list,
            control_regularization_pipe=ControlRegularizationPipe(self.regularization_list)
        )

        self.problem_cfgs[problem_name] = {
            'uh': uh, 'opt_problem': opt_problem
        }

    def compute_gradient(
            self,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ) -> Tuple[bool, Union[None, Dict[str, np.ndarray]]]:
        problem_grad = {}
        for name, cfg in self.problem_cfgs.items():
            opt_problem: OptimalControlProblem = cfg['opt_problem']
            is_solved, grads_dict = opt_problem.compute_gradient(
                self.mesh.domain.comm,
                state_kwargs={
                    'mat_assemble_method': mat_assemble_method,
                    'with_debug': with_debug
                },
                adjoint_kwargs={
                    'mat_assemble_method': mat_assemble_method,
                    'with_debug': with_debug
                },
                gradient_kwargs={
                    'mat_assemble_method': mat_assemble_method,
                    'with_debug': with_debug
                },
                update_state=True,
                update_adjoint=True
            )
            if not is_solved:
                print(f'[Error]: {grads_dict["log"]}')
                return False, None
            problem_grad[name] = np.copy(grads_dict[self.control_rho.name].x.array)
        return True, problem_grad

    def evaluate_cost(
            self,
            update_state,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        problem_loss_dict = {}
        for name, cfg in self.problem_cfgs.items():
            opt_problem: OptimalControlProblem = cfg['opt_problem']
            problem_loss_dict[name] = opt_problem.evaluate_cost_functional(
                self.mesh.domain.comm, update_state=update_state,
                mat_assemble_method=mat_assemble_method, with_debug=with_debug
            )
        return problem_loss_dict

    def update_control(self, grad_dict: Dict[str, np.ndarray]):
        grad_merge = np.zeros_like(self.control_rho.x.array, dtype=float)
        for name, grad in grad_dict.items():
            grad_merge += grad

        grad_merge = grad_merge.reshape((-1, 1))
        grad_merge = self.band_filter.run(grad_merge)
        grad_merge = np.asarray(grad_merge).reshape((-1))

        norm = np.max(np.abs(grad_merge))
        if norm > 0.1:
            grad_merge = grad_merge / norm * 0.1
        elif norm < 0.01:
            grad_merge = grad_merge / norm * 0.01

        control_np = np.copy(self.control_rho.x.array) - grad_merge
        self.control_rho.x.array[:] = np.maximum(np.minimum(control_np, 0.99), 0.01)

    def log(self, step):
        for name, cfg in self.problem_cfgs.items():
            uh: dolfinx.fem.Function = cfg['uh']

            vel_ref = dolfinx.fem.Function(self.vector_point_functionSpace, name='vel')
            vel_ref.interpolate(uh.sub(0).collapse())
            vel_ref_np = vel_ref.x.array.reshape((-1, self.mesh.topology_dim))
            if vel_ref_np.shape[1] == 2:
                vel_ref_np = np.concatenate([vel_ref_np, np.zeros((vel_ref_np.shape[0], 1))], axis=1)
            self.recorder.add_point_scalar(f'{name}_velocity', vel_ref_np)

            p_ref = uh.sub(1).collapse()
            self.recorder.add_point_scalar(f'{name}_pressure', p_ref.x.array)
        self.recorder.add_cell_scalar('control', self.control_rho.x.array)
        self.recorder.save(step)

    def run(
            self,
            max_iteration=200,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        step, min_loss = 0, np.inf
        while True:
            is_solved, grads_dict = self.compute_gradient(
                mat_assemble_method=mat_assemble_method, with_debug=with_debug
            )
            if not is_solved:
                return False

            loss_dict = self.evaluate_cost(update_state=False)
            loss_sum = np.sum(list(loss_dict.values()))
            print(f"[Debug]: Step_{step} loss:{loss_sum:.3f}")
            self.log(step)

            self.update_control(grads_dict)
            step += 1

            if step > max_iteration:
                break

            min_loss = np.minimum(min_loss, loss_sum)
            if loss_sum > min_loss * 1.05:
                break

        loss_dict = self.evaluate_cost(
            update_state=True, mat_assemble_method=mat_assemble_method, with_debug=with_debug
        )
        loss_sum = np.sum(list(loss_dict.values()))
        print(f"[Debug]: Step_{step} loss:{loss_sum:.3f}")
        self.log(step)

        return True

    def compute_alpha(
            self,
            rho: Union[dolfinx.fem.Function, np.ndarray], alpha_cfg: Dict,
            pow_operator=ufl.elem_pow
    ):
        alpha = self.alpha_fun(
            rho,
            alpha_min=alpha_cfg.get('alpha_min', 0.01),
            alpha_max=alpha_cfg.get('alpha_max', 1000.0),
            p=alpha_cfg.get('p', 3.0),
            pow_operator=pow_operator,
        )
        return alpha

    @staticmethod
    def alpha_fun(
            rho: Union[np.ndarray, dolfinx.fem.Function],
            alpha_min=0.01, alpha_max=1000.0, p=3.0, pow_operator=ufl.elem_pow
    ):
        return alpha_min + pow_operator(1.0 - rho, p) * (alpha_max - alpha_min)

    def compute_flux(self, velocity, marker: int):
        bc = dolfinx.fem.Function(self.V)
        bc.x.array[:] = velocity
        flux = -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(ufl.dot(bc, self.n_vec) * self.ds(marker)))
        return flux


def debug():
    mesh = MeshObj(mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example02/model.msh', dim=2)
    optim = TopoLogyFluidOptimizer(
        mesh=mesh, filter_radius=0.2, control_init=0.5,
        result_dir='/home/admin123456/Desktop/work/topopt_exps/new_test/example02/cache'
    )
    optim.add_regularization(
        reg_type='volume_regularization',
        target_ratio=0.25,
        weight=100.0,
    )
    optim.add_problem(
        problem_name='cond1',
        inlet_cfg={
            'inlet': {
                'marker': 18,
                'value': [1., 0., 0.]
            }
        },
        outlet_cfg={
            'outlet': {
                'marker': 19,
                'value': 0.0
            }
        },
        wall_cfg={
            'wall': {
                'marker': 20
            }
        },
        constant_nu=1. / 400.,
        fluid_type='stoke',
        loss_cfgs=[
            {
                'style': 'minimum_dissipation',
                'weight': 1.0
            }
        ],
    )
    optim.run(max_iteration=100)


if __name__ == '__main__':
    debug()
