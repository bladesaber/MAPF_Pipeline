import os
import shutil
import numpy as np
import dolfinx
import pandas as pd
import ufl
from typing import Dict, Union, List, Literal, Tuple
from ufl import grad, inner, div, dot
import pyvista
from sklearn.neighbors import KDTree

from scripts_py.version_9.dolfinx_grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshUtils
from scripts_py.version_9.dolfinx_grad.dynamic_recorder import VtuRecorder
from scripts_py.version_9.simulator.su2_simulator import Su2Simulator
from scripts_py.version_9.dolfinx_grad.vis_utils import VisUtils

from scripts_py.version_9.dolfinx_grad.lagrange_method.problem_state import StateProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.solver_optimize import OptimalShapeProblem
from scripts_py.version_9.dolfinx_grad.lagrange_method.type_database import create_shape_problem, create_state_problem
from scripts_py.version_9.dolfinx_grad.lagrange_method.shape_regularization import VolumeRegularization, \
    ShapeRegularizationPipe
from scripts_py.version_9.dolfinx_grad.lagrange_method.cost_functions import IntegralFunctional, \
    ScalarTrackingFunctional

from scripts_py.version_9.dolfinx_grad.gard_fuse_utils import GradRawFusion, GradSpringFeaFusion
from scripts_py.version_9.dolfinx_grad.spring_mesh_utils import SpringTorchMeshObj, SpringFeaMeshObj


# todo 使用SU2仿真并无法到达最优点，需要refit吗?


class Su2SimulatorPacker(Su2Simulator):
    def __init__(self, mesh_path, mesh: Union[SpringTorchMeshObj, SpringFeaMeshObj]):
        super().__init__(mesh_path=mesh_path)
        self.mesh = mesh
        self.su2_to_mesh_indices: np.ndarray[int] = None
        self.mesh_to_su2_indices: np.ndarray[int] = None
        self.generate_map_indices(self.mesh.grid)

    def generate_map_indices(self, grid: pyvista.UnstructuredGrid, tol=1e-4):
        grid_points = np.asarray(grid.points)
        if self.mesh.topology_dim == 2:
            grid_points = grid_points[:, :2]

        tree = KDTree(grid_points)
        dist_list, idxs_list = tree.query(self.points, k=1, return_distance=True)
        dist_list = np.concatenate(dist_list, axis=-1)
        idxs_list = np.concatenate(idxs_list, axis=-1)

        if np.max(dist_list) > tol:
            raise ValueError("[Error]: Mesh isn't compatible.")
        if len(np.unique(pd.Series(idxs_list).value_counts().values)) > 1:
            raise ValueError("[Error]: Mesh isn't compatible.")
        self.mesh_to_su2_indices = idxs_list
        self.su2_to_mesh_indices = np.argsort(idxs_list)

    def simulate(self, run_dir, with_monitor):
        res_vtu = self.run(run_dir=run_dir, with_monitor=with_monitor)
        velocity = np.asarray(res_vtu.point_data['Velocity'])
        velocity = velocity[self.su2_to_mesh_indices, :]
        pressure = np.asarray(res_vtu.point_data['Pressure'])
        pressure = pressure[self.su2_to_mesh_indices]
        return {
            'point_velocity': velocity,
            'point_pressure': pressure,
        }

    def update_geometry(self):
        if self.mesh.topology_dim == 2:
            self.points = self.mesh.points[self.mesh_to_su2_indices, :2]
        else:
            self.points = self.mesh.points[self.mesh_to_su2_indices, :]


class ShapeFluidOptimizer(object):
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

    def __init__(self, mesh_file: str, dim: int, result_dir):
        self.result_dir = result_dir
        assert mesh_file.endswith('.msh')
        self.mesh_file = mesh_file
        self.su2_file = os.path.join(result_dir, 'mesh.su2')
        Su2Simulator.msh_to_su2(self.mesh_file, su2_file=self.su2_file)
        self.mesh = SpringFeaMeshObj(mesh_file=mesh_file, dim=dim)

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

        self.surface_area = AssembleUtils.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.ds
        ))
        self.volume = AssembleUtils.assemble_scalar(dolfinx.fem.form(
            dolfinx.fem.Constant(self.mesh.domain, 1.0) * ufl.dx
        ))

        self.bcs_control = []
        self.regularization_list: List[Dict] = []
        self.problem_cfgs = {}

        self.recorder = VtuRecorder(self.mesh, save_dir=os.path.join(result_dir, 'result_vtu'), is_binary=True)

    def add_regularization(self, reg_type, **kwargs):
        if reg_type == 'volume_regularization':
            self.regularization_list.append({
                'style': reg_type,
                'target_volume_rho': kwargs['target_volume_rho'],
                'weight': kwargs['weight'],
                'method': 'percentage_div'
            })
        else:
            raise ValueError('[Error]: Non-valid regularization type.')

    def define_control_wall(self, bry_cfg: Dict):
        bc_free_marker, bc_free_dofs = [], []
        bc_fix_marker, bc_fix_dofs = [], []

        for name, cfg in bry_cfg.items():
            if cfg['is_free']:
                bc_free_marker.append(cfg['marker'])
                bc_dofs = MeshUtils.extract_entity_dofs(
                    self.vector_point_functionSpace, self.mesh.facet_dim,
                    MeshUtils.extract_entity_indices(
                        self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=cfg['marker']
                    )
                )
                bc_free_dofs.append(bc_dofs)

            else:
                bc_value = dolfinx.fem.Function(self.vector_point_functionSpace, name=f'bry_control_{name}')
                bc_dofs = MeshUtils.extract_entity_dofs(
                    self.vector_point_functionSpace, self.mesh.facet_dim,
                    MeshUtils.extract_entity_indices(
                        self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=cfg['marker']
                    )
                )
                bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)
                self.bcs_control.append((bc, self.vector_point_functionSpace, bc_dofs, bc_value))
                bc_fix_marker.append(cfg['marker'])
                bc_fix_dofs.append(bc_dofs)

        self.mesh.set_dofs(
            bc_fix_dofs=np.concatenate(bc_fix_dofs, axis=-1), bc_fix_marker=bc_fix_marker,
            bc_free_dofs=np.concatenate(bc_free_dofs, axis=-1), bc_free_marker=bc_free_marker,
        )

    def add_problem(
            self,
            problem_name: str,
            inlet_cfg: Dict[str, Dict[str, Union[int, np.ndarray, List]]],
            outlet_cfg: Dict[str, Dict[str, Union[int, float]]],
            wall_cfg: Dict[str, Dict],
            constant_nu: float,
            fluid_type: Literal['stoke', 'rans'],
            loss_cfgs: List[Dict[str, Union[str, int, float]]],
            state_ksp_option: Dict = None,
            adjoint_ksp_option: Dict = None,
            gradient_ksp_option: Dict = None,
            su2_cfg: Dict = None,
            scalar_product_method='default'
    ):
        uh = dolfinx.fem.Function(self.W, name=f"state")
        u, p = ufl.split(uh)  # please don't use up.split()
        vq = dolfinx.fem.Function(self.W, name=f"adjoint")
        v, q = ufl.split(vq)  # please don't use vq.split()

        f = dolfinx.fem.Constant(self.mesh.domain, np.zeros(self.mesh.topology_dim))
        nut = dolfinx.fem.Function(self.scalar_point_functionSpace, name=f"nut")
        nut.x.array[:] = constant_nu

        # ------ govern equation
        if fluid_type == 'stoke':
            govern_form = (
                    inner(grad(u), grad(v)) * ufl.dx
                    - div(v) * p * ufl.dx
                    - q * div(u) * ufl.dx
                    - inner(f, v) * ufl.dx
            )
        elif fluid_type == 'rans':
            govern_form = (
                    nut * inner(grad(u), grad(v)) * ufl.dx
                    + inner(grad(u) * u, v) * ufl.dx
                    - inner(p, div(v)) * ufl.dx
                    + inner(div(u), q) * ufl.dx
                    - inner(f, v) * ufl.dx
            )
        else:
            raise ValueError('[Error]: Non-valid simulator_type')

        # ------ state boundary
        bcs_state = []
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
            bcs_state.append((bc, self.W0, bc_dofs, bc_value))

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
            bcs_state.append((bc, self.W1, bc_dofs, bc_value))

        for name, cfg in wall_cfg.items():
            bc_value = dolfinx.fem.Function(self.V, name=f'bry_{name}')
            bc_dofs = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.mesh.facet_dim,
                MeshUtils.extract_entity_indices(
                    self.mesh.domain, style='facet', mesh_tags=self.mesh.facet_tags, marker=cfg['marker']
                )
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, self.W0)
            bcs_state.append((bc, self.W0, bc_dofs, bc_value))

        state_problem = create_state_problem(
            name=f"{problem_name}_state_part",
            F_form=govern_form, state=uh, adjoint=vq,
            is_linear=True if fluid_type == 'stoke' else False,
            bcs_info=bcs_state,
            state_ksp_option=state_ksp_option if (state_ksp_option is not None) else self.state_ksp_option,
            adjoint_ksp_option=adjoint_ksp_option if (adjoint_ksp_option is not None) else self.adjoint_ksp_option,
        )
        state_system = StateProblem([state_problem])

        # ------ simulator
        if fluid_type == 'rans':
            simulator = Su2SimulatorPacker(self.su2_file, self.mesh)

            su2_cache_dir = os.path.join(self.result_dir, f'{problem_name}_su2')
            simulator_cfg = {
                'proj_dir': su2_cache_dir,
                'mesh_filename': 'mesh.su2',
                'inlets_info': {},
                'outlets_info': {},
                'walls_info': {},
                'constant_viscosity': su2_cfg.get('constant_viscosity', 0.04),
                'conv_level': -12,
                'run_iter': su2_cfg.get('run_iter', 1000),
                'refit_by_fenics': su2_cfg.get('refit_by_fenics', False),
            }
            for name, cfg in inlet_cfg.items():
                vel_value = np.array(cfg['value'])
                vel_scale = np.linalg.norm(vel_value, ord=2)
                vel_direction: np.ndarray = vel_value / vel_scale
                simulator_cfg['inlets_info'][name] = {
                    'type': 'VELOCITY_INLET',
                    'temperature': cfg.get('temperature', 300.0),
                    'vel_scale': vel_scale,
                    'direction': (vel_direction[0], vel_direction[1], vel_direction[2])
                }

            for name, cfg in outlet_cfg.items():
                simulator_cfg['outlets_info'][name] = {
                    'type': 'PRESSURE_OUTLET',
                    'pressure': cfg['value'],
                }

            for name, cfg in wall_cfg.items():
                simulator_cfg['walls_info'][name] = {
                    'temperature': cfg.get('temperature', 300.0),
                }

        # ------ control problem
        shape_problem = create_shape_problem(
            domain=self.mesh.domain,
            bcs_info=self.bcs_control,
            gradient_ksp_option=gradient_ksp_option if (gradient_ksp_option is not None) else self.gradient_ksp_option
        )

        # ------ regularization
        reg_list = []
        for cfg in self.regularization_list:
            if cfg['style'] == 'volume_regularization':
                reg = VolumeRegularization(
                    name=f"{problem_name}_reg_vol",
                    shape_problem=shape_problem,
                    target_volume_rho=cfg['target_volume_rho'],
                    weight=cfg['weight'],
                    method='percentage_div'
                )
            else:
                raise ValueError('[Error]: Non-valid regularization type.')

            reg_list.append(reg)

        # ------ loss type
        cost_functional_list = []
        for cfg in loss_cfgs:
            if cfg['style'] == 'minimum_dissipation':
                cost_form = 0.5 * inner(u, u) * ufl.dx + inner(grad(u), grad(u)) * ufl.dx  # todo not sure
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

        # ------
        if scalar_product_method == 'default':
            opt_problem = OptimalShapeProblem(
                state_system=state_system,
                shape_problem=shape_problem,
                cost_functional_list=cost_functional_list,
                shape_regularization_list=ShapeRegularizationPipe(reg_list),
                scalar_product=None,
                scalar_product_method={'method': "default"}
            )
        elif scalar_product_method == 'Poincare-Steklov operator':
            opt_problem = OptimalShapeProblem(
                state_system=state_system,
                shape_problem=shape_problem,
                cost_functional_list=cost_functional_list,
                shape_regularization_list=ShapeRegularizationPipe(reg_list),
                scalar_product=None,
                scalar_product_method={
                    'method': 'Poincare-Steklov operator',
                    'cell_tags': self.mesh.cell_tags,
                    'facet_tags': self.mesh.facet_tags,
                    'bry_free_markers': self.mesh.bc_free_marker,
                    'bry_fixed_markers': self.mesh.bc_fix_marker,
                    "lambda_lame": 1.0,
                    "damping_factor": 0.2,
                    "mu_fix": 1.0,
                    "mu_free": 1.0,
                    "use_inhomogeneous": False,
                    "inhomogeneous_exponent": 1.0,
                    "update_inhomogeneous": False
                },
            )
        else:
            raise ValueError(f'[Error]: Non-valid scalar product method:{scalar_product_method}.')

        self.problem_cfgs[problem_name] = {
            'uh': uh, 'nut': nut, 'opt_problem': opt_problem,
            'fluid_type': fluid_type,
            'simulator': simulator if fluid_type == 'rans' else None,
            'simulator_cfg': simulator_cfg if fluid_type == 'rans' else None,
        }

    def compute_fluid_grad(
            self,
            state_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            adjoint_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            gradient_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ) -> Tuple[bool, Union[Dict[str, np.ndarray], None]]:
        self.simulate(
            mat_assemble_method=state_mat_assemble_method,
            with_debug=with_debug
        )

        problem_grad = {}
        for name, cfg in self.problem_cfgs.items():
            opt_problem: OptimalShapeProblem = cfg['opt_problem']
            is_solved, grad_var = opt_problem.compute_gradient(
                self.mesh.domain.comm,
                state_kwargs={
                    'mat_assemble_method': state_mat_assemble_method,
                    'with_debug': with_debug
                },
                adjoint_kwargs={
                    'mat_assemble_method': adjoint_mat_assemble_method,
                    'with_debug': with_debug
                },
                gradient_kwargs={
                    'mat_assemble_method': gradient_mat_assemble_method,
                    'with_debug': with_debug
                },
                scalar_product_kwargs={
                    'with_debug': with_debug
                },
                update_state=False,
                update_adjoint=True
            )
            if not is_solved:
                print(f'[Error]: {grad_var["log"]}')
                return False, None
            problem_grad[name] = np.copy(grad_var.x.array)
        return True, problem_grad

    # def compute_obstacle_grad(self, obs_pcd: np.ndarray, obs_radius: float, unit_lr: float, weight: float):
    #     """depreciated"""
    #     conf_mesh_idxs, conf_obs_idxs, require_dist = self.mesh.detect_obstacle(obs_pcd, obs_radius, unit_lr)
    #     if len(conf_mesh_idxs) > 0:
    #         mesh_pcd_t = TorchUtils.np2tensor(self.mesh.points, require_grad=True)
    #         obs_pcd_t = TorchUtils.np2tensor(obs_pcd, require_grad=False)
    #         require_dist_t = TorchUtils.np2tensor(require_dist.reshape(-1) + unit_lr, require_grad=False)
    #
    #         dist_t = torch.norm(mesh_pcd_t[conf_mesh_idxs, :] - obs_pcd_t[conf_obs_idxs, :], p=2, dim=1)
    #         loss = (F.relu(require_dist_t - dist_t)).sum() * weight
    #
    #         loss.backward()
    #         with torch.no_grad():
    #             grad_np = TorchUtils.grad2np(mesh_pcd_t, with_copy=True)
    #         return {'loss': loss.detach().numpy(), 'grad': grad_np}
    #     else:
    #         return {'loss': 0.0, 'grad': None}

    def simulate(
            self,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        for name, cfg in self.problem_cfgs.items():
            if cfg['fluid_type'] == 'stoke':
                opt_problem: OptimalShapeProblem = cfg['opt_problem']
                opt_problem.compute_state(
                    self.mesh.domain.comm,
                    mat_assemble_method=mat_assemble_method,
                    with_debug=with_debug
                )
            elif cfg['fluid_type'] == 'rans':
                simulator: Su2SimulatorPacker = cfg['simulator']
                simulator_cfg = cfg['simulator_cfg']
                simulator.prepare_sim_proj(
                    proj_dir=simulator_cfg['proj_dir'],
                    mesh_filename=simulator_cfg['mesh_filename'],
                    inlets_info=simulator_cfg['inlets_info'],
                    outlets_info=simulator_cfg['outlets_info'],
                    walls_info=simulator_cfg['walls_info'],
                    density=1.0,  # 由于上式fenics的naiver stoke form不包含密度，因此这里一定为1
                    constant_viscosity=simulator_cfg['constant_viscosity'],
                    conv_level=simulator_cfg['conv_level'],
                    run_iter=simulator_cfg['run_iter']
                )
                res_dict = simulator.simulate(simulator_cfg['proj_dir'], with_monitor=False)
                point_velocity = res_dict['point_velocity']
                if self.mesh.topology_dim == 2:
                    point_velocity = point_velocity[:, :2]
                point_pressure = res_dict['point_pressure']

                u_point_fun = dolfinx.fem.Function(self.vector_point_functionSpace)
                u_point_fun.x.array[:] = point_velocity.reshape(-1)
                u_order_fun = dolfinx.fem.Function(self.V)
                u_order_fun.interpolate(u_point_fun)

                p_point_fun = dolfinx.fem.Function(self.scalar_point_functionSpace)
                p_point_fun.x.array[:] = point_pressure.reshape(-1)
                p_order_fun = dolfinx.fem.Function(self.Q)
                p_order_fun.interpolate(p_point_fun)

                uh: dolfinx.fem.Function = cfg['uh']
                uh.x.array[self.V_to_W_dofs] = np.copy(u_order_fun.x.array)
                uh.x.array[self.Q_to_W_dofs] = np.copy(p_order_fun.x.array)

                if simulator_cfg['refit_by_fenics']:
                    opt_problem: OptimalShapeProblem = cfg['opt_problem']
                    opt_problem.compute_state(
                        self.mesh.domain.comm,
                        mat_assemble_method=mat_assemble_method,
                        with_debug=True
                    )

            else:
                raise ValueError('[Error]: Non-valid fluid type.')

    def evaluate_cost(
            self,
            update_state,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        if update_state:
            self.simulate(mat_assemble_method, with_debug)

        problem_loss_dict = {}
        for name, cfg in self.problem_cfgs.items():
            opt_problem: OptimalShapeProblem = cfg['opt_problem']
            problem_loss_dict[name] = opt_problem.evaluate_cost_functional(
                self.mesh.domain.comm, update_state=False,
            )
        return problem_loss_dict

    def get_loss_dict(
            self,
            update_state,
            mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False
    ):
        if update_state:
            self.simulate(mat_assemble_method, with_debug)

        cost_dict = {}
        for name, cfg in self.problem_cfgs.items():
            opt_problem: OptimalShapeProblem = cfg['opt_problem']
            for cost_name, cost_value in opt_problem.get_cost_info(
                    self.mesh.domain.comm, mat_assemble_method=mat_assemble_method, with_debug=with_debug
            ).items():
                cost_dict[cost_name] = cost_value
        return cost_dict

    def update_geometry(
            self, displacement: np.ndarray,
            deform_method: Literal['non_constraint', 'line_search', 'safe'] = 'line_search',
            min_deform=0.01, max_deform=0.1, line_search_step=5,
            with_vol_check=True, with_angle_check=True
    ):
        if deform_method == 'non_constraint':
            self.mesh.deform(displacement_np=displacement)
            is_valid = True
        elif deform_method == 'safe':
            is_valid = self.mesh.deform_safe(
                displacement_np=displacement,
                with_angle_check=with_angle_check, with_vol_check=with_vol_check
            )
        elif deform_method == 'line_search':
            # deform_direction = displacement / np.linalg.norm(displacement, ord=2)
            deform_direction = displacement / np.max(np.abs(displacement))
            scales = (np.linspace(min_deform, max_deform, line_search_step)[::-1]).tolist()
            is_valid = self.mesh.deform_line_search(
                deform_direction, scales=scales,
                with_vol_check=with_vol_check, vol_change=0.2,
                with_angle_check=with_angle_check, angle_tol_max=160.0, angle_tol_min=20.0
            )
        else:
            raise ValueError(f'[Error]: Non-valid deform method {deform_method}.')

        if not is_valid:
            return False

        for name, cfg in self.problem_cfgs.items():
            if cfg['fluid_type'] == 'rans':
                simulator: Su2SimulatorPacker = cfg['simulator']
                simulator.update_geometry()
        return True

    def run_debug_with_constraint(
            self,
            max_iteration=200,
            state_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            adjoint_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            gradient_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
            with_debug=False,
            obs_pcd: np.ndarray = None,
            obs_r=0.0
    ):
        """
        if isinstance(self.mesh, SpringTorchMeshObj):
            fuser = GradSpringFusionTorchObj(self.mesh, obs_pcd, obs_r=obs_r)
        elif isinstance(self.mesh, SpringFeaMeshObj):
            fuser = GradSpringFusionFeaObj(self.mesh, obs_pcd, obs_r=obs_r)
        else:
            raise ValueError('[Error]: Non-valid mesh.')

        step, min_fluid_loss = 0, np.inf
        while True:
            is_solved, grad_fluid_dict = self.compute_fluid_grad(
                state_mat_assemble_method=state_mat_assemble_method,
                adjoint_mat_assemble_method=adjoint_mat_assemble_method,
                gradient_mat_assemble_method=gradient_mat_assemble_method,
                with_debug=with_debug
            )
            if not is_solved:
                return False

            for name, pcd_grad in grad_fluid_dict.items():
                pcd_grad = pcd_grad.reshape((-1, self.mesh.topology_dim)) * -1.0
                fuser.add_var(name, pcd_grad)

            if isinstance(fuser, GradSpringFusionTorchObj):
                grad_fused = fuser.solve_disp(max_scale=0.04)
            elif isinstance(fuser, GradSpringFusionFeaObj):
                is_valid, grad_fused = fuser.solve_disp(
                    # method='poisson_eq',
                    # method='linear_elastic', kwargs={'stiff_cof': 1.0, 'lambda': 1.25, 'mu': 1.0},
                    method='linear_elastic_obs', kwargs={
                        'stiff_cof': 20.0, 'lambda': 1.25, 'mu': 1.0, 'step_scale': 0.05
                    },
                    max_scale=0.05,
                    with_debug=False
                )
                if not is_valid:
                    print('[Info]: solving linear elastic equation fail.')
                    break
            else:
                raise ValueError('[Error]: Non-valid fuser')

            if self.mesh.topology_dim == 2:
                grad_fused = np.concatenate([grad_fused, np.zeros((grad_fused.shape[0], 1))], axis=1)

            loss_fluid_dict = self.evaluate_cost(update_state=False)
            loss_fluid_sum = np.sum(list(loss_fluid_dict.values()))

            loss_dict = self.get_loss_dict(update_state=False)
            desc = f'[iter:{step}] '
            for loss_n, loss_f in loss_dict.items():
                desc += f'{loss_n}:{loss_f:.3f} '
            desc += f'loss:{loss_fluid_sum:.3f}'
            print(desc)

            self.log(step)

            is_valid = self.update_geometry(
                displacement=grad_fused, deform_method='safe',
                with_vol_check=False, with_angle_check=False
            )
            if not is_valid:
                print('[Info]: update geometry fail.')
                return False

            self.mesh.re_mesh()  # todo need more adjust

            step += 1
            if step > max_iteration:
                print('[Info]: reach max iteration.')
                break

            min_fluid_loss = np.minimum(min_fluid_loss, loss_fluid_sum)
            if loss_fluid_sum > min_fluid_loss * 1.1:
                print('[Info]: loss overfit.')
                break

        loss_fluid_dict = self.evaluate_cost(
            update_state=True,
            mat_assemble_method=state_mat_assemble_method,
            with_debug=with_debug
        )
        loss_fluid_sum = np.sum(list(loss_fluid_dict.values()))
        print(f"[Debug]: Step_{step} loss_fluid:{loss_fluid_sum:.3f}")
        self.log(step)

        return True
        """
        pass

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
        self.recorder.save(step)


def run_optimize(
        optim: ShapeFluidOptimizer,
        max_iteration=200,
        max_deformation=0.05,
        min_deformation=0.01,
        state_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
        adjoint_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
        gradient_mat_assemble_method: Literal['Identity_row', 'lift'] = 'lift',
        with_debug=False,
        obs_pcd: np.ndarray = None,
        obs_r=0.0,
        with_re_mesh=False
):
    if obs_pcd is None:
        fuser = GradRawFusion(mesh=optim.mesh)
    else:
        assert obs_r > 0.0
        fuser = GradSpringFeaFusion(optim.mesh, obs_pcd, obs_r)

    step, min_loss = 0, np.inf
    while True:
        is_solved, grad_fluid_dict = optim.compute_fluid_grad(
            state_mat_assemble_method=state_mat_assemble_method,
            adjoint_mat_assemble_method=adjoint_mat_assemble_method,
            gradient_mat_assemble_method=gradient_mat_assemble_method,
            with_debug=with_debug
        )
        if not is_solved:
            print('[Info]: solving linear elastic equation fail.')
            return False

        for name, pcd_grad in grad_fluid_dict.items():
            pcd_grad = pcd_grad.reshape((-1, optim.mesh.topology_dim)) * -1.0
            fuser.add_grad(name, pcd_grad)

        if fuser.style == 'GradRawFusion':
            disp = fuser.solve_disp(max_step=0.05)
        elif fuser.style == 'GradSpringFeaFusion':
            is_valid, disp = fuser.solve_disp(
                # method='poisson_eq',
                # method='linear_elastic', kwargs={'stiff_cof': 1.0, 'lambda': 1.25, 'mu': 1.0},
                method='linear_elastic_obs', kwargs={'stiff_cof': 20.0, 'lambda': 1.25, 'mu': 1.0, 'step_scale': 0.05},
                max_step=0.05,
                with_debug=False
            )
            if not is_valid:
                print('[Info]: solving linear elastic equation fail.')
                break
        else:
            raise ValueError('[Error]: Non-valid fuser type.')

        if optim.mesh.topology_dim == 2:
            disp = np.concatenate([disp, np.zeros((disp.shape[0], 1))], axis=1)

        # ------- test the sensitivity of gradient, gradient is very sensitivity
        # disp += np.random.normal(loc=0.0, scale=max_deformation * 0.1, size=grad_fluid.shape)
        # ------

        loss_dict = optim.get_loss_dict(
            update_state=False, mat_assemble_method=state_mat_assemble_method, with_debug=with_debug
        )
        # loss_fluid_dict = optim.evaluate_cost(update_state=False)
        loss_sum = np.sum(list(loss_dict.values()))

        # ------
        desc = f'[iter:{step}] '
        for loss_n, loss_f in loss_dict.items():
            desc += f'{loss_n}:{loss_f:.3f} '
        desc += f'loss:{loss_sum:.3f}'
        print(desc)
        optim.log(step)

        if fuser.style == 'GradRawFusion':
            is_valid = optim.update_geometry(
                displacement=disp,
                deform_method='line_search',
                min_deform=min_deformation,
                max_deform=max_deformation,
                line_search_step=5,
            )
        else:
            is_valid = optim.update_geometry(
                displacement=disp, deform_method='safe',
                with_vol_check=False, with_angle_check=False
            )
        if not is_valid:
            print('[Info]: update geometry fail.')
            return False

        if with_re_mesh:
            optim.mesh.re_mesh()  # todo need more adjust

        step += 1
        if step > max_iteration:
            print('[Info]: reach max iteration.')
            break

        min_loss = np.minimum(min_loss, loss_sum)
        if loss_sum > min_loss * 1.1:
            print('[Info]: loss overfit.')
            break

    # ------ last log
    loss_dict = optim.get_loss_dict(
        update_state=False, mat_assemble_method=state_mat_assemble_method, with_debug=with_debug
    )
    loss_sum = np.sum(list(loss_dict.values()))
    print(f"[Debug]: Step_{step} loss_fluid:{loss_sum:.3f}")
    desc = f'[iter:{step}] '
    for loss_n, loss_f in loss_dict.items():
        desc += f'{loss_n}:{loss_f:.3f} '
    desc += f'loss:{loss_sum:.3f}'
    print(desc)
    optim.log(step)

    return True


def debug_non_constraint():
    optim = ShapeFluidOptimizer(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example03/model.msh',
        dim=2,
        result_dir='/home/admin123456/Desktop/work/topopt_exps/new_test/example03/cache',
    )
    optim.add_regularization(
        reg_type='volume_regularization',
        target_volume_rho=1.0,
        weight=2000.0
    )
    optim.define_control_wall({
        'inlet': {'marker': 13, 'is_free': False},
        'outlet': {'marker': 14, 'is_free': False},
        'wallFixed': {'marker': 15, 'is_free': False},
        'wallFree': {'marker': 16, 'is_free': True},
    })
    # optim.mesh.draw_mesh()

    optim.add_problem(
        problem_name='cond1',
        inlet_cfg={
            'inlet': {
                'marker': 13,
                'value': [3., 0., 0.]
            }
        },
        outlet_cfg={
            'outlet': {
                'marker': 14,
                'value': 0.0
            }
        },
        wall_cfg={
            'wallFixed': {
                'marker': 15
            },
            'wallFree': {
                'marker': 16
            }
        },
        constant_nu=0.01,
        fluid_type='stoke',
        loss_cfgs=[
            {
                'style': 'minimum_dissipation',
                'weight': 1.0
            }
        ],
        su2_cfg={
            'run_iter': 1000,
            'constant_viscosity': 0.01,
            'refit_by_fenics': False
        }
    )

    # # ------ test simulation
    # optim.simulate()
    # optim.log(0)

    # ------ test optimize
    run_optimize(
        optim=optim,
        max_iteration=150,
        state_mat_assemble_method='lift',
        adjoint_mat_assemble_method='lift',
        gradient_mat_assemble_method='Identity_row',  # 必须使用Identity_row否则边界条件不正确
        with_debug=False
    )


def debug_constraint():
    obs_mesh = pyvista.read('/home/admin123456/Desktop/work/topopt_exps/new_test/example04/obs.stl')

    optim = ShapeFluidOptimizer(
        mesh_file='/home/admin123456/Desktop/work/topopt_exps/new_test/example04/model.msh',
        dim=2,
        result_dir='/home/admin123456/Desktop/work/topopt_exps/new_test/example04/cache',
    )
    optim.add_regularization(
        reg_type='volume_regularization',
        target_volume_rho=0.4,
        weight=50.0
    )
    optim.define_control_wall({
        'inlet': {'marker': 35, 'is_free': False},
        'outlet': {'marker': 36, 'is_free': False},
        'wall': {'marker': 37, 'is_free': True},
    })

    optim.add_problem(
        problem_name='cond1',
        inlet_cfg={
            'inlet': {
                'marker': 35,
                'value': [1., 0., 0.]
            }
        },
        outlet_cfg={
            'outlet': {
                'marker': 36,
                'value': 0.0
            }
        },
        wall_cfg={
            'wall': {
                'marker': 37
            }
        },
        constant_nu=0.01,
        fluid_type='stoke',
        loss_cfgs=[
            {
                'style': 'minimum_dissipation',
                'weight': 1.0
            }
        ],
        su2_cfg={
            'run_iter': 1000,
            'constant_viscosity': 0.01,
            'refit_by_fenics': False
        }
    )

    run_optimize(
        optim=optim,
        max_iteration=300,
        state_mat_assemble_method='lift',
        adjoint_mat_assemble_method='lift',
        gradient_mat_assemble_method='Identity_row',
        with_debug=False,
        obs_pcd=np.asarray(obs_mesh.points),
        obs_r=0.1,
    )


if __name__ == '__main__':
    # debug_non_constraint()
    debug_constraint()
