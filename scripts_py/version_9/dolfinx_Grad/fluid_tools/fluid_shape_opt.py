import numpy as np
import dolfinx
import os
import shutil
import pyvista
import ufl
from ufl import sym, grad, nabla_grad, dot, inner, div, Identity
from typing import List, Union, Dict, Tuple, Callable
import ctypes
import pickle

from ..recorder_utils import VTKRecorder, TensorBoardRecorder
from ..dolfinx_utils import MeshUtils, AssembleUtils, BoundaryUtils
from ..equation_solver import LinearProblemSolver, NonLinearProblemSolver
from ..lagrange_method.type_database import create_shape_problem, create_state_problem, ShapeDataBase
from ..lagrange_method.problem_state import StateProblem
from ..lagrange_method.cost_functions import CostFunctional_types
from ..lagrange_method.solver_optimize import OptimalShapeProblem
from ..lagrange_method.shape_regularization import ShapeRegularization
from ..remesh_helper import MeshDeformationRunner
from ..optimizer_utils import CostConvergeHandler, CostWeightHandler
from ..vis_mesh_utils import VisUtils
from .dolfin_simulator import DolfinSimulator
from .openfoam_simulator import OpenFoamSimulator
from ..collision_objs import MeshCollisionObj, ObstacleCollisionObj
from ..surface_fields import type_conflict_regulariztions
from ..remesh_helper import ReMesher
from ..simulator_convert import CrossSimulatorUtil


# todo
#  1.是不是使用不同的弹性密度能更凸显有效的变形而不是体积变形
#  2.平衡流量要做测试
#  3.网格被攻击了，remesh操作可能不可少

class FluidConditionalModel(object):
    """
    1. mesh不可以有90度角，否则Assemble后的矩阵有奇异
    2. msh文件中，每一个entity(curve, surface等)只能在一个physical group，否则xdmf会有歧义
    """

    def __init__(
            self,
            bry_markers: List[int],
            deformation_cfg: Dict,
            point_radius: float,
            run_strategy_cfg: dict,
            obs_avoid_cfg: dict,
            simulator: Union[OpenFoamSimulator],
            nu_value: float,
            save_opt_cfg: dict,
            velocity_order: int = 1,
            pressure_order: int = 1,
            nu_order: int = 1,
            tag_name: str = None,
            regularization_filter=['SparsePointsRegularization'],
    ):
        self.simulator = simulator
        self.name = self.simulator.name
        self.domain = self.simulator.domain
        self.cell_tags = self.simulator.cell_tags
        self.facet_tags = self.simulator.facet_tags
        self.tdim, self.fdim = self.simulator.tdim, self.simulator.fdim
        self.n_vec = MeshUtils.define_facet_norm(self.domain)
        self.ds = MeshUtils.define_ds(self.domain, self.facet_tags)

        self.W = dolfinx.fem.FunctionSpace(
            self.domain, ufl.MixedElement([
                ufl.VectorElement("Lagrange", self.domain.ufl_cell(), velocity_order),
                ufl.FiniteElement("Lagrange", self.domain.ufl_cell(), pressure_order)
            ])
        )
        self.W0, self.W1 = self.W.sub(0), self.W.sub(1)
        self.V, self.V_to_W_dofs = self.W0.collapse()
        self.Q, self.Q_to_W_dofs = self.W1.collapse()
        self.V_mapping_space = dolfinx.fem.VectorFunctionSpace(self.domain, ("CG", 1))
        self.Q_mapping_space = dolfinx.fem.FunctionSpace(self.domain, ("CG", 1))
        self.nu_function_space = dolfinx.fem.FunctionSpace(self.domain, ("CG", nu_order))

        self.up = dolfinx.fem.Function(self.W, name='state')
        self.u, self.p = ufl.split(self.up)
        self.vq = dolfinx.fem.Function(self.W, name='adjoint')
        v, q = ufl.split(self.vq)
        f = dolfinx.fem.Constant(self.domain, np.zeros(self.tdim))
        if isinstance(self.simulator, DolfinSimulator):
            self.nu = dolfinx.fem.Constant(self.domain, nu_value)
        elif isinstance(self.simulator, OpenFoamSimulator):
            self.nu = dolfinx.fem.Function(self.nu_function_space)
            self.nu.x.array[:] = nu_value
        else:
            raise ValueError("[ERROR]: Non-Valid Simulator")
        self.nu_value = nu_value

        self.F_form = (
                self.nu * inner(grad(self.u), grad(v)) * ufl.dx
                + inner(grad(self.u) * self.u, v) * ufl.dx
                - inner(self.p, div(v)) * ufl.dx
                + inner(div(self.u), q) * ufl.dx
                - inner(f, v) * ufl.dx
        )

        self.V_S = dolfinx.fem.FunctionSpace(self.domain, self.domain.ufl_domain().ufl_coordinate_element())
        self.energy_loss_form = dolfinx.fem.form(ufl.inner(grad(self.u), grad(self.u)) * ufl.dx)

        self.bcs_info_state = []
        self.bcs_info_control = []
        self.state_system: StateProblem = None
        self.control_problem: ShapeDataBase = None
        self.cost_weight: dict = {}
        self.opt_problem: OptimalShapeProblem = None
        self.deformation_handler = MeshDeformationRunner(self.domain, **deformation_cfg)

        # ------ conflict relative
        self.tag_name = tag_name if (tag_name is not None) else self.name
        self.mesh_obj = MeshCollisionObj(
            self.tag_name, self.domain, self.facet_tags, self.cell_tags, bry_markers, point_radius
        )
        self.mesh_obj.update_tree()
        self.conflict_regularization: type_conflict_regulariztions = None

        # ------ solving parameters
        self.solver_vars = {}
        self.run_strategy_cfg, self.obs_avoid_cfg = self.check_cfg(run_strategy_cfg, obs_avoid_cfg)
        self.regularization_filter = regularization_filter

        self.save_opt_cfg = save_opt_cfg
        self.save_opt_cfg['vertex_indices'] = ReMesher.reconstruct_vertex_indices(
            orig_msh_file=self.save_opt_cfg['orig_msh_file'], domain=self.domain
        )

    def check_cfg(self, run_strategy_cfg, obs_avoid_cfg):
        assert obs_avoid_cfg['bbox_rho'] < 1.0 and obs_avoid_cfg['bbox_w_lower'] > 0.0
        if obs_avoid_cfg['method'] == 'sigmoid_v1':
            obs_avoid_cfg['length_shift'] = run_strategy_cfg['max_step_limit']
            obs_avoid_cfg['length_scale'] = obs_avoid_cfg.get('length_scale', 1.0)
        elif obs_avoid_cfg['method'] == 'relu_v1':
            obs_avoid_cfg['length_shift'] = run_strategy_cfg['max_step_limit']
            obs_avoid_cfg['length_scale'] = obs_avoid_cfg.get('length_scale', 1.0)
        else:
            raise NotImplementedError
        return run_strategy_cfg, obs_avoid_cfg

    def add_state_boundary(self, name: str, value: Union[float, int, Callable], marker: int, is_velocity: bool):
        if is_velocity:
            bc_value = DolfinSimulator.get_boundary_function(name, value, self.V)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W0, self.V), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W0)
            self.bcs_info_state.append((bc, self.W0, bc_dof, bc_value))
        else:
            bc_value = DolfinSimulator.get_boundary_function(name, value, self.Q)
            bc_dof = MeshUtils.extract_entity_dofs(
                (self.W1, self.Q), self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
            )
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, self.W1)
            self.bcs_info_state.append((bc, self.W1, bc_dof, bc_value))

    def add_control_boundary(self, name: str, value: Union[float, int, Callable], marker: int):
        bc_value = DolfinSimulator.get_boundary_function(name, value, self.V_S)
        bc_dof = MeshUtils.extract_entity_dofs(
            self.V_S, self.fdim, MeshUtils.extract_facet_entities(self.domain, self.facet_tags, marker)
        )
        bc = dolfinx.fem.dirichletbc(bc_value, bc_dof, None)
        self.bcs_info_control.append((bc, self.V_S, bc_dof, bc_value))

    def state_initiation(
            self,
            snes_option={}, snes_criterion={},
            state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
            gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    ):
        self.state_system = StateProblem([
            create_state_problem(
                name='state_1',
                F_form=self.F_form,
                state=self.up,
                adjoint=self.vq,
                is_linear=False,
                bcs_info=self.bcs_info_state,
                state_ksp_option=state_ksp_option,
                adjoint_ksp_option=adjoint_ksp_option,
                snes_option=snes_option,
                snes_criterion=snes_criterion
            )
        ])

        self.control_problem = create_shape_problem(
            domain=self.domain,
            bcs_info=self.bcs_info_control,
            gradient_ksp_option=gradient_ksp_option
        )

    def optimization_initiation(
            self,
            cost_functional_list: List[CostFunctional_types],
            cost_weight: Dict,
            shapeRegularization: ShapeRegularization,
            scalar_product_method,
            conflict_regularization: type_conflict_regulariztions
    ):
        shapeRegularization.regularization_list.append(conflict_regularization)

        self.cost_weight = cost_weight
        self.opt_problem = OptimalShapeProblem(
            state_system=self.state_system,
            shape_problem=self.control_problem,
            shape_regulariztions=shapeRegularization,
            cost_functional_list=cost_functional_list,
            scalar_product=None,
            scalar_product_method=scalar_product_method
        )
        self.conflict_regularization = conflict_regularization

    # ------------ conflict related functions
    @staticmethod
    def scale_search_range(length, length_shift=0.0, length_scale=1.0):
        return (length + length_shift) * length_scale

    def find_conflicted_bbox(self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj]):
        other_objs = []
        other_objs.extend(obs_objs)
        other_objs.extend(mesh_objs)

        bbox_infos = {}
        for other_obj in other_objs:
            conflict_length = self.mesh_obj.point_radius + other_obj.point_radius
            search_range = self.scale_search_range(
                conflict_length, self.obs_avoid_cfg['length_shift'], self.obs_avoid_cfg['length_scale']
            )
            bbox_w = np.maximum(search_range * self.obs_avoid_cfg['bbox_rho'], self.obs_avoid_cfg['bbox_w_lower'])

            if isinstance(other_obj, ObstacleCollisionObj):
                obs_infos = other_obj.find_conflict_bbox_infos(self.mesh_obj.get_bry_coords(), search_range, bbox_w)

            elif isinstance(other_obj, MeshCollisionObj):
                if other_obj.name == self.tag_name:
                    continue
                obs_infos = other_obj.find_conflict_bbox_infos(self.mesh_obj.get_bry_coords(), search_range, bbox_w)

            else:
                raise ValueError("[ERROR]: Non-Valid Class")
            bbox_infos.update(obs_infos)
        return bbox_infos

    def update_conflict_regularization(self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj]):
        bbox_infos = self.find_conflicted_bbox(obs_objs, mesh_objs)
        if len(bbox_infos) > 1000:
            raise ValueError(f"[ERROR] Conflict Box are too much {len(bbox_infos)}")
        # else:
        #     print(f"[INFO]: Find {len(bbox_infos)} Conflict Boxes")

        self.conflict_regularization.update_expression(bbox_infos, self.mesh_obj.point_radius)
        self.opt_problem.gradient_system.update_gradient_equations()

        # # ---------
        # if len(bbox_infos) > 0:
        #     plt = pyvista.Plotter()
        #     vis_meshes = self.conflict_regularization._plot_bbox_meshes(bbox_infos, True, True)
        #     for vis_mesh in vis_meshes:
        #         plt.add_mesh(vis_mesh, style='wireframe')
        #     plt.add_mesh(VisUtils.convert_to_grid(self.domain), style='wireframe')
        #     plt.show()
        # # ---------

    # ------------ deformation related functions
    def obstacle_constraint_move(
            self, direction_np: np.ndarray, step_size: float,
            obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj], revert_move=False
    ):
        other_objs = []
        other_objs.extend(obs_objs)
        other_objs.extend(mesh_objs)

        while True:
            displacement_np = direction_np * step_size
            MeshUtils.move(self.domain, displacement_np)
            self.mesh_obj.update_tree()

            success_flag = True
            for other_obj in other_objs:
                if isinstance(other_obj, ObstacleCollisionObj):
                    conflict_idxs = self.mesh_obj.find_conflict_bry_nodes(
                        other_obj.coords, other_obj.point_radius + self.mesh_obj.point_radius
                    )

                elif isinstance(other_obj, MeshCollisionObj):
                    if other_obj.name == self.tag_name:
                        continue

                    conflict_idxs = self.mesh_obj.find_conflict_bry_nodes(
                        other_obj.get_bry_coords(), other_obj.point_radius + self.mesh_obj.point_radius
                    )

                else:
                    raise ValueError("[ERROR] Non-Valid Class")

                if len(conflict_idxs) > 0:
                    success_flag = False
                    break

            if success_flag:
                if revert_move:
                    MeshUtils.move(self.domain, displacement_np * -1.0)
                break

            else:
                MeshUtils.move(self.domain, displacement_np * -1.0)

                step_size = step_size * self.run_strategy_cfg['beta_rho']
                max_deformation = np.max(np.linalg.norm(step_size * direction_np, ord=2, axis=1))
                if max_deformation < self.run_strategy_cfg['deformation_lower']:
                    break

        return success_flag, step_size

    # ------------ gradient related functions
    def compute_simulation(self, **kwargs):
        if isinstance(self.simulator, DolfinSimulator):
            res_dict = self.simulator.simulate_navier_stoke_equation(
                snes_option=self.state_system.state_problems[0].snes_option,
                criterion=self.state_system.state_problems[0].snes_criterion,
                ksp_option=self.state_system.state_problems[0].state_ksp_option,
                guess_up=None,
                **kwargs
            )
            if not NonLinearProblemSolver.is_converged(res_dict['converged_reason']):
                raise ValueError(f"[ERROR] State Problem KSP Fail Converge {res_dict['converged_reason']}")

            # ------ convert simulator result to TaylorHood
            simulator_up: dolfinx.fem.Function = self.simulator.equation_map['navier_stoke']['up']
            u_fun, p_fun = simulator_up.sub(0).collapse(), simulator_up.sub(1).collapse()

            u_tmp, p_tmp = dolfinx.fem.Function(self.V), dolfinx.fem.Function(self.Q)
            u_tmp.interpolate(u_fun)
            p_tmp.interpolate(p_fun)

        elif isinstance(self.simulator, OpenFoamSimulator):
            tmp_dir = os.path.join(self.simulator.cache_dir, 'openfoam_simulation')
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.mkdir(tmp_dir)

            new_msh_file = os.path.join(tmp_dir, 'model.msh')
            ReMesher.convert_domain_to_new_msh(
                self.save_opt_cfg['orig_msh_file'], new_msh_file,
                self.domain, self.tdim, self.save_opt_cfg['vertex_indices']
            )
            res = self.simulator.run_simulate_process(tmp_dir, new_msh_file, True)
            res_vtk: pyvista.UnstructuredGrid = res['res_vtk']

            v_fun, p_fun, nut_fun = CrossSimulatorUtil.convert_to_simple_function(
                self.domain, coords=res_vtk.points, value_list=[
                    res_vtk.point_data['U'], res_vtk.point_data['p'], res_vtk.point_data['nut']
                ]
            )
            u_tmp, p_tmp = dolfinx.fem.Function(self.V), dolfinx.fem.Function(self.Q)
            u_tmp.interpolate(v_fun)
            p_tmp.interpolate(p_fun)
            self.nu.vector.aypx(0.0, nut_fun.vector)
            self.nu.x.array[:] = self.nu.x.array + self.nu_value

            shutil.rmtree(tmp_dir)
            del nut_fun

        else:
            raise ValueError("[ERROR]: Non-Valid Simulator")

        self.up.x.array[self.V_to_W_dofs] = u_tmp.x.array.copy()
        self.up.x.array[self.Q_to_W_dofs] = p_tmp.x.array.copy()
        del u_tmp, p_tmp

        # grid = VisUtils.convert_to_grid(self.domain)
        # VisUtils.show_arrow_res_vtk(grid, self.up.sub(0).collapse(), self.V, scale=0.001).show()

    def compute_shape_deformation(
            self,
            detect_cost_valid_func: Callable, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj],
            **kwargs
    ):
        self.compute_simulation(**kwargs)
        shape_grad: dolfinx.fem.Function = self.opt_problem.compute_gradient(
            self.domain.comm,
            state_kwargs={'with_debug': kwargs.get('with_debug', False)},
            adjoint_kwargs={'with_debug': kwargs.get('with_debug', False)},
            gradient_kwargs={
                'with_debug': kwargs.get('with_debug', False),
                'A_assemble_method': 'Identity_row'
            },
            update_state=False, update_adjoint=True
        )

        shape_grad_np = shape_grad.x.array
        shape_grad_np = shape_grad_np * -1.0
        direction_np = np.zeros(self.domain.geometry.x.shape)
        direction_np[:, :self.tdim] = shape_grad_np.reshape((-1, self.tdim))

        norm_max = np.max(np.linalg.norm(direction_np, axis=1))
        if norm_max > self.run_strategy_cfg['max_step_limit']:
            direction_np = direction_np * self.run_strategy_cfg['max_step_limit'] / norm_max

        # # ------ Just For Debug
        # grid = VisUtils.convert_to_grid(self.domain)
        # grid['grad_test'] = direction_np
        # VisUtils.show_arrow_from_grid(grid, 'grad_test', scale=1.0).show()
        # raise ValueError
        # # ------

        # ------ refit to safe step size based on constrained
        step_size = self.run_strategy_cfg['init_stepSize']
        if self.obs_avoid_cfg['method'] == 'sigmoid_v1':
            success_flag, step_size = self.obstacle_constraint_move(
                direction_np, step_size, obs_objs=obs_objs, mesh_objs=mesh_objs, revert_move=True
            )
        elif self.obs_avoid_cfg['method'] == 'relu_v1':
            success_flag = True
        else:
            raise NotImplementedError

        if not success_flag:
            return False, (None, None)

        # ------ refit to reasonable deformation
        success_flag, step_size = self.deformation_handler.move_mesh_by_line_search(
            direction_np, max_iter=10,
            init_stepSize=step_size,
            stepSize_lower=self.run_strategy_cfg['stepSize_lower'],
            detect_cost_valid_func=detect_cost_valid_func,
            max_step_limit=self.run_strategy_cfg['max_step_limit'],
            revert_move=True
        )

        return success_flag, (direction_np, step_size)

    # ------------ iteration related functions
    def init_optimize_state(self, record_dir, logger_dicts={}, with_debug=False):
        # ------ Step 1: Create Record Directory
        simulate_dir = os.path.join(record_dir, 'simulate')
        if os.path.exists(simulate_dir):
            shutil.rmtree(simulate_dir)
        os.mkdir(simulate_dir)
        u_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))

        # ------ Step 2: Init Calculation and log first step
        self.compute_simulation(with_debug=with_debug)

        for regularization_term in self.opt_problem.shape_regulariztions.regularization_list:
            if regularization_term.name in self.regularization_filter:
                continue
            cost_value = regularization_term.compute_objective() / regularization_term.mu
            regularization_term.update_weight(regularization_term.mu / cost_value)
            print(f"[Info] Regularization Weight {regularization_term.name}: "
                  f"cost:{cost_value} weight:{regularization_term.mu}")

        weight_handler = CostWeightHandler()
        for cost_func in self.opt_problem.cost_functional_list:
            weight_handler.add_cost(cost_func.name, cost_func.evaluate())
        weight_handler.compute_weight(self.cost_weight)
        for cost_func in self.opt_problem.cost_functional_list:
            cost_func.update_scale(weight_handler.get_weight(cost_func.name))
            cost_value, cost_num = weight_handler.get_cost(cost_func.name)
            print(f"[Info] Cost Weight {cost_func.name}: "
                  f"cost:{cost_value} num:{cost_num} weight:{cost_func.weight_value}")

        init_loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=False)
        print(f"[INFO] Init loss: {init_loss}")
        loss_storge_ctype = ctypes.c_double(init_loss)

        def detect_cost_valid_func(tol=self.run_strategy_cfg['loss_tol_rho']):
            self.compute_simulation(with_debug=with_debug)
            loss = self.opt_problem.evaluate_cost_functional(
                self.domain.comm, update_state=False, with_debug=with_debug
            )
            is_valid = loss < loss_storge_ctype.value + np.abs(loss_storge_ctype.value) * tol
            # print('Debug_remove: ', loss, is_valid)
            return is_valid

        self.solver_vars.update({
            'u_recorder': u_recorder,
            'loss_storge': loss_storge_ctype,
            'cost_valid_func': detect_cost_valid_func,
            'logger_dicts': logger_dicts,
        })
        log_dict = self.log_step(logger_dicts, step=0)

        return log_dict

    def compute_iteration_deformation(
            self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj], step: int, **kwargs
    ):
        self.update_conflict_regularization(obs_objs, mesh_objs)
        success_flag, (direction_np, step_size) = self.compute_shape_deformation(
            obs_objs=obs_objs, mesh_objs=mesh_objs,
            detect_cost_valid_func=self.solver_vars['cost_valid_func'],
            **kwargs
        )
        return {'state': success_flag, 'diffusion': direction_np * step_size}

    def move_mesh(self, diffusion: np.ndarray, check_intersection=False):
        if diffusion.shape != self.deformation_handler.shape:
            raise ValueError("[ERROR]: Shape UnCompatible")

        MeshUtils.move(self.domain, diffusion)

        if check_intersection:
            is_intersections = self.deformation_handler.detect_collision(self.domain)  # 对于多场景而言时必须的
        else:
            is_intersections = False

        res_dict = {'state': not is_intersections}
        if is_intersections:
            MeshUtils.move(self.domain, diffusion * -1.0)  # revert mesh
        else:
            max_deformation = np.max(np.linalg.norm(diffusion, ord=2, axis=1))
            res_dict['is_converge'] = max_deformation < self.run_strategy_cfg['deformation_lower']

        self.mesh_obj.update_tree()
        return res_dict

    def update_optimize_state(self, step, with_log_info=True):
        # --- Necessary Step
        # self.compute_simulation()
        loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=False)
        self.solver_vars['loss_storge'].value = loss

        # --- Get Log Info
        if with_log_info:
            log_dict = self.log_step(logger_dicts=self.solver_vars['logger_dicts'], step=step)
            log_dict.update({'loss': {f"{self.name}_loss": loss}})
            return loss, log_dict

        return loss

    def implicit_solve(
            self,
            record_dir, log_recorder: TensorBoardRecorder,
            obs_objs: List[ObstacleCollisionObj] = [],
            mesh_objs: List[MeshCollisionObj] = [],
            logger_dicts={}, with_debug=False,
            **kwargs
    ):
        log_dict = self.init_optimize_state(
            record_dir=record_dir, logger_dicts=logger_dicts, with_debug=with_debug
        )
        self.write_log_tensorboard(log_recorder, [log_dict], step=0)

        step = 0
        best_loss: float = np.inf
        best_loss_tol = 0.05
        while True:
            step += 1

            res_dict = self.compute_iteration_deformation(obs_objs, mesh_objs, step, **kwargs)
            if not res_dict['state']:
                print(f"[INFO]: {self.tag_name} without reasonable deformation")
                return -1

            res_dict = self.move_mesh(res_dict['diffusion'], check_intersection=False)
            # if not res_dict['state']:
            #     print(f"[INFO]: {self.tag_name} Move Mesh Fail")
            #     return -1

            is_converge = res_dict['is_converge']
            loss, log_dict = self.update_optimize_state(with_log_info=True, step=step)
            best_loss = np.minimum(best_loss, loss)

            self.write_log_tensorboard(log_recorder, [log_dict], step=step)

            if is_converge:
                break

            if np.any(loss > best_loss * (1.0 + best_loss_tol)):
                break

            if step > self.run_strategy_cfg['max_iter']:
                break

    def log_step(self, logger_dicts: dict, step):
        self.solver_vars['u_recorder'].write_function(self.up.sub(0).collapse(), step)

        res_dict = {}
        for tag_name in logger_dicts.keys():
            log_dict = logger_dicts[tag_name]

            data_cell = {}
            for marker_name in log_dict.keys():
                data_cell[marker_name] = AssembleUtils.assemble_scalar(log_dict[marker_name])
            res_dict.update({tag_name: data_cell})

        res_dict.update({
            'energy_loss': {f"{self.name}_energy_loss": AssembleUtils.assemble_scalar(self.energy_loss_form)}
        })
        return res_dict

    @staticmethod
    def write_log_tensorboard(log_recorder: TensorBoardRecorder, res_dicts: List[dict], step):
        for tag_name in list(res_dicts[0].keys()):
            data_cell = {}
            for res_dict in res_dicts:
                if isinstance(res_dict[tag_name], dict):
                    data_cell.update(res_dict[tag_name])
                else:
                    raise ValueError("[ERROR]: Wrong Format")

            if len(data_cell) == 1:
                log_recorder.write_scalar(tag_name, list(data_cell.values())[0], step)
            else:
                log_recorder.write_scalars(tag_name, data_cell, step)

    def save_msh(self, with_debug):
        print('[INFO]: Saving Result')
        self.compute_simulation(with_debug=with_debug)

        save_dir = self.save_opt_cfg['save_dir']
        ReMesher.convert_domain_to_new_msh(
            orig_msh_file=self.save_opt_cfg['orig_msh_file'],
            new_msh_file=os.path.join(save_dir, 'opt_model.msh'),
            domain=self.domain, dim=self.tdim, vertex_indices=self.save_opt_cfg['vertex_indices']
        )

        MeshUtils.save_XDMF(
            output_file=os.path.join(save_dir, 'opt_model.xdmf'), domain=self.domain,
            cell_tags=self.cell_tags, facet_tags=self.facet_tags
        )

        with open(os.path.join(save_dir, 'velocity.pkl'), 'wb') as f:
            pickle.dump(self.up.sub(0).collapse().x.array, f)

        with open(os.path.join(save_dir, 'pressure.pkl'), 'wb') as f:
            pickle.dump(self.up.sub(1).collapse().x.array, f)


class FluidShapeRecombineLayer(object):
    def __init__(self, tag_name):
        self.tag_name = tag_name
        assert tag_name is not None

        self.opt_dicts: Dict[str, FluidConditionalModel] = {}

    @property
    def name(self):
        return self.tag_name

    def add_condition_opt(self, opt: FluidConditionalModel):
        self.opt_dicts[opt.name] = opt

    def compute_iteration_deformation(
            self,
            obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj],
            step: int, diffusion_method='diffusion_loss_weight', **kwargs
    ):
        res_dict_list = {}
        for name in self.opt_dicts.keys():
            opt = self.opt_dicts[name]
            res_dict = opt.compute_iteration_deformation(obs_objs, mesh_objs, step=step, **kwargs)
            if not res_dict['state']:
                return {'state': False}

            res_dict_list[name] = {
                'diffusion': res_dict['diffusion'],
                'loss': opt.solver_vars['loss_storge'].value
            }

        diffusion = FluidShapeRecombineLayer.merge_diffusion(res_dict_list, method=diffusion_method)
        return {'state': True, 'diffusion': diffusion}

    def move_mesh(self, diffusion: np.ndarray):
        is_converge = True
        for i, name in enumerate(self.opt_dicts.keys()):
            if i == 0:
                res_dict = self.opt_dicts[name].move_mesh(diffusion, check_intersection=True)
            else:
                res_dict = self.opt_dicts[name].move_mesh(diffusion, check_intersection=False)
            if not res_dict['state']:
                return {'state': False}

            is_converge = is_converge and res_dict['is_converge']

        return {'state': True, 'is_converge': is_converge}

    def update_optimize_state(self, with_log_info=True, step=None):
        if with_log_info:
            log_list = []
        loss_list = []

        for name in self.opt_dicts.keys():
            if with_log_info:
                loss, log_dict = self.opt_dicts[name].update_optimize_state(with_log_info=with_log_info, step=step)
                log_list.append(log_dict)
            else:
                loss = self.opt_dicts[name].update_optimize_state(with_log_info=with_log_info, step=step)
            loss_list.append(loss)

        if with_log_info:
            return loss_list, log_list
        else:
            return loss_list

    @staticmethod
    def merge_diffusion(opt_res_dict: dict, method='diffusion_weight'):
        if method == 'diffusion_weight':
            weights = []
            for name in opt_res_dict.keys():
                weights.append(
                    np.abs(np.linalg.norm(opt_res_dict[name]['diffusion'], ord=2, axis=1, keepdims=True))
                )
            weights = np.concatenate(weights, axis=1)
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)

        elif method == 'loss_weight':
            weights = []
            for name in opt_res_dict.keys():
                weights.append(opt_res_dict[name]['loss'])
            weights = np.array(weights)
            weights = weights / weights.sum()

        elif method == 'diffusion_loss_weight':
            diffusion_weights = []
            for name in opt_res_dict.keys():
                diffusion_weights.append(
                    np.abs(np.linalg.norm(opt_res_dict[name]['diffusion'], ord=2, axis=1, keepdims=True))
                )
            diffusion_weights = np.concatenate(diffusion_weights, axis=1)
            diffusion_weights = diffusion_weights / (diffusion_weights.sum(axis=1, keepdims=True) + 1e-8)

            loss_weights = []
            for name in opt_res_dict.keys():
                loss_weights.append(opt_res_dict[name]['loss'])
            loss_weights = np.array(loss_weights)
            loss_weights = loss_weights / loss_weights.sum()

            weights = diffusion_weights * loss_weights.reshape((1, -1))

        else:
            raise NotImplementedError("[ERROR]: Non-Valid Method")

        diffusion = 0.0
        if weights.ndim == 1:
            for i, name in enumerate(opt_res_dict.keys()):
                diffusion += weights[i] * opt_res_dict[name]['diffusion']
        elif weights.ndim == 2:
            for i, name in enumerate(opt_res_dict.keys()):
                diffusion += weights[:, i: i + 1] * opt_res_dict[name]['diffusion']
        else:
            raise NotImplementedError("[ERROR]: Non-Valid Method")

        return diffusion

    def save_msh(self):
        self.opt_dicts[self.opt_dicts.keys()[0]].save_msh()
