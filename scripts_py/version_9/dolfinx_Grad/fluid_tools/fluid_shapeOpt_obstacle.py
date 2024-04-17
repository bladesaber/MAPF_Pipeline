import os
import shutil
from typing import Dict, List, Callable, Literal
import dolfinx
import numpy as np
import ctypes
import pyvista

from .fluid_shapeOpt_simple import FluidShapeOptSimple
from ..collision_objs import MeshCollisionObj, ObstacleCollisionObj
from ..dolfinx_utils import MeshUtils
from ..recorder_utils import VTKRecorder, TensorBoardRecorder
from ..optimizer_utils import CostWeightHandler
from ..dolfinx_utils import AssembleUtils
from ..vis_mesh_utils import VisUtils
from ..lagrange_method.cost_functions import CostFunctional_types
from ..lagrange_method.shape_regularization import ShapeRegularization
from ..surface_fields import type_conflict_regulariztions

# TODO
"""
1. 计算变形度
2. 根据变形度计算变形优先级(X)
3. 根据优先级产生权重场
    （这是不够的，我觉得需要使用场来模拟一个虚拟的斥力,这个力对于大权重mesh为0，对于小权重mesh为斥力，即是小权重mesh让位，
    但这不完备，因为大权重mesh不一定挤压小权重mesh）
4. 多次尝试不同优先级顺序来平移mesh
4. 检测范围内干涉，固定这些干涉面(错误)

所有cfg的name都不可重复
"""


class FluidObstacleAvoidModel(FluidShapeOptSimple):
    def __init__(
            self,
            name,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            Re: float,
            bry_markers: List[int],
            deformation_cfg: Dict,
            point_radius: float,
            run_strategy_cfg: dict,
            obs_avoid_cfg: dict,
            isStokeEqu=False,
            tag_name: str = None,
    ):
        super().__init__(
            domain=domain,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            Re=Re,
            deformation_cfg=deformation_cfg,
            isStokeEqu=isStokeEqu
        )

        self.name = name
        self.solve_mapper = {}

        self.tag_name = tag_name if (tag_name is not None) else name
        self.mesh_obj = MeshCollisionObj(
            self.tag_name, self.domain, self.facet_tags, self.cell_tags, bry_markers, point_radius
        )
        self.mesh_obj.update_tree()

        self.conflict_regularization: type_conflict_regulariztions = None
        self.run_strategy_cfg, self.obs_avoid_cfg = self.parse_cfgs(run_strategy_cfg, obs_avoid_cfg)

    @staticmethod
    def scale_search_range(length, length_shift=0.0, length_scale=1.0):
        return (length + length_shift) * length_scale

    @staticmethod
    def parse_cfgs(run_strategy_cfg, obs_avoid_cfg):
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

    def get_shape_derivatives(self, detect_cost_valid_func: Callable, **kwargs):
        shape_grad: dolfinx.fem.Function = self.opt_problem.compute_gradient(
            self.domain.comm,
            state_kwargs={'with_debug': kwargs.get('with_debug', False)},
            adjoint_kwargs={'with_debug': kwargs.get('with_debug', False)},
            gradient_kwargs={
                'with_debug': kwargs.get('with_debug', False),
                'A_assemble_method': 'Identity_row'
            },
        )

        shape_grad_np = shape_grad.x.array
        shape_grad_np = shape_grad_np * -1.0
        direction_np = np.zeros(self.domain.geometry.x.shape)
        direction_np[:, :self.tdim] = shape_grad_np.reshape((-1, self.tdim))

        # # ------ Just For Debug
        # grid = VisUtils.convert_to_grid(self.domain)
        # grid['grad_test'] = direction_np
        # VisUtils.show_arrow_from_grid(grid, 'grad_test', scale=1.0).show()
        # raise ValueError
        # # ------

        success_flag, step_size = self.deformation_handler.move_mesh_by_line_search(
            direction_np, max_iter=10,
            init_stepSize=self.run_strategy_cfg['init_stepSize'],
            stepSize_lower=self.run_strategy_cfg['stepSize_lower'],
            detect_cost_valid_func=detect_cost_valid_func,
            max_step_limit=self.run_strategy_cfg['max_step_limit'],
            revert_move=True
        )

        return success_flag, (direction_np, step_size)

    def init_solve_cfg(self, record_dir, logger_dicts={}, with_debug=False):
        # ------ Step 1: Create Record Directory
        simulate_dir = os.path.join(record_dir, 'simulate')
        if os.path.exists(simulate_dir):
            shutil.rmtree(simulate_dir)
        os.mkdir(simulate_dir)
        u_recorder = VTKRecorder(os.path.join(simulate_dir, 'simulate_u.pvd'))

        # ------ Step 2: Init Calculation and log first step
        self.opt_problem.state_system.solve(self.domain.comm, with_debug=with_debug)
        if self.cost_weight is not None:
            weight_handler = CostWeightHandler()
            for cost_func in self.opt_problem.cost_functional_list:
                cost = cost_func.evaluate()
                weight_handler.add_cost(cost_func.name, cost)
            weight_handler.compute_weight(self.cost_weight)
            for cost_func in self.opt_problem.cost_functional_list:
                cost_func.update_scale(weight_handler.get_weight(cost_func.name))
                print(f"[Info] Cost Weight: {cost_func.name}: {weight_handler.get_weight(cost_func.name)}")

        init_loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=False)
        loss_storge_ctype = ctypes.c_double(init_loss)

        def detect_cost_valid_func(tol=self.run_strategy_cfg['loss_tol_rho']):
            loss = self.opt_problem.evaluate_cost_functional(
                self.domain.comm, update_state=True, with_debug=with_debug
            )
            is_valid = loss < loss_storge_ctype.value + np.abs(loss_storge_ctype.value) * tol
            return is_valid

        log_dict = self.log_step(logger_dicts)
        self.solve_mapper.update({
            'u_recorder': u_recorder,
            'loss_storge': loss_storge_ctype,
            'cost_valid_func': detect_cost_valid_func,
            'logger_dicts': logger_dicts,
        })
        return log_dict

    def optimization_initiation(
            self,
            cost_functional_list: List[CostFunctional_types],
            cost_weight: Dict,
            shapeRegularization: ShapeRegularization,
            scalar_product_method,
            conflict_regularization: type_conflict_regulariztions
    ):
        shapeRegularization.regularization_list.append(conflict_regularization)

        super().optimization_initiation(
            cost_functional_list=cost_functional_list,
            cost_weight=cost_weight,
            shapeRegularization=shapeRegularization,
            scalar_product_method=scalar_product_method
        )
        self.conflict_regularization = conflict_regularization

    def find_conflicted_bbox(self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj]):
        bbox_infos = {}
        for obs_obj in obs_objs:
            conflict_length = self.mesh_obj.point_radius + obs_obj.point_radius
            search_range = self.scale_search_range(
                conflict_length, self.obs_avoid_cfg['length_shift'], self.obs_avoid_cfg['length_scale']
            )
            bbox_w = np.maximum(conflict_length * self.obs_avoid_cfg['bbox_rho'], self.obs_avoid_cfg['bbox_w_lower'])

            obs_bbox_infos = obs_obj.find_conflict_bbox_infos(self.mesh_obj.get_bry_coords(), search_range, bbox_w)
            bbox_infos.update(obs_bbox_infos)

        for other_mesh_obj in mesh_objs:
            if other_mesh_obj.name == self.tag_name:
                continue

            conflict_length = self.mesh_obj.point_radius + other_mesh_obj.point_radius
            search_range = self.scale_search_range(
                conflict_length, self.obs_avoid_cfg['length_shift'], self.obs_avoid_cfg['length_scale']
            )
            bbox_w = np.maximum(conflict_length * self.obs_avoid_cfg['bbox_rho'], self.obs_avoid_cfg['bbox_w_lower'])

            mesh_bbox_infos = other_mesh_obj.find_conflict_bbox_infos(
                self.mesh_obj.get_bry_coords(), search_range, bbox_w
            )
            bbox_infos.update(mesh_bbox_infos)

        return bbox_infos

    def safe_move(
            self,
            direction_np: np.ndarray, step_size: float,
            obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj], revert_move=False
    ):
        while True:
            displacement_np = direction_np * step_size
            MeshUtils.move(self.domain, displacement_np)
            self.mesh_obj.update_tree()

            success_flag = True
            for obs_obj in obs_objs:
                conflict_idxs = self.mesh_obj.find_conflict_bry_nodes(
                    obs_obj.coords, obs_obj.point_radius + self.mesh_obj.point_radius
                )
                if len(conflict_idxs) > 0:
                    success_flag = False
                    break

            if success_flag:
                for mesh_obj in mesh_objs:
                    conflict_idxs = self.mesh_obj.find_conflict_bry_nodes(
                        mesh_obj.get_bry_coords(), mesh_obj.point_radius + self.mesh_obj.point_radius
                    )
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

    def update_conflict_regularization(self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj]):
        bbox_infos = self.find_conflicted_bbox(obs_objs, mesh_objs)
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

    def single_solve(
            self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj], step: int, **kwargs
    ):
        self.update_conflict_regularization(obs_objs, mesh_objs)

        grad_flag, (direction_np, step_size) = self.get_shape_derivatives(
            self.solve_mapper['cost_valid_func'], **kwargs
        )
        if not grad_flag:
            return {'state': grad_flag}

        res_dict = {'state': False}
        if self.obs_avoid_cfg['method'] == 'sigmoid_v1':
            move_flag, step_size = self.safe_move(
                direction_np=direction_np, step_size=step_size, obs_objs=obs_objs, mesh_objs=mesh_objs,
            )  # safe_move已包含update_tree()
            res_dict.update({'state': move_flag})

        elif self.obs_avoid_cfg['method'] == 'relu_v1':
            displacement_np = direction_np * step_size
            MeshUtils.move(self.domain, displacement_np)
            self.mesh_obj.update_tree()

            res_dict.update({'state': True})

        else:
            raise NotImplementedError

        max_deformation = np.max(np.linalg.norm(step_size * direction_np, ord=2, axis=1))
        res_dict.update({'is_converge': max_deformation < self.run_strategy_cfg['deformation_lower']})

        if res_dict['state']:
            loss, log_dict = self.update_opt_info(with_log_info=True, step=step)
            res_dict.update({'log_dict': log_dict})
            print(f"[###Info {step}] loss:{loss:.8f} step_size:{step_size}")

        return res_dict

    def update_opt_info(self, with_log_info=True, step=None):
        # --- Necessary Step
        loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=True)
        self.solve_mapper['loss_storge'].value = loss

        # --- Get Log Info
        if with_log_info:
            self.solve_mapper['u_recorder'].write_function(self.up.sub(0).collapse(), step)
            log_dict = self.log_step(logger_dicts=self.solve_mapper['logger_dicts'])
            log_dict.update({'loss': {f"{self.name}_loss": loss}})
            return loss, log_dict

        return loss

    def log_step(self, logger_dicts: dict):
        res_dict = {}
        for tag_name in logger_dicts.keys():
            log_dict = logger_dicts[tag_name]

            data_cell = {}
            for marker_name in log_dict.keys():
                data_cell[marker_name] = AssembleUtils.assemble_scalar(log_dict[marker_name])
            res_dict.update({tag_name: data_cell})

        res_dict.update({
            'energy': {f"{self.name}_energy": AssembleUtils.assemble_scalar(self.energy_form)},
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

    def solve(self, **kwargs):
        raise ValueError("[Error]: Depreciate For This Class.")


class FluidConditionalModel(FluidObstacleAvoidModel):
    def single_solve(
            self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj], step: int, **kwargs
    ):
        self.update_conflict_regularization(obs_objs, mesh_objs)

        grad_flag, (direction_np, step_size) = self.get_shape_derivatives(
            self.solve_mapper['cost_valid_func'], **kwargs
        )
        res_dict = {'state': False}
        if self.obs_avoid_cfg['method'] == 'sigmoid_v1':
            move_flag, step_size = self.safe_move(
                direction_np=direction_np, step_size=step_size, obs_objs=obs_objs, mesh_objs=mesh_objs,
                revert_move=True
            )
            res_dict.update({'state': move_flag, 'diffusion': direction_np * step_size})

        elif self.obs_avoid_cfg['method'] == 'relu_v1':
            res_dict.update({'state': True, 'diffusion': direction_np * step_size})

        else:
            raise NotImplementedError

        return res_dict

    def move_mesh(self, diffusion: np.ndarray):
        if diffusion.shape != self.deformation_handler.shape:
            raise ValueError("[ERROR]: Shape UnCompatible")

        MeshUtils.move(self.domain, diffusion)
        is_intersections = self.deformation_handler.detect_collision(self.domain)

        res_dict = {'state': not is_intersections}
        if is_intersections:
            MeshUtils.move(self.domain, diffusion * -1.0)  # revert mesh
        else:
            max_deformation = np.max(np.linalg.norm(diffusion, ord=2, axis=1))
            res_dict['is_converge'] = max_deformation < self.run_strategy_cfg['deformation_lower']
        self.mesh_obj.update_tree()
        return res_dict


class FluidShapeRecombineLayer(object):
    def __init__(self, tag_name):
        self.tag_name = tag_name
        assert tag_name is not None

        self.opt_dicts: Dict[str, FluidConditionalModel] = {}

    def add_condition_opt(self, opt: FluidConditionalModel):
        self.opt_dicts[opt.name] = opt

    def single_solve(
            self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj],
            step: int, diffusion_method='diffusion_loss_weight', **kwargs
    ):
        res_dict = {}
        for name in self.opt_dicts.keys():
            opt = self.opt_dicts[name]
            sub_res_dict = opt.single_solve(
                obs_objs=obs_objs, mesh_objs=mesh_objs, step=step, with_debug=kwargs.get("with_debug", False)
            )

            if not sub_res_dict['state']:
                return {'state': False}

            res_dict[name] = {
                'diffusion': sub_res_dict['diffusion'],
                'loss': opt.solve_mapper['loss_storge'].value
            }

        diffusion = FluidShapeRecombineLayer.merge_diffusion(res_dict, method=diffusion_method)
        return {'state': True, 'diffusion': diffusion}

    def move_mesh(self, diffusion: np.ndarray):
        is_converge = True
        for name in self.opt_dicts.keys():
            res_dict = self.opt_dicts[name].move_mesh(diffusion)
            if not res_dict['state']:
                return {'state': False}
            is_converge = is_converge and res_dict['is_converge']
        return {'state': True, 'is_converge': is_converge}

    def update_opt_info(self, with_log_info=True, step=None):
        if with_log_info:
            log_list = []
        loss_list = []

        for name in self.opt_dicts.keys():
            if with_log_info:
                loss, log_dict = self.opt_dicts[name].update_opt_info(with_log_info=with_log_info, step=step)
                log_list.append(log_dict)
            else:
                loss = self.opt_dicts[name].update_opt_info(with_log_info=with_log_info, step=step)
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
