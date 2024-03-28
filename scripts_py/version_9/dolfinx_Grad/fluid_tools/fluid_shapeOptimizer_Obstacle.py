import os
import shutil
from typing import Dict, List, Callable, Literal
import dolfinx
import numpy as np
import ctypes
import pyvista

from .fluid_shapeOptimizer_simple import FluidShapeOptSimple
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
2. 根据变形度计算变形优先级
3. 根据优先级依次变形
4. 检测范围内干涉，固定这些干涉面
"""


class FluidShapeFreeObsModel1(FluidShapeOptSimple):
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
            opt_strategy_cfg: dict = None,
            isStokeEqu=False,
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
        self.opt_strategy_cfg = self.parse_opt_strategy_cfg(opt_strategy_cfg)
        self.mesh_obj = MeshCollisionObj(name, self.domain, self.facet_tags, self.cell_tags, bry_markers, point_radius)
        self.conflict_regularization: type_conflict_regulariztions = None

    @staticmethod
    def scale_search_range(length, length_shift=0.0, length_scale=1.0):
        return (length + length_shift) * length_scale

    @staticmethod
    def parse_opt_strategy_cfg(opt_strategy_cfg=None):
        opt_strategy_cfg['method'] = opt_strategy_cfg.get('method', 'sigmoid_v1')
        opt_strategy_cfg['opt_tol_rho'] = opt_strategy_cfg.get('opt_tol_rho', 0.01)
        opt_strategy_cfg['max_step_limit'] = opt_strategy_cfg.get('max_step_limit', 0.1)
        opt_strategy_cfg['deformation_lower'] = opt_strategy_cfg.get('deformation_lower', 1e-2)

        opt_strategy_cfg['bbox_rho'] = opt_strategy_cfg.get('bbox_rho', 0.85)
        opt_strategy_cfg['bbox_w_lower'] = opt_strategy_cfg.get('bbox_w_lower', -0.1)
        assert opt_strategy_cfg['bbox_rho'] < 1.0 and opt_strategy_cfg['bbox_w_lower'] > 0.0

        if opt_strategy_cfg['method'] == 'sigmoid_v1':
            opt_strategy_cfg['beta_rho'] = opt_strategy_cfg.get('beta_rho', 0.75)
            opt_strategy_cfg['length_shift'] = opt_strategy_cfg['max_step_limit']
            opt_strategy_cfg['length_scale'] = 1.0

        elif opt_strategy_cfg['method'] == 'relu_v1':
            opt_strategy_cfg['length_shift'] = 0.0
            opt_strategy_cfg['length_scale'] = 1.0

        else:
            raise NotImplementedError

        return opt_strategy_cfg

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
            init_stepSize=self.opt_strategy_cfg['init_stepSize'],
            stepSize_lower=self.opt_strategy_cfg['stepSize_lower'],
            detect_cost_valid_func=detect_cost_valid_func,
            revert_move=True,
            max_step_limit=self.opt_strategy_cfg['max_step_limit']
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

        def detect_cost_valid_func():
            loss = self.opt_problem.evaluate_cost_functional(
                self.domain.comm, update_state=True, with_debug=with_debug
            )
            is_valid = (
                    loss < loss_storge_ctype.value +
                    np.abs(loss_storge_ctype.value) * self.opt_strategy_cfg['opt_tol_rho']
            )
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

    def find_conflicted_bbox(
            self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj], opt_strategy_cfg: dict
    ):
        bbox_infos = {}
        for obs_obj in obs_objs:
            conflict_length = self.mesh_obj.point_radius + obs_obj.point_radius
            search_range = self.scale_search_range(
                conflict_length, opt_strategy_cfg['length_shift'], opt_strategy_cfg['length_scale']
            )
            bbox_w = np.maximum(conflict_length * opt_strategy_cfg['bbox_rho'], opt_strategy_cfg['bbox_w_lower'])

            obs_bbox_infos = obs_obj.find_conflict_bbox_infos(self.mesh_obj.get_bry_coords(), search_range, bbox_w)
            bbox_infos.update(obs_bbox_infos)

        for other_mesh_obj in mesh_objs:
            conflict_length = self.mesh_obj.point_radius + other_mesh_obj.point_radius
            search_range = self.scale_search_range(
                conflict_length, opt_strategy_cfg['length_shift'], opt_strategy_cfg['length_scale']
            )
            bbox_w = np.maximum(conflict_length * opt_strategy_cfg['bbox_rho'], opt_strategy_cfg['bbox_w_lower'])

            mesh_bbox_infos = other_mesh_obj.find_conflict_bbox_infos(
                self.mesh_obj.get_bry_coords(), search_range, bbox_w
            )
            bbox_infos.update(mesh_bbox_infos)

        return bbox_infos

    def safe_move(
            self,
            direction_np: np.ndarray, step_size: float,
            obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj],
            opt_strategy_cfg: dict
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
                break
            else:
                MeshUtils.move(self.domain, displacement_np * -1.0)

                step_size = step_size * opt_strategy_cfg['beta_rho']
                max_deformation = np.max(np.linalg.norm(step_size * direction_np, ord=2, axis=1))
                if max_deformation < opt_strategy_cfg['deformation_lower']:
                    break

        return success_flag, step_size

    def single_solve(
            self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj], step: int, **kwargs
    ):
        bbox_infos = self.find_conflicted_bbox(obs_objs, mesh_objs, self.opt_strategy_cfg)
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

        grad_flag, (direction_np, step_size) = self.get_shape_derivatives(
            self.solve_mapper['cost_valid_func'], **kwargs
        )
        if not grad_flag:
            return {'state': grad_flag}

        res_dict = {'state': False}
        if self.opt_strategy_cfg['method'] == 'sigmoid_v1':
            move_flag, step_size = self.safe_move(
                direction_np=direction_np, step_size=step_size,
                obs_objs=obs_objs, mesh_objs=mesh_objs,
                opt_strategy_cfg=self.opt_strategy_cfg
            )

            max_deformation = np.max(np.linalg.norm(step_size * direction_np, ord=2, axis=1))
            res_dict.update({
                'state': move_flag,
                'is_converge': max_deformation < self.opt_strategy_cfg['deformation_lower'],
            })

        elif self.opt_strategy_cfg['method'] == 'relu_v1':
            displacement_np = direction_np * step_size
            MeshUtils.move(self.domain, displacement_np)
            self.mesh_obj.update_tree()

            max_deformation = np.max(np.linalg.norm(step_size * direction_np, ord=2, axis=1))
            res_dict.update({
                'state': True,
                'is_converge': max_deformation < self.opt_strategy_cfg['deformation_lower'],
            })

        else:
            raise NotImplementedError

        if res_dict['state']:
            loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=True)
            self.solve_mapper['loss_storge'].value = loss

            self.solve_mapper['u_recorder'].write_function(self.up.sub(0).collapse(), step)
            log_dict = self.log_step(logger_dicts=self.solve_mapper['logger_dicts'])
            log_dict.update({'loss': loss})
            res_dict.update({'log_dict': log_dict})

            print(f"[###Info {step}] loss:{loss:.8f} step_size:{step_size}")

        return res_dict

    def log_step(self, logger_dicts: dict):
        res_dict = {}
        for tag_name in logger_dicts.keys():
            log_dict = logger_dicts[tag_name]
            if len(log_dict) == 1:
                marker_name = list(log_dict.keys())[0]
                value = AssembleUtils.assemble_scalar(log_dict[marker_name])
                res_dict.update({marker_name: value})
            else:
                data_cell = {}
                for marker_name in log_dict.keys():
                    data_cell[marker_name] = AssembleUtils.assemble_scalar(log_dict[marker_name])
                res_dict.update({tag_name: data_cell})

        res_dict.update({
            'energy': AssembleUtils.assemble_scalar(self.energy_form),
            'energy_loss': AssembleUtils.assemble_scalar(self.energy_loss_form),
            'name': self.name
        })
        return res_dict

    @staticmethod
    def log_dict(log_recorder: TensorBoardRecorder, res_dicts: List[dict], step):
        tag_names = list(res_dicts[0].keys())
        tag_names.remove('name')

        for tag_name in tag_names:
            data_cell = {}
            for res_dict in res_dicts:
                if isinstance(res_dict[tag_name], dict):
                    data_cell.update(res_dict[tag_name])
                else:
                    data_cell[f"{res_dict['name']}_{tag_name}"] = res_dict[tag_name]

            if len(data_cell) == 1:
                log_recorder.write_scalars(tag_name, data_cell, step)

    def solve(self, **kwargs):
        raise NotImplementedError("[Error]: Depreciate For This Class.")
