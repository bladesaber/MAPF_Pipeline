import os
import shutil
from typing import Dict, List, Callable
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

# TODO
"""
1. 计算变形度
2. 根据变形度计算变形优先级
3. 根据优先级依次变形
4. 检测范围内干涉，固定这些干涉面
"""


class FluidShapeOptObstacle(FluidShapeOptSimple):
    def __init__(
            self,
            name,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            Re: float,
            deformation_cfg: Dict,
            bry_markers: List[int],
            conflict_radius: float,
            conflict_offset: float,
            freeze_radius: float = 0.0,
            isStokeEqu=False,
            beta_rho=0.75,
            deformation_lower=1e-2,
    ):
        """
        Param:
            beta_rho: step_size的衰减系数，必须小于1.0，越接近1.0，搜索精度越大
            deformation_lower: 最小形变程度，用于判断是否收敛
        """
        assert 1.0 > beta_rho > 0.0

        super().__init__(
            domain=domain,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            Re=Re,
            deformation_cfg=deformation_cfg,
            isStokeEqu=isStokeEqu
        )

        self.collision_obj = MeshCollisionObj(self.domain, self.facet_tags, self.cell_tags, bry_markers)

        self.name = name
        self.conflict_radius = conflict_radius
        self.conflict_offset = conflict_offset
        self.freeze_radius = freeze_radius
        self.beta_rho = beta_rho
        self.deformation_lower = deformation_lower
        self.solve_mapper = {}

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
            direction_np, max_iter=10, init_stepSize=1.0, stepSize_lower=1e-3,
            detect_cost_valid_func=detect_cost_valid_func, revert_move=True
        )

        return success_flag, (direction_np, step_size)

    def safe_move(
            self,
            direction_np: np.ndarray, step_size: float,
            obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj],
            beta_rho: float, deformation_lower
    ):
        while True:
            displacement_np = direction_np * step_size
            MeshUtils.move(self.domain, displacement_np)
            self.collision_obj.create_tree()

            success_flag = True
            for obs_obj in obs_objs:
                conflict_idxs, _ = self.collision_obj.find_conflict_bry_nodes(
                    obs_obj.get_coords(), self.conflict_radius
                )
                if len(conflict_idxs) > 0:
                    success_flag = False
                    break

            if success_flag:
                for mesh_obj in mesh_objs:
                    conflict_idxs, _ = self.collision_obj.find_conflict_bry_nodes(
                        mesh_obj.get_bry_coords(), self.conflict_radius
                    )
                    if len(conflict_idxs) > 0:
                        success_flag = False
                        break

            self.collision_obj.release_tree()
            if success_flag:
                break
            else:
                MeshUtils.move(self.domain, displacement_np * -1.0)

                step_size = step_size * beta_rho
                max_deformation = np.max(np.linalg.norm(step_size * direction_np, ord=2, axis=1))
                if max_deformation < deformation_lower:
                    break

        return success_flag, step_size

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

        def detect_cost_valid_func(tol_rho=0.05):
            loss = self.opt_problem.evaluate_cost_functional(
                self.domain.comm, update_state=True, with_debug=with_debug
            )
            is_valid = loss < loss_storge_ctype.value + np.abs(loss_storge_ctype.value) * tol_rho
            return is_valid

        log_dict = self.log_step(logger_dicts)
        self.solve_mapper = {
            'u_recorder': u_recorder,
            'loss_storge': loss_storge_ctype,
            'cost_valid_func': detect_cost_valid_func,
            'logger_dicts': logger_dicts,
        }
        return log_dict

    def single_solve(
            self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj], step: int, **kwargs
    ):
        if len(self.solve_mapper) == 0:
            raise ValueError("[Error]: Solving Need to Be Init First.")

        self.update_shape_boundary_cond(obs_objs, mesh_objs)

        success_flag, (direction_np, step_size) = self.get_shape_derivatives(
            self.solve_mapper['cost_valid_func'], **kwargs
        )
        if not success_flag:
            return {'state': success_flag}

        success_flag, step_size = self.safe_move(
            direction_np=direction_np, step_size=step_size,
            obs_objs=obs_objs, mesh_objs=mesh_objs,
            beta_rho=self.beta_rho, deformation_lower=self.deformation_lower
        )

        res_dict = {'state': success_flag}
        if success_flag:
            loss = self.opt_problem.evaluate_cost_functional(self.domain.comm, update_state=True)
            self.solve_mapper['loss_storge'].value = loss

            self.solve_mapper['u_recorder'].write_function(self.up.sub(0).collapse(), step)
            log_dict = self.log_step(logger_dicts=self.solve_mapper['logger_dicts'])
            log_dict.update({'loss': loss})

            max_deformation = np.max(np.linalg.norm(step_size * direction_np, ord=2, axis=1))
            res_dict.update({
                'is_converge': max_deformation < self.deformation_lower,
                'log_dict': log_dict
            })

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

    def solve(self, **kwargs):
        raise NotImplementedError("[Error]: Depreciate For This Class.")

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

    def optimization_initiation(
            self,
            cost_functional_list: List[CostFunctional_types],
            cost_weight: Dict,
            shapeRegularization: ShapeRegularization,
            scalar_product_method,
    ):
        super().optimization_initiation(
            cost_functional_list=cost_functional_list,
            cost_weight=cost_weight,
            shapeRegularization=shapeRegularization,
            scalar_product_method=scalar_product_method
        )
        self.opt_problem.shape_problem.conflict_bc_info = None

    def find_conflict_area_dofs(self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj]):
        self.collision_obj.create_tree()

        dofs = []
        for obs_obj in obs_objs:
            conflict_idxs, _ = self.collision_obj.find_conflict_bry_nodes(
                obs_obj.get_coords(), self.conflict_radius + self.conflict_offset
            )
            dofs.append(conflict_idxs)
        for mesh_obj in mesh_objs:
            conflict_idxs, _ = self.collision_obj.find_conflict_bry_nodes(
                mesh_obj.get_bry_coords(), self.conflict_radius + self.conflict_offset
            )
            dofs.append(conflict_idxs)

        if len(dofs) == 0:
            return np.array([])

        dofs = np.concatenate(dofs, axis=0)
        dofs = np.unique(dofs)

        if self.freeze_radius > 0.0 and len(dofs) > 0:
            relate_dofs = self.collision_obj.find_node_neighbors(
                coords=self.domain.geometry.x[dofs, :self.tdim],
                radius=self.freeze_radius
            )
            dofs = np.concatenate([dofs, relate_dofs], axis=0)
            dofs = np.unique(dofs)

        self.collision_obj.release_tree()

        return dofs

    def update_shape_boundary_cond(self, obs_objs: List[ObstacleCollisionObj], mesh_objs: List[MeshCollisionObj]):
        # ------ Step 1: Resume Initiation
        if self.opt_problem.shape_problem.conflict_bc_info is not None:
            self.opt_problem.shape_problem.bcs.pop()  # pop last one
            self.opt_problem.shape_problem.bcs_infos.pop()  # pop last one
            self.opt_problem.shape_problem.conflict_bc_info = None

        # ------ Step 2: Check Conflict Area
        bc_dofs = self.find_conflict_area_dofs(obs_objs, mesh_objs)
        if bc_dofs.shape[0] > 0:
            bc_value = dolfinx.fem.Function(self.V_S, name="conflict_boundary")
            bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)

            self.opt_problem.shape_problem.conflict_bc_info = (bc, self.V_S, bc_dofs, bc_value)
            self.opt_problem.shape_problem.add_bc(bc, self.V_S, bc_dofs, bc_value)


class FluidShapeOptField(FluidShapeOptSimple):
    def __init__(
            self,
            name,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            Re: float,
            deformation_cfg: Dict,
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
