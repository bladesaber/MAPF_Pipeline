import numpy as np
from typing import List, Union, Dict, Tuple, Literal
import torch
from sklearn.neighbors import KDTree
import h5py

from scripts_py.version_9.mapf_pkg import smooth_seg_utils
from scripts_py.version_9.mapf_pkg.torch_utils import TorchUtils, CostUtils
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.utils import TensorBoardRecorder


class PathConflictCell(object):
    def __init__(
            self,
            group_idx0: int, cell_idx0: int, cell_type0: Literal["segment", "connector"],
            conflict_pcd0_idxs: Union[np.ndarray, List[int]],
            group_idx1: int, cell_idx1: int, cell_type1: Literal["segment", "connector"],
            conflict_pcd1_idxs: Union[np.ndarray, List[int]],
            require_distance: Union[np.ndarray, List[float]]
    ):
        self.group_idx0 = group_idx0
        self.cell_idx0 = cell_idx0
        self.cell_type0 = cell_type0
        self.conflict_pcd0_idxs = conflict_pcd0_idxs

        self.group_idx1 = group_idx1
        self.cell_idx1 = cell_idx1
        self.cell_type1 = cell_type1
        self.conflict_pcd1_idxs = conflict_pcd1_idxs
        self.require_distance = require_distance


class ObstacleConflictCell(object):
    def __init__(
            self,
            group_idx: int, cell_idx: int, conflict_pcd_idxs: Union[np.ndarray, List[int]],
            obs_pcd_idxs: Union[np.ndarray, List[int]], require_distance: Union[np.ndarray, List[float]],
            cell_type: Literal["segment", "connector"]
    ):
        self.group_idx = group_idx
        self.cell_idx = cell_idx
        self.conflict_pcd_idxs = conflict_pcd_idxs
        self.obs_pcd_idxs = obs_pcd_idxs
        self.require_distance = require_distance
        self.cell_type = cell_type


class PathOptimizer(object):
    def __init__(self):
        self.group_cells: Dict[int, smooth_seg_utils.GroupCell] = {}

        self.obstacle_xyzr: np.ndarray = None
        self.obstacle_num = 0
        self.obstacle_tensor: torch.Tensor = None
        self.obstacle_tree: KDTree = None

        self.obstacle_conflict_cells: List[ObstacleConflictCell] = []
        self.path_conflict_cells: List[PathConflictCell] = []

        self.conflict_cfg: Dict = None

    def init_cfg(
            self,
            init_cfg: Dict[int, Dict[str, Union[np.ndarray, Dict]]],
            obstacle_xyzr: np.ndarray, conflict_cfg: Dict
    ):
        for group_idx, group_cfg in init_cfg.items():
            group_cell = smooth_seg_utils.GroupCell(group_idx)
            group_cell.assemble_discrete_graph(
                path_dict=group_cfg['path_dict'],
                assemble_cfg=group_cfg['assemble_cfg'],
                aid_path_info=group_cfg['aid_path_info'],
            )
            self.group_cells[group_idx] = group_cell

        self.obstacle_xyzr = obstacle_xyzr
        self.obstacle_num = self.obstacle_xyzr.shape[0]
        if self.obstacle_num > 0:
            self.obstacle_tensor = TorchUtils.np2tensor(obstacle_xyzr[:, :3], require_grad=False)
            self.obstacle_tree = KDTree(obstacle_xyzr[:, :3])

        self.conflict_cfg = conflict_cfg

    def restore(self):
        raise NotImplementedError

    def init_optimize_cfg(self, group_idx, setting_cfg: Dict):
        self.group_cells[group_idx].init_cfg(setting_cfg)

    def init_cross_seg_cfg(self, group_idx, setting_cfg: Dict):
        self.group_cells[group_idx].init_cross_seg_cfg(setting_cfg)

    def update_spline_NpCfg(self, group_idx: int):
        group_cell = self.group_cells[group_idx]
        table_xyzr = group_cell.c_table[['x', 'y', 'z', 'radius']].values
        for seg_idx, seg_cell in group_cell.seg_dict.items():
            seg_cell.update_spline_NpCfg(table_xyzr)
        for con_idx, con_cell in group_cell.connector_dict.items():
            con_cell.update_spline_NpCfg(table_xyzr)

    def prepare_tensor(self):
        for group_idx, group_cell in self.group_cells.items():
            table_xyzr = group_cell.c_table[['x', 'y', 'z', 'radius']].values
            for seg_idx, seg_cell in group_cell.seg_dict.items():
                seg_cell.update_spline_NpCfg(table_xyzr, update_threshold=0.1)
            for con_idx, con_cell in group_cell.connector_dict.items():
                con_cell.update_spline_NpCfg(table_xyzr, update_threshold=0.1)
            group_cell.prepare_tensor()

    def update_tensor(self, iteration: int, update_iteration=10):
        for group_idx, group_cell in self.group_cells.items():
            if iteration % update_iteration == 0:
                group_cell.update_state(with_tensor=True, with_np=True)
                table_xyzr: np.ndarray = group_cell.c_table[['x', 'y', 'z', 'radius']].values

                for seg_idx, seg_cell in group_cell.seg_dict.items():
                    is_update = seg_cell.update_spline_NpCfg(table_xyzr, update_threshold=0.1)
                    if is_update:
                        seg_cell.prepare_tensor()
                        seg_cell.update_tensor(group_cell.table_xyz_t)

                for con_idx, con_cell in group_cell.connector_dict.items():
                    is_update = con_cell.update_spline_NpCfg(table_xyzr, update_threshold=0.1)
                    if is_update:
                        con_cell.prepare_tensor()
                        con_cell.update_tensor(group_cell.table_xyz_t)

                for path_idx, path_cell in group_cell.path_dict.items():
                    path_cell.update_tensor(with_analytical_tangent=False)

            else:
                group_cell.update_state(with_tensor=True, with_np=False)

    def find_obstacle_conflict_cells(self, unit_lr: float):
        self.obstacle_conflict_cells.clear()
        if self.obstacle_num == 0:
            return self.obstacle_conflict_cells

        for group_idx, group_cell in self.group_cells.items():
            is_fix_bool = group_cell.c_table['is_fix'].values == 1
            fix_pcd = group_cell.c_table[['x', 'y', 'z']].values[is_fix_bool]

            cell_list: List[Union[smooth_seg_utils.SegmentCell, smooth_seg_utils.ConnectorCell]] = []
            for _, seg_cell in group_cell.seg_dict.items():
                cell_list.append(seg_cell)
            for _, con_cell in group_cell.connector_dict.items():
                cell_list.append(con_cell)

            for cell in cell_list:
                xyz_np = TorchUtils.tensor2np(cell.spline_xyz_t)
                radius = cell.radius

                detection_radius = radius + cell.obstacle_thickness + unit_lr
                search_radius = detection_radius + np.max(self.obstacle_xyzr[:, -1])
                idxs_list, dist_list = self.obstacle_tree.query_radius(xyz_np, r=search_radius, return_distance=True)

                conflict_pcd_idxs, obs_pcd_idxs, require_radius = [], [], []
                for pcd_idx, (obs_idxs, dists) in enumerate(zip(idxs_list, dist_list)):
                    if len(obs_idxs) == 0:
                        continue

                    is_fix_point = np.linalg.norm(xyz_np[pcd_idx] - fix_pcd, ord=2, axis=1) <= 1e-6
                    if np.any(is_fix_point):
                        continue

                    for obs_idx, dist in zip(obs_idxs, dists):
                        if dist <= detection_radius[pcd_idx] + self.obstacle_xyzr[obs_idx, 3]:
                            conflict_pcd_idxs.append(pcd_idx)
                            obs_pcd_idxs.append(obs_idx)
                            seg_radius = radius[pcd_idx]
                            obs_radius = self.obstacle_xyzr[obs_idx, 3]
                            aux_radius = cell.obstacle_thickness
                            require_radius.append(seg_radius + obs_radius + aux_radius)

                if len(conflict_pcd_idxs) > 0:
                    conflict_pcd_idxs = np.array(conflict_pcd_idxs)
                    obs_pcd_idxs = np.array(obs_pcd_idxs)
                    require_radius = np.array(require_radius)
                    self.obstacle_conflict_cells.append(
                        ObstacleConflictCell(
                            group_idx, cell.idx, conflict_pcd_idxs, obs_pcd_idxs, require_radius, cell.cell_type
                        )
                    )

        return self.obstacle_conflict_cells

    def find_path_conflict_cells(self, unit_lr: float):
        self.path_conflict_cells.clear()

        group_cell_dict: Dict[int, List[Union[smooth_seg_utils.SegmentCell, smooth_seg_utils.ConnectorCell]]] = {}
        for group_idx, group_cell in self.group_cells.items():
            cell_list = []
            for _, seg_cell in group_cell.seg_dict.items():
                cell_list.append(seg_cell)
            for _, con_cell in group_cell.connector_dict.items():
                cell_list.append(con_cell)
            group_cell_dict[group_idx] = cell_list

        group_idxs = list(group_cell_dict.keys())
        num = len(group_idxs)
        for i in range(num):
            group_idx_i = group_idxs[i]
            cell_list_i = group_cell_dict[group_idx_i]

            for j in range(i + 1, num, 1):
                group_idx_j = group_idxs[j]
                cell_list_j = group_cell_dict[group_idx_j]

                # ------ find conflict
                for cell_i in cell_list_i:
                    xyz_i = TorchUtils.tensor2np(cell_i.spline_xyz_t)
                    radius_i = cell_i.radius
                    aux_radius_i = cell_i.path_thickness
                    tree_i = KDTree(xyz_i)

                    for cell_j in cell_list_j:
                        xyz_j = TorchUtils.tensor2np(cell_j.spline_xyz_t)
                        radius_j = cell_j.radius
                        aux_radius_j = cell_j.path_thickness

                        search_radius = (
                                np.max(radius_i) + np.max(radius_j)
                                + max(np.max(aux_radius_i), np.max(aux_radius_j))
                                + unit_lr * 2.0
                        )
                        idxs_list, dist_list = tree_i.query_radius(xyz_j, search_radius, return_distance=True)

                        pcd0_idxs, pcd1_idxs, require_radius = [], [], []
                        for pcd_j_idx, (pcd_i_idx_list, dists) in enumerate(zip(idxs_list, dist_list)):
                            if len(pcd_i_idx_list) == 0:
                                continue

                            for pcd_i_idx, dist in zip(pcd_i_idx_list, dists):
                                detection_radius = (
                                        radius_i[pcd_i_idx] + radius_j[pcd_j_idx]
                                        + max(aux_radius_i, aux_radius_j)
                                        + unit_lr * 2.0
                                )
                                if dist <= detection_radius:
                                    pcd0_idxs.append(pcd_i_idx)
                                    pcd1_idxs.append(pcd_j_idx)
                                    seg_i_radius = radius_i[pcd_i_idx]
                                    seg_j_radius = radius_j[pcd_j_idx]
                                    aux_radius = max(aux_radius_i, aux_radius_j)
                                    require_radius.append(seg_i_radius + seg_j_radius + aux_radius)

                        if len(pcd0_idxs) > 0:
                            pcd0_idxs = np.array(pcd0_idxs)
                            pcd1_idxs = np.array(pcd1_idxs)
                            require_radius = np.array(require_radius)
                            self.path_conflict_cells.append(PathConflictCell(
                                group_idx0=group_idxs[i], cell_idx0=cell_i.idx,
                                cell_type0=cell_i.cell_type, conflict_pcd0_idxs=pcd0_idxs,
                                group_idx1=group_idxs[j], cell_idx1=cell_j.idx,
                                cell_type1=cell_j.cell_type, conflict_pcd1_idxs=pcd1_idxs,
                                require_distance=require_radius
                            ))

        return self.path_conflict_cells

    def compute_conflict_cost(self) -> Dict[str, Union[torch.Tensor, float]]:
        cost_dict = {'obstacle_cost': 0.0, 'path_cost': 0.0}

        for obs_cell in self.obstacle_conflict_cells:
            if obs_cell.cell_type == 'segment':
                sub_cell = self.group_cells[obs_cell.group_idx].seg_dict[obs_cell.cell_idx]
            else:
                sub_cell = self.group_cells[obs_cell.group_idx].connector_dict[obs_cell.cell_idx]

            costs = CostUtils.compute_shape_conflict_cost(
                pcd0=sub_cell.spline_xyz_t[obs_cell.conflict_pcd_idxs, :],
                pcd1=self.obstacle_tensor[obs_cell.obs_pcd_idxs, :],
                require_dist=obs_cell.require_distance,
                weight=1.0
            )
            cost_dict['obstacle_cost'] += costs.mean() * self.conflict_cfg['obstacle_conflict_weight']

        for path_cell in self.path_conflict_cells:
            if path_cell.cell_type0 == 'segment':
                cell0 = self.group_cells[path_cell.group_idx0].seg_dict[path_cell.cell_idx0]
            else:
                cell0 = self.group_cells[path_cell.group_idx0].connector_dict[path_cell.cell_idx0]

            if path_cell.cell_type1 == 'segment':
                cell1 = self.group_cells[path_cell.group_idx1].seg_dict[path_cell.cell_idx1]
            else:
                cell1 = self.group_cells[path_cell.group_idx1].connector_dict[path_cell.cell_idx1]

            costs = CostUtils.compute_shape_conflict_cost(
                pcd0=cell0.spline_xyz_t[path_cell.conflict_pcd0_idxs, :],
                pcd1=cell1.spline_xyz_t[path_cell.conflict_pcd1_idxs, :],
                require_dist=path_cell.require_distance,
                weight=1.0
            )
            cost_dict['path_cost'] += costs.mean() * self.conflict_cfg['path_conflict_weight']

        return cost_dict

    def run_by_hand(self, max_iter: int, lr: float, with_momentum=False):
        lr_unit = np.linalg.norm(np.array([lr, lr, lr]))
        if with_momentum:
            optimizer = TorchUtils.get_optimizer(
                [group_cell.table_xyz_t for group_idx, group_cell in self.group_cells.items()], lr=lr
            )

        loss_record, loss_info = [], {}
        for it in range(1, max_iter + 1):
            # ------ find conflict
            self.find_obstacle_conflict_cells(lr_unit)
            self.find_path_conflict_cells(lr_unit)

            # ------ compute all cost
            loss_info.clear()
            for group_idx, group_cell in self.group_cells.items():
                loss_info.update(group_cell.compute_loss(lr_unit, with_default_weight=False))
            loss_info.update(self.compute_conflict_cost())

            loss: torch.Tensor = 0.0
            for _, cost_value in loss_info.items():
                loss += cost_value

            # ------ log information
            if (it % 100 == 0) or (it == 1):
                loss_record.append(loss.detach().numpy())
                print(f"[iter:{it}] loss:{loss_record[-1]:.6f}")

            # ------ compute gradient
            loss.backward()
            with torch.no_grad():
                for group_idx, group_cell in self.group_cells.items():
                    grad = group_cell.table_xyz_t.grad
                    grad[group_cell.is_fix, :] = 0.0

            # ------ update parameters
            if with_momentum:
                optimizer.step()
                optimizer.zero_grad()

            else:
                with torch.no_grad():
                    for group_idx, group_cell in self.group_cells.items():
                        # grad = grad / (torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
                        grad = grad / grad.abs().max()  # it works
                        group_cell.table_xyz_t -= grad * lr
                        group_cell.table_xyz_t.grad.zero_()

            self.update_tensor(iteration=it, update_iteration=50)

        for _, group_cell in self.group_cells.items():
            group_cell.update_state(with_tensor=False, with_np=True)

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = val.detach().numpy()
            else:
                loss_info_np[key] = val

        return loss_record, loss_info_np

    def auto_run(self, cfg: Dict, tensor_writer: TensorBoardRecorder = None):
        with_tensor_writer = False
        if tensor_writer is not None:
            with_tensor_writer = True

        loss_record, loss_info = [], {}
        iteration = -1
        for out_iter in range(cfg['outer_loop']):
            optimizer = TorchUtils.get_optimizer(
                [group_cell.table_xyz_t for group_idx, group_cell in self.group_cells.items()], lr=cfg['lr']
            )

            scheduler = TorchUtils.get_lr_scheduler(
                optimizer, factor=cfg['factor'], patience=cfg['patience'],
                cooldown=cfg['cooldown'], min_lr=cfg['min_lr']
            )

            while True:
                iteration += 1
                lr = scheduler.get_last_lr()[-1]
                lr_unit = np.linalg.norm(np.array([lr, lr, lr]))

                # ------ find conflict
                self.find_obstacle_conflict_cells(lr_unit)
                self.find_path_conflict_cells(lr_unit)

                # ------ compute all cost
                loss_info.clear()
                for group_idx, group_cell in self.group_cells.items():
                    group_cost_info = group_cell.compute_loss(lr_unit, with_default_weight=False)
                    loss_info.update(group_cost_info)
                    if with_tensor_writer:
                        tensor_writer.write_scalars(f"{group_idx}_loss", group_cost_info, step=iteration)

                conflict_loos_info = self.compute_conflict_cost()
                loss_info.update(conflict_loos_info)
                if with_tensor_writer:
                    tensor_writer.write_scalars(f"conflict_loss", conflict_loos_info, step=iteration)

                loss: torch.Tensor = 0.0
                for _, cost_value in loss_info.items():
                    loss += cost_value

                # ------ log information
                if (iteration % 100 == 0) or (iteration == 1):
                    loss_record.append(loss.detach().numpy())
                    print(f"[outIter{out_iter}/iter:{iteration}/lr:{lr:.5f}] loss:{loss_record[-1]:.6f}")
                if with_tensor_writer:
                    tensor_writer.write_scalar('loss', loss, step=iteration)

                # ------ compute gradient
                loss.backward()
                with torch.no_grad():
                    for group_idx, group_cell in self.group_cells.items():
                        grad = group_cell.table_xyz_t.grad
                        grad[group_cell.is_fix, :] = 0.0

                # ------ update parameters
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step(loss)

                self.update_tensor(iteration=iteration, update_iteration=15)

                # ------ prepare to break
                if iteration > cfg['max_iter']:
                    break

                if lr <= cfg['min_lr']:
                    break

        for _, group_cell in self.group_cells.items():
            group_cell.update_state(with_tensor=False, with_np=True)

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = val.detach().numpy()
            else:
                loss_info_np[key] = val

        return loss_record, loss_info_np

    def save(self, file: str = None, writer: h5py.File = None):
        close_auto = False
        if writer is None:
            assert file.endswith('.hdf5')
            writer = h5py.File(file, mode='w')
            close_auto = True

        for group_idx, group_cell in self.group_cells.items():
            group_cell.save(writer=writer)

        if close_auto:
            writer.close()

    def loss_check(self, lr):
        lr_unit = np.linalg.norm(np.array([lr, lr, lr]))

        loss_info = {}
        for group_idx, group_cell in self.group_cells.items():
            loss_info.update(group_cell.compute_loss(lr_unit, with_default_weight=False))

        self.find_obstacle_conflict_cells(lr_unit)
        self.find_path_conflict_cells(lr_unit)
        loss_info.update(self.compute_conflict_cost())

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = val.detach().numpy()
            else:
                loss_info_np[key] = val

        return loss_info_np

    def draw_constraints(
            self, lr_unit=None, vis: VisUtils = None,
            with_path_conflict=True, with_obstacle_conflict=True, with_real_conflict=False
    ):
        if lr_unit is not None:
            self.find_obstacle_conflict_cells(lr_unit)
            self.find_path_conflict_cells(lr_unit)

        show_plot = False
        if vis is None:
            vis = VisUtils()
            show_plot = True

        for group_idx, group_cell in self.group_cells.items():
            color = np.random.uniform(0.0, 1.0, size=(3,))

            cell_list: List[Union[smooth_seg_utils.SegmentCell, smooth_seg_utils.ConnectorCell]] = []
            for seg_idx, seg_cell in group_cell.seg_dict.items():
                cell_list.append(seg_cell)
            for con_idx, con_cell in group_cell.connector_dict.items():
                cell_list.append(con_cell)

            for cell in cell_list:
                spline_xyz = TorchUtils.tensor2np(cell.spline_xyz_t)
                line_set = VisUtils.create_line_set(np.arange(0, spline_xyz.shape[0], 1))
                mesh = VisUtils.create_line(spline_xyz[:, :3], line_set)
                # vis.plot(mesh, color=color, style='wireframe', point_size=0.1, line_width=2.0)
                vis.plot(mesh, color=color, point_size=5)

        if with_obstacle_conflict:
            mesh = VisUtils.create_point_cloud(self.obstacle_xyzr[:, :3])
            vis.plot(mesh, color=(0.3, 0.3, 0.3), opacity=1.0)

            for obs_cell in self.obstacle_conflict_cells:
                group_cell = self.group_cells[obs_cell.group_idx]
                if obs_cell.cell_type == 'segment':
                    sub_cell = group_cell.seg_dict[obs_cell.cell_idx]
                elif obs_cell.cell_type == 'connector':
                    sub_cell = group_cell.connector_dict[obs_cell.cell_idx]
                else:
                    raise ValueError(f"[ERROR]: Non-valid obstacle cell type:{obs_cell.cell_type}")

                pcd0 = TorchUtils.tensor2np(sub_cell.spline_xyz_t[obs_cell.conflict_pcd_idxs, :])
                pcd1 = self.obstacle_xyzr[obs_cell.obs_pcd_idxs, :3]

                if with_real_conflict:
                    dists = np.linalg.norm(pcd0 - pcd1, axis=1)
                    conflict_bool = dists < obs_cell.require_distance
                    if conflict_bool.sum() == 0:
                        continue

                    pcd0 = pcd0[conflict_bool]
                    pcd1 = pcd1[conflict_bool]
                    max_conflict = np.max(obs_cell.require_distance[conflict_bool] - dists[conflict_bool])
                    print(f'[INFO {obs_cell.group_idx}/{obs_cell.cell_type}_{obs_cell.cell_idx}]: '
                          f'obstacle max conflict {max_conflict};')

                pcd0_idxs = np.arange(0, pcd0.shape[0], 1)
                pcd1_idxs = pcd0_idxs + pcd0.shape[0]
                pcd = np.concatenate([pcd0, pcd1], axis=0)
                line_set = np.concatenate([
                    np.full(shape=(pcd0.shape[0], 1), fill_value=2),
                    pcd0_idxs.reshape((-1, 1)), pcd1_idxs.reshape((-1, 1))
                ], axis=1)
                mesh = VisUtils.create_line(pcd, line_set)
                vis.plot(mesh, color=(1.0, 0.0, 0.0), point_size=4)

        if with_path_conflict:
            for path_cell in self.path_conflict_cells:
                if path_cell.cell_type0 == 'segment':
                    sub0_cell = self.group_cells[path_cell.group_idx0].seg_dict[path_cell.cell_idx0]
                elif path_cell.cell_type0 == 'connector':
                    sub0_cell = self.group_cells[path_cell.group_idx0].connector_dict[path_cell.cell_idx0]
                else:
                    raise ValueError(f"[ERROR]: Non-valid path cell type:{path_cell.cell_type0}")

                if path_cell.cell_type1 == 'segment':
                    sub1_cell = self.group_cells[path_cell.group_idx1].seg_dict[path_cell.cell_idx1]
                elif path_cell.cell_type1 == 'connector':
                    sub1_cell = self.group_cells[path_cell.group_idx1].connector_dict[path_cell.cell_idx1]
                else:
                    raise ValueError(f"[ERROR]: Non-valid path cell type:{path_cell.cell_type1}")

                pcd0 = TorchUtils.tensor2np(sub0_cell.spline_xyz_t[path_cell.conflict_pcd0_idxs, :])
                pcd1 = TorchUtils.tensor2np(sub1_cell.spline_xyz_t[path_cell.conflict_pcd1_idxs, :])

                if with_real_conflict:
                    dists = np.linalg.norm(pcd0 - pcd1, axis=1)
                    conflict_bool = dists < path_cell.require_distance
                    if conflict_bool.sum() == 0:
                        continue
                    pcd0 = pcd0[conflict_bool]
                    pcd1 = pcd1[conflict_bool]
                    max_conflict = np.max(path_cell.require_distance[conflict_bool] - dists[conflict_bool])
                    print(f'[INFO {path_cell.group_idx0}/{path_cell.cell_type0}_{path_cell.cell_idx0}-'
                          f'{path_cell.group_idx1}/{path_cell.cell_type1}_{path_cell.cell_idx1}]: '
                          f'path max conflict {max_conflict};')

                pcd0_idxs = np.arange(0, pcd0.shape[0], 1)
                pcd1_idxs = pcd0_idxs + pcd0.shape[0]
                pcd = np.concatenate([pcd0, pcd1], axis=0)
                line_set = np.concatenate([
                    np.full(shape=(pcd0.shape[0], 1), fill_value=2),
                    pcd0_idxs.reshape((-1, 1)), pcd1_idxs.reshape((-1, 1))
                ], axis=1)
                mesh = VisUtils.create_line(pcd, line_set)
                vis.plot(mesh, color=(0.0, 0.0, 1.0), point_size=4)

        if show_plot:
            vis.show()

    def draw_curvature(self):
        vis = VisUtils()
        for group_idx, group_cell in self.group_cells.items():
            group_cell.draw_curvature(vis)
        vis.show()
