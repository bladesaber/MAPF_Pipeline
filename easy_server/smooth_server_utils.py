import os
import shutil
import numpy as np
import pandas as pd
import torch
from queue import Queue
import time
import json
from typing import Generator

from scripts_py.mapf_pkg.smooth_optimizer import PathOptimizer
from scripts_py.mapf_pkg.utils import TensorBoardRecorder
from scripts_py.mapf_pkg.torch_utils import TorchUtils


class SmoothInfo:
    obstacle_xyzr: list = []
    optimizer: PathOptimizer
    group_name2idx: dict[str, int] = {}
    group_idx2name: dict[int, str] = {}
    is_obstacle_pcd_ready: bool = False


class SmoothServerUtils:
    smooth_map: dict[str, SmoothInfo] = {}

    @staticmethod
    def init_smooth_map(smooth_name: str, smooth_json: dict):
        smooth_info = SmoothServerUtils.smooth_map[smooth_name]

        pipes_cfg = smooth_json['pipes_cfg']
        obs_xyzr = np.array(smooth_info.obstacle_xyzr)

        init_cfg = {}
        for group_name, group_setting in smooth_json['setting'].items():
            if group_name not in smooth_info.group_name2idx:
                smooth_info.group_name2idx[group_name] = len(smooth_info.group_name2idx)
                smooth_info.group_idx2name[len(smooth_info.group_idx2name)] = group_name

            init_cfg[smooth_info.group_name2idx[group_name]] = {
                'path_dict': {},
                'assemble_cfg': {
                    'seg_node_min': group_setting['graph_cfg']['seg_node_min'],
                    'seg_node_max': group_setting['graph_cfg']['seg_node_max'],
                },
                'aid_path_info': [],
            }

            for aid_name, info in group_setting['aid_path_info'].items():
                info_copy = info.copy()
                info_copy.update({
                    'name': aid_name,
                    'from_position': np.array(pipes_cfg[info['from_name']]['position']),
                    'from_vector': np.array(pipes_cfg[info['from_name']]['direction']),
                    'from_is_input': pipes_cfg[info['from_name']]['is_input'],
                    'to_position': np.array(pipes_cfg[info['to_name']]['position']),
                    'to_vector': np.array(pipes_cfg[info['to_name']]['direction']),
                    'to_is_input': pipes_cfg[info['to_name']]['is_input'],
                })
                init_cfg[smooth_info.group_name2idx[group_name]]['aid_path_info'].append(info_copy)

            for name, xyzr in group_setting['path_result'].items():
                init_cfg[smooth_info.group_name2idx[group_name]]['path_dict'][name] = np.array(xyzr)
        
        opt = PathOptimizer()
        opt.init_cfg(init_cfg, obs_xyzr, smooth_json['conflict_setting'])

        for group_name, group_setting in smooth_json['setting'].items():
            group_setting['segments'] = []
            group_setting['connectors'] = []
            group_setting['paths'] = []

            for seg_idx, seg_cell in opt.group_cells[smooth_info.group_name2idx[group_name]].seg_dict.items():
                seg_run_cfg = {
                    'seg_idx': seg_idx,
                    'bspline_degree': 3,
                    'bspline_density': 0.25,
                    'left_overlap_ratio': 0.25,
                    'right_overlap_ratio': 0.25,
                    'left_intercept_ratio': 0.5,
                    'right_intercept_ratio': 0.5,
                    'segment_scale': 1.0,
                    'with_analytical_curvature_computed': True,
                    "with_analytical_tangent_computed": False,
                    'cost_setting': {
                        'path_thickness': 0.01,
                        'obstacle_thickness': 0.01,
                        'cost_cfgs': [
                            {
                                'cost_type': 'length',
                                'data_type': 'control_points',
                                'method': 'square_sum',
                                'weight': 1.0
                            },
                            {
                                'cost_type': 'length_distribution',
                                'data_type': 'control_points',
                                'method': 'length_variance',
                                'weight': 100.0
                            },
                            {
                                "cost_type": "curvature",
                                "curvature_method": "analytical_curvature",
                                "method": "sigmoid_mean",
                                "curvature_scale": 3.0,
                                "weight": 10.0
                            }
                        ]
                    }
                }

                if seg_cell.left_clamp:
                    seg_run_cfg['cost_setting']['cost_cfgs'].append({
                        "cost_type": "clamp_vector",
                        "vector_length": 1.0,
                        "vector": seg_cell.left_vector.tolist(),
                        "weight": 100.0
                    })
                if seg_cell.right_clamp:
                    seg_run_cfg['cost_setting']['cost_cfgs'].append({
                        "cost_type": "clamp_vector",
                        "vector_length": 1.0,
                        "vector": seg_cell.right_vector.tolist(),
                        "weight": 100.0
                    })
                group_setting['segments'].append(seg_run_cfg)
            
            for con_idx, con_cell in opt.group_cells[smooth_info.group_name2idx[group_name]].connector_dict.items():
                con_run_cfg = {
                    'con_idx': con_idx,
                    'bspline_degree': 3,
                    'bspline_density': 0.25,
                    'left_seg_idx': con_cell.left_seg_idx,
                    'right_seg_idx': con_cell.right_seg_idx,
                    'with_analytical_curvature_computed': True,
                    "with_analytical_tangent_computed": False,
                    'cost_setting': {
                        'path_thickness': 0.01,
                        'obstacle_thickness': 0.01,
                        'cost_cfgs': [
                            {
                                "cost_type": "curvature",
                                "curvature_method": "analytical_curvature",
                                "method": "sigmoid_mean",
                                "curvature_scale": 3.0,
                                "weight": 10.0
                            },
                        ]
                    }
                }
                group_setting['connectors'].append(con_run_cfg)
            
            for path_idx, path_cell in opt.group_cells[smooth_info.group_name2idx[group_name]].path_dict.items():
                path_run_cfg = {
                    'path_idx': path_idx,
                    'path_name': path_cell.path_name,
                    'cell_idxs': path_cell.cell_idxs,
                    'cell_types': path_cell.cell_types,
                    'cost_setting': {
                    'cost_cfgs': [
                        {
                            'cost_type': 'tangent_diffusion',
                            'method': 'abs_variance_max',
                            'weight': 10.0
                        }
                    ],
                    'element_cost_cfgs': []
                    },
                }
                group_setting['paths'].append(path_run_cfg)

        for group_name, group_setting in smooth_json['setting'].items():
            opt.init_optimize_cfg(smooth_info.group_name2idx[group_name], {
                'segments': group_setting['segments'],
                'connectors': group_setting['connectors'],
                'paths': group_setting['paths']
            })
        
        for group_name, group_setting in smooth_json['setting'].items():
            for seg_idx, seg_cell in opt.group_cells[smooth_info.group_name2idx[group_name]].seg_dict.items():
                for seg_setting in group_setting['segments']:
                    if seg_setting['seg_idx'] == seg_idx:
                        seg_setting.update({
                            'cell_setting': seg_cell.export_cfg()
                        })
            for con_idx, con_cell in opt.group_cells[smooth_info.group_name2idx[group_name]].connector_dict.items():
                for con_setting in group_setting['connectors']:
                    if con_setting['con_idx'] == con_idx:
                        con_setting.update({
                            'cell_setting': con_cell.export_cfg()
                        })
            for path_idx, path_cell in opt.group_cells[smooth_info.group_name2idx[group_name]].path_dict.items():
                for path_setting in group_setting['paths']:
                    if path_setting['path_idx'] == path_idx:
                        path_setting.update({
                            'cell_setting': path_cell.export_cfg()
                        })
            group_setting['cell_setting'] = opt.group_cells[smooth_info.group_name2idx[group_name]].export_cfg()
            del group_setting['path_result']

        return smooth_json

    @staticmethod
    def restore_smooth_json(smooth_name: str, smooth_json: dict):
        opt = PathOptimizer()
        smooth_info = SmoothServerUtils.smooth_map[smooth_name]
        obs_xyzr = np.array(smooth_info.obstacle_xyzr)
        opt.restore(smooth_json, obs_xyzr)
        for group_idx in opt.group_cells.keys():
            opt.update_spline_NpCfg(group_idx)
        return opt

    @staticmethod
    def get_smooth_result(smooth_name: str, opt: PathOptimizer) -> dict:
        smooth_info = SmoothServerUtils.smooth_map[smooth_name]
        result_json = {}
        for group_idx, group_cell in opt.group_cells.items():
            result_json[smooth_info.group_idx2name[group_idx]] = group_cell.export_cfg()
        return result_json

    @staticmethod
    def manual_smooth(
        smooth_name: str, smooth_json: dict, run_times: int, lr: float, log_iter: int = 50,
        with_momentum=False, 
        data_queue: Queue = None, mode: str = 'process_stream'
    ) -> tuple[list[float], dict[str, float]] | Generator[str, None, None]:
        if mode == 'stream':
            start_time = time.time()

        # ------ restore smooth json
        opt = SmoothServerUtils.restore_smooth_json(smooth_name, smooth_json)

        # ------ prepare tensor
        opt.prepare_tensor()

        # ------ get optimizer
        lr_unit = np.linalg.norm(np.array([lr, lr, lr]))
        if with_momentum:
            optimizer = TorchUtils.get_optimizer(
                [group_cell.table_xyz_t for group_idx, group_cell in opt.group_cells.items()], lr=lr
            )

        # ------ run by hand
        loss_record, loss_info = [], {}
        for it in range(1, run_times + 1):
            # ------ find conflict
            opt.find_obstacle_conflict_cells(lr_unit)
            opt.find_path_conflict_cells(lr_unit)

            # ------ compute all cost
            loss_info.clear()
            for group_idx, group_cell in opt.group_cells.items():
                loss_info.update(group_cell.compute_loss(lr_unit, with_default_weight=False))
            loss_info.update(opt.compute_conflict_cost())

            loss: torch.Tensor = 0.0
            for _, cost_value in loss_info.items():
                loss += cost_value

            # ------ log information
            if (it % log_iter == 0) or (it == 1):
                if isinstance(loss, torch.Tensor):
                    loss_record.append(float(loss.detach().numpy()))
                else:
                    loss_record.append(float(loss))
                print(f"[iter:{it}] loss:{loss_record[-1]:.6f}")

                if it != 1:
                    smooth_result = SmoothServerUtils.get_smooth_result(smooth_name, opt)
                    if mode == 'process_stream':
                        data_queue.put({
                            'is_finished': False,
                            'result': {'smooth_result': smooth_result}
                        })
                    elif mode == 'stream':
                        yield json.dumps({
                            'is_finished': False,
                            'result': {'smooth_result': smooth_result},
                            'name': 'manual_smooth',
                            'progress': time.time() - start_time,
                            'msg': f"[iter:{it}] loss:{loss_record[-1]:.6f}"
                        }) + '\n'

            # ------ compute gradient
            loss.backward()
            with torch.no_grad():
                for group_idx, group_cell in opt.group_cells.items():
                    grad = group_cell.table_xyz_t.grad
                    grad[group_cell.is_fix, :] = 0.0

            # ------ update parameters
            if with_momentum:
                optimizer.step()
                optimizer.zero_grad()

            else:
                with torch.no_grad():
                    for group_idx, group_cell in opt.group_cells.items():
                        grad = group_cell.table_xyz_t.grad
                        # grad = grad / (torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
                        grad = grad / grad.abs().max()  # it works
                        group_cell.table_xyz_t -= grad * lr
                        group_cell.table_xyz_t.grad.zero_()

            opt.update_tensor(iteration=it, update_iteration=50)

        for _, group_cell in opt.group_cells.items():
            group_cell.update_state(with_tensor=False, with_np=True)

        loss_info_np = {}
        for key in list(loss_info.keys()):
            val = loss_info.pop(key)
            if isinstance(val, torch.Tensor):
                loss_info_np[key] = float(val.detach().numpy())
            else:
                loss_info_np[key] = float(val)

        # ------ put the final result
        if mode == 'process_stream':
            data_queue.put({
                'is_finished': True,
                'result': {
                'smooth_result': SmoothServerUtils.get_smooth_result(smooth_name, opt),
                'loss_record': loss_record,
                'loss_info': loss_info_np
                }
            })
        elif mode == 'stream':
            yield json.dumps({
                'is_success': True,
                'is_finished': True,
                'total_finished': True,
                'result': {
                    'smooth_result': SmoothServerUtils.get_smooth_result(smooth_name, opt),
                    'loss_record': loss_record,
                    'loss_info': loss_info_np
                },
                'name': 'manual_smooth',
                'progress': time.time() - start_time,
                'msg': f"loss:{loss_record[-1]:.6f}"
            }) + '\n'

    @staticmethod
    def sample_surface_points_from_path(path: np.ndarray) -> np.ndarray:
        """
        path: (n, 4) means (x, y, z, radius)
        assume 
        """

