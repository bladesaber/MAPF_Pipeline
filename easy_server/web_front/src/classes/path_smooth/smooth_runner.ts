import * as THREE from 'three';

import type { TreeNode } from '@/types/data_type';
import {
  color_to_hex,
  create_points,
} from '@/utils/visulize_utils';

import { MeshObj } from '../mesh_type';
import {
  ConnectorCell,
  GroupCell,
  PathCell,
  SegmentCell,
} from './smooth_nodes';

export class PipeCfg {
  name: string;
  radius: number;
  position: number[];
  direction: number[];
  group_name: string;
  is_input: boolean;
  radius_scale: number;

  constructor(
    name: string,
    radius: number, position: number[], direction: number[],
    group_name: string, is_input: boolean, radius_scale: number
  ) {
    this.name = name;
    this.radius = radius;
    this.position = position;
    this.direction = direction;
    this.group_name = group_name;
    this.is_input = is_input;
    this.radius_scale = radius_scale;
  }

  export_json(): any {
    return {
      name: this.name,
      group_name: this.group_name,
      is_input: this.is_input,
      radius: this.radius,
      position: this.position,
      direction: this.direction,
      radius_scale: this.radius_scale,
    }
  }

  export_tree(smooth_name: string, task_name: string, project_name: string): TreeNode {
    const node: TreeNode = {
      label: this.name,
      style: 'SmoothPipeCfg',
      data: {
        'Project': project_name,
        'SmoothName': smooth_name,
        'TaskName': task_name,
        'GroupName': this.group_name,
        'Name': this.name,
      },
    }
    return node;
  }
}

export class ObstacleCell extends MeshObj {
  name: string;
  original_xyzr: number[][];
  filtered_xyzr: number[][];

  constructor(name: string, original_xyzr: number[][], scene: THREE.Scene) {
    super(name);
    this.name = name;
    this.original_xyzr = original_xyzr;
    this.filtered_xyzr = original_xyzr;
    this.add_mesh(scene);
  }

  add_mesh(scene: THREE.Scene): void {
    const points: number[][] = []
    this.filtered_xyzr.forEach(xyzr => {
      points.push([xyzr[0], xyzr[1], xyzr[2]]);
    })
    const mesh = create_points(points, { color: color_to_hex([0, 0, 0]) });
    mesh.name = this.name;
    this.mesh_dict[mesh.name] = mesh;
    super.add_mesh(scene);
  }

  // 更新点集并重建可视化
  update_xyzr(new_xyzr: number[][], pipe_dict: Map<string, PipeCfg>, scene: THREE.Scene): void {
    this.original_xyzr = new_xyzr;
    this.filtered_xyzr = ObstacleCell.filter_xyzr(pipe_dict, new_xyzr);
    this.destory(scene);
    this.add_mesh(scene);
  }

  static filter_xyzr(pipe_dict: Map<string, PipeCfg>, original_xyzr: number[][]): number[][] {
    return original_xyzr.filter(xyzr => {
      // if distance between point and pipe is less than pipe radius plus obstacle radius, then it is filtered
      for (const pipe of pipe_dict.values()) {
        const distance = Math.sqrt(
          Math.pow(xyzr[0] - pipe.position[0], 2) +
          Math.pow(xyzr[1] - pipe.position[1], 2) +
          Math.pow(xyzr[2] - pipe.position[2], 2)
        )
        if (distance < pipe.radius + xyzr[3]) {
          return false;
        }
      }
      return true;
    })
  }

  export_tree(smooth_name: string, task_name: string, project_name: string): TreeNode {
    const node: TreeNode = {
      label: this.name,
      style: 'SmoothObstacleCell',
      data: {
        'Project': project_name,
        'SmoothName': smooth_name,
        'TaskName': task_name,
        'Name': this.name,
      },
    }
    return node;
  }
}

export class SmoothRunner {
  name: string;
  task_name: string;
  project_name: string;
  conflict_setting: {
    obstacle_conflict_weight: number,
    path_conflict_weight: number,
    self_conflict_weight: number
  }

  auto_run_cfg: {
    lr: number,
    factor: number,
    patience: number,
    cooldown: number,
    min_lr: number,
    max_iter: number,
    outer_loop: number
  }

  pipes_cfg: Map<string, PipeCfg> = new Map();
  group_cells: Map<string, GroupCell> = new Map();
  obstacle_cell: ObstacleCell | null = null;

  constructor(name: string, task_name: string, project_name: string) {
    this.name = name;
    this.task_name = task_name;
    this.project_name = project_name;
    this.conflict_setting = {
      obstacle_conflict_weight: 1000.0,
      path_conflict_weight: 100.0,
      self_conflict_weight: 100.0
    }
    this.auto_run_cfg = {
      lr: 0.015,
      factor: 0.5,
      patience: 1500,
      cooldown: 1000,
      min_lr: 0.005,
      max_iter: 20000,
      outer_loop: 1
    }
  }

  export_json(): any {
    const pipes_cfg_json: { [key: string]: any } = {}
    this.pipes_cfg.forEach((pipe_cfg, pipe_name) => {
      pipes_cfg_json[pipe_name] = {
        radius: pipe_cfg.radius,
        position: pipe_cfg.position,
        direction: pipe_cfg.direction,
        group_name: pipe_cfg.group_name,
        is_input: pipe_cfg.is_input,
        radius_scale: pipe_cfg.radius_scale,
      }
    })

    const setting_json: any[] = []
    this.group_cells.forEach((group_cell, group_name) => {
      setting_json.push(group_cell.export_json())
    })

    return {
      smooth_name: this.name,
      task_name: this.task_name,
      project_name: this.project_name,

      pipes_cfg: pipes_cfg_json,
      conflict_setting: this.conflict_setting,
      auto_run_cfg: this.auto_run_cfg,

      setting: setting_json,
    }
  }

  export_tree(): TreeNode {
    const pipe_cfg_node: TreeNode = {
      id: `${this.project_name}-${this.name}-pipe-config`,
      label: 'Pipe Config',
      style: 'PipeCfg',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
      children: Array.from(this.pipes_cfg.values()).map(pipe_cfg => pipe_cfg.export_tree(this.name, this.task_name, this.project_name)),
    }

    const obstacle_cell_node: TreeNode = {
      id: `${this.project_name}-${this.name}-obstacle-cell`,
      label: 'Obstacle Cell',
      style: 'ObstacleCell',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
      children: this.obstacle_cell ? [this.obstacle_cell.export_tree(this.name, this.task_name, this.project_name)] : [],
    }

    const auto_run_cfg_node: TreeNode = {
      id: `${this.project_name}-${this.name}-auto-run-config`,
      label: 'Auto Run Config',
      style: 'AutoRunCfg',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
      children: []
    }

    const conflict_setting_node: TreeNode = {
      id: `${this.project_name}-${this.name}-conflict-setting`,
      label: 'Conflict Setting',
      style: 'ConflictSetting',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
    }

    const group_setting_node: TreeNode = {
      id: `${this.project_name}-${this.name}-group-setting`,
      label: 'Group Setting',
      style: 'GroupSetting',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
      children: Array.from(this.group_cells.values()).map(group_cell => group_cell.export_tree()) || [],
    }

    // 若 group_cells 为空，添加占位节点
    if (!group_setting_node.children || group_setting_node.children.length === 0) {
      group_setting_node.children = [{
        id: `${this.project_name}-${this.name}-no-groups`,
        label: 'No Groups',
        style: 'Empty',
        data: { 'Project': this.project_name, 'SmoothName': this.name, 'TaskName': this.task_name },
      }];
    }

    const root: TreeNode = {
      id: `${this.project_name}-${this.name}`,
      label: `${this.task_name}/${this.name}`,
      style: 'SmoothRunner',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
      children: [pipe_cfg_node, obstacle_cell_node, conflict_setting_node, group_setting_node, auto_run_cfg_node],
    }
    return root;
  }

  static load_json(json: any, proj_manager: any, scene: THREE.Scene): SmoothRunner {
    const runner = new SmoothRunner(json.smooth_name, json.task_name, json.project_name);

    // 还原 pipes_cfg
    if (json.pipes_cfg) {
      for (const [pipe_name, pipe_cfg] of Object.entries(json.pipes_cfg)) {
        const cfg = pipe_cfg as any;
        runner.pipes_cfg.set(pipe_name, new PipeCfg(
          pipe_name,
          cfg.radius,
          cfg.position,
          cfg.direction,
          cfg.group_name,
          cfg.is_input,
          cfg.radius_scale
        ));
      }
    }

    // 还原 conflict_setting 和 auto_run_cfg
    runner.conflict_setting = json.conflict_setting || runner.conflict_setting;
    runner.auto_run_cfg = json.auto_run_cfg || runner.auto_run_cfg;

    // 还原 group_cells
    if (Array.isArray(json.setting)) {
      for (const group of json.setting) {
        try {
          const group_cell = new GroupCell(
            json.smooth_name,
            json.task_name,
            json.project_name,
            group.group_name,
            group.cell_setting?.group_idx ?? 0
          );
          group_cell.aid_path_info = group.aid_path_info || [];
          group_cell.graph_cfg = group.graph_cfg || { seg_node_min: 10, seg_node_max: 50 };
          group_cell.c_table = group.cell_setting?.c_table || {};

          // 还原 segments（第一步：全部 set_result）
          if (Array.isArray(group.segments)) {
            for (const seg of group.segments) {
              try {
                const degree = seg.bspline_degree ?? 3;
                const c_idxs = seg.cell_setting?.c_idxs;
                if (!Array.isArray(c_idxs) || !group_cell.c_table) {
                  console.warn('segment 缺少 c_idxs 或 c_table，已跳过', seg);
                  continue;
                }
                const left_intercept_num = Math.max(seg.cell_setting?.left_intercept_num ?? degree, degree);
                const right_intercept_num = Math.max(seg.cell_setting?.right_intercept_num ?? degree, degree);
                const seg_cell = new SegmentCell(
                  json.smooth_name,
                  json.task_name,
                  json.project_name,
                  `${group.group_name}/seg_${seg.seg_idx}`,
                  seg.seg_idx,
                  seg.group_idx,
                  c_idxs,
                  group_cell,
                  scene,
                  left_intercept_num,
                  right_intercept_num,
                  seg.cell_setting?.left_clamp ?? false,
                  seg.cell_setting?.right_clamp ?? false
                );
                seg_cell.degree = degree;
                seg_cell.density = seg.bspline_density;
                seg_cell.left_overlap_ratio = seg.left_overlap_ratio;
                seg_cell.right_overlap_ratio = seg.right_overlap_ratio;
                seg_cell.left_intercept_ratio = seg.left_intercept_ratio;
                seg_cell.right_intercept_ratio = seg.right_intercept_ratio;
                seg_cell.segment_scale = seg.segment_scale;
                seg_cell.with_analytical_curvature_computed = seg.with_analytical_curvature_computed;
                seg_cell.with_analytical_tangent_computed = seg.with_analytical_tangent_computed;
                seg_cell.cost_setting = seg.cost_setting;
                
                // 设置 left_vector 和 right_vector
                seg_cell.left_vector = seg.cell_setting?.left_vector || [0,0,0];
                seg_cell.right_vector = seg.cell_setting?.right_vector || [0,0,0];
                
                // 如果 left_clamp 为 true，从 pipes_cfg 中获取正确的 left_vector
                if (seg_cell.left_clamp) {
                  const x = group_cell.c_table['x'][seg_cell.c_idxs[0]];
                  const y = group_cell.c_table['y'][seg_cell.c_idxs[0]];
                  const z = group_cell.c_table['z'][seg_cell.c_idxs[0]];
                  runner.pipes_cfg.forEach((pipe_cfg, pipe_name) => {
                    if (pipe_cfg.position[0] === x && pipe_cfg.position[1] === y && pipe_cfg.position[2] === z) {
                      seg_cell.left_vector = pipe_cfg.direction;
                      seg_cell.is_left_input = pipe_cfg.is_input;
                    }
                  });
                }
                
                // 如果 right_clamp 为 true，从 pipes_cfg 中获取正确的 right_vector
                if (seg_cell.right_clamp) {
                  const x = group_cell.c_table['x'][seg_cell.c_idxs[seg_cell.c_idxs.length - 1]];
                  const y = group_cell.c_table['y'][seg_cell.c_idxs[seg_cell.c_idxs.length - 1]];
                  const z = group_cell.c_table['z'][seg_cell.c_idxs[seg_cell.c_idxs.length - 1]];
                  runner.pipes_cfg.forEach((pipe_cfg, pipe_name) => {
                    if (pipe_cfg.position[0] === x && pipe_cfg.position[1] === y && pipe_cfg.position[2] === z) {
                      seg_cell.right_vector = pipe_cfg.direction;
                      seg_cell.is_right_input = pipe_cfg.is_input;
                    }
                  });
                }
                
                seg_cell.path_points = seg.path_points || [];
                seg_cell.radius_list = seg.radius_list || [];
                group_cell.seg_dict[seg_cell.name] = seg_cell;
              } catch (e) {
                console.error('segment 导入异常:', e, seg);
              }
            }
            // 补充：全部 set_result
            for (const seg of group.segments) {
              const seg_cell = group_cell.seg_dict[`${group.group_name}/seg_${seg.seg_idx}`];
              if (seg_cell) {
                seg_cell.set_result(seg.path_points || [], seg.radius_list || [], scene);
              }
            }
          }

          // 还原 connectors（第二步：推断 path_points 并 set_result）
          if (Array.isArray(group.connectors)) {
            for (const con of group.connectors) {
              try {
                const con_degree = con.bspline_degree ?? 3;
                const con_left_intercept_num = Math.max(con.cell_setting?.left_intercept_num ?? con_degree, con_degree);
                const con_right_intercept_num = Math.max(con.cell_setting?.right_intercept_num ?? con_degree, con_degree);
                
                // 获取连接的segments（用于后续的path_points生成）
                const left_seg = group_cell.seg_dict[`${group.group_name}/seg_${con.left_seg_idx}`];
                const right_seg = group_cell.seg_dict[`${group.group_name}/seg_${con.right_seg_idx}`];
                
                // 优先使用后端返回的c_idxs，如果没有才重新生成
                let c_idxs: number[] = [];
                if (con.cell_setting?.c_idxs && Array.isArray(con.cell_setting.c_idxs) && con.cell_setting.c_idxs.length > 0) {
                  // 使用后端返回的c_idxs
                  c_idxs = con.cell_setting.c_idxs;
                } else {
                  // 从连接的 segments 的端点获取控制点
                  if (left_seg && right_seg && left_seg.c_idxs && right_seg.c_idxs) {
                    // 使用左段的最后一个控制点和右段的第一个控制点
                    const left_last_c_idx = left_seg.c_idxs[left_seg.c_idxs.length - 1];
                    const right_first_c_idx = right_seg.c_idxs[0];
                    
                    // 如果两个控制点不同，则使用它们
                    if (left_last_c_idx !== right_first_c_idx) {
                      c_idxs = [left_last_c_idx, right_first_c_idx];
                    } else {
                      // 如果相同，则只使用一个控制点
                      c_idxs = [left_last_c_idx];
                    }
                  }
                  console.log(`[ConnectorCell] 重新生成c_idxs:`, c_idxs);
                }
                
                const con_cell = new ConnectorCell(
                  json.smooth_name,
                  json.task_name,
                  json.project_name,
                  `${group.group_name}/con_${con.con_idx}`,
                  con.con_idx,
                  con.group_idx,
                  con.left_seg_idx,
                  con.right_seg_idx,
                  group_cell,
                  scene,
                  c_idxs, // 使用后端返回的或生成的控制点索引
                  con_left_intercept_num,
                  con_right_intercept_num,
                  con.cell_setting?.left_clamp ?? false,
                  con.cell_setting?.right_clamp ?? false
                );
                con_cell.degree = con_degree;
                con_cell.density = con.bspline_density;
                con_cell.with_analytical_curvature_computed = con.with_analytical_curvature_computed;
                con_cell.with_analytical_tangent_computed = con.with_analytical_tangent_computed;
                con_cell.cost_setting = con.cost_setting;
                
                // 设置 left_vector 和 right_vector
                con_cell.left_vector = con.cell_setting?.left_vector || [0,0,0];
                con_cell.right_vector = con.cell_setting?.right_vector || [0,0,0];
                
                // 如果 left_clamp 为 true，从 pipes_cfg 中获取正确的 left_vector
                if (con_cell.left_clamp && c_idxs.length > 0) {
                  const x = group_cell.c_table['x'][c_idxs[0]];
                  const y = group_cell.c_table['y'][c_idxs[0]];
                  const z = group_cell.c_table['z'][c_idxs[0]];
                  runner.pipes_cfg.forEach((pipe_cfg, pipe_name) => {
                    if (pipe_cfg.position[0] === x && pipe_cfg.position[1] === y && pipe_cfg.position[2] === z) {
                      con_cell.left_vector = pipe_cfg.direction;
                    }
                  });
                }
                
                // 如果 right_clamp 为 true，从 pipes_cfg 中获取正确的 right_vector
                if (con_cell.right_clamp && c_idxs.length > 0) {
                  const x = group_cell.c_table['x'][c_idxs[c_idxs.length - 1]];
                  const y = group_cell.c_table['y'][c_idxs[c_idxs.length - 1]];
                  const z = group_cell.c_table['z'][c_idxs[c_idxs.length - 1]];
                  runner.pipes_cfg.forEach((pipe_cfg, pipe_name) => {
                    if (pipe_cfg.position[0] === x && pipe_cfg.position[1] === y && pipe_cfg.position[2] === z) {
                      con_cell.right_vector = pipe_cfg.direction;
                    }
                  });
                }
                
                con_cell.path_points = con.path_points || [];
                con_cell.radius_list = con.radius_list || [];
                // 打印日志辅助排查
                // 只要left_seg/right_seg有效，就直接插值生成50个点
                if (
                  left_seg && right_seg &&
                  Array.isArray(left_seg.path_points) && left_seg.path_points.length > 0 &&
                  Array.isArray(right_seg.path_points) && right_seg.path_points.length > 0
                ) {
                  const start = left_seg.path_points[left_seg.path_points.length - 1];
                  const end = right_seg.path_points[0];
                  const start_r = (left_seg.radius_list && left_seg.radius_list[left_seg.radius_list.length - 1]) || 1;
                  const end_r = (right_seg.radius_list && right_seg.radius_list[0]) || 1;
                  // 插值 50 个点
                  const N = 50;
                  const new_points = [];
                  const new_radius = [];
                  for (let i = 0; i < N; i++) {
                    const t = i / (N - 1);
                    new_points.push([
                      start[0] * (1 - t) + end[0] * t,
                      start[1] * (1 - t) + end[1] * t,
                      start[2] * (1 - t) + end[2] * t
                    ]);
                    new_radius.push(start_r * (1 - t) + end_r * t);
                  }
                  con_cell.path_points = new_points;
                  con_cell.radius_list = new_radius;
                }
                group_cell.connector_dict[con_cell.name] = con_cell;
                con_cell.set_result(con_cell.path_points, con_cell.radius_list, scene);
              } catch (e) {
                console.error('connector 导入异常:', e, con);
              }
            }
          }

          // 还原 paths
          if (Array.isArray(group.paths)) {
            for (const path of group.paths) {
              try {
                const cell_idxs = path.cell_idxs;
                const cell_types = path.cell_types;
                if (!Array.isArray(cell_idxs) || !Array.isArray(cell_types)) {
                  console.warn('path 缺少 cell_idxs 或 cell_types，已跳过', path);
                  continue;
                }
                const path_cell = new PathCell(
                  json.smooth_name,
                  json.task_name,
                  json.project_name,
                  path.path_name,
                  path.path_idx,
                  path.group_idx,
                  cell_idxs,
                  cell_types,
                  group_cell
                );
                path_cell.cost_cfgs = path.cost_setting?.cost_cfgs || [];
                path_cell.element_cost_cfgs = path.cost_setting?.element_cost_cfgs || [];
                // path_cell.path_points = path.path_points || [];
                // path_cell.radius_list = path.radius_list || [];
                group_cell.path_dict[path.path_name] = path_cell;
              } catch (e) {
                console.error('path 导入异常:', e, path);
              }
            }
          }

          runner.group_cells.set(group.group_name, group_cell);
          // 补充：主动生成 mesh
          Object.values(group_cell.seg_dict).forEach(seg => seg.add_mesh(scene));
          Object.values(group_cell.connector_dict).forEach(con => con.add_mesh(scene));
        } catch (e) {
          console.error('group_cell 导入异常:', e, group);
        }
      }
    }
    return runner;
  }
  
}