import { GroupCell } from "./smooth_nodes";
import type { TreeNode } from '@/types/data_type';
import { MeshObj } from "../mesh_type";
import * as THREE from "three";
import { remove_mesh, add_mesh, create_points, color_to_hex } from "@/utils/visulize_utils";
import { ElMessage } from "element-plus";
import { distance } from "three/tsl";

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
      label: 'Obstacle Cell',
      style: 'ObstacleCell',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
      children: [this.obstacle_cell!.export_tree(this.name, this.task_name, this.project_name)],
    }

    const auto_run_cfg_node: TreeNode = {
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
      label: 'Conflict Setting',
      style: 'ConflictSetting',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
    }

    const group_setting_node: TreeNode = {
      label: 'Group Setting',
      style: 'GroupSetting',
      data: {
        'Project': this.project_name,
        'SmoothName': this.name,
        'TaskName': this.task_name,
      },
      children: Array.from(this.group_cells.values()).map(group_cell => group_cell.export_tree()),
    }

    const root: TreeNode = {
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
  
}