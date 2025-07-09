import type { Scene } from 'three';
import * as THREE from 'three';

import type { TreeNode } from '@/types/data_type';
import {
  color_to_hex,
  create_tube,
} from '@/utils/visulize_utils';

import { MeshObj } from '../mesh_type';
import { BsplineUtils } from './bspline_utils';
import { create_circle_set, create_surface_mesh_from_cells } from '@/utils/visulize_utils';


export class CoarseTubeMesh extends MeshObj {
  center_points: number[][];
  radius_list: number[];
  direction_list: number[][];
  color: number[];

  constructor(
    name: string,
    center_points: number[][],
    radius_list: number[],
    direction_list: number[][],
    color: number[],
    scene: Scene
  ) {
    super(name);
    this.center_points = center_points;
    this.radius_list = radius_list;
    this.direction_list = direction_list;
    this.color = color;

    this.add_mesh(scene);
  }

  add_mesh(scene: Scene): void {
    const coarse_mesh = create_circle_set(
      this.center_points, this.radius_list, this.direction_list, { color: color_to_hex(this.color) }
    );
    coarse_mesh.name = this.name;
    this.mesh_dict[this.name] = coarse_mesh;
    scene.add(coarse_mesh);
  }
}

interface PartSurfaceMeshInfo {
  'name': string;
  'path': number[][];
  'left_vec': number[] | null;
  'right_vec': number[] | null;
  'is_left_input': boolean;
  'is_right_input': boolean;
  'is_left_cut': boolean;
  'is_right_cut': boolean;
}

export class FineSurfaceMesh extends MeshObj {
  vertices: number[][];
  cells: number[][];
  color: number[];

  constructor(name: string, vertices: number[][], cells: number[][], color: number[], scene: Scene) {
    super(name);
    this.vertices = vertices;
    this.cells = cells;
    this.color = color;
    this.add_mesh(scene);
  }

  add_mesh(scene: Scene): void {
    const surface_mesh = create_surface_mesh_from_cells(
      this.vertices, this.cells, { color: color_to_hex(this.color) }
    );
    surface_mesh.name = this.name;
    this.mesh_dict[surface_mesh.name] = surface_mesh;
    scene.add(surface_mesh);
  }
}

// -------------------------------------------------
export class SegmentCell extends MeshObj {
  smooth_name: string;
  task_name: string;
  project_name: string;

  idx: number;
  group_idx: number;
  c_idxs: number[];
  parent_cell: GroupCell;

  left_clamp: boolean = false;
  right_clamp: boolean = false;
  left_vector: number[] = [0., 0., 0.];
  right_vector: number[] = [0., 0., 0.];
  is_left_input: boolean = false;
  is_right_input: boolean = false;

  density: number = 0.5;
  degree: number = 3;
  segment_scale: number = 1.0;

  with_analytical_curvature_computed: boolean = true;
  with_analytical_tangent_computed: boolean = false;

  left_overlap_ratio: number = 0.25;
  right_overlap_ratio: number = 0.25;
  left_intercept_ratio: number = 0.5;
  right_intercept_ratio: number = 0.5;
  left_intercept_num: number;
  right_intercept_num: number;

  cost_setting: {
    path_thickness: number;
    obstacle_thickness: number;
    cost_cfgs: {
      cost_type: string;
      data_type: string;
      method: string;
      weight: number;
    }[];
  } = {
    path_thickness: 0.0,
    obstacle_thickness: 0.0,
    cost_cfgs: [],
  };

  color: number[] = [];

  tube_mesh: MeshObj | null = null;
  line_mesh: THREE.Mesh | null = null;
  path_points: number[][] = [];
  radius_list: number[] = [];

  constructor(
    smooth_name: string, task_name: string, project_name: string, segment_name: string,
    idx: number, group_idx: number, c_idxs: number[], parent_cell: GroupCell, scene: Scene,
    // just for front end visualization
    left_intercept_num: number, right_intercept_num: number, 
    left_clamp: boolean, right_clamp: boolean
  ) {
    super(segment_name);
    this.smooth_name = smooth_name;
    this.task_name = task_name;
    this.project_name = project_name;
    
    this.idx = idx;
    this.group_idx = group_idx;
    this.c_idxs = c_idxs;
    this.parent_cell = parent_cell;
    this.left_intercept_num = left_intercept_num;
    this.right_intercept_num = right_intercept_num;
    this.left_clamp = left_clamp;
    this.right_clamp = right_clamp;
    this.color = [Math.random() * 255, Math.random() * 255, Math.random() * 255];
    this.add_mesh(scene);
  }

  protected matrix_multiply(matrix: number[][], points: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < matrix.length; i++) {
      result[i] = [];
      for (let j = 0; j < points[0].length; j++) {
        result[i][j] = 0;
        for (let k = 0; k < matrix[0].length; k++) {
          result[i][j] += matrix[i][k] * points[k][j];
        }
      }
    }
    return result;
  }

  add_mesh(scene: Scene): void {
    const control_points: number[][] = []
    this.c_idxs.forEach(c_idx => {
      control_points.push(
        [this.parent_cell.c_table['x'][c_idx], this.parent_cell.c_table['y'][c_idx], this.parent_cell.c_table['z'][c_idx]
      ]);
    })

    const n_controls = this.c_idxs.length;
    const knot_cfg = BsplineUtils.create_knot_cfg(n_controls, this.degree, this.left_clamp, this.right_clamp)
    const knot = knot_cfg['knot'];
    const fill_last_one = knot_cfg['fill_last_one'];

    const sample_num = 100
    const ts = BsplineUtils.create_ts(
      sample_num, this.degree, knot, this.left_clamp, this.right_clamp,
      this.left_intercept_num, this.right_intercept_num
    )
    const spline_mat = BsplineUtils.compute_mat(ts, n_controls, this.degree, knot, fill_last_one)
    const points = this.matrix_multiply(spline_mat, control_points);
    this.path_points = points;

    const control_radius: number[][] = []
    this.c_idxs.forEach(c_idx => {
      control_radius.push(
        [this.parent_cell.c_table['radius'][c_idx]]
      );
    })
    const radius_list: number[] = this.matrix_multiply(spline_mat, control_radius).map(radius => radius[0]);
    this.radius_list = radius_list;

    // 粗管道延迟创建，不在add_mesh中创建
    this.tube_mesh = null;

    // 细线
    this.line_mesh = create_tube(points, 0.05, { color: color_to_hex(this.color), opacity: 0.8 }) as THREE.Mesh;
    this.line_mesh.name = this.name + '_line';
    this.line_mesh.visible = true;
    this.mesh_dict[this.line_mesh.name] = this.line_mesh;

    super.add_mesh(scene);
  }

  show_tube_mesh(scene: Scene) {
    if (!this.tube_mesh) {
      const direction_list: number[][] = []
      for (let i = 0; i < this.path_points.length; i++) {
        if (i == 0) {
          if (this.left_clamp) {
            direction_list.push(this.left_vector);
          } else {
            const direction = [
              this.path_points[i + 1][0] - this.path_points[i][0],
              this.path_points[i + 1][1] - this.path_points[i][1],
              this.path_points[i + 1][2] - this.path_points[i][2]
            ]
            direction_list.push(direction);
          }
        } else if (i == this.path_points.length - 1) {
          if (this.right_clamp) {
            direction_list.push(this.right_vector);
          } else {
            const direction = [
              this.path_points[i][0] - this.path_points[i - 1][0],
              this.path_points[i][1] - this.path_points[i - 1][1],
              this.path_points[i][2] - this.path_points[i - 1][2]
            ]
            direction_list.push(direction);
          }
        } else {
          const direction = [
            this.path_points[i + 1][0] - this.path_points[i][0],
            this.path_points[i + 1][1] - this.path_points[i][1],
            this.path_points[i + 1][2] - this.path_points[i][2]
          ]
          direction_list.push(direction);
        }
      }

      this.tube_mesh = new CoarseTubeMesh(
        this.name + '_coarse',
        this.path_points, this.radius_list, direction_list, this.color, scene
      );

    } else {
      this.tube_mesh.show_mesh();
    }
  }

  hide_tube_mesh() {
    if (this.tube_mesh) {
      this.tube_mesh.hide_mesh();
    }
  }

  show_line_mesh() {
    if (this.line_mesh) {
      this.line_mesh.visible = true;
    }
  }

  hide_line_mesh() {
    if (this.line_mesh) {
      this.line_mesh.visible = false;
    }
  }

  export_json(): any {
    return {
      seg_idx: this.idx,
      group_idx: this.group_idx,
      bspline_degree: this.degree,
      bspline_density: this.density,
      left_overlap_ratio: this.left_overlap_ratio,
      right_overlap_ratio: this.right_overlap_ratio,
      left_intercept_ratio: this.left_intercept_ratio,
      right_intercept_ratio: this.right_intercept_ratio,
      segment_scale: this.segment_scale,
      with_analytical_curvature_computed: this.with_analytical_curvature_computed,
      with_analytical_tangent_computed: this.with_analytical_tangent_computed,
      cost_setting: this.cost_setting,
      cell_setting: {
        seg_idx: this.idx,
        group_idx: this.group_idx,
        c_idxs: this.c_idxs,
        left_clamp: this.left_clamp,
        right_clamp: this.right_clamp,
        left_vector: this.left_vector,
        right_vector: this.right_vector,
      }
    }
  }

  export_tree(): TreeNode {
    return {
      label: this.name,
      style: 'SegmentCell',
      data: {
        'SegIdx': this.idx,
        'GroupIdx': this.group_idx,
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
      },
    }
  }

  export_surface_mesh_json(): PartSurfaceMeshInfo {
    const path: number[][] = []
    for (let i = 0; i < this.path_points.length; i++) {
      path.push([
        this.path_points[i][0], this.path_points[i][1], this.path_points[i][2], this.radius_list[i]
      ])
    }
    return {
      name: this.name,
      path: path,
      left_vec: this.left_clamp ? this.left_vector : null,
      right_vec: this.right_clamp ? this.right_vector : null,
      is_left_input: false,
      is_right_input: false,
      is_left_cut: false,
      is_right_cut: false,
    }
  }
}

export class ConnectorCell extends SegmentCell {
  left_seg_idx: number;
  right_seg_idx: number;

  constructor(
    smooth_name: string, task_name: string, project_name: string, connector_name: string,
    idx: number, group_idx: number, left_seg_idx: number, right_seg_idx: number, parent_cell: GroupCell, scene: Scene,
    // just for front end visualization
    c_idxs: number[], left_intercept_num: number, right_intercept_num: number
  ) {
    super(
      smooth_name, task_name, project_name, 
      connector_name, idx, group_idx, c_idxs, parent_cell, scene,
      // just for front end visualization
      left_intercept_num, right_intercept_num,
      false, false
    );
    this.left_seg_idx = left_seg_idx;
    this.right_seg_idx = right_seg_idx;
  }

  export_json(): any {
    return {
      con_idx: this.idx,
      group_idx: this.group_idx,
      bspline_degree: this.degree,
      bspline_density: this.density,
      left_seg_idx: this.left_seg_idx,
      right_seg_idx: this.right_seg_idx,
      with_analytical_curvature_computed: this.with_analytical_curvature_computed,
      with_analytical_tangent_computed: this.with_analytical_tangent_computed,
      cost_setting: this.cost_setting,
      cell_setting: {
        con_idx: this.idx,
        group_idx: this.group_idx,
        left_seg_idx: this.left_seg_idx,
        right_seg_idx: this.right_seg_idx
      }
    }
  }

  export_tree(): TreeNode {
    return {
      label: this.name,
      style: 'ConnectorCell',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'ConIdx': this.idx,
        'GroupIdx': this.group_idx,
      },
    }
  }
}

export class PathCell {
  smooth_name: string;
  task_name: string;
  project_name: string;
  name: string;

  idx: number;
  group_idx: number;
  cell_idxs: number[];
  cell_types: string[];
  parent_cell: GroupCell;

  cost_cfgs: {
    cost_type: string;
    data_type: string;
    method: string;
    weight: number;
  }[] = [];
  element_cost_cfgs: {
    cost_type: string;
    vector_length: number;
    vector: number[];
    weight: number;
  }[] = [];
  
  constructor(
    smooth_name: string, task_name: string, project_name: string,
    name: string, idx: number, group_idx: number, cell_idxs: number[], cell_types: string[], parent_cell: GroupCell
  ) {
    this.smooth_name = smooth_name;
    this.task_name = task_name;
    this.project_name = project_name;

    this.name = name;
    this.idx = idx;
    this.group_idx = group_idx;
    this.cell_idxs = cell_idxs;
    this.cell_types = cell_types;
    this.parent_cell = parent_cell;

    this.cost_cfgs = [];
    this.element_cost_cfgs = [];
  }

  export_json(): any {
    return {
      path_name: this.name,
      path_idx: this.idx,
      group_idx: this.group_idx,
      cell_idxs: this.cell_idxs,
      cell_types: this.cell_types,
      cost_setting: {
        cost_cfgs: this.cost_cfgs,
        element_cost_cfgs: this.element_cost_cfgs,
      },
      cell_setting: {
        path_name: this.name,
        path_idx: this.idx,
        group_idx: this.group_idx,
        cell_idxs: this.cell_idxs,
        cell_types: this.cell_types,
      }
    }
  }

  export_tree(): TreeNode {
    return {
      label: this.name,
      style: 'PathCell',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'PathIdx': this.idx,
        'GroupIdx': this.group_idx,
      },
    }
  }
}

export class GroupCell {
  smooth_name: string;
  task_name: string;
  project_name: string;
  group_name: string;
  group_idx: number;

  aid_path_info: {
      name: string;
      from_name: string;
      to_name: string;
  }[];
  graph_cfg: {
      seg_node_min: number;
      seg_node_max: number;
  };

  c_table: {
    [key: string]: number[];
  };
  seg_dict: {
    [key: string]: SegmentCell;
  };
  connector_dict: {
    [key: string]: ConnectorCell;
  };
  path_dict: {
    [key: string]: PathCell;
  };

  fine_surface_mesh: FineSurfaceMesh | null = null;

  constructor(smooth_name: string, task_name: string, project_name: string, group_name: string, group_idx: number) {
    this.smooth_name = smooth_name;
    this.task_name = task_name;
    this.project_name = project_name;
    this.group_name = group_name;
    this.group_idx = group_idx;

    this.aid_path_info = [];
    this.graph_cfg = {
        seg_node_min: 15,
        seg_node_max: 70
    };
    this.c_table = {};
    this.seg_dict = {};
    this.connector_dict = {};
    this.path_dict = {};
  }

  update_c_table(c_table: { [key: string]: number[] }, scene: Scene): void {
    this.c_table = c_table;
    for (const seg of Object.values(this.seg_dict)) {
      seg.update_mesh(scene);
    }
    for (const con of Object.values(this.connector_dict)) {
      con.update_mesh(scene);
    }
  }

  export_json(): any {
    return {
      group_name: this.group_name,
      aid_path_info: this.aid_path_info,
      graph_cfg: this.graph_cfg,
      segments: Object.values(this.seg_dict).map(seg => seg.export_json()),
      connectors: Object.values(this.connector_dict).map(con => con.export_json()),
      paths: Object.values(this.path_dict).map(path => path.export_json()),
      cell_setting: {
        group_idx: this.group_idx,
        c_table: this.c_table
      }
    }
  }
  
  export_tree(): TreeNode {
    const seg_node: TreeNode = {
      label: 'Segments',
      style: 'SegmentCells',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'GroupIdx': this.group_idx,
      },
      children: [],
    };
    const con_node: TreeNode = {
      label: 'Connectors',
      style: 'ConnectorCells',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'GroupIdx': this.group_idx,
      },
      children: [],
    };
    const path_node: TreeNode = {
      label: 'Paths',
      style: 'PathCells',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'GroupIdx': this.group_idx,
      },
      children: [],
    };

    for (const seg of Object.values(this.seg_dict)) {
      seg_node.children!.push(seg.export_tree());
    }

    for (const con of Object.values(this.connector_dict)) {
      con_node.children!.push(con.export_tree());
    }

    for (const path of Object.values(this.path_dict)) {
      path_node.children!.push(path.export_tree());
    }

    // ---------------------------------------
    const aid_path_node: TreeNode = {
      label: 'Aid Paths',
      style: 'AidPaths',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'GroupIdx': this.group_idx,
      },
      children: this.aid_path_info.map(info => ({
        label: `${info.name} (${info.from_name} -> ${info.to_name})`,
        style: 'AidPath',
        data: {
          'Project': this.project_name,
          'SmoothName': this.smooth_name,
          'TaskName': this.task_name,
          'GroupIdx': this.group_idx,
          'Name': info.name,
        },
      })),
    };

    // ---------------------------------------
    const graph_node: TreeNode = {
      label: 'Graph',
      style: 'Graph',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'GroupIdx': this.group_idx,
      },
    };

    return {
      label: `${this.group_name}`,
      style: 'GroupCell',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'GroupIdx': this.group_idx,
      },
      children: [aid_path_node, graph_node, seg_node, con_node, path_node],
    }
  }

  export_surface_mesh_json(): { [key: string]: PartSurfaceMeshInfo } {
    const part_surface_mesh_infos: { [key: string]: PartSurfaceMeshInfo } = {};
    for (const seg of Object.values(this.seg_dict)) {
      part_surface_mesh_infos[seg.name] = seg.export_surface_mesh_json();
    }
    return part_surface_mesh_infos;
  }

  generate_surface_mesh(vertices: number[][], cells: number[][], color: number[], scene: Scene): void {
    this.fine_surface_mesh = new FineSurfaceMesh(
      this.group_name + '_surface', vertices, cells, color, scene
    );
  }

  hide_surface_mesh(): void {
    if (this.fine_surface_mesh) {
      this.fine_surface_mesh.hide_mesh();
    }
  }
}
