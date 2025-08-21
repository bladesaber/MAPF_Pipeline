import type { Scene } from 'three';
import * as THREE from 'three';

import type { TreeNode } from '@/types/data_type';
import {
  color_to_hex,
  create_circle_set,
  create_surface_mesh_from_cells,
  create_tube,
} from '@/utils/visulize_utils';

import { MeshObj } from '../mesh_type';
import { BsplineUtils } from './bspline_utils';

// 动态颜色生成函数
function generateDistinctColors(count: number): number[][] {
  const colors: number[][] = [];
  
  if (count <= 10) {
    // 如果组数不多，使用预定义的高质量颜色
    const predefinedColors: number[][] = [
      [0.90, 0.11, 0.14], // 红
      [0.22, 0.49, 0.72], // 蓝
      [0.30, 0.68, 0.31], // 绿
      [1.00, 0.60, 0.00], // 橙
      [0.60, 0.31, 0.64], // 紫
      [1.00, 0.85, 0.18], // 黄
      [0.65, 0.34, 0.16], // 棕
      [0.40, 0.76, 0.65], // 青
      [0.99, 0.51, 0.75], // 粉
      [0.60, 0.60, 0.60], // 灰
    ];
    return predefinedColors.slice(0, count);
  }
  
  // 动态生成颜色：使用黄金角算法确保颜色分布均匀
  const goldenAngle = Math.PI * (3 - Math.sqrt(5)); // 黄金角
  const saturation = 0.7; // 饱和度
  const lightness = 0.6;  // 亮度
  
  for (let i = 0; i < count; i++) {
    const hue = (i * goldenAngle) % (2 * Math.PI); // 色相
    const r = Math.cos(hue) * saturation + lightness;
    const g = Math.cos(hue + 2 * Math.PI / 3) * saturation + lightness;
    const b = Math.cos(hue + 4 * Math.PI / 3) * saturation + lightness;
    
    // 确保颜色值在 [0, 1] 范围内
    colors.push([
      Math.max(0, Math.min(1, r)),
      Math.max(0, Math.min(1, g)),
      Math.max(0, Math.min(1, b))
    ]);
  }
  
  return colors;
}

// 全局颜色缓存，避免重复计算
const colorCache = new Map<number, number[][]>();

// 获取指定数量的颜色
function getColorsForGroupCount(count: number): number[][] {
  if (!colorCache.has(count)) {
    colorCache.set(count, generateDistinctColors(count));
  }
  return colorCache.get(count)!;
}

// 全局组计数器，用于跟踪当前已创建的组数
let globalGroupCounter = 0;

// 工具函数：提取纯name
function extractPureName(name: string, group_name: string) {
  return name.startsWith(group_name + '/') ? name.slice(group_name.length + 1) : name;
}

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
    // 将0-1范围的颜色转换为0-255范围
    const color_255 = this.color.map(c => Math.floor(c * 255));
    const coarse_mesh = create_circle_set(
      this.center_points, this.radius_list, this.direction_list, { color: color_to_hex(color_255) }
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
  protected scene: Scene;

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
    super(extractPureName(segment_name, parent_cell.group_name));
    this.smooth_name = smooth_name;
    this.task_name = task_name;
    this.project_name = project_name;
    
    this.idx = idx;
    this.group_idx = group_idx;
    this.c_idxs = c_idxs;
    this.parent_cell = parent_cell;
    this.scene = scene;
    this.left_intercept_num = left_intercept_num;
    this.right_intercept_num = right_intercept_num;
    this.left_clamp = left_clamp;
    this.right_clamp = right_clamp;
    this.color = parent_cell.group_color;
   
    // 注册对控制点的引用
    const referenceId = `${parent_cell.group_name}/${this.name}`;
    this.c_idxs.forEach(c_idx => {
      parent_cell.register_control_point_reference(c_idx, referenceId, scene);
    });
   
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
    const meshName = `${this.parent_cell.group_name}/${this.name}`;
    const color = this.color.length > 0 ? this.color : [0.5, 0.5, 0.5];
    
    // 清理所有同名且userData.type==='smooth'的mesh
    scene.children
      .filter(obj => obj.name === meshName && obj.userData && obj.userData.type === 'smooth')
      .forEach(mesh => {
        scene.remove(mesh);
        if ((mesh as THREE.Mesh).geometry) (mesh as THREE.Mesh).geometry.dispose();
        if ((mesh as THREE.Mesh).material && (mesh as THREE.Mesh).material instanceof THREE.Material) {
          ((mesh as THREE.Mesh).material as THREE.Material).dispose();
        }
      });
    if (!this.c_idxs || !this.parent_cell?.c_table || this.c_idxs.length === 0) {
      console.warn('SegmentCell.add_mesh 缺少 c_idxs 或 c_table,跳过渲染', this.name, this.c_idxs, this.parent_cell?.c_table);
      return;
    }
    const control_points: number[][] = []
    this.c_idxs.forEach(c_idx => {
      control_points.push([
        this.parent_cell.c_table['x'][c_idx], 
        this.parent_cell.c_table['y'][c_idx], 
        this.parent_cell.c_table['z'][c_idx]
      ]);
    })

    const n_controls = this.c_idxs.length;
    const knot_cfg = BsplineUtils.create_knot_cfg(n_controls, this.degree, this.left_clamp, this.right_clamp)
    const knot = knot_cfg['knot'];
    const fill_last_one = knot_cfg['fill_last_one'];

    const sample_num = 200
    const ts = BsplineUtils.create_ts(
      sample_num, this.degree, knot, this.left_clamp, this.right_clamp,
      this.left_intercept_num, this.right_intercept_num
    )
    const spline_mat = BsplineUtils.compute_mat(ts, n_controls, this.degree, knot, fill_last_one)
    const points = this.matrix_multiply(spline_mat, control_points);
    this.path_points = points;

    const control_radius: number[][] = []
    this.c_idxs.forEach(c_idx => {
      control_radius.push([
        this.parent_cell.c_table['radius'][c_idx]
      ]);
    })
    const radius_list: number[] = this.matrix_multiply(spline_mat, control_radius).map(radius => radius[0]);
    this.radius_list = radius_list;

    // 粗管道延迟创建，不在add_mesh中创建
    this.tube_mesh = null;

    // 细线
    if (!this.path_points || this.path_points.length < 2 || !Array.isArray(this.path_points[0])) {
      console.warn('add_mesh: path_points 无效，无法创建 line_mesh', this.name, this.path_points);
      return;
    }
    // 将0-1范围的颜色转换为0-255范围
    const color_255 = this.color.map(c => Math.floor(c * 255));
    // 新命名格式 group_name/name
    this.line_mesh = create_tube(this.path_points, 0.05, { color: color_to_hex(color_255), opacity: 0.8 }) as THREE.Mesh;
    this.line_mesh.name = meshName;
    this.line_mesh.visible = true;
    this.line_mesh.userData = { group_name: this.parent_cell.group_name, name: this.name, group_idx: this.group_idx, type: 'smooth' };
    this.mesh_dict[meshName] = this.line_mesh;
    if (scene && !scene.children.includes(this.line_mesh)) {
      scene.add(this.line_mesh);
    }

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
    if (!this.line_mesh) this.add_mesh(this.scene);
    if (this.line_mesh) {
      this.line_mesh.visible = true;
    } else {
      console.warn('[show_line_mesh] 无法显示 line_mesh', this.name);
    }
  }

  hide_line_mesh() {
    // 循环移除所有同名mesh，彻底清理
    const meshName = `${this.parent_cell.group_name}/${this.name}`;
    let mesh;
    while ((mesh = this.scene.getObjectByName(meshName))) {
      this.scene.remove(mesh);
      if ((mesh as THREE.Mesh).geometry) (mesh as THREE.Mesh).geometry.dispose();
      if ((mesh as THREE.Mesh).material && (mesh as THREE.Mesh).material instanceof THREE.Material) {
        ((mesh as THREE.Mesh).material as THREE.Material).dispose();
      }
    }
    this.line_mesh = null;
  }

  update_mesh(scene: Scene) {
    this.hide_line_mesh();
    this.hide_tube_mesh();
    this.add_mesh(scene);
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
      id: `${this.project_name}-${this.smooth_name}-${this.parent_cell.group_name}-segment-${this.name}`,
      label: `${this.parent_cell.group_name}/${this.name}`,
      style: 'SegmentCell',
      data: {
        meshName: `${this.parent_cell.group_name}/${this.name}`,
        'SegIdx': this.idx,
        'GroupIdx': this.group_idx,
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'groupName': this.parent_cell.group_name, // 新增 groupName 字段
        'name': this.name,
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
      is_left_input: this.is_left_input,
      is_right_input: this.is_right_input,
      is_left_cut: this.left_clamp,
      is_right_cut: this.right_clamp,
    }
  }

  set_result(path_points: number[][], radius_list: number[], scene: Scene) {
    this.path_points = path_points;
    this.radius_list = radius_list;
    this.add_mesh(scene);
  }
}

export class ControlPointCell extends MeshObj {
  smooth_name: string;
  task_name: string;
  project_name: string;
  name: string;

  c_idx: number;
  group_idx: number;
  parent_cell: GroupCell;
  protected scene: Scene;

  // 控制点可视化
  private control_point_mesh: THREE.Mesh | null = null;
  
  // 引用计数，用于管理共享控制点
  private reference_count: number = 0;
  
  // 引用者列表，用于跟踪哪些对象引用了这个控制点
  private references: Set<string> = new Set();
  
  // 显示状态跟踪：记录哪些引用者要求显示控制点
  private visible_references: Set<string> = new Set();

  constructor(
    smooth_name: string, 
    task_name: string, 
    project_name: string, 
    control_point_name: string,
    c_idx: number, 
    group_idx: number, 
    parent_cell: GroupCell, 
    scene: Scene
  ) {
    super(control_point_name);
    this.smooth_name = smooth_name;
    this.task_name = task_name;
    this.project_name = project_name;
    this.name = control_point_name;
    
    this.c_idx = c_idx;
    this.group_idx = group_idx;
    this.parent_cell = parent_cell;
    this.scene = scene;
  }

  // 添加引用者
  addReference(referenceId: string): void {
    if (!this.references.has(referenceId)) {
      this.references.add(referenceId);
      this.reference_count++;
      
      // 如果是第一个引用者，创建控制点mesh但不显示
      if (this.reference_count === 1) {
        this.create_control_point_mesh();
        // 默认隐藏控制点
        this.hide_control_point();
      } else {
        // 更新颜色
        this.update_control_point_color();
      }
    }
  }

  // 移除引用者
  removeReference(referenceId: string): void {
    if (this.references.has(referenceId)) {
      this.references.delete(referenceId);
      this.visible_references.delete(referenceId); // 同时移除显示状态
      this.reference_count--;
      
      // 如果没有引用者了，清理控制点mesh
      if (this.reference_count === 0) {
        this.hide_control_point();
      } else {
        // 更新显示状态和颜色
        this.update_visibility();
        this.update_control_point_color();
      }
    }
  }

  // 为特定引用者显示控制点
  show_for_reference(referenceId: string): void {
    if (this.references.has(referenceId)) {
      this.visible_references.add(referenceId);
      this.update_visibility();
    }
  }

  // 为特定引用者隐藏控制点
  hide_for_reference(referenceId: string): void {
    if (this.references.has(referenceId)) {
      this.visible_references.delete(referenceId);
      this.update_visibility();
    }
  }

  // 更新控制点可见性
  private update_visibility(): void {
    if (this.control_point_mesh) {
      this.control_point_mesh.visible = this.visible_references.size > 0;
    }
  }

  // 获取引用计数
  getReferenceCount(): number {
    return this.reference_count;
  }

  // 获取引用者列表
  getReferences(): string[] {
    return Array.from(this.references);
  }

  // 获取显示状态
  isVisible(): boolean {
    return this.visible_references.size > 0;
  }

  // 创建控制点mesh
  private create_control_point_mesh(): void {
    if (!this.parent_cell?.c_table || this.c_idx >= this.parent_cell.c_table['x'].length) {
      // console.warn(`[${this.name}] Cannot create control point: c_idx=${this.c_idx}, c_table=${!!this.parent_cell?.c_table}`);
      return;
    }
    
    const x = this.parent_cell.c_table['x'][this.c_idx];
    const y = this.parent_cell.c_table['y'][this.c_idx];
    const z = this.parent_cell.c_table['z'][this.c_idx];
    
    // 根据引用者数量选择颜色和大小
    let color: number;
    let size: number = 0.15;
    
    if (this.reference_count === 1) {
      // 单个引用者：绿色（普通点）
      color = 0x00ff00;
    } else if (this.reference_count === 2) {
      // 两个引用者：蓝色（共用点）
      color = 0x0080ff;
      size = 0.18; // 稍微大一点，突出共用点
    } else if (this.reference_count === 3) {
      // 三个引用者：紫色（多共用点）
      color = 0x8000ff;
      size = 0.20;
    } else {
      // 更多引用者：红色（高度共用点）
      color = 0xff0000;
      size = 0.22;
    }
    
    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(size, 16, 16),
      new THREE.MeshStandardMaterial({ 
        color: color, 
        opacity: 0.8, 
        transparent: true,
        emissive: color,
        emissiveIntensity: this.reference_count > 1 ? 0.2 : 0.0 // 共用点有发光效果
      })
    );
    sphere.position.set(x, y, z);
    sphere.name = `${this.parent_cell.group_name}/${this.name}`;
    sphere.userData = { 
      c_idx: this.c_idx, 
      parent: this, 
      type: 'ControlPoint',
      references: Array.from(this.references),
      reference_count: this.reference_count,
      position: [x, y, z],
      radius: this.parent_cell.c_table['radius'][this.c_idx],
      is_fix: this.parent_cell.c_table['is_fix'][this.c_idx],
      is_unique: this.parent_cell.c_table['is_unique'][this.c_idx],
      group_name: this.parent_cell.group_name,
      group_idx: this.group_idx,
      project_name: this.project_name,
      smooth_name: this.smooth_name,
      task_name: this.task_name
    };
    
    this.control_point_mesh = sphere;
    this.mesh_dict[sphere.name] = sphere;
    this.scene.add(sphere);
    
    // 默认隐藏控制点
    this.control_point_mesh.visible = false;
    
    // console.log(`[${this.name}] Created control point at (${x}, ${y}, ${z}) with ${this.reference_count} references (hidden by default)`);
  }

  // 更新控制点颜色
  private update_control_point_color(): void {
    if (!this.control_point_mesh) return;
    
    // 根据引用者数量选择颜色和大小
    let color: number;
    let size: number = 0.15;
    
    if (this.reference_count === 1) {
      // 单个引用者：绿色（普通点）
      color = 0x00ff00;
    } else if (this.reference_count === 2) {
      // 两个引用者：蓝色（共用点）
      color = 0x0080ff;
      size = 0.18;
    } else if (this.reference_count === 3) {
      // 三个引用者：紫色（多共用点）
      color = 0x8000ff;
      size = 0.20;
    } else {
      // 更多引用者：红色（高度共用点）
      color = 0xff0000;
      size = 0.22;
    }
    
    if (this.control_point_mesh.material instanceof THREE.Material) {
      const material = this.control_point_mesh.material as THREE.MeshStandardMaterial;
      material.color.setHex(color);
      material.emissive.setHex(color);
      material.emissiveIntensity = this.reference_count > 1 ? 0.2 : 0.0; // 共用点有发光效果
    }
    
    // 更新几何体大小
    if (this.control_point_mesh.geometry instanceof THREE.SphereGeometry) {
      const currentRadius = (this.control_point_mesh.geometry as THREE.SphereGeometry).parameters.radius;
      if (Math.abs(currentRadius - size) > 0.01) {
        // 只有当大小差异足够大时才重新创建几何体
        this.control_point_mesh.geometry.dispose();
        this.control_point_mesh.geometry = new THREE.SphereGeometry(size, 16, 16);
      }
    }
    
    // 更新userData
    this.control_point_mesh.userData.reference_count = this.reference_count;
    this.control_point_mesh.userData.references = Array.from(this.references);
    // 更新位置和半径信息
    if (this.parent_cell?.c_table && this.c_idx < this.parent_cell.c_table['x'].length) {
      this.control_point_mesh.userData.position = [
        this.parent_cell.c_table['x'][this.c_idx],
        this.parent_cell.c_table['y'][this.c_idx],
        this.parent_cell.c_table['z'][this.c_idx]
      ];
      this.control_point_mesh.userData.radius = this.parent_cell.c_table['radius'][this.c_idx];
      this.control_point_mesh.userData.is_fix = this.parent_cell.c_table['is_fix'][this.c_idx];
      this.control_point_mesh.userData.is_unique = this.parent_cell.c_table['is_unique'][this.c_idx];
    }
  }

  // 显示控制点
  show_control_point(): void {
    if (this.reference_count > 0 && !this.control_point_mesh) {
      this.create_control_point_mesh();
    }
    if (this.control_point_mesh) {
      this.control_point_mesh.visible = true;
    }
  }

  // 隐藏控制点
  hide_control_point(): void {
    if (this.control_point_mesh) {
      this.control_point_mesh.visible = false;
    }
  }

  // 完全移除控制点
  remove_control_point(): void {
    if (this.control_point_mesh) {
      this.scene.remove(this.control_point_mesh);
      this.control_point_mesh.geometry.dispose();
      if (this.control_point_mesh.material instanceof THREE.Material) {
        this.control_point_mesh.material.dispose();
      }
      delete this.mesh_dict[this.control_point_mesh.name];
      this.control_point_mesh = null;
    }
  }

  // 更新控制点位置和相关信息
  update_position(): void {
    if (!this.parent_cell?.c_table || this.c_idx >= this.parent_cell.c_table['x'].length) {
      return;
    }
    
    const x = this.parent_cell.c_table['x'][this.c_idx];
    const y = this.parent_cell.c_table['y'][this.c_idx];
    const z = this.parent_cell.c_table['z'][this.c_idx];
    
    if (this.control_point_mesh) {
      this.control_point_mesh.position.set(x, y, z);
      // 更新userData中的所有相关信息
      this.control_point_mesh.userData.position = [x, y, z];
      this.control_point_mesh.userData.radius = this.parent_cell.c_table['radius'][this.c_idx];
      this.control_point_mesh.userData.is_fix = this.parent_cell.c_table['is_fix'][this.c_idx];
      this.control_point_mesh.userData.is_unique = this.parent_cell.c_table['is_unique'][this.c_idx];
      this.control_point_mesh.userData.references = Array.from(this.references);
      this.control_point_mesh.userData.reference_count = this.reference_count;
    }
  }

  // 获取控制点位置
  get_position(): number[] {
    if (!this.parent_cell?.c_table || this.c_idx >= this.parent_cell.c_table['x'].length) {
      return [0, 0, 0];
    }
    
    return [
      this.parent_cell.c_table['x'][this.c_idx],
      this.parent_cell.c_table['y'][this.c_idx],
      this.parent_cell.c_table['z'][this.c_idx]
    ];
  }

  // 设置控制点位置
  set_position(x: number, y: number, z: number): void {
    if (!this.parent_cell?.c_table || this.c_idx >= this.parent_cell.c_table['x'].length) {
      return;
    }
    
    this.parent_cell.c_table['x'][this.c_idx] = x;
    this.parent_cell.c_table['y'][this.c_idx] = y;
    this.parent_cell.c_table['z'][this.c_idx] = z;
    
    this.update_position();
  }

  export_json(): any {
    return {
      c_idx: this.c_idx,
      group_idx: this.group_idx,
      position: this.get_position(),
      reference_count: this.reference_count,
      references: Array.from(this.references)
    }
  }

  export_tree(): TreeNode {
    return {
      label: `${this.parent_cell.group_name}/${this.name}`,
      style: 'ControlPointCell',
      data: {
        meshName: `${this.parent_cell.group_name}/${this.name}`,
        'CIdx': this.c_idx,
        'GroupIdx': this.group_idx,
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'groupName': this.parent_cell.group_name,
        'name': this.name,
        'referenceCount': this.reference_count,
        'references': Array.from(this.references)
      },
    }
  }
}

export class ConnectorCell extends SegmentCell {
  left_seg_idx: number;
  right_seg_idx: number;

  constructor(
    smooth_name: string, task_name: string, project_name: string, connector_name: string,
    idx: number, group_idx: number, left_seg_idx: number, right_seg_idx: number, parent_cell: GroupCell, scene: Scene,
    c_idxs: number[], left_intercept_num: number, right_intercept_num: number, left_clamp: boolean, right_clamp: boolean
  ) {
    super(
      smooth_name, task_name, project_name, 
      extractPureName(connector_name, parent_cell.group_name),
      idx, group_idx, c_idxs, parent_cell, scene,
      left_intercept_num, right_intercept_num,
      left_clamp, right_clamp
    );
    this.left_seg_idx = left_seg_idx;
    this.right_seg_idx = right_seg_idx;
   
    // 如果c_idxs为空，先更新控制点
    if (!c_idxs || c_idxs.length === 0) {
      this.update_connector_control_points();
    }
   
    // 注册对控制点的引用
    const referenceId = `${parent_cell.group_name}/${this.name}`;
    this.c_idxs.forEach(c_idx => {
      parent_cell.register_control_point_reference(c_idx, referenceId, scene);
    });
  }
  // 确保连接器的控制点引用
  ensure_control_point_references(scene: Scene): void {
    const referenceId = `${this.parent_cell.group_name}/${this.name}`;
    
    // 如果连接器有控制点，确保它们被正确注册
    if (this.c_idxs && this.c_idxs.length > 0) {
      this.c_idxs.forEach(c_idx => {
        this.parent_cell.register_control_point_reference(c_idx, referenceId, scene);
      });
    } else {
      // 如果连接器没有控制点，尝试从左右段获取共享的控制点
      this.update_connector_control_points();
      
      // 重新注册控制点引用
      this.c_idxs.forEach(c_idx => {
        this.parent_cell.register_control_point_reference(c_idx, referenceId, scene);
      });
    }
  }

  // 更新连接器的控制点，从左右段获取共享的控制点
  update_connector_control_points(): void {
    const left_seg = this.parent_cell.seg_dict[`seg_${this.left_seg_idx}`];
    const right_seg = this.parent_cell.seg_dict[`seg_${this.right_seg_idx}`];
    
    if (left_seg && right_seg) {
      // 获取左段的最后一个控制点和右段的第一个控制点
      const left_c_idxs = left_seg.c_idxs;
      const right_c_idxs = right_seg.c_idxs;
      
      if (left_c_idxs && right_c_idxs && left_c_idxs.length > 0 && right_c_idxs.length > 0) {
        const left_last_idx = left_c_idxs[left_c_idxs.length - 1];
        const right_first_idx = right_c_idxs[0];
        
        // 如果左段的最后一个和右段的第一个是同一个控制点，需要获取更多控制点
        if (left_last_idx === right_first_idx) {
          // 这是一个共用点，我们需要获取更多的控制点来生成连接线
          const connector_c_idxs: number[] = [];
          
          // 添加左段的最后几个控制点（如果存在）
          if (left_c_idxs.length >= 2) {
            connector_c_idxs.push(left_c_idxs[left_c_idxs.length - 2]);
          }
          connector_c_idxs.push(left_last_idx); // 共用点
          
          // 添加右段的前几个控制点（如果存在）
          if (right_c_idxs.length >= 2) {
            connector_c_idxs.push(right_c_idxs[1]);
          }
          
          // 确保至少有2个不同的控制点
          if (connector_c_idxs.length >= 2) {
            this.c_idxs = connector_c_idxs;
          } else {
            // 如果仍然只有一个控制点，创建一个虚拟的控制点
            // console.warn(`[${this.name}] Only one control point available, creating virtual control point`);
            this.c_idxs = [left_last_idx, left_last_idx]; // 临时使用同一个点
          }
        } else {
          // 左段的最后一个和右段的第一个是不同的控制点
          this.c_idxs = [left_last_idx, right_first_idx];
        }
      }
    }
  }

 

  add_mesh(scene: Scene): void {
    const meshName = `${this.parent_cell.group_name}/${this.name}`;
    // 清理所有同名且userData.type==='smooth'的mesh
    scene.children
      .filter(obj => obj.name === meshName && obj.userData && obj.userData.type === 'smooth')
      .forEach(mesh => {
        scene.remove(mesh);
        if ((mesh as THREE.Mesh).geometry) (mesh as THREE.Mesh).geometry.dispose();
        if ((mesh as THREE.Mesh).material && (mesh as THREE.Mesh).material instanceof THREE.Material) {
          ((mesh as THREE.Mesh).material as THREE.Material).dispose();
        }
      });
    // 使用B样条曲线生成平滑路径
    if (!this.path_points || this.path_points.length < 2) {
      if (!this.c_idxs || !this.parent_cell?.c_table || this.c_idxs.length === 0) {
        console.warn('ConnectorCell.add_mesh 缺少 c_idxs 或 c_table,跳过渲染', this.name, this.c_idxs, this.parent_cell?.c_table);
        return;
      }
      
      // 从控制点生成B样条曲线
      const control_points: number[][] = []
      this.c_idxs.forEach(c_idx => {
        control_points.push([
          this.parent_cell.c_table['x'][c_idx], 
          this.parent_cell.c_table['y'][c_idx], 
          this.parent_cell.c_table['z'][c_idx]
        ]);
      })

      const n_controls = this.c_idxs.length;
      const knot_cfg = BsplineUtils.create_knot_cfg(n_controls, this.degree, this.left_clamp, this.right_clamp)
      const knot = knot_cfg['knot'];
      const fill_last_one = knot_cfg['fill_last_one'];

      const sample_num = 200
      const ts = BsplineUtils.create_ts(
        sample_num, this.degree, knot, this.left_clamp, this.right_clamp,
        this.left_intercept_num, this.right_intercept_num
      )
      const spline_mat = BsplineUtils.compute_mat(ts, n_controls, this.degree, knot, fill_last_one)
      const points = this.matrix_multiply(spline_mat, control_points);
      this.path_points = points;

      const control_radius: number[][] = []
      this.c_idxs.forEach(c_idx => {
        control_radius.push([
          this.parent_cell.c_table['radius'][c_idx]
        ]);
      })
      const radius_list: number[] = this.matrix_multiply(spline_mat, control_radius).map(radius => radius[0]);
      this.radius_list = radius_list;
    }
    // 将0-1范围的颜色转换为0-255范围
    const color_255 = this.color.map(c => Math.floor(c * 255));
    this.line_mesh = create_tube(this.path_points, 0.05, { color: color_to_hex(color_255), opacity: 0.8 }) as THREE.Mesh;
    this.line_mesh.name = meshName;
    this.line_mesh.visible = true;
    this.line_mesh.userData = { group_name: this.parent_cell.group_name, name: this.name, group_idx: this.group_idx, type: 'smooth' };
    this.mesh_dict[meshName] = this.line_mesh;
    scene.add(this.line_mesh);
  }

  remove_mesh(scene: Scene) {
    for (const key in this.mesh_dict) {
      scene.remove(this.mesh_dict[key]);
      this.mesh_dict[key].geometry.dispose();
      if (this.mesh_dict[key].material instanceof THREE.Material) {
        this.mesh_dict[key].material.dispose();
      }
    }
    this.mesh_dict = {};
    this.line_mesh = null;
    this.tube_mesh = null;
  }

  update_mesh(scene: Scene) {
    this.hide_tube_mesh();
    this.remove_mesh(scene);
    this.add_mesh(scene);
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

  show_tube_mesh() {
    if (!this.tube_mesh && this.path_points && this.path_points.length > 1) {
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

      // 确保所有方向向量都有足够的大小
      for (let i = 0; i < direction_list.length; i++) {
        const dir = direction_list[i];
        const magnitude = Math.sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        if (magnitude < 0.001) {
          // 如果方向向量太小，使用默认的Z轴方向
          direction_list[i] = [0, 0, 1];
        } else {
          // 归一化方向向量
          direction_list[i] = [dir[0] / magnitude, dir[1] / magnitude, dir[2] / magnitude];
        }
      }

      this.tube_mesh = new CoarseTubeMesh(
        this.name + '_coarse',
        this.path_points, this.radius_list, direction_list, this.color, this.scene
      );

    } else if (this.tube_mesh) {
      this.tube_mesh.show_mesh();
    }
  }

  hide_tube_mesh() {
    if (this.tube_mesh) {
      this.tube_mesh.hide_mesh();
    }
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
        c_idxs: this.c_idxs,
        cell_type: 'connector',
        con_idx: this.idx,
        group_idx: this.group_idx,
        left_seg_idx: this.left_seg_idx,
        right_seg_idx: this.right_seg_idx,
        left_intercept_num: this.left_intercept_num,
        right_intercept_num: this.right_intercept_num,
      }
    }
  }

  export_tree(): TreeNode {
    return {
      id: `${this.project_name}-${this.smooth_name}-${this.parent_cell.group_name}-connector-${this.name}`,
      label: `${this.parent_cell.group_name}/${this.name}`,
      style: 'ConnectorCell',
      data: {
        meshName: `${this.parent_cell.group_name}/${this.name}`,
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'ConIdx': this.idx,
        'GroupIdx': this.group_idx,
        'groupName': this.parent_cell.group_name, // 新增 groupName 字段
        'name': this.name,
      },
    }
  }

  set_result(path_points: number[][], radius_list: number[], scene: Scene) {
    this.path_points = path_points;
    this.radius_list = radius_list;
    this.add_mesh(scene);
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

  path_points: number[][] = [];
  radius_list: number[] = [];

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
  group_color: number[];

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
  control_point_dict: { [key: string]: ControlPointCell } = {};

  constructor(smooth_name: string, task_name: string, project_name: string, group_name: string, group_idx: number) {
    this.smooth_name = smooth_name;
    this.task_name = task_name;
    this.project_name = project_name;
    this.group_name = group_name;
    this.group_idx = group_idx;
    
    // 动态分配颜色：根据组索引分配颜色，确保每个组都有不同的颜色
    const colors = getColorsForGroupCount(Math.max(globalGroupCounter + 1, group_idx + 1));
    this.group_color = colors[group_idx % colors.length];
    globalGroupCounter = Math.max(globalGroupCounter, group_idx + 1);

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
    
    // 更新所有控制点的位置
    for (const controlPoint of Object.values(this.control_point_dict)) {
      controlPoint.update_position();
    }
    
    for (const seg of Object.values(this.seg_dict)) {
      seg.update_mesh(scene);
    }
    for (const con of Object.values(this.connector_dict)) {
      con.update_mesh(scene);
    }
  }

  ensure_connector_control_points(scene: Scene): void {
    for (const connector of Object.values(this.connector_dict)) {
      // 确保连接器有正确的控制点
      if (!connector.c_idxs || connector.c_idxs.length === 0) {
        connector.update_connector_control_points();
        
        // 重新注册控制点引用
        const referenceId = `${this.group_name}/${connector.name}`;
        connector.c_idxs.forEach(c_idx => {
          this.register_control_point_reference(c_idx, referenceId, scene);
        });
        
      }
    }
  }

  // 调试方法：检查控制点的引用状态
  debug_control_points(): void {
    // console.log(`=== Debug Control Points for Group: ${this.group_name} ===`);
    
    // 检查段的控制点
    for (const [segmentName, segment] of Object.entries(this.seg_dict)) {
      console.log(`  ${segmentName}: c_idxs = [${segment.c_idxs.join(', ')}]`);
    }
    
    // 检查连接器的控制点
    console.log('Connectors:');
    for (const [connectorName, connector] of Object.entries(this.connector_dict)) {
      console.log(`  ${connectorName}: c_idxs = [${connector.c_idxs.join(', ')}]`);
    }
    
    // 检查控制点的引用状态
    console.log('Control Points:');
    for (const [controlPointName, controlPoint] of Object.entries(this.control_point_dict)) {
      const refCount = controlPoint.getReferenceCount();
      const references = controlPoint.getReferences();
      const isVisible = controlPoint.isVisible();
      console.log(`  ${controlPointName}: refCount=${refCount}, visible=${isVisible}, references=[${references.join(', ')}]`);
    }
    
    console.log('=== End Debug ===');
  }

  export_json(): any {
    return {
      group_name: this.group_name,
      aid_path_info: this.aid_path_info,
      graph_cfg: this.graph_cfg,
      segments: Object.values(this.seg_dict).map(seg => seg.export_json()),
      connectors: Object.values(this.connector_dict).map(con => con.export_json()),
      paths: Object.values(this.path_dict).map(path => path.export_json()),
      control_points: Object.values(this.control_point_dict).map(cp => cp.export_json()),
      cell_setting: {
        group_idx: this.group_idx,
        c_table: this.c_table
      }
    }
  }
  
  export_tree(): TreeNode {
    const seg_node: TreeNode = {
      id: `${this.project_name}-${this.smooth_name}-${this.group_name}-segments`,
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
      id: `${this.project_name}-${this.smooth_name}-${this.group_name}-connectors`,
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
      id: `${this.project_name}-${this.smooth_name}-${this.group_name}-paths`,
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
    const control_point_node: TreeNode = {
      id: `${this.project_name}-${this.smooth_name}-${this.group_name}-control-points`,
      label: 'Control Points',
      style: 'ControlPointCells',
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

    for (const controlPoint of Object.values(this.control_point_dict)) {
      control_point_node.children!.push(controlPoint.export_tree());
    }

    // ---------------------------------------
    const aid_path_node: TreeNode = {
      id: `${this.project_name}-${this.smooth_name}-${this.group_name}-aid-paths`,
      label: 'Aid Paths',
      style: 'AidPaths',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'GroupIdx': this.group_idx,
      },
      children: this.aid_path_info.map(info => ({
        id: `${this.project_name}-${this.smooth_name}-${this.group_name}-aid-path-${info.name}`,
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
      id: `${this.project_name}-${this.smooth_name}-${this.group_name}-graph`,
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
      id: `${this.project_name}-${this.smooth_name}-${this.group_name}`,
      label: `${this.group_name}`,
      style: 'GroupCell',
      data: {
        'Project': this.project_name,
        'SmoothName': this.smooth_name,
        'TaskName': this.task_name,
        'GroupIdx': this.group_idx,
      },
      children: [aid_path_node, graph_node, seg_node, con_node, path_node, control_point_node],
    }
  }

  export_surface_mesh_json(): { [key: string]: PartSurfaceMeshInfo } {
    const part_surface_mesh_infos: { [key: string]: PartSurfaceMeshInfo } = {};
    for (const seg of Object.values(this.seg_dict)) {
      part_surface_mesh_infos[seg.name] = seg.export_surface_mesh_json();
    }
    for (const con of Object.values(this.connector_dict)) {
      part_surface_mesh_infos[con.name] = con.export_surface_mesh_json();
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

  // 显示特定段或连接器的控制点
  show_control_points_for_cell(cellName: string): void {
    // 查找该段或连接器使用的所有控制点
    let cell: SegmentCell | ConnectorCell | null = null;
    
    // 检查是否是段
    if (this.seg_dict[cellName]) {
      cell = this.seg_dict[cellName];
    }
    // 检查是否是连接器
    else if (this.connector_dict[cellName]) {
      cell = this.connector_dict[cellName];
    }
    
    if (cell) {
      const referenceId = `${this.group_name}/${cellName}`;
      cell.c_idxs.forEach(c_idx => {
        const control_point_name = `control_point_${c_idx}`;
        const key = `${this.group_name}/${control_point_name}`;
        
        if (this.control_point_dict[key]) {
          this.control_point_dict[key].show_for_reference(referenceId);
        }
      });
    }
  }

  // 隐藏特定段或连接器的控制点
  hide_control_points_for_cell(cellName: string): void {
    // 查找该段或连接器使用的所有控制点
    let cell: SegmentCell | ConnectorCell | null = null;
    
    // 检查是否是段
    if (this.seg_dict[cellName]) {
      cell = this.seg_dict[cellName];
    }
    // 检查是否是连接器
    else if (this.connector_dict[cellName]) {
      cell = this.connector_dict[cellName];
    }
    
    if (cell) {
      const referenceId = `${this.group_name}/${cellName}`;
      cell.c_idxs.forEach(c_idx => {
        const control_point_name = `control_point_${c_idx}`;
        const key = `${this.group_name}/${control_point_name}`;
        
        if (this.control_point_dict[key]) {
          this.control_point_dict[key].hide_for_reference(referenceId);
        }
      });
    }
  }

  show_control_points() {
    // 显示所有段和连接器的控制点
    for (const [segmentName, segment] of Object.entries(this.seg_dict)) {
      this.show_control_points_for_cell(segmentName);
    }
    for (const [connectorName, connector] of Object.entries(this.connector_dict)) {
      this.show_control_points_for_cell(connectorName);
    }
  }

  hide_control_points() {
    // 隐藏所有段和连接器的控制点
    for (const [segmentName, segment] of Object.entries(this.seg_dict)) {
      this.hide_control_points_for_cell(segmentName);
    }
    for (const [connectorName, connector] of Object.entries(this.connector_dict)) {
      this.hide_control_points_for_cell(connectorName);
    }
  }

  // 只显示段的控制点
  show_segments_control_points(): void {
    // 先隐藏所有控制点
    this.hide_control_points();
    
    // 显示所有段的控制点
    for (const [segmentName, segment] of Object.entries(this.seg_dict)) {
      const referenceId = `${this.group_name}/${segmentName}`;
      segment.c_idxs.forEach(c_idx => {
        const control_point_name = `control_point_${c_idx}`;
        const key = `${this.group_name}/${control_point_name}`;
        
        if (this.control_point_dict[key]) {
          this.control_point_dict[key].show_for_reference(referenceId);
        }
      });
    }
  }
  
  // 只显示连接器的控制点
  show_connectors_control_points(scene: Scene): void {
    // 先隐藏所有控制点
    this.hide_control_points();
    
    console.log(`[show_connectors_control_points] 显示连接器控制点，组名: ${this.group_name}`);
    
    // 显示所有连接器使用的控制点（使用原本的c_idxs，不重新生成）
    for (const [connectorName, connector] of Object.entries(this.connector_dict)) {
      const referenceId = `${this.group_name}/${connectorName}`;
      console.log(`[show_connectors_control_points] 连接器: ${connectorName}, c_idxs: [${connector.c_idxs.join(', ')}]`);
      
      // 确保连接器有控制点引用（但不重新生成c_idxs）
      if (connector.c_idxs && connector.c_idxs.length > 0) {
        connector.c_idxs.forEach((c_idx, index) => {
          const control_point_name = `control_point_${c_idx}`;
          const key = `${this.group_name}/${control_point_name}`;
          
          // 确保控制点存在并注册引用
          if (!this.control_point_dict[key]) {
            this.register_control_point_reference(c_idx, referenceId, scene);
          } else {
            // 如果控制点已存在，直接显示
            this.control_point_dict[key].show_for_reference(referenceId);
          }
        });
      } else {
        // 只有在没有控制点索引时才重新生成
        console.warn(`[${connectorName}] 没有控制点索引，重新生成控制点`);
        connector.update_connector_control_points();
        
        // 重新注册控制点引用
        connector.c_idxs.forEach(c_idx => {
          this.register_control_point_reference(c_idx, referenceId, scene);
        });
      }
    }
    
    // 调试信息
    this.debug_control_points();
  }

  // 只隐藏段的控制点
  hide_segments_control_points(): void {
    // 隐藏所有段的控制点
    for (const [segmentName, segment] of Object.entries(this.seg_dict)) {
      const referenceId = `${this.group_name}/${segmentName}`;
      segment.c_idxs.forEach(c_idx => {
        const control_point_name = `control_point_${c_idx}`;
        const key = `${this.group_name}/${control_point_name}`;
        
        if (this.control_point_dict[key]) {
          this.control_point_dict[key].hide_for_reference(referenceId);
        }
      });
    }
  }
  
  // 只隐藏连接器的控制点
  hide_connectors_control_points(): void {
    // 隐藏所有连接器使用的控制点
    for (const [connectorName, connector] of Object.entries(this.connector_dict)) {
      const referenceId = `${this.group_name}/${connectorName}`;
      connector.c_idxs.forEach(c_idx => {
        const control_point_name = `control_point_${c_idx}`;
        const key = `${this.group_name}/${control_point_name}`;
        
        if (this.control_point_dict[key]) {
          this.control_point_dict[key].hide_for_reference(referenceId);
        }
      });
    }
  }

  // 获取或创建控制点
  get_or_create_control_point(c_idx: number, scene: Scene): ControlPointCell {
    const control_point_name = `control_point_${c_idx}`;
    const key = `${this.group_name}/${control_point_name}`;
    
    if (!this.control_point_dict[key]) {
      this.control_point_dict[key] = new ControlPointCell(
        this.smooth_name,
        this.task_name,
        this.project_name,
        control_point_name,
        c_idx,
        this.group_idx,
        this,
        scene
      );
    }
    
    return this.control_point_dict[key];
  }

  // 注册段或连接器对控制点的引用
  register_control_point_reference(c_idx: number, referenceId: string, scene: Scene): void {
    const controlPoint = this.get_or_create_control_point(c_idx, scene);
    controlPoint.addReference(referenceId);
  }

  // 注销段或连接器对控制点的引用
  unregister_control_point_reference(c_idx: number, referenceId: string): void {
    const control_point_name = `control_point_${c_idx}`;
    const key = `${this.group_name}/${control_point_name}`;
    
    if (this.control_point_dict[key]) {
      this.control_point_dict[key].removeReference(referenceId);
      
      // 如果没有引用者了，从字典中移除
      if (this.control_point_dict[key].getReferenceCount() === 0) {
        this.control_point_dict[key].remove_control_point();
        delete this.control_point_dict[key];
      }
    }
  }
}
