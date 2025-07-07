import * as THREE from "three";

import { PathResultMesh } from "./result_mesh";
import { GridMesh } from "./grid";
export class CellTask {
  public name: string;
  public type: string = 'CellTask';
  public start_name: string;
  public end_name: string;
  public radius: number;
  public result: PathResultMesh | null = null;
  public has_result: boolean = false;
  private color: number[] = [1, 0, 0];

  constructor(name: string, start_name: string, end_name: string, radius: number, color: number[]) {
    this.name = name;
    this.start_name = start_name;
    this.end_name = end_name;
    this.radius = radius;
    this.color = color;
  }

  set_result(result: number[][], scene: THREE.Scene, is_discrete: boolean = false, grid: GridMesh|null = null) {
    if (this.has_result) {
      return;
    }

    if (is_discrete) {
      result = result.map((point) => {
        return [
          grid!.grid_to_value_scaled(point[0]), 
          grid!.grid_to_value_scaled(point[1]), 
          grid!.grid_to_value_scaled(point[2])
        ];
      });
    }
    this.result = new PathResultMesh(this.name, result, this.radius, this.color, scene);
    this.has_result = true;
  }

  clear_result(scene: THREE.Scene) {
    if (this.result) {
      this.result.destory(scene);
    }
    this.result = null;
    this.has_result = false;
  }

  hide_line_mesh() {
    if (this.result) {
      this.result.hide_line_mesh();
    }
  }

  hide_tube_mesh() {
    if (this.result) {
      this.result.hide_tube_mesh();
    }
  }

  show_line_mesh() {
    if (this.result) {
      this.result.show_line_mesh();
    }
  }

  show_tube_mesh() {
    if (this.result) {
      this.result.show_tube_mesh();
    }
  }

  toJSON(is_discrete: boolean = false, grid: GridMesh|null = null): any {
    let points = null;
    if (this.result) {
      if (is_discrete) {
        points = this.result.points.map((point) => {
          return [grid!.value_to_grid(point[0]), grid!.value_to_grid(point[1]), grid!.value_to_grid(point[2])];
        });
      } else {
        points = this.result.points;
      }
    }

    return {
      name: this.name,
      start_name: this.start_name,
      end_name: this.end_name,
      radius: is_discrete ? grid!.value_to_grid_scaled(this.radius) : this.radius,
      result: this.result ? points : [],
    };
  }

  destroy(scene: THREE.Scene) {
    this.clear_result(scene);
  }
}

export class GroupTask {
  public name: string;
  public type: string = 'GroupTask';
  public cell_tasks: CellTask[] = [];
  public has_result: boolean = false;
  public run_config: Map<string, string | number | boolean> = new Map();

  constructor(name: string, run_config: Map<string, string | number | boolean>) {
    this.name = name;
    this.run_config = run_config;
  }

  hide_line_mesh() {
    this.cell_tasks.forEach((cell_task) => {
      cell_task.hide_line_mesh();
    });
  }

  hide_tube_mesh() {
    this.cell_tasks.forEach((cell_task) => {
      cell_task.hide_tube_mesh();
    });
  }

  show_line_mesh() {
    this.cell_tasks.forEach((cell_task) => {
      cell_task.show_line_mesh();
    });
  }

  show_tube_mesh() {
    this.cell_tasks.forEach((cell_task) => {
      cell_task.show_tube_mesh();
    });
  }

  set_result(results: Map<string, number[][]>, scene: THREE.Scene, is_discrete: boolean = false, grid: GridMesh|null = null) {
    if (this.has_result) {
      this.clear_result(scene);
    }
    this.cell_tasks.forEach((cell_task, index) => {
      if (results.get(cell_task.name) && results.get(cell_task.name)!.length > 0) {
        cell_task.set_result(results.get(cell_task.name)!, scene, is_discrete, grid);
      }
    });
    this.has_result = true;
  }

  clear_result(scene: THREE.Scene) {
    this.cell_tasks.forEach((cell_task) => {
      cell_task.clear_result(scene);
    });
    this.has_result = false;
  }

  check_finish() {
    if (this.cell_tasks.length === 0) {
      this.has_result = false;
      return;
    }
    if (Array.from(this.cell_tasks.values()).every((cell_task) => cell_task.has_result)) {
      this.has_result = true;
    }
  }

  toJSON(is_discrete: boolean = false, grid: GridMesh|null = null): any {
    const run_config = Object.fromEntries(this.run_config);

    if (is_discrete) {
      if (['ScaledWeightedAstarSearchBoost', 'ScaledWeightedAstarSearchBucket'].includes(run_config.planner_method as string)) {
        run_config.threshold1 = grid!.value_to_grid_scaled(this.run_config.get('threshold1') as number);
      } else if (['DoubleScaledWeightedAstarSearchBoost', 'DoubleScaledWeightedAstarSearchBucket'].includes(run_config.planner_method as string)) {
        run_config.threshold1 = grid!.value_to_grid_scaled(this.run_config.get('threshold1') as number);
        run_config.threshold2 = grid!.value_to_grid_scaled(this.run_config.get('threshold2') as number);
      }
    }

    return {
      name: this.name,
      run_config: run_config,
      cell_tasks: this.cell_tasks.map((cell_task) => cell_task.toJSON(is_discrete, grid)),
    };
  }

  destroy(scene: THREE.Scene) {
    this.cell_tasks.forEach((cell_task) => {
      cell_task.destroy(scene);
    });
  }
}

export class PriorityTask {
  public name: string;
  public type: string = 'PriorityTask';
  public group_tasks: Map<string, GroupTask> = new Map();
  public has_result: boolean = false;
  public run_config: Map<string, string | number | boolean> = new Map();
  
  constructor(name: string, run_config: Map<string, string | number | boolean>) {
    this.name = name;
    this.run_config = run_config;
  }

  hide_line_mesh() {
    this.group_tasks.forEach((group_task) => {
      group_task.hide_line_mesh();
    });
  }

  hide_tube_mesh() {
    this.group_tasks.forEach((group_task) => {
      group_task.hide_tube_mesh();
    });
  }

  show_line_mesh() {
    this.group_tasks.forEach((group_task) => {
      group_task.show_line_mesh();
    });
  }

  show_tube_mesh() {
    this.group_tasks.forEach((group_task) => {
      group_task.show_tube_mesh();
    });
  }

  set_result(results: Map<string, Map<string, number[][]>>, scene: THREE.Scene, is_discrete: boolean = false, grid: GridMesh|null = null) {
    if (this.has_result) {
      this.clear_result(scene);
    }

    results.forEach((group_results, group_name) => {
      if (group_results.size > 0) {
        const group_task = this.group_tasks.get(group_name);
        group_task!.set_result(group_results, scene, is_discrete, grid);
      }
    });

    this.has_result = true;
  }

  clear_result(scene: THREE.Scene) {
    this.group_tasks.forEach((group_task) => {
      group_task.clear_result(scene);
    });
    this.has_result = false;
  }

  check_finish() {
    if (this.group_tasks.size === 0) {
      this.has_result = false;
      return;
    }
    this.group_tasks.forEach((group_task) => {
      group_task.check_finish();
    });
    if (Array.from(this.group_tasks.values()).every((group_task) => group_task.has_result)) {
      this.has_result = true;
    }
  }

  toJSON(is_discrete: boolean = false, grid: GridMesh|null = null): any {
    return {
      name: this.name,
      run_config: Object.fromEntries(this.run_config),
      group_tasks: Array.from(this.group_tasks.values()).map((group_task) => group_task.toJSON(is_discrete, grid)),
    };
  }

  destroy(scene: THREE.Scene) {
    this.group_tasks.forEach((group_task) => {
      group_task.destroy(scene);
    });
  }
}

export class SearchTask {
  public name: string;
  public type: string = 'SearchTask';
  public priority_tasks: PriorityTask[] = [];
  public has_result: boolean = false;

  constructor(name: string) {
    this.name = name;
  }

  hide_line_mesh() {
    this.priority_tasks.forEach((priority_task) => {
      priority_task.hide_line_mesh();
    });
  }

  hide_tube_mesh() {
    this.priority_tasks.forEach((priority_task) => {
      priority_task.hide_tube_mesh();
    });
  }

  show_line_mesh() {
    this.priority_tasks.forEach((priority_task) => {
      priority_task.show_line_mesh();
    });
  }

  show_tube_mesh() {
    this.priority_tasks.forEach((priority_task) => {
      priority_task.show_tube_mesh();
    });
  }

  set_result(results: Map<string, Map<string, number[][]>>, scene: THREE.Scene, is_discrete: boolean = false, grid: GridMesh|null = null) {
    if (this.has_result) {
      this.clear_result(scene);
    }
    this.priority_tasks.forEach((priority_task) => {
      priority_task.set_result(results, scene, is_discrete, grid);
    });
    this.has_result = true;
  }

  clear_result(scene: THREE.Scene) {
    this.priority_tasks.forEach((priority_task) => {
      priority_task.clear_result(scene);
    });
    this.has_result = false;
  }

  check_finish() {
    if (this.priority_tasks.length === 0) {
      this.has_result = false;
      return;
    }
    this.priority_tasks.forEach((priority_task) => {
      priority_task.check_finish();
    });
    if (Array.from(this.priority_tasks.values()).every((priority_task) => priority_task.has_result)) {
      this.has_result = true;
    }
  }

  toJSON(is_discrete: boolean = false, grid: GridMesh|null = null): any {
    return {
      name: this.name,
      priority_tasks: this.priority_tasks.map((priority_task) => priority_task.toJSON(is_discrete, grid)),
    };
  }

  destroy(scene: THREE.Scene) {
    this.priority_tasks.forEach((priority_task) => {
      priority_task.destroy(scene);
    });
  }
}

