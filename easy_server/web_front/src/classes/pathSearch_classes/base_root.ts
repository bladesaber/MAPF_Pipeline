import * as THREE from "three";

import { BinaryConstraint } from "./constraint_utils.ts";
import { TaskManager } from "./base_task.ts";
import { IntermediatePointBlock } from "./intermediatePoint_block.ts";

export class BlockRoot {
    private scene: THREE.Scene;
    private task_manager_map: Map<string, TaskManager> = new Map();
    private constraint_tree: Map<string, BinaryConstraint> = new Map();
    private intermediate_point_map: Map<string, IntermediatePointBlock> = new Map();

    constructor(scene: THREE.Scene) {
      this.scene = scene;
    }
  
    get_scene() {
      return this.scene;
    }
  
    add_task_manager(task_manager: TaskManager): [boolean, string] {
      if (this.task_manager_map.get(task_manager.name)) {
        return [false, 'Task manager already exists'];
      }
      this.task_manager_map.set(task_manager.name, task_manager);
      return [true, ''];
    }
  
    remove_task_manager(name: string) {
      const task_manager = this.task_manager_map.get(name);
      if (!task_manager) {
        return;
      }
      task_manager.destory(this);
      this.task_manager_map.delete(name);
    }
  
    get_task_manager(name: string) {
      return this.task_manager_map.get(name);
    }
  
    has_task_manager(name: string) {
      return this.task_manager_map.has(name);
    }

    get_task_manager_names() {
      return Array.from(this.task_manager_map.keys());
    }
  
    add_constraint(cell: BinaryConstraint) {
      this.constraint_tree.set(cell.name, cell);
    }
  
    remove_constraint(cell: BinaryConstraint) {
      this.constraint_tree.delete(cell.name);
    }
  
    get_constraint(name: string) {
      return this.constraint_tree.get(name);
    }
  
    traverse_tree(block_name: string) {
      this.constraint_tree.forEach((cell, key) => {
        cell.reset_updated();
      });
  
      let active_names: string[] = [block_name];
      let next_names: string[] = [];
  
      while (active_names.length > 0) {
        for (const main_name of active_names) {
          this.constraint_tree.forEach((cell, key) => {
            if (cell.get_block_names().includes(main_name) && !cell.is_updated()) {
              cell.set_updated();
  
              const [success, passive_name] = cell.parse_constraint(this, main_name);
              if (success) {
                next_names.push(passive_name);
              }
            }
          });
        }
        active_names = next_names;
        next_names = [];
      }
    }
    
    add_intermediate_point(name: string, x: number, y: number, z: number): [boolean, string] {
      if (this.intermediate_point_map.get(name)) {
        return [false, 'Intermediate point already exists'];
      }
      const intermediate_point = new IntermediatePointBlock(name, x, y, z, this);
      this.intermediate_point_map.set(name, intermediate_point);
      return [true, ''];
    }

    remove_intermediate_point(name: string) {
      if (!this.intermediate_point_map.get(name)) {
        return;
      }
      this.intermediate_point_map.delete(name);
    }

    get_intermediate_point_names() {
      return Array.from(this.intermediate_point_map.keys());
    }

    export_json() {
      const save_data: {
        task_managers: any[];
        intermediate_points: any[];
      } = {
        task_managers: [],
        intermediate_points: [],
      }
      this.task_manager_map.forEach((task_manager) => {
        save_data.task_managers.push(task_manager.export_json())
      })
      this.intermediate_point_map.forEach((point) => {
        save_data.intermediate_points.push(point.toJson())
      })
      return save_data;
    }

    load_json(data: any) {
      data.task_managers.forEach((task_cfg: any) => {
        if (!this.task_manager_map.has(task_cfg.name)) {
          this.add_task_manager(TaskManager.load_json(task_cfg, this));
        }
      })
      data.intermediate_points.forEach((point_cfg: any) => {
        if (!this.intermediate_point_map.has(point_cfg.name)) {
          this.add_intermediate_point(point_cfg.name, point_cfg.x, point_cfg.y, point_cfg.z);
        }
      })
    }
}
  