import * as THREE from "three";

import { BinaryConstraint } from "./constraint_utils.ts";
import type { TaskManager } from "./base_task.ts";

export class BlockRoot {
    private scene: THREE.Scene;
    private task_manager_map: Map<string, TaskManager> = new Map();
    private constraint_tree: Map<string, BinaryConstraint> = new Map();
  
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
    
}
  