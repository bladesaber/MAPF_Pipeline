import * as THREE from 'three'
import { BaseGeometryObj } from './point_obj.ts'

export class RootObj {
  private scene: THREE.Scene;
  private obj_map: Map<string, BaseGeometryObj>;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.obj_map = new Map();
  }

  get_scene() {
    return this.scene;
  }

  add_obj(obj: BaseGeometryObj) {
    this.obj_map.set(obj.name, obj);
  }

  remove_obj(obj: BaseGeometryObj) {
    let update_objs = []
    for (const cur_obj of this.obj_map.values()) {
      if (cur_obj.hooks.includes(obj.name)) {
        cur_obj.remove_var(obj.name);
        update_objs.push(cur_obj);
      }
    }

    for (const update_obj of update_objs) {
      this.update_objs_from_obj(update_obj);
    }

    obj.destory(this.scene);
    this.obj_map.delete(obj.name);
  }

  get_obj(name: string) {
    return this.obj_map.get(name);
  }

  get_obj_list() {
    return Array.from(this.obj_map.values());
  }

  update_objs_from_obj(updated_obj: BaseGeometryObj) {
    updated_obj.update_mesh(this.scene);
    let current_hook = [updated_obj.name]
    let next_hook = []

    for (let layer_num = updated_obj.layer + 1; layer_num < 3; layer_num++) {
      for (const obj of this.obj_map.values()) {
        if (obj.layer === layer_num && obj.hooks.some(hook => current_hook.includes(hook))) {
          obj.update_mesh(this.scene);
          next_hook.push(obj.name)
        }
      }

      current_hook = next_hook.slice()
      next_hook = []
      if (current_hook.length === 0) {
        break;
      }
    }
  }

  update_all_objs() {
    for (const obj of this.obj_map.values()) {
      obj.update_mesh(this.scene);
    }
  }

} 