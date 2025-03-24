import * as THREE from 'three'
import { GroupTypes } from '../types/data_type.ts'
import { BaseGeometryObj, BaseNode, GroupNode, GroupGeometryObj } from './base_obj.ts'

export class RootObj {
  private scene: THREE.Scene;
  private obj_map: Map<string, BaseGeometryObj|GroupGeometryObj>;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.obj_map = new Map();
  }

  get_scene() {
    return this.scene;
  }

  add_objs(objs: (BaseGeometryObj|GroupGeometryObj)[]) {
    objs.forEach(obj => {
      this.obj_map.set(obj.name, obj);
    })
  }

  remove_obj(obj: BaseGeometryObj|GroupGeometryObj) {
    obj.destory(this);
    this.update_objs_from_obj(obj);
    this.obj_map.delete(obj.name);
  }

  request_obj(node_style: BaseNode|GroupNode){
    if (node_style.style === 'BaseNode') {
      return this.obj_map.get(node_style.name)
    } 
    else if (node_style.style === 'GroupNode') {
      const group_name = (node_style as GroupNode).group_name
      if (!this.obj_map.has(group_name)) {
        return null
      }
      if (!(this.obj_map.get(group_name) as GroupGeometryObj).has_cell_mesh(node_style.name)){
        return null
      }
      return (this.obj_map.get(group_name) as GroupGeometryObj).get_cell_mesh(node_style.name)
    }
  }
  
  get_obj(name: string): BaseGeometryObj|GroupGeometryObj|null {
    const obj = this.obj_map.get(name);
    if (!obj) {
      return null;
    }
    return obj;
  }

  get_obj_deep(name: string): BaseGeometryObj|GroupGeometryObj|null {
    for (const obj of this.obj_map.values()) {
      if (obj.name === name) {
        return obj;
      }
      else {
        if (GroupTypes.includes((obj as BaseGeometryObj|GroupGeometryObj).style as typeof GroupTypes[number])){
          const cell_obj = (obj as GroupGeometryObj).get_cell_mesh(name);
          if (cell_obj) {
            return cell_obj;
          }
        }
      }
    }
    return null;
  }

  get_obj_list() {
    return Array.from(this.obj_map.values());
  }

  has_obj(name: string) {
    return this.obj_map.has(name);
  }

  update_objs_from_obj(updated_obj: BaseGeometryObj|GroupGeometryObj) {
    let current_hook: string[] = []
    if (updated_obj.geometry_style === 'base_geometry') {
      (updated_obj as BaseGeometryObj).update_mesh(this);
      current_hook.push(updated_obj.name)

    } else if (updated_obj.geometry_style === 'group_geometry') {
      (updated_obj as GroupGeometryObj).update_mesh(this, [updated_obj.name]);
      current_hook.push((updated_obj.node_style as GroupNode).group_name)

    } else {
      throw new Error(`Unknown geometry style: ${updated_obj.geometry_style}`)
    }

    let next_hook = []
    while (true) {
      for (const obj of this.obj_map.values()) {
        if (obj.hooks.some(hook => current_hook.includes(hook))) {

          if (obj.geometry_style === 'base_geometry') {
            (obj as BaseGeometryObj).update_mesh(this);
            next_hook.push(obj.name)

          } else if (obj.geometry_style === 'group_geometry') {
            (obj as GroupGeometryObj).update_mesh(this, current_hook);
            next_hook.push((obj.node_style as GroupNode).group_name)

          } else {
            throw new Error(`Unknown geometry style: ${obj.geometry_style}`)
          }
        }
      }

      current_hook = next_hook.slice()
      next_hook = []
      if (current_hook.length === 0) {
        break;
      }
    }
  }

} 