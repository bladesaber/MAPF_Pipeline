import { add_mesh, remove_mesh, create_sphere } from "../../utils/visulize_utils.ts";
import * as THREE from "three";
import { RootObj } from "./store_obj.ts";

export class BaseNode {
  style: string = "BaseNode";
  name: string;
  constructor(name: string) {
    this.name = name;
  }
}

export class GroupNode extends BaseNode {
  style: string = "GroupNode";
  group_name: string;
  constructor(name: string, group_name: string) {
    super(name);
    this.group_name = group_name;
  }
}

export class BaseGeometryObj {
  name: string;
  mesh: THREE.Mesh | null = null;
  hooks: string[] = [];
  style: string = "base_geometry";
  node_style: BaseNode | GroupNode;
  geometry_style: string = "base_geometry";

  constructor(name: string, node_style: BaseNode | GroupNode) {
    this.name = name;
    this.node_style = node_style;
  }

  add_mesh(rootObj: RootObj) {
    throw new Error("Not implemented");
  }

  remove_mesh(rootObj: RootObj) {
    remove_mesh(rootObj.get_scene(), this.name);
  }

  add_var(var_obj: BaseGeometryObj|GroupGeometryObj, index: number = -1) {
    throw new Error("Not implemented");
  }

  remove_var(var_obj: BaseGeometryObj|GroupGeometryObj) {
    throw new Error("Not implemented");
  }

  destory(rootObj: RootObj) {
    this.remove_mesh(rootObj);
    if (this.mesh) {
      this.mesh.geometry.dispose();
      this.mesh = null;
    }
  }

  update_mesh(rootObj: RootObj) {
    this.destory(rootObj);
    this.add_mesh(rootObj);
  }

  hide_mesh() {
    if (this.mesh) {
      this.mesh.visible = false;
    }
  }

  show_mesh() {
    if (this.mesh) {
      this.mesh.visible = true;
    }
  }

  update_hooks(hooks_name: string[], clear: boolean = false) {
    if (clear) {
      this.hooks = [];
    }
    this.hooks.push(...hooks_name);
  }

  toJSON() {
    return {
      name: this.name,
      style: this.style,
    };
  }
}

export class GroupGeometryObj {
  name: string;
  parent_nodes: Map<string, BaseNode|GroupNode> | (BaseNode|GroupNode)[] = [];
  mesh_objs: Map<string, BaseGeometryObj> = new Map();
  hooks: string[] = [];
  style: string = "group_geometry";
  node_style: BaseNode | GroupNode;
  geometry_style: string = "group_geometry";

  constructor(name: string, node_style: BaseNode | GroupNode) {
    this.name = name;
    this.node_style = node_style;
  }

  add_mesh(rootObj: RootObj) {
    throw new Error("Not implemented");
  }

  remove_mesh(rootObj: RootObj) {
    this.mesh_objs.forEach((mesh) => {
      mesh.remove_mesh(rootObj);
    });
  }

  destory(rootObj: RootObj) {
    this.remove_mesh(rootObj);
    this.mesh_objs.forEach((mesh) => {
      mesh.destory(rootObj);
    });
    this.mesh_objs = new Map();
  }

  get_cell_mesh(name: string) {
    return this.mesh_objs.get(name);
  }

  has_cell_mesh(name: string) {
    return this.mesh_objs.has(name);
  }

  update_mesh(rootObj: RootObj, changed_objs_name: string[]) {
    // if the changed obj is parent
    this.hooks.forEach((hook_name) => {
      if (changed_objs_name.includes(hook_name)) {
        this.destory(rootObj);
        this.add_mesh(rootObj);
        return;
      }
    });

    // if the changed obj is owned by the group
    changed_objs_name.forEach((name) => {
      if (this.has_cell_mesh(name)) {
        this.get_cell_mesh(name)?.update_mesh(rootObj);
      }
    });
  }

  hide_mesh() {
    this.mesh_objs.forEach((mesh) => {
      mesh.hide_mesh();
    });
  }

  show_mesh() {
    this.mesh_objs.forEach((mesh) => {
      mesh.show_mesh();
    });
  }

  update_hooks(hooks_name: string[], clear: boolean = false) {
    if (clear) {
      this.hooks = [];
    }
    this.hooks.push(...hooks_name);
  }

  toJSON() {
    return {
      name: this.name,
      style: this.style,
      mesh_objs: Array.from(this.mesh_objs.values()).map((obj) => obj.toJSON()),
    };
  }
}
