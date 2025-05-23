import { add_mesh, remove_mesh } from "../../utils/visulize_utils.ts";
import * as THREE from "three";

import type { BlockRoot } from "./base_root.ts";

export class BaseBlock {
  name: string;
  properties: string[] = [];
  mesh_objs: Map<string, THREE.Mesh|THREE.Points> = new Map();

  constructor(name: string) {
    this.name = name;
  }

  add_mesh(root: BlockRoot) {
    this.mesh_objs.forEach((mesh, key) => {
      add_mesh(root.get_scene(), mesh);
    });
  }

  remove_mesh(root: BlockRoot) {
    this.mesh_objs.forEach((mesh, key) => {
      remove_mesh(root.get_scene(), key);
    });
    this.mesh_objs.forEach((mesh, key) => {
        mesh.geometry.dispose();
      //   mesh.material.dispose();
    });
    this.mesh_objs.clear();
  }

  update_mesh(root: BlockRoot) {
    this.remove_mesh(root);
    this.add_mesh(root);
  }

  destory(root: BlockRoot) {
    this.remove_mesh(root);
  }

  set_param(param: any) {
    throw new Error("Not implemented");
  }

  set_property(property: string, value: any) {
    throw new Error("Not implemented");
  }

  get_property(property: string) {
    throw new Error("Not implemented");
  }

  hide_mesh() {
    this.mesh_objs.forEach((mesh, key) => {
      mesh.visible = false;
    });
  }
  
  show_mesh() {
    this.mesh_objs.forEach((mesh, key) => {
      mesh.visible = true;
    });
  }

  toJson() {
    return {
      name: this.name
    };
  }
}

