import * as THREE from "three";

import type { BlockRoot } from "./base_root.ts";

import { BaseBlock } from "./base_block.ts";

export class EnvironmentBlock extends BaseBlock {
  xlength: number;
  ylength: number;
  zlength: number;
  xsize: number;
  ysize: number;
  zsize: number;
  resolution: number;
  
  constructor(name: string, xlength: number, ylength: number, zlength: number, resolution: number, root: BlockRoot) {
    super(name);
    this.xlength = xlength;
    this.ylength = ylength;
    this.zlength = zlength;
    this.resolution = resolution;
    this.xsize = Math.ceil(xlength / resolution);
    this.ysize = Math.ceil(ylength / resolution);
    this.zsize = Math.ceil(zlength / resolution);
    this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
    const environment_mesh = new THREE.Mesh(
      new THREE.BoxGeometry(this.xsize, this.ysize, this.zsize),
      new THREE.MeshBasicMaterial({ color: 0x00ff00, opacity: 0.3, transparent: true })
    );
    environment_mesh.position.set(
        this.xsize / 2, 
        this.ysize / 2, 
        this.zsize / 2
    );
    environment_mesh.name = `${this.name}_environment`;
    this.mesh_objs.set(environment_mesh.name, environment_mesh);
    super.add_mesh(root);
  }

  get_property(property: string): number {
    throw new Error(`Invalid property: ${property}`);
  }

  set_properties(property: string, value: number): void {
    throw new Error(`Invalid property: ${property}`);
  }

  toJson(): any {
    return {
      name: this.name,
      xlength: this.xlength,
      ylength: this.ylength,
      zlength: this.zlength,
      resolution: this.resolution,
    };
  }
}
