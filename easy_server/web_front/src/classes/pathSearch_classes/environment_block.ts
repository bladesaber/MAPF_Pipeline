import * as THREE from "three";

import type { BlockRoot } from "./base_root.ts";

import { BaseBlock } from "./base_block.ts";

export class EnvironmentBlock extends BaseBlock {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
  zmin: number;
  zmax: number;

  constructor(name: string, xmin: number, xmax: number, ymin: number, ymax: number, zmin: number, zmax: number, root: BlockRoot) {
    super(name);
    this.xmin = xmin;
    this.xmax = xmax;
    this.ymin = ymin;
    this.ymax = ymax;
    this.zmin = zmin;
    this.zmax = zmax;
    this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
    const environment_mesh = new THREE.Mesh(
      new THREE.BoxGeometry(this.xmax - this.xmin, this.ymax - this.ymin, this.zmax - this.zmin),
      new THREE.MeshBasicMaterial({ color: 0x00ff00, opacity: 0.3, transparent: true })
    );
    environment_mesh.position.set(
        this.xmin + (this.xmax - this.xmin) / 2, 
        this.ymin + (this.ymax - this.ymin) / 2, 
        this.zmin + (this.zmax - this.zmin) / 2
    );
    environment_mesh.name = `${this.name}_environment`;
    this.mesh_objs.set(environment_mesh.name, environment_mesh);
    super.add_mesh(root);
  }

  get_property(property: string): number {
    if (property === "xmin") {
      return this.xmin;
    }
    else if (property === "xmax") {
      return this.xmax;
    }
    else if (property === "ymin") {
      return this.ymin;
    }
    else if (property === "ymax") {
      return this.ymax;
    }
    else if (property === "zmin") {
      return this.zmin;
    }
    else if (property === "zmax") {
      return this.zmax;
    }
    else {
      throw new Error(`Invalid property: ${property}`);
    }
  }

  set_properties(property: string, value: number): void {
    if (property === "xmin") {
      this.xmin = value;
    }
    else if (property === "xmax") {
      this.xmax = value;
    }
    else if (property === "ymin") {
      this.ymin = value;
    }
    else if (property === "ymax") {
      this.ymax = value;
    }
    else if (property === "zmin") {
      this.zmin = value;
    }
    else if (property === "zmax") {
      this.zmax = value;
    }
    else {
      throw new Error(`Invalid property: ${property}`);
    }
  }

  toJson(): any {
    return {
      name: this.name,
      xmin: this.xmin,
      xmax: this.xmax,
      ymin: this.ymin,
      ymax: this.ymax,
      zmin: this.zmin,
      zmax: this.zmax,
    };
  }
}
