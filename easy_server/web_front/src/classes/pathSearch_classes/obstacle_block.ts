import type { BlockRoot } from "./base_root.ts";
import * as THREE from "three";
import { read_mesh_json } from "../../utils/visulize_utils.ts";

import { BaseBlock } from "./base_block.ts";

export class ObstacleBlock extends BaseBlock {
  relax_groups: string[];

  constructor(name: string, relax_groups: string[] = [], root: BlockRoot) {
    super(name);
    this.relax_groups = relax_groups;
  }

  convert_to_discretize_cfg(): any {
    throw new Error("Not implemented");
  }
}

export class CylinderObstacleBlock extends ObstacleBlock {
  x: number;
  y: number;
  z: number;
  radius: number;
  height: number;
  vector_x: number;
  vector_y: number;
  vector_z: number;

  constructor(
    name: string, 
    x: number, y: number, z: number, 
    vector_x: number, vector_y: number, vector_z: number, 
    radius: number, height: number, relax_groups: string[], root: BlockRoot
) {
    super(name, relax_groups, root);
    this.x = x;
    this.y = y;
    this.z = z;
    this.vector_x = vector_x;
    this.vector_y = vector_y;
    this.vector_z = vector_z;
    this.radius = radius;
    this.height = height;
    this.add_mesh(root);
  }
  
  add_mesh(root: BlockRoot): void {
    const cylinder_mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(this.radius, this.radius, this.height, 32),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    );
    cylinder_mesh.name = this.name;
    cylinder_mesh.position.set(this.x, this.y, this.z);

    const direction = new THREE.Vector3(this.vector_x, this.vector_y, this.vector_z).normalize();
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    
    cylinder_mesh.setRotationFromQuaternion(quaternion);

    this.mesh_objs.set(cylinder_mesh.name, cylinder_mesh);
    super.add_mesh(root);
  }

  get_property(property: string): number {
    if (property === "x") {
      return this.x;
    }
    else if (property === "y") {
      return this.y;
    }
    else if (property === "z") {
      return this.z;
    }
    else if (property === "radius") {
      return this.radius;
    }
    else if (property === "height") {
      return this.height;
    }
    else if (property === "vector_x") {
      return this.vector_x;
    }
    else if (property === "vector_y") {
      return this.vector_y;
    }
    else if (property === "vector_z") {
      return this.vector_z;
    }
    else {
      throw new Error(`Invalid property: ${property}`);
    }
  }

  set_properties(property: string, value: number): void {
    if (property === "x") {
      this.x = value;
    }
    else if (property === "y") {
      this.y = value;
    }
    else if (property === "z") {
      this.z = value;
    }
    else if (property === "radius") {
      this.radius = value;
    }
    else if (property === "height") {
      this.height = value;
    }
    else if (property === "vector_x") {
      this.vector_x = value;
    }
    else if (property === "vector_y") {
      this.vector_y = value;
    }
    else if (property === "vector_z") {
      this.vector_z = value;
    }
    else {
      throw new Error(`Invalid property: ${property}`);
    }
  }

  toJson(): any {
    return {
      name: this.name,
      x: this.x,
      y: this.y,
      z: this.z,
      radius: this.radius,
      height: this.height,
      vector_x: this.vector_x,
      vector_y: this.vector_y,
      vector_z: this.vector_z,
      relax_groups: this.relax_groups,
      type: "CylinderObstacle",
    };
  }
}

export class BoxObstacleBlock extends ObstacleBlock {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
  zmin: number;
  zmax: number;

  constructor(
    name: string, 
    xmin: number, xmax: number, ymin: number, ymax: number, zmin: number, zmax: number, 
    relax_groups: string[], root: BlockRoot
  ) {
    super(name, relax_groups, root);
    this.xmin = xmin;
    this.xmax = xmax;
    this.ymin = ymin;
    this.ymax = ymax;
    this.zmin = zmin;
    this.zmax = zmax;
    this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
    const box_mesh = new THREE.Mesh(
      new THREE.BoxGeometry(this.xmax - this.xmin, this.ymax - this.ymin, this.zmax - this.zmin),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    );
    box_mesh.position.set(
      this.xmin + (this.xmax - this.xmin) / 2, 
      this.ymin + (this.ymax - this.ymin) / 2, 
      this.zmin + (this.zmax - this.zmin) / 2
    );
    box_mesh.name = this.name;
    this.mesh_objs.set(box_mesh.name, box_mesh);
    super.add_mesh(root);
  }

  get_property(property: string): number {
    if (property === "x") {
      return this.xmin + (this.xmax - this.xmin) / 2;
    }
    else if (property === "y") {
      return this.ymin + (this.ymax - this.ymin) / 2;
    }
    else if (property === "z") {
      return this.zmin + (this.zmax - this.zmin) / 2;
    }
    else if (property === "xmin") {
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
      relax_groups: this.relax_groups,
      type: "BoxObstacle",
    };
  }
}

export class SphereObstacleBlock extends ObstacleBlock {
  x: number;
  y: number;
  z: number;
  radius: number;
  
  constructor(
    name: string, x: number, y: number, z: number, radius: number, 
    relax_groups: string[], root: BlockRoot
  ) {
    super(name, relax_groups, root);
    this.x = x;
    this.y = y;
    this.z = z;
    this.radius = radius;
    this.add_mesh(root);
  }
  
  add_mesh(root: BlockRoot): void {
    const sphere_mesh = new THREE.Mesh(
      new THREE.SphereGeometry(this.radius, 32, 32),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    );
    sphere_mesh.position.set(this.x, this.y, this.z);
    sphere_mesh.name = this.name;
    this.mesh_objs.set(sphere_mesh.name, sphere_mesh);
    super.add_mesh(root);
  }

  get_property(property: string): number {
    if (property === "x") {
      return this.x;
    }
    else if (property === "y") {
      return this.y;
    }
    else if (property === "z") {
      return this.z;
    }
    else if (property === "radius") {
      return this.radius;
    }
    else {
      throw new Error(`Invalid property: ${property}`);
    }
  }

  set_properties(property: string, value: number): void {
    if (property === "x") {
      this.x = value;
    }
    else if (property === "y") {
      this.y = value;
    }
    else if (property === "z") {
      this.z = value;
    }
    else if (property === "radius") {
      this.radius = value;
    }
    else {
      throw new Error(`Invalid property: ${property}`);
    }
  }

  toJson(): any {
    return {
      name: this.name,
      x: this.x,
      y: this.y,
      z: this.z,
      radius: this.radius,
      relax_groups: this.relax_groups,
      type: "SphereObstacle",
    };
  }
}

export class HttpObstacleBlock extends ObstacleBlock {
  constructor(
      name: string, mesh_json: any, relax_groups: string[], root: BlockRoot
  ) {
      super(name, relax_groups, root);
      const mesh = read_mesh_json(mesh_json);
      mesh.name = this.name;
      this.mesh_objs.set(mesh.name, mesh);
      this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
      super.add_mesh(root);
  }

  toJson(): any {
      return {
          name: this.name,
          relax_groups: this.relax_groups,
          type: "HttpObstacle",
      };
  }
}
