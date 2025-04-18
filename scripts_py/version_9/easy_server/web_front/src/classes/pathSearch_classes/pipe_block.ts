import type { BlockRoot } from "./base_root.ts";
import * as THREE from "three";

import { BaseBlock } from "./base_block.ts";
import { norm_vector } from "@/utils/math_utils.ts";

export class PipeBlock extends BaseBlock {
  x: number;
  y: number;
  z: number;
  vector_x: number;
  vector_y: number;
  vector_z: number;
  radius: number;
  is_inlet: boolean;
  color: string;
  group_name: string;
  properties: string[] = ["x", "y", "z", "vector_x", "vector_y", "vector_z", "radius"];

  constructor(
    name: string, 
    x: number, y: number, z: number, vector_x: number, vector_y: number, vector_z: number, 
    radius: number, is_inlet: boolean, color: string, group_name: string, root: BlockRoot
) {
    super(name);
    this.x = x;
    this.y = y;
    this.z = z;
    const vector_norm = norm_vector([vector_x, vector_y, vector_z]);
    this.vector_x = vector_norm[0];
    this.vector_y = vector_norm[1];
    this.vector_z = vector_norm[2];
    this.radius = radius;
    this.is_inlet = is_inlet;
    this.color = color;
    this.group_name = group_name;
    this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
    // Set orientation using quaternion
    const direction = new THREE.Vector3(this.vector_x, this.vector_y, this.vector_z).normalize();
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);

    const cylinder_length = Math.max(1.0, this.radius * 0.5);
    const pipe_mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(
        this.radius * 0.4, 
        this.radius * 0.4, 
        cylinder_length, 
        32
      ),
      new THREE.MeshBasicMaterial({ color: this.color })
    );

    pipe_mesh.position.set(this.x, this.y, this.z);
    // pipe_mesh.setRotationFromQuaternion(quaternion);
    pipe_mesh.name = `${this.name}_pipe`;
    this.mesh_objs.set(pipe_mesh.name, pipe_mesh);
    
    const cone_height = this.radius * 0.5;
    const cone_mesh = new THREE.Mesh(
      new THREE.ConeGeometry(this.radius * 0.8, cone_height, 32),
      new THREE.MeshBasicMaterial({ color: this.color })
    );
    cone_mesh.position.set(
      pipe_mesh.position.x + this.vector_x * (cylinder_length * 0.5 + cone_height * 0.5),
      pipe_mesh.position.y + this.vector_y * (cylinder_length * 0.5 + cone_height * 0.5),
      pipe_mesh.position.z + this.vector_z * (cylinder_length * 0.5 + cone_height * 0.5)
    );
    cone_mesh.setRotationFromQuaternion(quaternion);
    cone_mesh.name = `${this.name}_cone`;
    this.mesh_objs.set(cone_mesh.name, cone_mesh);

    super.add_mesh(root);
  }

  get_property(property: string): number| boolean {
    if (property === "x") {
      return this.x;
    }
    else if (property === "y") {
      return this.y;
    }
    else if (property === "z") {
      return this.z;
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
    else if (property === "radius") {
      return this.radius;
    }
    else if (property === "is_inlet") {
      return this.is_inlet;
    }
    else {
      throw new Error(`Invalid property: ${property}`);
    }
  }

  set_properties(property: string, value: any) {
    if (property === "x") {
      this.x = value;
    }
    else if (property === "y") {
      this.y = value;
    }
    else if (property === "z") {
      this.z = value;
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
    else if (property === "radius") {
      this.radius = value;
    }
    else if (property === "is_inlet") {
      this.is_inlet = value;
    }
  }

  toJson(): any {
    return {
      name: this.name,
      x: this.x,
      y: this.y,
      z: this.z,
      vector_x: this.vector_x,
      vector_y: this.vector_y,
      vector_z: this.vector_z,
      radius: this.radius,
      is_inlet: this.is_inlet,
      group_name: this.group_name,
      color: this.color,
    };
  }
  
}