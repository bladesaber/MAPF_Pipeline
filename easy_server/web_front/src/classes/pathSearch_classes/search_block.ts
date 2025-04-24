import type { BlockRoot } from "./base_root.ts";
import * as THREE from "three";

import { BaseBlock } from "./base_block.ts";
import { create_tube, create_colored_points } from "@/utils/visulize_utils.ts";

export class DiscreteGridBlock extends BaseBlock {
  points: number[][] = [];
  distance_to_obstacle: number[] = [];
  
  constructor(name: string, points: number[][], distance_to_obstacle: number[], root: BlockRoot) {
    super(name);
    this.points = points;
    this.distance_to_obstacle = distance_to_obstacle;
    this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
    const points_mesh: THREE.Points = create_colored_points(
      this.points, this.distance_to_obstacle, { color: 0x0000ff, size: 0.25 }
    );
    points_mesh.name = this.name;
    this.mesh_objs.set(points_mesh.name, points_mesh);
    super.add_mesh(root);
  }
}

export class PathBlock extends BaseBlock {
  points: number[][] = [];
  radius: number = 0;
  color: string = "#f00f0f";

  constructor(name: string, points: number[][], radius: number, color: string, root: BlockRoot) {
    super(name);
    this.points = points;
    this.radius = radius;
    this.color = color;
    this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
    const lines = create_tube(this.points, this.radius, { 
      color: parseInt(this.color.replace('#', '0x')),
      opacity: 1.0,
      transparent: true
    }) as THREE.Mesh;
    lines.name = this.name;
    this.mesh_objs.set(this.name, lines);
    super.add_mesh(root);
  }
  
  get_property(property: string): number|boolean {
    throw new Error("Not supported");
  }

  set_property(property: string, value: number|boolean): void {
    throw new Error("Not supported");
  }
  
  toJson(): any {
    return {
      name: this.name,
      points: this.points,
      radius: this.radius,
    };
  }
}
