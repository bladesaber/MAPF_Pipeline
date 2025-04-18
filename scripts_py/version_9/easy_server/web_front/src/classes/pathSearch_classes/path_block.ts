import type { BlockRoot } from "./base_root.ts";
import * as THREE from "three";

import { BaseBlock } from "./base_block.ts";
import { create_multi_lines } from "@/utils/visulize_utils.ts";

export class PathBlock extends BaseBlock {
  points: number[][] = [];
  radius: number = 0;
  color: number = 0xf00f0f;

  constructor(name: string, points: number[][], radius: number, color: number, root: BlockRoot) {
    super(name);
    this.points = points;
    this.radius = radius;
    this.color = color;
    this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
    const lines = create_multi_lines(this.points, { color: this.color, output_mesh: true }) as THREE.Mesh;
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
      color: this.color,
    };
  }
}