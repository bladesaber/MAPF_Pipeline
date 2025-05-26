import * as THREE from "three";

import { MeshObj } from "../mesh_type";
import { color_to_hex } from "@/utils/visulize_utils.ts";
import { create_tube, create_colored_points } from "@/utils/visulize_utils.ts";

export class PathResultMesh extends MeshObj {
  points: number[][] = [];
  radius: number = -1;
  color: number[] = [1, 0, 0];

  line_mesh: THREE.Mesh | null = null;
  tube_mesh: THREE.Mesh | null = null;

  constructor(name: string, points: number[][], radius: number, color: number[], scene: THREE.Scene) {
    super(name);
    this.points = points;
    this.radius = radius;
    this.color = color;
    this.add_mesh(scene);
  }
  
  add_mesh(scene: THREE.Scene) {
    this.tube_mesh = create_tube(this.points, this.radius, { 
        color: color_to_hex(this.color),
        opacity: 0.5
    }) as THREE.Mesh;
    this.tube_mesh.name = this.name;

    this.line_mesh = create_tube(this.points, 0.5, { 
      color: color_to_hex(this.color),
      opacity: 0.5
    }) as THREE.Mesh;
    this.line_mesh.name = this.name + '_line';
    
    this.mesh_dict[this.tube_mesh.name] = this.tube_mesh;
    this.mesh_dict[this.line_mesh.name] = this.line_mesh;
    super.add_mesh(scene);

    this.tube_mesh.visible = false;
  }

  show_tube_mesh() {
    if (this.tube_mesh) {
      this.tube_mesh.visible = true;
    }
  }

  hide_tube_mesh() {
    if (this.tube_mesh) {
      this.tube_mesh.visible = false;
    }
  }

  show_line_mesh() {
    if (this.line_mesh) {
      this.line_mesh.visible = true;
    }
  }

  hide_line_mesh() {
    if (this.line_mesh) {
      this.line_mesh.visible = false;
    }
  }
  
}