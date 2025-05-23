import * as THREE from "three";

import { MeshObj } from "../mesh_type";
import { color_to_hex } from "@/utils/visulize_utils.ts";
import { create_tube, create_colored_points } from "@/utils/visulize_utils.ts";

export class PathResultMesh extends MeshObj {
  points: number[][] = [];
  radius: number = -1;
  color: number[] = [1, 0, 0];

  constructor(name: string, points: number[][], radius: number, color: number[], scene: THREE.Scene) {
    super(name);
    this.points = points;
    this.radius = radius;
    this.color = color;
    this.add_mesh(scene);
  }
  
  add_mesh(scene: THREE.Scene) {
    const mesh = create_tube(this.points, this.radius, { 
        color: color_to_hex(this.color),
        opacity: 0.5
    }) as THREE.Mesh;
    mesh.name = this.name;
    
    this.mesh_dict[mesh.name] = mesh;
    super.add_mesh(scene);
  }
  
}