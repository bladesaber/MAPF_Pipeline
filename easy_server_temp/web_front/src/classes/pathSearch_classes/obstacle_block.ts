import type { BlockRoot } from "./base_root.ts";
import * as THREE from "three";
import { read_mesh_json, create_points } from "../../utils/visulize_utils.ts";
import { add_mesh, remove_mesh } from "../../utils/visulize_utils.ts";

import { BaseBlock } from "./base_block.ts";

export class ObstacleBlock extends BaseBlock {
  relax_groups: string[];
  discretize_points: number[][] = [];
  discretize_mesh: THREE.Points|null = null;
  has_discretize_mesh: boolean = false;
  sample_resolution: number = 0.1;

  constructor(name: string, relax_groups: string[] = [], sample_resolution: number = 0.1, root: BlockRoot) {
    super(name);
    this.relax_groups = relax_groups;
    this.sample_resolution = sample_resolution;
  }

  discrete(root: BlockRoot): void {
    if (this.has_discretize_mesh) {
      remove_mesh(root.get_scene(), this.discretize_mesh!.name);
      this.discretize_mesh!.geometry.dispose();
      this.has_discretize_mesh = false;
    }
    this.discretize_points = this.sample_surface_points();
    this.discretize_mesh = create_points(this.discretize_points, {color: 0x000000, size: 0.5});
    this.discretize_mesh.name = `${this.name}_discretize`;
    this.has_discretize_mesh = true;
    // console.log('this.discretize_mesh', this.discretize_mesh)
    add_mesh(root.get_scene(), this.discretize_mesh);
  }

  hide_discretize_mesh() {
    if (this.discretize_mesh) {
      this.discretize_mesh.visible = false;
    }
  }

  show_discretize_mesh() {
    if (this.discretize_mesh) {
      this.discretize_mesh.visible = true;
    }
  }

  sample_surface_points(): number[][] {
    throw new Error("Not implemented");
  }

  remove_mesh(root: BlockRoot): void {
    super.remove_mesh(root);
    if (this.discretize_mesh) {
      remove_mesh(root.get_scene(), this.discretize_mesh.name);
      this.discretize_mesh.geometry.dispose();
      this.has_discretize_mesh = false;
    }
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
  
  discrete_x: number;
  discrete_y: number;
  discrete_z: number;
  discrete_radius: number;
  discrete_height: number;
  grid_resolution: number;
  
  constructor(
    name: string, 
    x: number, y: number, z: number, 
    vector_x: number, vector_y: number, vector_z: number, 
    radius: number, height: number, 
    sample_resolution: number = 0.1, grid_resolution: number = 0.1, relax_groups: string[], root: BlockRoot
) {
    super(name, relax_groups, sample_resolution, root);
    this.x = x;
    this.y = y;
    this.z = z;
    this.vector_x = vector_x;
    this.vector_y = vector_y;
    this.vector_z = vector_z;
    this.radius = radius;
    this.height = height;

    this.discrete_x = x / grid_resolution;
    this.discrete_y = y / grid_resolution;
    this.discrete_z = z / grid_resolution;
    this.discrete_radius = radius / grid_resolution;
    this.discrete_height = height / grid_resolution;
    this.grid_resolution = grid_resolution;

    this.add_mesh(root);
  }
  
  add_mesh(root: BlockRoot): void {
    const cylinder_mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(this.discrete_radius, this.discrete_radius, this.discrete_height, 32),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    );
    cylinder_mesh.name = this.name;
    cylinder_mesh.position.set(this.discrete_x, this.discrete_y, this.discrete_z);

    const direction = new THREE.Vector3(this.vector_x, this.vector_y, this.vector_z).normalize();
    const quaternion = new THREE.Quaternion();
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    
    cylinder_mesh.setRotationFromQuaternion(quaternion);

    this.mesh_objs.set(cylinder_mesh.name, cylinder_mesh);
    super.add_mesh(root);
  }

  get_property(property: string): number {
    throw new Error(`Invalid property: ${property}`);
  }

  set_properties(property: string, value: number): void {
    throw new Error(`Invalid property: ${property}`);
  }

  sample_surface_points(): number[][] {
    const resolution = this.sample_resolution / this.grid_resolution;
    const points: number[][] = [];
    
    // Calculate number of segments for side and caps
    const circumference = 2 * Math.PI * this.discrete_radius;
    const numRadialSegments = Math.max(8, Math.floor(circumference / resolution));
    const numHeightSegments = Math.max(2, Math.floor(this.discrete_height / resolution));
    const numCapRings = Math.max(2, Math.floor(this.discrete_radius / (2 * resolution)));

    // Create orthonormal basis for the cylinder orientation
    const axis = new THREE.Vector3(this.vector_x, this.vector_y, this.vector_z).normalize();
    let tempUp = new THREE.Vector3(0, 1, 0);
    
    // Handle case where axis is parallel to up vector
    if (Math.abs(axis.dot(tempUp)) > 0.999) {
        tempUp = new THREE.Vector3(1, 0, 0);
    }
    
    const right = new THREE.Vector3().crossVectors(tempUp, axis).normalize();
    const up = new THREE.Vector3().crossVectors(axis, right).normalize();

    // Helper function to transform local to world coordinates
    const localToWorld = (x: number, y: number, z: number) => {
        return new THREE.Vector3(
            this.discrete_x + right.x * x + up.x * y + axis.x * z,
            this.discrete_y + right.y * x + up.y * y + axis.y * z,
            this.discrete_z + right.z * x + up.z * y + axis.z * z
        );
    };

    // Sample side surface points
    for (let h = 0; h <= numHeightSegments; h++) {
        const height = (h / numHeightSegments) * this.discrete_height - this.discrete_height / 2;
        
        for (let i = 0; i < numRadialSegments; i++) {
            const angle = (i / numRadialSegments) * Math.PI * 2;
            const x = Math.cos(angle) * this.discrete_radius;
            const y = Math.sin(angle) * this.discrete_radius;
            
            const worldPoint = localToWorld(x, y, height);
            points.push([worldPoint.x, worldPoint.y, worldPoint.z]);
        }
    }

    // Sample bottom cap points
    for (let r = 1; r <= numCapRings; r++) {
        const radius = (r / numCapRings) * this.discrete_radius;
        const segments = Math.max(4, Math.floor(numRadialSegments * (r / numCapRings)));
        
        for (let i = 0; i < segments; i++) {
            const angle = (i / segments) * Math.PI * 2;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            
            const worldPoint = localToWorld(x, y, -this.discrete_height / 2);
            points.push([worldPoint.x, worldPoint.y, worldPoint.z]);
        }
    }

    // Sample top cap points
    for (let r = 1; r <= numCapRings; r++) {
        const radius = (r / numCapRings) * this.discrete_radius;
        const segments = Math.max(4, Math.floor(numRadialSegments * (r / numCapRings)));
        
        for (let i = 0; i < segments; i++) {
            const angle = (i / segments) * Math.PI * 2;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            
            const worldPoint = localToWorld(x, y, this.discrete_height / 2);
            points.push([worldPoint.x, worldPoint.y, worldPoint.z]);
        }
    }

    // Add center points for caps
    points.push(localToWorld(0, 0, -this.discrete_height / 2).toArray());
    points.push(localToWorld(0, 0, this.discrete_height / 2).toArray());

    return points;
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
      sample_resolution: this.sample_resolution,
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

  discrete_xmin: number;
  discrete_xmax: number;
  discrete_ymin: number;
  discrete_ymax: number;
  discrete_zmin: number;
  discrete_zmax: number;  
  grid_resolution: number;

  constructor(
    name: string, 
    xmin: number, xmax: number, ymin: number, ymax: number, zmin: number, zmax: number, 
    sample_resolution: number = 0.1, grid_resolution: number = 0.1, relax_groups: string[], root: BlockRoot
  ) {
    super(name, relax_groups, sample_resolution, root);
    this.xmin = xmin;
    this.xmax = xmax;
    this.ymin = ymin;
    this.ymax = ymax;
    this.zmin = zmin;
    this.zmax = zmax;
    this.discrete_xmin = xmin / grid_resolution;
    this.discrete_xmax = xmax / grid_resolution;
    this.discrete_ymin = ymin / grid_resolution;
    this.discrete_ymax = ymax / grid_resolution;
    this.discrete_zmin = zmin / grid_resolution;
    this.discrete_zmax = zmax / grid_resolution;
    this.grid_resolution = grid_resolution;
    this.add_mesh(root);
  }

  add_mesh(root: BlockRoot): void {
    const box_mesh = new THREE.Mesh(
      new THREE.BoxGeometry(
        this.discrete_xmax - this.discrete_xmin, 
        this.discrete_ymax - this.discrete_ymin, 
        this.discrete_zmax - this.discrete_zmin
      ),
      new THREE.MeshBasicMaterial({ color: 0x00ff00 })
    );
    box_mesh.position.set(
      this.discrete_xmin + (this.discrete_xmax - this.discrete_xmin) / 2, 
      this.discrete_ymin + (this.discrete_ymax - this.discrete_ymin) / 2, 
      this.discrete_zmin + (this.discrete_zmax - this.discrete_zmin) / 2
    );
    box_mesh.name = this.name;
    this.mesh_objs.set(box_mesh.name, box_mesh);
    super.add_mesh(root);
  }

  get_property(property: string): number {
    throw new Error(`Invalid property: ${property}`);
  }

  set_properties(property: string, value: number): void {
    throw new Error(`Invalid property: ${property}`);
  }

  sample_surface_points(): number[][] {
    const resolution = this.sample_resolution / this.grid_resolution;
    const points: number[][] = [];
    const xSegments = Math.max(2, Math.floor((this.discrete_xmax - this.discrete_xmin) / resolution));
    const ySegments = Math.max(2, Math.floor((this.discrete_ymax - this.discrete_ymin) / resolution));
    const zSegments = Math.max(2, Math.floor((this.discrete_zmax - this.discrete_zmin) / resolution));

    // Sample points on each face of the box
    // Front and back faces (constant z)
    for (let x = 0; x <= xSegments; x++) {
      for (let y = 0; y <= ySegments; y++) {
        const xPos = this.discrete_xmin + (x / xSegments) * (this.discrete_xmax - this.discrete_xmin);
        const yPos = this.discrete_ymin + (y / ySegments) * (this.discrete_ymax - this.discrete_ymin);
        points.push([xPos, yPos, this.discrete_zmin]); // Front face
        points.push([xPos, yPos, this.discrete_zmax]); // Back face
      }
    }

    // Left and right faces (constant x)
    for (let y = 0; y <= ySegments; y++) {
      for (let z = 0; z <= zSegments; z++) {
        const yPos = this.discrete_ymin + (y / ySegments) * (this.discrete_ymax - this.discrete_ymin);
        const zPos = this.discrete_zmin + (z / zSegments) * (this.discrete_zmax - this.discrete_zmin);
        points.push([this.discrete_xmin, yPos, zPos]); // Left face
        points.push([this.discrete_xmax, yPos, zPos]); // Right face
      }
    }

    // Top and bottom faces (constant y)
    for (let x = 0; x <= xSegments; x++) {
      for (let z = 0; z <= zSegments; z++) {
        const xPos = this.discrete_xmin + (x / xSegments) * (this.discrete_xmax - this.discrete_xmin);
        const zPos = this.discrete_zmin + (z / zSegments) * (this.discrete_zmax - this.discrete_zmin);
        points.push([xPos, this.discrete_ymin, zPos]); // Bottom face
        points.push([xPos, this.discrete_ymax, zPos]); // Top face
      }
    }
    return points;
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
      sample_resolution: this.sample_resolution,
    };
  }
}

export class HttpObstacleBlock extends ObstacleBlock {
  constructor(
    name: string, mesh_json: any, 
    sample_resolution: number = 0.1, grid_resolution: number = 0.1, relax_groups: string[], root: BlockRoot
  ) {
    super(name, relax_groups, sample_resolution, root);
    throw new Error("Not implemented");
  }
  
  // constructor(
  //     name: string, mesh_json: any, 
  //     sample_resolution: number = 0.1, grid_resolution: number = 0.1, relax_groups: string[], root: BlockRoot
  // ) {
  //     super(name, relax_groups, sample_resolution, root);
  //     const mesh = read_mesh_json(mesh_json);
  //     mesh.name = this.name;
  //     this.mesh_objs.set(mesh.name, mesh);
  //     this.add_mesh(root);
  // }

  // add_mesh(root: BlockRoot): void {
  //     super.add_mesh(root);
  // }

  // toJson(): any {
  //     return {
  //         name: this.name,
  //         relax_groups: this.relax_groups,
  //         type: "HttpObstacle",
  //     };
  // }
}
