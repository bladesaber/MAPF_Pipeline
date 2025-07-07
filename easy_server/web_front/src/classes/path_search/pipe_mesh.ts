import * as THREE from "three";

import { MeshObj } from "../mesh_type";
import { create_cone, color_to_hex, add_mesh, remove_mesh, create_points, create_sphere } from "@/utils/visulize_utils";
import type { GridMesh } from "./grid";
import { pcd2plane_dist } from "@/utils/utils";

export class PipeMesh extends MeshObj {
    x: number;
    y: number;
    z: number;
    radius: number;
    vector_x: number;
    vector_y: number;
    vector_z: number;
    color: number[];
    group_name: string;
    pipe_mode: "inlet" | "outlet" | "intermediate" = "inlet";

    discretize_points: number[][] = [];
    has_discrete: boolean = false;
    discrete_mesh: THREE.Points | null = null;

    constructor(
        name: string, 
        x: number, y: number, z: number, radius: number,
        vector_x: number, vector_y: number, vector_z: number,
        color: number[], group_name: string, pipe_mode: "inlet" | "outlet" | "intermediate",
        scene: THREE.Scene
    ) {
        super(name);
        this.x = x;
        this.y = y;
        this.z = z;
        this.radius = radius;
        this.vector_x = vector_x;
        this.vector_y = vector_y;
        this.vector_z = vector_z;
        this.color = color;
        this.group_name = group_name;
        this.pipe_mode = pipe_mode;
        this.add_mesh(scene);
    }

    add_mesh(scene: THREE.Scene): void {
        let mesh: THREE.Mesh;
        if (this.pipe_mode === "intermediate"){
            mesh = create_sphere(
                this.x, this.y, this.z, this.radius, 
                { color: color_to_hex(this.color), output_mesh: true }
            ) as THREE.Mesh;
        }
        else {
            mesh = create_cone(
                this.x + this.vector_x * this.radius * 0.5, 
                this.y + this.vector_y * this.radius * 0.5, 
                this.z + this.vector_z * this.radius * 0.5, 
                this.radius, this.radius, 
                this.vector_x, this.vector_y, this.vector_z, 
                { color: color_to_hex(this.color) }
            );
        }
        mesh.name = this.name;

        // Add edge lines
        const edges = new THREE.EdgesGeometry(mesh.geometry);
        const line = new THREE.LineSegments(
            edges,
            new THREE.LineBasicMaterial({ color: 0x000000 })
        );
        line.name = `${this.name}_edges`;
        line.position.copy(mesh.position);
        line.quaternion.copy(mesh.quaternion);
        
        this.mesh_dict[mesh.name] = mesh;
        this.mesh_dict[line.name] = line;
        super.add_mesh(scene);
    }

    discrete(
        radius: number, resolution: number, 
        pipe_dict: { x: number, y: number, z: number, radius: number }[], 
        scene: THREE.Scene
    ): void {
        if (this.pipe_mode === "intermediate"){
            return;
        }

        // create half sphere shell of pipe
        radius = radius + 0.2
        let obstacle_points = []
        const sub_resolution = this.compute_sub_resolution(resolution, radius);

        const phi_steps = Math.ceil(2 * Math.PI * radius / sub_resolution);
        const theta_steps = Math.ceil(Math.PI * radius / sub_resolution);
            
        for (let phi_idx = 0; phi_idx < phi_steps; phi_idx++) {
            const phi = (phi_idx / phi_steps) * 2 * Math.PI;
            for (let theta_idx = 0; theta_idx < theta_steps; theta_idx++) {
                const theta = (theta_idx / theta_steps) * Math.PI;
                const dx = radius * Math.sin(theta) * Math.cos(phi);
                const dy = radius * Math.sin(theta) * Math.sin(phi);
                const dz = radius * Math.cos(theta);
                obstacle_points.push([this.x + dx, this.y + dy, this.z + dz]);
            }
        }

        if (this.pipe_mode === "inlet"){
            const norm = [this.vector_x, this.vector_y, this.vector_z]
            const center = [this.x, this.y, this.z]
            const dist = pcd2plane_dist(obstacle_points, norm, center)
            obstacle_points = obstacle_points.filter((point, index) => dist[index] <= -0.1)

        } else if (this.pipe_mode === "outlet"){
            const norm = [this.vector_x, this.vector_y, this.vector_z]
            const center = [this.x, this.y, this.z]
            const dist = pcd2plane_dist(obstacle_points, norm, center)
            obstacle_points = obstacle_points.filter((point, index) => dist[index] >= 0.1)
        }
        
        this.discretize_points = obstacle_points
        this.discretize_points = this.discretize_points.filter(point => {
            const [x, y, z] = point;
            // Check if point conflicts with any pipe
            return !pipe_dict.some(pipe => {
                const dx = x - pipe.x;
                const dy = y - pipe.y;
                const dz = z - pipe.z;
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                return distance <= pipe.radius + 0.1;
            });
        });

        this.discrete_mesh = create_points(this.discretize_points, {
            color: color_to_hex(this.color),
            size: 1.0
        });
        this.discrete_mesh.name = `${this.name}_discrete`;
        add_mesh(scene, this.discrete_mesh);
    }

    compute_sub_resolution(resolution: number, radius: number): number {
        resolution =  Math.min(resolution * 0.85, radius / 2);
        return resolution;
    }

    remove_mesh(scene: THREE.Scene): void {
        if (this.discrete_mesh) {
            remove_mesh(scene, this.discrete_mesh.name);
            this.discrete_mesh = null;
        }
        super.remove_mesh(scene);
        this.has_discrete = false;
    }

    hide_discrete_mesh(): void {
        if (this.discrete_mesh) {
            this.discrete_mesh.visible = false;
        }
    }

    show_discrete_mesh(): void {
        if (this.discrete_mesh) {
            this.discrete_mesh.visible = true;
        }
    }

    toJSON(is_discrete: boolean = false, grid: GridMesh|null = null): any {
        return {
            name: this.name,
            x: is_discrete ? grid!.value_to_grid(this.x) : this.x,
            y: is_discrete ? grid!.value_to_grid(this.y) : this.y,
            z: is_discrete ? grid!.value_to_grid(this.z) : this.z,
            radius: is_discrete ? grid!.value_to_grid_scaled(this.radius) : this.radius,
            vector_x: this.vector_x,
            vector_y: this.vector_y,
            vector_z: this.vector_z,
            group_name: this.group_name,
            pipe_mode: this.pipe_mode,
        };
    }
}


