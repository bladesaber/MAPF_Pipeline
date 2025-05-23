import * as THREE from "three";

import { MeshObj } from "../mesh_type";
import { create_cone, color_to_hex, add_mesh, remove_mesh } from "@/utils/visulize_utils";
import type { GridMesh } from "./grid";

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

    discrete_x: number = -1;
    discrete_y: number = -1;
    discrete_z: number = -1;
    has_discrete: boolean = false;
    discrete_mesh: THREE.Mesh | null = null;

    constructor(
        name: string, 
        x: number, y: number, z: number, radius: number,
        vector_x: number, vector_y: number, vector_z: number,
        color: number[], group_name: string,
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
        this.add_mesh(scene);
    }

    add_mesh(scene: THREE.Scene): void {
        const mesh = create_cone(
            this.x, this.y, this.z, this.radius, this.radius, 
            this.vector_x, this.vector_y, this.vector_z, 
            { color: color_to_hex(this.color) }
        );
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

    discrete(scene: THREE.Scene, grid: GridMesh): void {
        if (this.has_discrete) {
            this.discrete_mesh = null;
        }

        this.discrete_x = grid.value_to_grid(this.x);
        this.discrete_y = grid.value_to_grid(this.y);
        this.discrete_z = grid.value_to_grid(this.z);
        this.has_discrete = true;

        this.discrete_mesh = create_cone(
            this.discrete_x, this.discrete_y, this.discrete_z, 
            this.radius, this.radius, 
            this.vector_x, this.vector_y, this.vector_z, 
            { color: color_to_hex(this.color) }
        );
        this.discrete_mesh.name = `${this.name}_discrete`;
        add_mesh(scene, this.discrete_mesh);
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
        };
    }
}