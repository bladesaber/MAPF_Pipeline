import * as THREE from 'three';

import {
  add_mesh,
  color_to_hex,
  create_colored_points,
  create_points,
  remove_mesh,
} from '@/utils/visulize_utils';

import { MeshObj } from '../mesh_type';

export class GridMesh extends MeshObj {
    xlength: number;
    ylength: number;
    zlength: number;
    xsize: number;
    ysize: number;
    zsize: number;
    resolution: number;

    discretize_source: string = '';

    surface_obstacle_points: number[][] = [];
    surface_obstacle_mesh: THREE.Points | null = null;

    inner_framework_points: number[][] = [];
    inner_framework_distance: number[] = [];
    inner_framework_mesh: THREE.Points | null = null;

    constructor(
        name: string, xlength: number, ylength: number, zlength: number, resolution: number,
        scene: THREE.Scene
    ) {
        super(name);
        this.xlength = xlength;
        this.ylength = ylength;
        this.zlength = zlength;
        this.xsize = Math.ceil(this.xlength / resolution);
        this.ysize = Math.ceil(this.ylength / resolution);
        this.zsize = Math.ceil(this.zlength / resolution);
        this.resolution = resolution;
        this.add_mesh(scene);
    }

    value_to_grid(value: number): number {
        return Math.round(value / this.resolution);
    }

    value_to_grid_scaled(value: number): number {
        return value / this.resolution;
    }

    grid_to_value(value: number): number {
        return value * this.resolution;
    }

    grid_to_value_scaled(value: number): number {
        return value * this.resolution;
    }

    add_mesh(scene: THREE.Scene): void {
        // Create the main box mesh with higher transparency
        const mesh = new THREE.Mesh(
            new THREE.BoxGeometry(this.xlength, this.ylength, this.zlength),
            new THREE.MeshBasicMaterial({
                color: 0x00ff00,
                opacity: 0.025,  // More transparent
                transparent: true,
                depthWrite: false,  // Disable depth writing for transparent objects
                depthTest: true,
                side: THREE.DoubleSide  // Render both sides of the faces
            })
        );
        mesh.position.set(
            this.xlength / 2,
            this.ylength / 2,
            this.zlength / 2
        );
        mesh.name = this.name;

        // Add edge lines to make the box more visible
        const edges = new THREE.EdgesGeometry(mesh.geometry);
        const line = new THREE.LineSegments(
            edges,
            new THREE.LineBasicMaterial({
                color: 0x00ff00,
                linewidth: 2,
                transparent: true,
                opacity: 0.8,
                depthWrite: true,  // Enable depth writing for lines
                depthTest: true,
                depthFunc: THREE.LessEqualDepth  // Use LessEqual for depth testing
            })
        );
        line.position.copy(mesh.position);
        line.name = `${this.name}_edges`;

        // Add a small offset to the line position to prevent z-fighting
        line.position.y += 0.001;

        this.mesh_dict[mesh.name] = mesh;
        this.mesh_dict[line.name] = line;
        super.add_mesh(scene);
    }

    hide_discretize_mesh() {
        if (this.surface_obstacle_mesh) {
            this.surface_obstacle_mesh.visible = false;
        }
    }

    show_discretize_mesh() {
        if (this.surface_obstacle_mesh) {
            this.surface_obstacle_mesh.visible = true;
        }
    }

    hide_inner_framework_mesh() {
        if (this.inner_framework_mesh) {
            this.inner_framework_mesh.visible = false;
        }
    }

    show_inner_framework_mesh() {
        if (this.inner_framework_mesh) {
            this.inner_framework_mesh.visible = true;
        }
    }

    has_discretize_mesh(source: string): boolean {
        return this.discretize_source === source;
    }

    discretize(
        scene: THREE.Scene, resolution: number, source: string,
        pipe_dict: { x: number, y: number, z: number, radius: number }[]
    ): void {
        if (this.surface_obstacle_mesh) {
            remove_mesh(scene, this.surface_obstacle_mesh.name);
            this.surface_obstacle_mesh = null;
            
            if (this.inner_framework_mesh) {
                remove_mesh(scene, this.inner_framework_mesh.name);
                this.inner_framework_mesh = null;
            }
        }

        this.surface_obstacle_points = []
        const usedPositions = new Set<string>() // Track used positions

        const xSegments = Math.max(2, Math.floor((this.xlength - 0) / resolution));
        const ySegments = Math.max(2, Math.floor((this.ylength - 0) / resolution));
        const zSegments = Math.max(2, Math.floor((this.zlength - 0) / resolution));

        // Sample points on each face of the box
        // Front and back faces (constant z)
        for (let x = 0; x <= xSegments; x++) {
            for (let y = 0; y <= ySegments; y++) {
                const xPos = 0 + (x / xSegments) * (this.xlength - 0);
                const yPos = 0 + (y / ySegments) * (this.ylength - 0);
                if (!usedPositions.has(`${xPos},${yPos},0`)) {
                    this.surface_obstacle_points.push([xPos, yPos, 0]); // Front face
                    usedPositions.add(`${xPos},${yPos},0`);
                }
                if (!usedPositions.has(`${xPos},${yPos},${this.zlength}`)) {
                    this.surface_obstacle_points.push([xPos, yPos, this.zlength]); // Back face
                    usedPositions.add(`${xPos},${yPos},${this.zlength}`);
                }
            }
        }

        // Left and right faces (constant x)
        for (let y = 0; y <= ySegments; y++) {
            for (let z = 0; z <= zSegments; z++) {
                const yPos = 0 + (y / ySegments) * (this.ylength - 0);
                const zPos = 0 + (z / zSegments) * (this.zlength - 0);
                if (!usedPositions.has(`0,${yPos},${zPos}`)) {
                    this.surface_obstacle_points.push([0, yPos, zPos]); // Left face
                    usedPositions.add(`0,${yPos},${zPos}`);
                }
                if (!usedPositions.has(`${this.xlength},${yPos},${zPos}`)) {
                    this.surface_obstacle_points.push([this.xlength, yPos, zPos]); // Right face
                    usedPositions.add(`${this.xlength},${yPos},${zPos}`);
                }
            }
        }

        // Top and bottom faces (constant y)
        for (let x = 0; x <= xSegments; x++) {
            for (let z = 0; z <= zSegments; z++) {
                const xPos = 0 + (x / xSegments) * (this.xlength - 0);
                const zPos = 0 + (z / zSegments) * (this.zlength - 0);
                if (!usedPositions.has(`${xPos},0,${zPos}`)) {
                    this.surface_obstacle_points.push([xPos, 0, zPos]); // Bottom face
                    usedPositions.add(`${xPos},0,${zPos}`);
                }
                if (!usedPositions.has(`${xPos},${this.ylength},${zPos}`)) {
                    this.surface_obstacle_points.push([xPos, this.ylength, zPos]); // Top face
                    usedPositions.add(`${xPos},${this.ylength},${zPos}`);
                }
            }
        }

        // Filter points that conflict with pipes
        const filtered_points = this.surface_obstacle_points.filter(point => {
            const [x, y, z] = point;
            // Check if point conflicts with any pipe
            return !pipe_dict.some(pipe => {
                const dx = x - pipe.x;
                const dy = y - pipe.y;
                const dz = z - pipe.z;
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                return distance <= pipe.radius + 0.25;
            });
        });
        this.surface_obstacle_points = filtered_points;

        // Create points visualization
        this.surface_obstacle_mesh = create_points(this.surface_obstacle_points, {
            color: color_to_hex([0, 0, 0]),
            size: 0.5
        });
        this.surface_obstacle_mesh.name = `${this.name}_surface_obstacle`;
        add_mesh(scene, this.surface_obstacle_mesh);
        this.discretize_source = source;
        this.surface_obstacle_mesh.visible = false;
    }

    add_inner_grid(inner_grid_points: number[][], inner_grid_distance: number[], scene: THREE.Scene) {
        if (this.inner_framework_mesh) {
            remove_mesh(scene, this.inner_framework_mesh.name);
            this.inner_framework_mesh = null;
        }

        this.inner_framework_points = []
        this.inner_framework_distance = []
        for (let i = 0; i < inner_grid_points.length; i++) {
            this.inner_framework_points.push([
                inner_grid_points[i][0] * this.resolution,
                inner_grid_points[i][1] * this.resolution,
                inner_grid_points[i][2] * this.resolution
            ])
            this.inner_framework_distance.push(inner_grid_distance[i])
        }

        this.inner_framework_mesh = create_colored_points(
            this.inner_framework_points, this.inner_framework_distance,
            { size: 0.3 }
        );
        this.inner_framework_mesh.name = `${this.name}_inner_framework`;
        add_mesh(scene, this.inner_framework_mesh);
        this.inner_framework_mesh.visible = false;

        console.log(scene)
    }

    update_inner_grid_visibility(threshold: number, method: string) {
        if (this.inner_framework_mesh) {
            const positions = this.inner_framework_mesh.geometry.getAttribute('position')
            const colors = this.inner_framework_mesh.geometry.getAttribute('color')
            const visible = new Float32Array(positions.count)

            for (let i = 0; i < positions.count; i++) {
                const distance = this.inner_framework_distance[i]
                if (method === 'less_than') {
                    visible[i] = distance < threshold ? 1.0 : 0.0
                } else if (method === 'greater_than') {
                    visible[i] = distance > threshold ? 1.0 : 0.0
                } else {
                    visible[i] = 1.0
                }
            }

            this.inner_framework_mesh.geometry.setAttribute('visible', new THREE.BufferAttribute(visible, 1))
            this.inner_framework_mesh.geometry.attributes.visible.needsUpdate = true
        }
    }

    toJSON() {
        return {
            name: this.name,
            xlength: this.xlength,
            ylength: this.ylength,
            zlength: this.zlength,
            resolution: this.resolution,
        };
    }
}