import * as THREE from "three";

import { MeshObj } from "../mesh_type";
import { remove_mesh, add_mesh, create_points, color_to_hex } from "@/utils/visulize_utils";
import { create_sphere } from "@/utils/visulize_utils";
import { pass } from "three/tsl";

export class ObstacleMesh extends MeshObj {
    relax_groups: string[] = [];
    discretize_points: number[][] = [];
    discretize_radius: number[] = [];
    discretize_mesh: THREE.Points | null = null;
    discretize_source: string = '';
    color: number[] = [0, 0, 0];

    constructor(name: string, color: number[], relax_groups: string[]) {
        super(name);
        this.color = color;
        this.relax_groups = relax_groups;
    }

    remove_mesh(scene: THREE.Scene): void {
        super.remove_mesh(scene);
        if (this.discretize_mesh) {
            remove_mesh(scene, this.discretize_mesh.name);
            this.discretize_mesh.geometry.dispose();
            this.discretize_source = '';
        }
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

    has_discretize_mesh(source: string): boolean {
        return this.discretize_source === source;
    }

    discrete(
        scene: THREE.Scene, resolution: number, source: string,
        pipe_dict: { x: number, y: number, z: number, radius: number }[]
    ): void {
        if (this.discretize_mesh !== null) {
            remove_mesh(scene, this.discretize_mesh!.name);
            this.discretize_mesh!.geometry.dispose();
        }
        
        resolution = this.compute_resolution(resolution);
        [this.discretize_points, this.discretize_radius] = this.sample_surface_points(resolution);

        this.discretize_points = this.discretize_points.filter(point => {
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

        this.discretize_mesh = create_points(this.discretize_points, {
            color: color_to_hex(this.color),
            size: 0.5
        });
        this.discretize_mesh.name = `${this.name}_discretize`;
        this.discretize_source = source;
        add_mesh(scene, this.discretize_mesh);
    }

    compute_resolution(resolution: number): number {
        return resolution;
    }

    sample_surface_points(resolution: number): [number[][], number[]] {
        throw new Error("Not implemented");
    }
}

export class CylinderObstacle extends ObstacleMesh {
    x: number;
    y: number;
    z: number;
    radius: number;
    height: number;
    vector_x: number;
    vector_y: number;
    vector_z: number;

    constructor(
        name: string, color: number[],
        x: number, y: number, z: number, radius: number, height: number,
        vector_x: number, vector_y: number, vector_z: number,
        relax_groups: string[],
        scene: THREE.Scene
    ) {
        super(name, color, relax_groups);
        this.x = x;
        this.y = y;
        this.z = z;
        this.radius = radius;
        this.height = height;
        this.vector_x = vector_x;
        this.vector_y = vector_y;
        this.vector_z = vector_z;
        this.add_mesh(scene);
    }

    add_mesh(scene: THREE.Scene): void {
        const mesh = new THREE.Mesh(
            new THREE.CylinderGeometry(this.radius, this.radius, this.height, 32),
            new THREE.MeshBasicMaterial({ color: 0x00ff00 })
        );
        mesh.name = this.name;

        mesh.position.set(this.x, this.y, this.z);
        const direction = new THREE.Vector3(this.vector_x, this.vector_y, this.vector_z).normalize();
        const quaternion = new THREE.Quaternion();
        quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);    
        mesh.setRotationFromQuaternion(quaternion);
        
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

    sample_surface_points(resolution: number): [number[][], number[]] {
        const points: number[][] = [];

        // Calculate number of segments for side and caps
        const circumference = 2 * Math.PI * this.radius;
        const numRadialSegments = Math.max(8, Math.floor(circumference / resolution));
        const numHeightSegments = Math.max(2, Math.floor(this.height / resolution));
        const numCapRings = Math.max(2, Math.floor(this.radius / (2 * resolution)));

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
                this.x + right.x * x + up.x * y + axis.x * z,
                this.y + right.y * x + up.y * y + axis.y * z,
                this.z + right.z * x + up.z * y + axis.z * z
            );
        };

        // Sample side surface points
        for (let h = 0; h <= numHeightSegments; h++) {
            const height = (h / numHeightSegments) * this.height - this.height / 2;

            for (let i = 0; i < numRadialSegments; i++) {
                const angle = (i / numRadialSegments) * Math.PI * 2;
                const x = Math.cos(angle) * this.radius;
                const y = Math.sin(angle) * this.radius;

                const worldPoint = localToWorld(x, y, height);
                points.push([worldPoint.x, worldPoint.y, worldPoint.z]);
            }
        }

        // Sample bottom cap points
        for (let r = 1; r <= numCapRings; r++) {
            const radius = (r / numCapRings) * this.radius;
            const segments = Math.max(4, Math.floor(numRadialSegments * (r / numCapRings)));

            for (let i = 0; i < segments; i++) {
                const angle = (i / segments) * Math.PI * 2;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;

                const worldPoint = localToWorld(x, y, -this.height / 2);
                points.push([worldPoint.x, worldPoint.y, worldPoint.z]);
            }
        }

        // Sample top cap points
        for (let r = 1; r <= numCapRings; r++) {
            const radius = (r / numCapRings) * this.radius;
            const segments = Math.max(4, Math.floor(numRadialSegments * (r / numCapRings)));

            for (let i = 0; i < segments; i++) {
                const angle = (i / segments) * Math.PI * 2;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;

                const worldPoint = localToWorld(x, y, this.height / 2);
                points.push([worldPoint.x, worldPoint.y, worldPoint.z]);
            }
        }

        // Add center points for caps
        points.push(localToWorld(0, 0, -this.height / 2).toArray());
        points.push(localToWorld(0, 0, this.height / 2).toArray());

        const radius = []
        for (let i = 0; i < points.length; i++) {
            radius.push(0.0);
        }

        return [points, radius];
    }

    compute_resolution(resolution: number): number {
        resolution =  Math.min(resolution, this.radius / 2);
        resolution = Math.min(resolution, this.height / 2);
        return resolution;
    }

    toJSON() {
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
            color: this.color,
            type: 'Cylinder',
            relax_groups: this.relax_groups,
        };
    }
}

export class BoxObstacle extends ObstacleMesh {
    xmin: number;
    xmax: number;
    ymin: number;
    ymax: number;
    zmin: number;
    zmax: number;

    constructor(
        name: string, color: number[],
        xmin: number, xmax: number,
        ymin: number, ymax: number,
        zmin: number, zmax: number,
        relax_groups: string[],
        scene: THREE.Scene
    ) {
        super(name, color, relax_groups);
        this.xmin = xmin;
        this.xmax = xmax;
        this.ymin = ymin;
        this.ymax = ymax;
        this.zmin = zmin;
        this.zmax = zmax;
        this.add_mesh(scene);
    }

    add_mesh(scene: THREE.Scene): void {
        const mesh = new THREE.Mesh(
            new THREE.BoxGeometry(
              this.xmax - this.xmin, 
              this.ymax - this.ymin, 
              this.zmax - this.zmin
            ),
            new THREE.MeshBasicMaterial({ color: 0x00ff00 })
        );
        mesh.name = this.name;
        mesh.position.set(
            this.xmin + (this.xmax - this.xmin) / 2, 
            this.ymin + (this.ymax - this.ymin) / 2, 
            this.zmin + (this.zmax - this.zmin) / 2
        );

        // Add edge lines
        const edges = new THREE.EdgesGeometry(mesh.geometry);
        const line = new THREE.LineSegments(
            edges,
            new THREE.LineBasicMaterial({ color: 0x000000 })
        );
        line.name = `${this.name}_edges`;
        line.position.copy(mesh.position);
        
        this.mesh_dict[mesh.name] = mesh;
        this.mesh_dict[line.name] = line;
        super.add_mesh(scene);
    }

    sample_surface_points(resolution: number): [number[][], number[]] {
        const points: number[][] = [];
        const xSegments = Math.max(2, Math.floor((this.xmax - this.xmin) / resolution));
        const ySegments = Math.max(2, Math.floor((this.ymax - this.ymin) / resolution));
        const zSegments = Math.max(2, Math.floor((this.zmax - this.zmin) / resolution));

        // Sample points on each face of the box
        // Front and back faces (constant z)
        for (let x = 0; x <= xSegments; x++) {
            for (let y = 0; y <= ySegments; y++) {
                const xPos = this.xmin + (x / xSegments) * (this.xmax - this.xmin);
                const yPos = this.ymin + (y / ySegments) * (this.ymax - this.ymin);
                points.push([xPos, yPos, this.zmin]); // Front face
                points.push([xPos, yPos, this.zmax]); // Back face
            }
        }

        // Left and right faces (constant x)
        for (let y = 0; y <= ySegments; y++) {
            for (let z = 0; z <= zSegments; z++) {
                const yPos = this.ymin + (y / ySegments) * (this.ymax - this.ymin);
                const zPos = this.zmin + (z / zSegments) * (this.zmax - this.zmin);
                points.push([this.xmin, yPos, zPos]); // Left face
                points.push([this.xmax, yPos, zPos]); // Right face
            }
        }

        // Top and bottom faces (constant y)
        for (let x = 0; x <= xSegments; x++) {
            for (let z = 0; z <= zSegments; z++) {
                const xPos = this.xmin + (x / xSegments) * (this.xmax - this.xmin);
                const zPos = this.zmin + (z / zSegments) * (this.zmax - this.zmin);
                points.push([xPos, this.ymin, zPos]); // Bottom face
                points.push([xPos, this.ymax, zPos]); // Top face
            }
        }

        const radius = []
        for (let i = 0; i < points.length; i++) {
            radius.push(0.0);
        }

        return [points, radius];
    }

    compute_resolution(resolution: number): number {
        resolution =  Math.min(resolution, (this.xmax - this.xmin) / 2);
        resolution = Math.min(resolution, (this.ymax - this.ymin) / 2);
        resolution = Math.min(resolution, (this.zmax - this.zmin) / 2);
        return resolution;
    }

    toJSON() {
        return {
            name: this.name,
            xmin: this.xmin,
            xmax: this.xmax,
            ymin: this.ymin,
            ymax: this.ymax,
            zmin: this.zmin,
            zmax: this.zmax,
            color: this.color,
            type: 'Box',
            relax_groups: this.relax_groups,
        };
    }
}

export class PointsObstacle extends ObstacleMesh {
    points: number[][] = [];

    constructor(name: string, color: number[], points: number[][], relax_groups: string[], scene: THREE.Scene) {
        super(name, color, relax_groups);
        this.points = points;
        this.add_mesh(scene);
    }

    add_mesh(scene: THREE.Scene): void {
        const mesh = create_points(this.points, {
            color: color_to_hex(this.color),
            size: 0.5
        });
        mesh.name = this.name;
        
        this.mesh_dict[mesh.name] = mesh;
        super.add_mesh(scene);
    }

    sample_surface_points(resolution: number): [number[][], number[]] {
        const radius = []
        for (let i = 0; i < this.points.length; i++) {
            radius.push(0.0);
        }
        return [this.points, radius];
    }

    toJSON() {
        return {
            name: this.name,
            points: this.points,
            color: this.color,
            type: 'Points',
            relax_groups: this.relax_groups,
        };
    }
}

export class SpheresObstacle extends ObstacleMesh {
    points: number[][] = [];
    radius: number[] = [];

    constructor(name: string, color: number[], points: number[][], radius: number[], relax_groups: string[], scene: THREE.Scene) {
        super(name, color, relax_groups);
        this.points = points;
        this.radius = radius;
        this.add_mesh(scene);
    }

    add_mesh(scene: THREE.Scene): void {
        for (let i = 0; i < this.points.length; i++) {
            const sphere = create_sphere(
                this.points[i][0], this.points[i][1], this.points[i][2], this.radius[i], 
                {
                    color: color_to_hex(this.color),
                    size: this.radius[i]
                });
            sphere.name = `${this.name}_${i}`;
            this.mesh_dict[sphere.name] = sphere;
        }
        super.add_mesh(scene);
    }

    sample_surface_points(resolution: number): [number[][], number[]] {
        const points: number[][] = [];
        for (let i = 0; i < this.points.length; i++) {
            const x = this.points[i][0];
            const y = this.points[i][1];
            const z = this.points[i][2];
            const radius = this.radius[i];
            const sub_resolution = this.compute_sub_resolution(resolution, radius);

            // Generate points on sphere surface using spherical coordinates
            const phi_steps = Math.ceil(2 * Math.PI * radius / sub_resolution);
            const theta_steps = Math.ceil(Math.PI * radius / sub_resolution);
            
            for (let phi_idx = 0; phi_idx < phi_steps; phi_idx++) {
                const phi = (phi_idx / phi_steps) * 2 * Math.PI;
                for (let theta_idx = 0; theta_idx < theta_steps; theta_idx++) {
                    const theta = (theta_idx / theta_steps) * Math.PI;
                    
                    // Convert spherical to Cartesian coordinates
                    const dx = radius * Math.sin(theta) * Math.cos(phi);
                    const dy = radius * Math.sin(theta) * Math.sin(phi);
                    const dz = radius * Math.cos(theta);
                    
                    points.push([x + dx, y + dy, z + dz]);
                }
            }
        }
        const radius = []
        for (let i = 0; i < points.length; i++) {
            radius.push(0.0);
        }
        return [points, radius];
    }

    compute_sub_resolution(resolution: number, radius: number): number {
        resolution =  Math.min(resolution * 0.85, radius / 2);
        return resolution;
    }

    toJSON() {
        return {
            name: this.name,
            points: this.points,
            radius: this.radius,
            color: this.color,
            type: 'Spheres',
            relax_groups: this.relax_groups,
        };
    }
}

export class SquaresObstacle extends ObstacleMesh {
    points: number[][] = [];
    size: number[] = [];

    constructor(name: string, color: number[], points: number[][], size: number[], relax_groups: string[], scene: THREE.Scene) {
        super(name, color, relax_groups);
        this.points = points;
        this.size = size;
        this.add_mesh(scene);
    }

    add_mesh(scene: THREE.Scene): void {
        for (let i = 0; i < this.points.length; i++) {
            const mesh = new THREE.Mesh(
                new THREE.BoxGeometry(
                  this.size[i], this.size[i], this.size[i]
                ),
                new THREE.MeshBasicMaterial({ color: color_to_hex(this.color) })
            );
            mesh.position.set(
                this.points[i][0], this.points[i][1], this.points[i][2]
            );
            mesh.name = `${this.name}_${i}`;
            this.mesh_dict[mesh.name] = mesh;
        }
        super.add_mesh(scene);
    }

    sample_surface_points(resolution: number): [number[][], number[]] {
        const points: number[][] = [];
        for (let i = 0; i < this.points.length; i++) {
            const size = this.size[i];
            const xSegments = Math.max(2, Math.ceil(size / resolution));
            const ySegments = Math.max(2, Math.ceil(size / resolution));
            const zSegments = Math.max(2, Math.ceil(size / resolution));

            // Sample points on each face of the box
            // Front and back faces (constant z)
            for (let x = 0; x <= xSegments; x++) {
                for (let y = 0; y <= ySegments; y++) {
                    const xPos = this.points[i][0] - size / 2 + (x / xSegments) * size;
                    const yPos = this.points[i][1] - size / 2 + (y / ySegments) * size;
                    points.push([xPos, yPos, this.points[i][2] - size / 2]); // Front face
                    points.push([xPos, yPos, this.points[i][2] + size / 2]); // Back face
                }
            }

            // Left and right faces (constant x)
            for (let y = 0; y <= ySegments; y++) {
                for (let z = 0; z <= zSegments; z++) {
                    const yPos = this.points[i][1] - size / 2 + (y / ySegments) * size;
                    const zPos = this.points[i][2] - size / 2 + (z / zSegments) * size;
                    points.push([this.points[i][0] - size / 2, yPos, zPos]); // Left face
                    points.push([this.points[i][0] + size / 2, yPos, zPos]); // Right face
                }
            }

            // Top and bottom faces (constant y)
            for (let x = 0; x <= xSegments; x++) {
                for (let z = 0; z <= zSegments; z++) {
                    const xPos = this.points[i][0] - size / 2 + (x / xSegments) * size;
                    const zPos = this.points[i][2] - size / 2 + (z / zSegments) * size;
                    points.push([xPos, this.points[i][1] - size / 2, zPos]); // Bottom face
                    points.push([xPos, this.points[i][1] + size / 2, zPos]); // Top face
                }
            }
        }
        const radius = []
        for (let i = 0; i < points.length; i++) {
            radius.push(0.0);
        }
        return [points, radius];
    }

    toJSON() {
        return {
            name: this.name,
            points: this.points,
            size: this.size,
            color: this.color,
            type: 'Squares',
            relax_groups: this.relax_groups,
        };
    }
}

export class SphereLineObstacle extends ObstacleMesh {
    points: number[][] = [];
    radius: number[] = [];

    constructor(name: string, color: number[], points: number[][], radius: number[], relax_groups: string[], scene: THREE.Scene) {
        super(name, color, relax_groups);
        this.points = points;
        this.radius = radius;
        this.add_mesh(scene);
    }

    add_mesh(scene: THREE.Scene): void {
        for (let i = 0; i < this.points.length; i++) {
            const sphere = create_sphere(
                this.points[i][0], this.points[i][1], this.points[i][2], this.radius[i], 
                {
                    color: color_to_hex(this.color),
                    size: this.radius[i],
                    output_mesh: false
                }
                // {
                //     color: color_to_hex(this.color),
                //     size: this.radius[i],
                //     output_mesh: true,
                //     transparent: true,
                //     opacity: 0.5
                // }
            );
            sphere.name = `${this.name}_${i}`;
            this.mesh_dict[sphere.name] = sphere;
        }
        super.add_mesh(scene);
    }

    sample_surface_points(resolution: number): [number[][], number[]] {
        return [this.points, this.radius];
    }

    discrete(
        scene: THREE.Scene, resolution: number, source: string,
        pipe_dict: { x: number, y: number, z: number, radius: number }[]
    ): void {
        if (this.discretize_mesh !== null) {
            remove_mesh(scene, this.discretize_mesh!.name);
            this.discretize_mesh!.geometry.dispose();
        }
        resolution = this.compute_resolution(resolution);
        [this.discretize_points, this.discretize_radius] = this.sample_surface_points(resolution);
        this.discretize_source = source;
    }

    toJSON() {
        return {
            name: this.name,
            points: this.points,
            radius: this.radius,
            color: this.color,
            type: 'SphereLine',
            relax_groups: this.relax_groups,
        };
    }
}