import type { Scene } from "three";
import { MeshObj } from "../mesh_type";
import * as THREE from "three";
import { color_to_hex, create_line_set } from "@/utils/visulize_utils";
import type { TreeNode } from '@/types/data_type';

export class SimulateNetworkMesh extends MeshObj {
    name: string;
    n_dim: number;
    n_elem: number;
    n_point: number;
    n_mark: number;
    cells: number[][];
    points: number[][];
    marker_dict: Map<string, Map<string, number | number[][]>>;

    constructor(
        name: string,
        points: number[][], 
        cells: number[][], 
        marker_dict: Map<string, Map<string, number | number[][]>>, 
        scene: Scene,
    ) {
        super(name);
        this.name = name;
        this.n_dim = 3;
        this.n_elem = cells.length;
        this.n_point = points.length;
        this.n_mark = 0;
        this.cells = cells;
        this.points = points;
        this.marker_dict = marker_dict;
        this.add_mesh(scene);
    }

    convert_indices_to_line_set(indices: number[][]) {
        const line_set = new Set<string>()
        indices.forEach((row) => {
            if (row.length == 2) {
                line_set.add(`${Math.min(row[0], row[1])},${Math.max(row[0], row[1])}`)
            }
            else if (row.length == 3) {
                line_set.add(`${Math.min(row[0], row[1])},${Math.max(row[0], row[1])}`)
                line_set.add(`${Math.min(row[1], row[2])},${Math.max(row[1], row[2])}`)
                line_set.add(`${Math.min(row[2], row[0])},${Math.max(row[2], row[0])}`)
            }
            else if (row.length == 4) {
                line_set.add(`${Math.min(row[0], row[1])},${Math.max(row[0], row[1])}`)
                line_set.add(`${Math.min(row[0], row[2])},${Math.max(row[0], row[2])}`)
                line_set.add(`${Math.min(row[0], row[3])},${Math.max(row[0], row[3])}`)
                line_set.add(`${Math.min(row[1], row[2])},${Math.max(row[1], row[2])}`)
                line_set.add(`${Math.min(row[1], row[3])},${Math.max(row[1], row[3])}`)
                line_set.add(`${Math.min(row[2], row[3])},${Math.max(row[2], row[3])}`)
            }
            else {
                throw new Error(`Unsupported number of points: ${row.length}`)
                return
            }
        })
        return Array.from(line_set).map(str => str.split(',').map(Number))
    }

    add_mesh(scene: Scene) {
        const internal_mesh = create_line_set(
            this.points,
            this.convert_indices_to_line_set(this.cells.map(cell => cell.slice(1)))
        )
        internal_mesh.name = this.name + "_internal"
        internal_mesh.visible = false
        this.mesh_dict[internal_mesh.name] = internal_mesh
        
        let color = [255, 0, 0]
        this.marker_dict.forEach((marker_dict, marker_name) => {
            if (marker_name === 'wall') {
                color = [0, 255, 0]
            }
            else {
                color = [255, 0, 0]
            }
            
            const boundary_mesh = create_line_set(
                this.points,
                this.convert_indices_to_line_set((marker_dict.get("marker_cell") as number[][]).map(cell => cell.slice(1))),
                {color: color_to_hex(color)}
            )
            boundary_mesh.name = this.name + "_" + marker_name
            this.mesh_dict[boundary_mesh.name] = boundary_mesh
        })

        super.add_mesh(scene);
    }

    export_tree(project_name: string, simulate_task_name: string): TreeNode {
        const root: TreeNode = {
            label: this.name,
            style: 'SimulateNetwork',
            data: {
                'Project': project_name,
                'SimulateTask': simulate_task_name,
                'SimulateNetwork': this.name,
            },
            children: Array.from(this.marker_dict.entries()).map(([marker_name, marker_dict]) => {
                return {
                    label: marker_name,
                    style: 'SimulateBoundary',
                    data: {
                        'Project': project_name,
                        'SimulateTask': simulate_task_name,
                        'SimulateNetwork': this.name,
                        'SimulateBoundary': marker_name,
                    },
                }
            }),
        };
        return root
    }

    hide_internal_mesh() {
        this.mesh_dict[this.name + "_internal"].visible = false
    }

    show_internal_mesh() {
        this.mesh_dict[this.name + "_internal"].visible = true
    }

    hide_boundary_mesh(marker_name: string) {
        this.mesh_dict[this.name + "_" + marker_name].visible = false
    }

    show_boundary_mesh(marker_name: string) {
        this.mesh_dict[this.name + "_" + marker_name].visible = true
    }

    get_boundary_list(): string[] {
        return Array.from(this.marker_dict.keys())
    }
}