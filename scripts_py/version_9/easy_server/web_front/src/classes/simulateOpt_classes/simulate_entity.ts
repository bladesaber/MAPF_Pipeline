import * as THREE from 'three'

import { read_mesh_json, add_mesh, remove_mesh } from "@/utils/visulize_utils";
import type { SimulateTreeNode, SimulateCondition, SimulateInletCfg, SimulateOutletCfg, SimulateWallCfg } from '@/types/data_type.ts'
import { pass } from 'three/tsl';
import { create_line_set, create_arrow_set } from '@/utils/visulize_utils';

export class CadEntity {
    name: string;
    mesh: THREE.Mesh | null = null;

    constructor(name: string, mesh_json: any, rootObj: SimulateRootObj) {
        this.name = name;
        this.add_mesh(mesh_json, rootObj)
    }

    add_mesh(mesh_json: any, rootObj: SimulateRootObj) {
        if (!mesh_json) return
        this.mesh = read_mesh_json(mesh_json['shapes'][0])
        this.mesh.name = this.name
        add_mesh(rootObj.get_scene(), this.mesh)
    }

    remove_mesh(rootObj: SimulateRootObj) {
        if (!this.mesh) return
        remove_mesh(rootObj.get_scene(), this.mesh.name)
    }

    destory(rootObj: SimulateRootObj) {
        this.remove_mesh(rootObj)
        if (this.mesh) {
            this.mesh.geometry.dispose();
            this.mesh = null;
        }
    }

    hide_mesh() {
        if (this.mesh) {
          this.mesh.visible = false;
        }
      }
    
    show_mesh() {
        if (this.mesh) {
          this.mesh.visible = true;
        }
    }
}

export class SimulateNetwork{
    name: string;
    n_dim: number;
    n_elem: number;
    n_point: number;
    n_mark: number;
    cells: number[][];
    points: number[][];
    marker_dict: Map<string, Map<string, number|number[][]>>;

    mesh_dict: Map<string, THREE.LineSegments> = new Map();

    constructor(name: string, points: number[][], cells: number[][], marker_dict: Map<string, Map<string, number|number[][]>>, rootObj: SimulateRootObj) {
        this.name = name;
        this.n_dim = 3;
        this.n_elem = cells.length;
        this.n_point = points.length;
        this.n_mark = 0;
        this.cells = cells;
        this.points = points;
        this.marker_dict = marker_dict;
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

    get_random_color () {
        // 随机生成一个亮色
        const r = Math.floor(Math.random() * 128 + 128)
        const g = Math.floor(Math.random() * 128 + 128)
        const b = Math.floor(Math.random() * 128 + 128)
        return (r << 16) | (g << 8) | b // 转换为 0xRRGGBB 格式
    }

    add_mesh(rootObj: SimulateRootObj) {
        this.mesh_dict.clear();
        const internal_mesh = create_line_set(
            this.points,
            this.convert_indices_to_line_set(this.cells.map(cell => cell.slice(1)))
        )
        internal_mesh.name = "internal"
        this.mesh_dict.set("internal", internal_mesh)

        this.marker_dict.forEach((marker_dict, marker_name) => {
            const boundary_mesh = create_line_set(
                this.points,
                this.convert_indices_to_line_set((marker_dict.get("marker_cell") as number[][]).map(cell => cell.slice(1))),
                {color: this.get_random_color()}
            )
            boundary_mesh.name = marker_name
            this.mesh_dict.set(marker_name, boundary_mesh)
        })

        this.mesh_dict.forEach((mesh, mesh_name) => {
            add_mesh(rootObj.get_scene(), mesh);
        })
    }

    remove_mesh(rootObj: SimulateRootObj) {
        this.mesh_dict.forEach((mesh, mesh_name) => {
            remove_mesh(rootObj.get_scene(), mesh_name)
        })
    }

    destory(rootObj: SimulateRootObj) {
        this.remove_mesh(rootObj)
        this.mesh_dict.forEach((mesh, mesh_name) => {
            mesh.geometry.dispose();
        })
        this.mesh_dict.clear();
    }

    hide_mesh() {
        this.mesh_dict.forEach((mesh, mesh_name) => {
            mesh.visible = false;
        })
    }

    show_mesh() {
        this.mesh_dict.forEach((mesh, mesh_name) => {
            mesh.visible = true;
        })
    }
}

export class SimulateResult{
    name: string;
    velocity: number[][];
    pressure: number[];
    mesh: THREE.LineSegments | null = null;

    constructor(name: string, velocity: number[][], pressure: number[]) {
        this.name = name;
        this.velocity = velocity;
        this.pressure = pressure;
    }

    add_mesh(points: number[][], arrow_length: number, rootObj: SimulateRootObj) {
        this.mesh = create_arrow_set(
            points,
            this.velocity,
            arrow_length,
            this.pressure
        )
        this.mesh.name = this.name
        add_mesh(rootObj.get_scene(), this.mesh)
    }

    remove_mesh(rootObj: SimulateRootObj) {
        if (!this.mesh) return
        remove_mesh(rootObj.get_scene(), this.mesh.name)
    }

    destory(rootObj: SimulateRootObj) { 
        this.remove_mesh(rootObj)
        if (this.mesh) {
            this.mesh.geometry.dispose();
            this.mesh = null;
        }
    }

    hide_mesh() {
        if (this.mesh) {
          this.mesh.visible = false;
        }
    }

    show_mesh() {
        if (this.mesh) {
          this.mesh.visible = true;
        }
    }
}

export class SimulateEntity{
    name: string;
    entity_map: Map<string, CadEntity> = new Map();
    entity_groups: Map<string, string[]> = new Map();

    simulate_conditions: Map<string, SimulateCondition> = new Map();
    simulate_network: SimulateNetwork| null = null;
    simulate_results: Map<string, SimulateResult> = new Map();

    constructor(name: string) {
        this.name = name;
    }

    add_entity(entity: CadEntity) {
        this.entity_map.set(entity.name, entity);
    }

    remove_entity(entity: CadEntity, rootObj: SimulateRootObj) {
        this.entity_groups.forEach((group) => {
            group.includes(entity.name) && group.splice(group.indexOf(entity.name), 1);
        });
        this.entity_map.delete(entity.name);
        entity.destory(rootObj);
    }

    define_entity_group(group_name: string, entity_names: string[]) {
        entity_names.forEach((entity_name) => {
            this.entity_groups.forEach((group) => {
                if (group.includes(entity_name)) {
                    throw new Error(`Entity ${entity_name} already exists in group ${group_name}`);
                    return
                }
            })
        })
        this.entity_groups.set(group_name, entity_names);
    }

    remove_entity_group(group_name: string) {
        this.entity_groups.delete(group_name);
    }

    get_entity_group_names(): string[] {
        return Array.from(this.entity_groups.keys());
    }

    add_simulate_network(
        points: number[][], cells: number[][], marker_dict: Map<string, Map<string, number|number[][]>>, 
        rootObj: SimulateRootObj
    ) {
        if (this.simulate_network) {
            this.remove_simulate_network(rootObj);
        }
        this.simulate_network = new SimulateNetwork(this.name, points, cells, marker_dict, rootObj);
        this.simulate_network.add_mesh(rootObj);
    }

    remove_simulate_network(rootObj: SimulateRootObj) {
        if (this.simulate_network) {
            this.simulate_network.destory(rootObj);
            this.simulate_network = null;
        }
    }

    has_simulate_condition(name: string) {
        return this.simulate_conditions.has(name);
    }

    add_simulate_condition(condition: SimulateCondition) {
        this.simulate_conditions.set(condition.name, condition);
    }

    remove_simulate_condition(name: string) {
        this.simulate_conditions.delete(name);
    }

    get_simulate_condition(name: string) {
        return this.simulate_conditions.get(name);
    }

    add_simulate_result(condition_name: string, velocity: number[][], pressure: number[], rootObj: SimulateRootObj) {
        if (this.simulate_results.has(condition_name)) {
            this.remove_simulate_result(condition_name, rootObj);
        }
        if (!this.simulate_network) {
            throw new Error('Simulate network not found');
            return
        }
        this.simulate_results.set(condition_name, new SimulateResult(condition_name, velocity, pressure));
        this.simulate_results.get(condition_name)?.add_mesh(this.simulate_network.points, 0.1, rootObj);
    }

    remove_simulate_result(condition_name: string, rootObj: SimulateRootObj) {
        if (this.simulate_results.has(condition_name)) {
            this.simulate_results.get(condition_name)?.destory(rootObj);
            this.simulate_results.delete(condition_name);
        }
    }

    has_simulate_result(condition_name: string) {
        return this.simulate_results.has(condition_name);
    }

    get_simulate_result(condition_name: string) {
        return this.simulate_results.get(condition_name);
    }

    has_entity(name: string) {
        return this.entity_map.has(name);
    }

    get_entity(name: string) {
        return this.entity_map.get(name);
    }

    destory(rootObj: SimulateRootObj) {
        this.entity_map.forEach((entity) => {
            entity.destory(rootObj);
        });
        this.entity_map.clear();
        this.simulate_network?.destory(rootObj);
        this.simulate_network = null;
    }

    export_tree_data(): SimulateTreeNode {
        const entity_tree: SimulateTreeNode = {
            label: 'Entities',
            children: Array.from(this.entity_map.entries()).map(([entity_name, entity]) => {
                return {
                    label: entity_name, 
                    tag_map: new Map([
                        ['SimulateEntity', this.name]
                    ]),
                    type: 'Entity'
                }
            }),
            tagDict: new Map([['SimulateEntity', this.name]]),
            type: 'Node_Entities'
        }

        const group_tree: SimulateTreeNode = {
            label: 'Groups',
            children: [],
            tagDict: new Map([
                ['SimulateEntity', this.name]
            ]),
            type: 'Node_Groups'
        }
        this.entity_groups.forEach((group_entities, group_name) => {
            group_tree.children?.push({
                label: group_name,
                children: group_entities.map((entity_name) => {
                    return {
                        label: entity_name, 
                        tagDict: new Map([
                            ['SimulateEntity', this.name]
                        ]),
                        type: 'Entity'
                    }
                }),
                tagDict: new Map([
                    ['SimulateEntity', this.name]
                ]),
                type: 'Group'
            })
        })

        const condition_tree: SimulateTreeNode = {
            label: 'Conditions',
            children: [],
            tagDict: new Map([
                ['SimulateEntity', this.name]
            ]),
            type: 'Node_Conditions'
        }
        this.simulate_conditions.forEach((condition, condition_name) => {
            condition_tree.children?.push({
                label: condition_name,
                children: [
                    {
                        label: 'physical_cfg', 
                        tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'PhysicalCfg'
                    },
                    {
                        label: 'material_cfg', 
                        tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'MaterialCfg'
                    },
                    {
                        label: 'run_cfg', 
                        tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'RunCfg'
                    },
                    {
                        label: 'initialization_cfg', 
                        tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'InitializationCfg'
                    },
                    {
                        label: 'inlet_cfg', 
                        children: Array.from(condition.inlet_cfg.entries()).map(([inlet_name, inlet_cfg]) => {
                            return {
                                label: inlet_name, 
                                tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                                type: 'InletCfg'
                            }
                        }),
                        tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'InletCfgs'
                    },
                    {
                        label: 'outlet_cfg',
                        children: Array.from(condition.outlet_cfg.entries()).map(([outlet_name, outlet_cfg]) => {
                            return {
                                label: outlet_name, 
                                tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                                type: 'OutletCfg'
                            }
                        }),
                        tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'OutletCfgs'
                    },
                    {
                        label: 'wall_cfg',
                        children: Array.from(condition.wall_cfg.entries()).map(([wall_name, wall_cfg]) => {
                            return {
                                label: wall_name, 
                                tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                                type: 'WallCfg'
                            }
                        }),
                        tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'WallCfgs'
                    }
                ],
                tagDict: new Map([['SimulateEntity', this.name]]),
                type: 'Condition'
            })

            if (this.has_simulate_result(condition_name)) {
                condition_tree.children?.push({
                    label: 'result',
                    tagDict: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                    type: 'SimulateResult'
                })
            }
        })

        const tree_data: SimulateTreeNode = {
            label: this.name,
            children: [
                entity_tree,
                group_tree,
                condition_tree,
            ],
            type: 'Node'
        }

        if (this.simulate_network) {
            const network_tree: SimulateTreeNode = {
                label: 'Network',
                children: [],
                tagDict: new Map([['SimulateEntity', this.name]]),
                type: 'Node_Network'
            }
            this.simulate_network.mesh_dict.forEach((mesh, mesh_name) => {
                network_tree.children?.push({
                    label: mesh_name,
                    tagDict: new Map([
                        ['SimulateEntity', this.name]
                    ]),
                    type: 'CellNetwork'
                })
            })
            tree_data.children?.push(network_tree)
        }

        return tree_data;
    }

    get_storage_content(): Uint8Array | null {
        const stored = sessionStorage.getItem(this.name);
        if (!stored) return null;
        
        const parsedData = JSON.parse(stored);
        if (!parsedData.content) return null;

        // const decoder = new TextDecoder()
        // const text = decoder.decode(uint8Array)

        return new Uint8Array(parsedData.content);
    }

    hide_group(group_name: string) {
        this.entity_groups.get(group_name)?.forEach((entity_name) => {
            this.entity_map.get(entity_name)?.hide_mesh();
        });
    }

    show_group(group_name: string) {
        this.entity_groups.get(group_name)?.forEach((entity_name) => {
            this.entity_map.get(entity_name)?.show_mesh();
        });
    }

    get_rest_items(): string[] {
        const rest_items: string[] = []
        this.entity_map.forEach((entity) => {
            let included = false
            this.entity_groups.forEach((group) => {
                if (group.includes(entity.name)) {
                    included = true
                    return
                }
            })
            if (!included) {
                rest_items.push(entity.name)
            }
        })
        return rest_items
    }

    hide() {
        this.entity_map.forEach((entity) => {
            entity.hide_mesh();
        });
    }

    show() {
        this.entity_map.forEach((entity) => {
            entity.show_mesh();
        });
    }

    hide_network() {
        if (this.simulate_network) {
            this.simulate_network.hide_mesh();
        }
    }

    show_network() {
        if (this.simulate_network) {
            this.simulate_network.show_mesh();
        }
    }

    hide_result(condition_name: string) {
        if (this.simulate_results.has(condition_name)) {
            this.simulate_results.get(condition_name)?.hide_mesh();
        }
    }

    show_result(condition_name: string) {
        if (this.simulate_results.has(condition_name)) {
            this.simulate_results.get(condition_name)?.show_mesh();
        }
    }
}

export class SimulateRootObj {
    private scene: THREE.Scene;
    model_map: Map<string, SimulateEntity> = new Map();

    constructor(scene: THREE.Scene) {
        this.scene = scene;
    }

    add_simulate_entity(entity: SimulateEntity) {
        this.model_map.set(entity.name, entity);
    }

    remove_simulate_entity(entity: SimulateEntity, rootObj: SimulateRootObj) {
        this.model_map.delete(entity.name);
        entity.destory(rootObj);
    }

    get_simulate_entity(name: string) {
        return this.model_map.get(name);
    }

    get_scene() {
        return this.scene;
    }
}