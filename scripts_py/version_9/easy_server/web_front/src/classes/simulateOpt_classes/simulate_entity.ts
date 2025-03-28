import * as THREE from 'three'

import { read_mesh_json, add_mesh, remove_mesh } from "@/utils/visulize_utils";
import type { SimulateTreeNode, SimulateCondition, SimulateInletCfg, SimulateOutletCfg, SimulateWallCfg } from '@/types/data_type.ts'
import { pass } from 'three/tsl';

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

export class SimulateMesh{
    name: string;
    mesh: THREE.Mesh | null = null;

    constructor(name: string, rootObj: SimulateRootObj) {
        this.name = name;
    }
}

export class SimulateEntity{
    name: string;
    storage_key: string|null = null;
    entity_map: Map<string, CadEntity> = new Map();

    entity_groups: Map<string, string[]> = new Map();
    simulate_conditions: Map<string, SimulateCondition> = new Map();
    simulate_meshes: SimulateMesh| null = null;

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
            tag_map: new Map([
                ['SimulateEntity', this.name]
            ]),
            type: 'Node_Entities'
        }

        const group_tree: SimulateTreeNode = {
            label: 'Groups',
            children: [],
            tag_map: new Map([
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
                        tag_map: new Map([
                            ['SimulateEntity', this.name]
                        ]),
                        type: 'Entity'
                    }
                }),
                tag_map: new Map([
                    ['SimulateEntity', this.name]
                ]),
                type: 'Group'
            })
        })

        const condition_tree: SimulateTreeNode = {
            label: 'Conditions',
            children: [],
            tag_map: new Map([
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
                        tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'PhysicalCfg'
                    },
                    {
                        label: 'material_cfg', 
                        tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'MaterialCfg'
                    },
                    {
                        label: 'run_cfg', 
                        tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'RunCfg'
                    },
                    {
                        label: 'initialization_cfg', 
                        tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'InitializationCfg'
                    },
                    {
                        label: 'inlet_cfg', 
                        children: Array.from(condition.inlet_cfg.entries()).map(([inlet_name, inlet_cfg]) => {
                            return {
                                label: inlet_name, 
                                tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                                type: 'InletCfg'
                            }
                        }),
                        tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'InletCfgs'
                    },
                    {
                        label: 'outlet_cfg',
                        children: Array.from(condition.outlet_cfg.entries()).map(([outlet_name, outlet_cfg]) => {
                            return {
                                label: outlet_name, 
                                tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                                type: 'OutletCfg'
                            }
                        }),
                        tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'OutletCfgs'
                    },
                    {
                        label: 'wall_cfg',
                        children: Array.from(condition.wall_cfg.entries()).map(([wall_name, wall_cfg]) => {
                            return {
                                label: wall_name, 
                                tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                                type: 'WallCfg'
                            }
                        }),
                        tag_map: new Map([['SimulateEntity', this.name], ['SimulateCondition', condition_name]]),
                        type: 'WallCfgs'
                    },
                ],
                tag_map: new Map([['SimulateEntity', this.name]]),
                type: 'Condition'
            })
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

        return tree_data;
    }

    hide_mesh() {
        this.entity_map.forEach((entity) => {
            entity.hide_mesh();
        });
    }

    show_mesh() {
        this.entity_map.forEach((entity) => {
            entity.show_mesh();
        });
    }
    
    test() {
        console.log('storage_content:', this.get_storage_content())
    }

    get_storage_content() {
        const stored = sessionStorage.getItem(this.name);
        if (!stored) return null;
        return JSON.parse(stored).content;
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