import type { BlockRoot } from "./base_root.ts";
import * as THREE from "three";

import { BaseBlock } from "./base_block.ts";
import { create_sphere } from "@/utils/visulize_utils.ts";

export class IntermediatePointBlock extends BaseBlock {
    x: number;
    y: number;
    z: number;
    properties: string[] = ["x", "y", "z"];

    constructor(name: string, x: number, y: number, z: number, root: BlockRoot) {
        super(name);
        this.x = x;
        this.y = y;
        this.z = z;
        this.add_mesh(root);
    }

    add_mesh(root: BlockRoot): void {
        const sphere = create_sphere(this.x, this.y, this.z, 0.5, {color: 0xff00ff});
        sphere.name = this.name;
        this.mesh_objs.set(sphere.name, sphere);
        super.add_mesh(root);
    }

    get_property(property: string): number {
        if (property === "x") {
            return this.x;
        }
        else if (property === "y") {
            return this.y;
        }
        else if (property === "z") {
            return this.z;
        }
        else {
            throw new Error(`Invalid property: ${property}`);
        }
    }

    set_properties(property: string, value: number) {
        throw new Error("IntermediatePointBlock does not support set_properties");
    }

    toJson(): any {
        return {
            name: this.name,
            x: this.x,
            y: this.y,
            z: this.z,
        };
    }
}