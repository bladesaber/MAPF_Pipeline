import * as THREE from 'three';

import {
  add_mesh,
  remove_mesh,
} from '../utils/visulize_utils.ts';

export class MeshObj {
    name: string;
    mesh_dict: { [key: string]: THREE.Mesh | THREE.LineSegments | THREE.Points } = {};

    constructor(name: string) {
        this.name = name;
    }

    add_mesh(scene: THREE.Scene) {
        for (const key in this.mesh_dict) {
            add_mesh(scene, this.mesh_dict[key]);
        }
    }

    remove_mesh(scene: THREE.Scene) {
        for (const key in this.mesh_dict) {
            remove_mesh(scene, key);
        }
    }

    destory(scene: THREE.Scene) {
        this.remove_mesh(scene);
        for (const key in this.mesh_dict) {
            this.mesh_dict[key].geometry.dispose();
        }
        this.mesh_dict = {};
    }

    update_mesh(scene: THREE.Scene) {
        this.destory(scene);
        this.add_mesh(scene);
    }

    hide_mesh() {
        for (const key in this.mesh_dict) {
            this.mesh_dict[key].visible = false;
        }
    }

    show_mesh() {
        for (const key in this.mesh_dict) {
            this.mesh_dict[key].visible = true;
        }
    }

    toJSON() {
        return { name: this.name };
    }
}
