import type { BlockRoot } from "./base_root";
import type { BaseBlock } from "./base_block";
import type { PathSearchTreeNode } from "../../types/data_type";

import { PipeBlock } from "./pipe_block";
import { EnvironmentBlock } from "./environment_block";
import { CylinderObstacleBlock, BoxObstacleBlock, SphereObstacleBlock, HttpObstacleBlock } from "./obstacle_block";
import { color } from "three/tsl";

export class TaskManager {
    name: string;
  
    env_block: EnvironmentBlock|null = null;
    obstacle_blocks: Map<string, BaseBlock> = new Map();
    group_pipes: Map<string, string[]> = new Map();
    pipe_blocks: Map<string, PipeBlock> = new Map();
    group_colors: Map<string, string> = new Map();
  
    constructor(name: string, xmin: number, xmax: number, ymin: number, ymax: number, zmin: number, zmax: number, root: BlockRoot) {
      this.name = name;
      this.env_block = new EnvironmentBlock(name, xmin, xmax, ymin, ymax, zmin, zmax, root);
    }
  
    add_pipe(
      name: string, 
      x: number, y: number, z: number, 
      vec_x: number, vec_y: number, vec_z: number,
      radius: number, group_name: string, is_inlet: boolean, root: BlockRoot
    ): [boolean, string] {
      if (this.group_pipes.get(group_name)?.includes(name)) {
        return [false, 'Pipe already exists in this group'];
      }

      const pipe_block = new PipeBlock(
        name, x, y, z, vec_x, vec_y, vec_z, radius, is_inlet, 
        this.group_colors.get(group_name)!, root
      );
      this.pipe_blocks.set(name, pipe_block);
      this.group_pipes.get(group_name)?.push(name);
      return [true, ''];
    }

    get_pipe(name: string): PipeBlock|null {
      return this.pipe_blocks.get(name) ?? null;
    }
  
    remove_pipe(name: string, root: BlockRoot) {
      const pipe_block = this.pipe_blocks.get(name);
      if (!pipe_block) {
        return;
      }
  
      pipe_block.destory(root);
      this.pipe_blocks.delete(name);
      this.group_pipes.forEach((pipe_names, group_name) => {
        if (pipe_names.includes(name)) {
          pipe_names.splice(pipe_names.indexOf(name), 1);
        }
      });
    }
  
    add_pipe_group(name: string): [boolean, string] {
      if (this.group_pipes.get(name)) {
        return [false, 'Pipe group already exists'];
      }
      this.group_pipes.set(name, []);
      this.group_colors.set(name, '#' + Math.floor(Math.random()*16777215).toString(16).padStart(6, '0'));
      return [true, ''];
    }
  
    update_pipe_group(pipe_name: string, group_name: string) {
      this.group_pipes.forEach((names, group_name) => {
        if (names.includes(pipe_name)) {
          names.splice(names.indexOf(pipe_name), 1);
        }
      });
      if (!this.group_pipes.has(group_name)) {
        this.group_pipes.set(group_name, []);
      }
      this.group_pipes.get(group_name)?.push(pipe_name);
    }

    get_pipe_group(name: string): string[]|null {
      return this.group_pipes.get(name) ?? null;
    }

    get_group_names(): string[] {
      return Array.from(this.group_pipes.keys());
    }
  
    remove_pipe_group(group_name: string, root: BlockRoot) {
      this.group_pipes.forEach((names, group_name) => {
        this.remove_pipe(names[0], root);
      });
      this.group_pipes.delete(group_name);
      this.group_colors.delete(group_name);
    }
    
    add_obstacle(name: string, properties: any, root: BlockRoot): [boolean, string] {
      if (this.obstacle_blocks.get(name)) {
        return [false, 'Obstacle already exists'];
      }

      let obstacle_block: BaseBlock;
      if (properties.type === 'CylinderObstacle') {
        obstacle_block = new CylinderObstacleBlock(
          name, properties.x, properties.y, properties.z, 
          properties.vector_x, properties.vector_y, properties.vector_z, 
          properties.radius, properties.height, properties.relax_groups, root
        );
      }
      else if (properties.type === 'BoxObstacle') {
        console.log('Box properties:', properties)

        obstacle_block = new BoxObstacleBlock(
          name, properties.xmin, properties.xmax, properties.ymin, 
          properties.ymax, properties.zmin, properties.zmax, 
          properties.relax_groups, root
        );
      }
      else if (properties.type === 'SphereObstacle') {
        obstacle_block = new SphereObstacleBlock(
          name, properties.x, properties.y, properties.z, properties.radius, 
          properties.relax_groups, root
        );
      }
      else if (properties.type === 'HttpObstacle') {
        obstacle_block = new HttpObstacleBlock(name, properties.mesh_json, properties.relax_groups, root);
      }
      else {
        return [false, 'Invalid obstacle type'];
      }

      this.obstacle_blocks.set(name, obstacle_block);
      return [true, ''];
    }

    get_obstacle(name: string): BaseBlock|null {
      return this.obstacle_blocks.get(name) ?? null;
    }
  
    remove_obstacle(name: string, root: BlockRoot) {
      const obstacle_block = this.obstacle_blocks.get(name);
      if (!obstacle_block) {
        return;
      }
      obstacle_block.destory(root);
      this.obstacle_blocks.delete(name);
    }
  
    export_tree(): PathSearchTreeNode {
      const tree: PathSearchTreeNode = {
        id: this.name,
        label: this.name,
        type: 'TaskManager',
        children: [
          {
            id: 'Environment',
            label: 'Environment',
            type: 'TaskEnvironment',
            tagDict: new Map([
              ['Task', this.name],
            ]),
          },
          {
            id: 'Obstacle',
            label: 'Obstacle',
            type: 'TaskObstacle',
            tagDict: new Map([
              ['Task', this.name],
            ]),
            children: Array.from(this.obstacle_blocks.entries()).map(([name, block]) => {
              return {
                  label: name, 
                  tagDict: new Map([['Task', this.name], ['Obstacle', name]]),
                  type: 'Obstacle'
              }
            })
          }
        ],
      };
  
      const group_tree: PathSearchTreeNode = {
        id: 'PipeGroup',
        label: 'PipeGroup',
        type: 'TaskPipeGroup',
        tagDict: new Map([
          ['Task', this.name],
        ]),
        children: []
      };
  
      this.group_pipes.forEach((pipe_names, group_name) => {
        group_tree.children?.push({
          label: group_name,
          tagDict: new Map([['Task', this.name], ['PipeGroup', group_name]]),
          type: 'PipeGroup',
          children: pipe_names.map(pipe_name => {
            return {
              label: pipe_name,
              tagDict: new Map([['Task', this.name], ['PipeGroup', group_name], ['Pipe', pipe_name]]),
              type: 'Pipe'
            }
          })
        })
      });
  
      tree.children?.push(group_tree);
      return tree;
    }
  
    destory(root: BlockRoot) {
      this.env_block?.destory(root);
      this.obstacle_blocks.forEach((block, name) => {
        block.destory(root);
      });
      this.pipe_blocks.forEach((block, name) => {
        block.destory(root);
      });
      this.obstacle_blocks.clear();
      this.pipe_blocks.clear();
      this.group_pipes.clear();
      this.group_colors.clear();
    }
    
}