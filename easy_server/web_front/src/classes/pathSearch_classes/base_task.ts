import type { BlockRoot } from "./base_root";
import type { BaseBlock } from "./base_block";
import type { PathSearchTreeNode, PathSearchMainTask, 
  PathSearchPriorityTask, PathSearchGroupTask, PathSearchCellTask 
} from "../../types/data_type";

import { PipeBlock } from "./pipe_block";
import { EnvironmentBlock } from "./environment_block";
import { CylinderObstacleBlock, BoxObstacleBlock, SphereObstacleBlock, HttpObstacleBlock } from "./obstacle_block";
import { PathBlock } from "./path_block";

export class TaskManager {
    name: string;
  
    env_block: EnvironmentBlock|null = null;
    obstacle_blocks: Map<string, BaseBlock> = new Map();
    
    group_cfg: Map<string, Map<string, string | string[]>> = new Map();
    pipe_blocks: Map<string, PipeBlock> = new Map();

    main_tasks: Map<string, PathSearchMainTask> = new Map();
    path_blocks: Map<string, PathBlock> = new Map();
  
    constructor(name: string, xmin: number, xmax: number, ymin: number, ymax: number, zmin: number, zmax: number, root: BlockRoot) {
      this.name = name;
      this.env_block = new EnvironmentBlock(name, xmin, xmax, ymin, ymax, zmin, zmax, root);
    }

    // --- pipe group ---
    add_pipe_group(name: string, color: string=''): [boolean, string] {
      if (this.group_cfg.get(name)) {
        return [false, 'Pipe group already exists'];
      }
      const innerMap = new Map<string, string | string[]>();
      if (color === '') {
        innerMap.set('color', '#' + Math.floor(Math.random()*16777215).toString(16).padStart(6, '0'));
      }
      else {
        innerMap.set('color', color);
      }
      innerMap.set('pipes', []);
      this.group_cfg.set(name, innerMap);
      return [true, ''];
    }

    update_pipe_group(pipe_name: string, group_name: string) {
      (this.group_cfg.get(this.pipe_blocks.get(pipe_name)!.group_name)?.get('pipes')! as string[]).splice(
        (this.group_cfg.get(this.pipe_blocks.get(pipe_name)!.group_name)?.get('pipes') as string[]).indexOf(pipe_name), 1
      );
      (this.group_cfg.get(group_name)!.get('pipes') as string[]).push(pipe_name);
    }

    get_pipe_group(name: string): string[]|null {
      return this.group_cfg.get(name)?.get('pipes') as string[] | null;
    }

    get_group_names(): string[] {
      return Array.from(this.group_cfg.keys());
    }
  
    remove_pipe_group(group_name: string, root: BlockRoot) {
      (this.group_cfg.get(group_name)?.get('pipes') as string[]).forEach((pipe_name) => {
        this.remove_pipe(pipe_name, root);
      });
      this.group_cfg.delete(group_name);
    }
  
    // --- pipe ---
    add_pipe(
      name: string, 
      x: number, y: number, z: number, 
      vec_x: number, vec_y: number, vec_z: number,
      radius: number, group_name: string, is_inlet: boolean, root: BlockRoot
    ): [boolean, string] {
      if (this.group_cfg.get(group_name)?.get('pipes')?.includes(name)) {
        return [false, 'Pipe already exists in this group'];
      }
      const pipe_block = new PipeBlock(
        name, x, y, z, vec_x, vec_y, vec_z, radius, is_inlet, 
        this.group_cfg.get(group_name)!.get('color') as string, group_name, root
      );
      this.pipe_blocks.set(name, pipe_block);

      (this.group_cfg.get(group_name)!.get('pipes') as string[]).push(name);
      return [true, ''];
    }

    get_pipe(name: string): PipeBlock|null {
      return this.pipe_blocks.get(name) ?? null;
    }

    get_pipe_names(): string[] {
      return Array.from(this.pipe_blocks.keys());
    }
  
    remove_pipe(name: string, root: BlockRoot) {
      const pipe_block = this.pipe_blocks.get(name);
      if (!pipe_block) {
        return;
      }
  
      pipe_block.destory(root);
      this.pipe_blocks.delete(name);
      (this.group_cfg.get(pipe_block.group_name)?.get('pipes') as string[]).splice(
        (this.group_cfg.get(pipe_block.group_name)?.get('pipes') as string[]).indexOf(name), 1
      );
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

    // --- task ---
    add_main_task(name: string): [boolean, string] {
      if (this.main_tasks.get(name)) {
        return [false, 'Main task already exists'];
      }

      const main_task: PathSearchMainTask = {
        label: name,
        type: 'MainTask',
        priority_tasks: []
      }

      this.main_tasks.set(name, main_task);
      return [true, ''];
    }

    add_priority_task(name: string, main_task_name: string): [boolean, string] {
      const main_task = this.main_tasks.get(main_task_name);
      if (!main_task) {
        return [false, 'Main task not found'];
      }
      
      if (main_task.priority_tasks.find((task) => task.label === name)) {
        return [false, 'Priority task already exists'];
      }

      const priority_task: PathSearchPriorityTask = {
        label: name,
        type: 'PriorityTask',
        priority: main_task.priority_tasks.length,
        group_tasks: []
      };

      main_task.priority_tasks.push(priority_task);
      return [true, ''];
    }

    add_group_task(name: string, main_task_name: string, priority_task_name: string): [boolean, string] {
      const main_task = this.main_tasks.get(main_task_name);
      if (!main_task) {
        return [false, 'Main task not found'];
      }

      const priority_task = main_task.priority_tasks.find(task => task.label === priority_task_name);
      if (!priority_task) {
        return [false, 'Priority task not found'];
      }
      
      if (priority_task.group_tasks.find(task => task.label === name)) {
        return [false, 'Group task already exists'];
      }

      const group_task: PathSearchGroupTask = {
        label: name,
        type: 'GroupTask',
        group_name: name,
        cell_tasks: []
      };

      priority_task.group_tasks.push(group_task);
      return [true, ''];
    }
    
    add_cell_task(
      name: string, 
      main_task_name: string, 
      priority_task_name: string, 
      group_task_name: string,
      start_points: string[],
      end_points: string[],
      radius: number,
      run_cfg: Map<string, string|number|boolean>,
    ): [boolean, string] {
      const main_task = this.main_tasks.get(main_task_name);
      if (!main_task) {
        return [false, 'Main task not found'];
      }

      const priority_task = main_task.priority_tasks.find(task => task.label === priority_task_name);
      if (!priority_task) {
        return [false, 'Priority task not found'];
      }

      const group_task = priority_task.group_tasks.find(task => task.label === group_task_name);
      if (!group_task) {
        return [false, 'Group task not found'];
      }

      if (group_task.cell_tasks.find(task => task.label === name)) {
        return [false, 'Cell task already exists'];
      }
      
      const cell_task: PathSearchCellTask = {
        label: name,
        type: 'CellTask',
        start_points: start_points,
        end_points: end_points,
        radius: radius,
        run_cfg: run_cfg,
        result_name: 'None'
      };

      group_task.cell_tasks.push(cell_task);
      return [true, ''];
    }

    remove_cell_task(name: string, main_task_name: string, priority_task_name: string, group_task_name: string) {
      const main_task = this.main_tasks.get(main_task_name);
      if (!main_task) {
        return;
      }

      const priority_task = main_task.priority_tasks.find(task => task.label === priority_task_name);
      if (!priority_task) {
        return;
      }

      const group_task = priority_task.group_tasks.find(task => task.label === group_task_name);
      if (!group_task) {
        return;
      }

      group_task.cell_tasks = group_task.cell_tasks.filter(task => task.label !== name);
    }

    remove_group_task(name: string, main_task_name: string, priority_task_name: string) {
      const main_task = this.main_tasks.get(main_task_name);
      if (!main_task) {
        return;
      }

      const priority_task = main_task.priority_tasks.find(task => task.label === priority_task_name);
      if (!priority_task) {
        return;
      }
      priority_task.group_tasks = priority_task.group_tasks.filter(task => task.label !== name);
    }

    remove_priority_task(name: string, main_task_name: string) {
      const main_task = this.main_tasks.get(main_task_name);
      if (!main_task) {
        return;
      }
      main_task.priority_tasks = main_task.priority_tasks.filter(task => task.label !== name);
    }

    remove_main_task(name: string) {
      this.main_tasks.delete(name);
    }

    // --- export tree ---
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
  
      this.group_cfg.forEach((cfg, group_name) => {
        group_tree.children?.push({
          label: group_name,
          tagDict: new Map([['Task', this.name], ['PipeGroup', group_name]]),
          type: 'PipeGroup',
          children: (cfg.get('pipes') as string[]).map(pipe_name => {
            return {
              label: pipe_name,
              tagDict: new Map([['Task', this.name], ['PipeGroup', group_name], ['Pipe', pipe_name]]),
              type: 'Pipe'
            }
          })
        })
      });
      tree.children?.push(group_tree);
      
      const tasks_tree: PathSearchTreeNode = {
        id: 'TasksRoot',
        label: 'TasksRoot',
        type: 'TasksRoot',
        tagDict: new Map([
          ['Task', this.name],
        ]),
        children: []
      };

      this.main_tasks.forEach((main_task, name) => {
        tasks_tree.children?.push({
          label: name,
          type: 'MainTask',
          tagDict: new Map([
            ['Task', this.name], 
            ['MainTask', name]
          ]),
          children: main_task.priority_tasks.map((priority_task) => {
            return {
              label: priority_task.label,
              type: 'PriorityTask',
              tagDict: new Map([
                ['Task', this.name], 
                ['MainTask', name], 
                ['PriorityTask', priority_task.label]
              ]),
              children: priority_task.group_tasks.map((group_task) => {
                return {
                  label: group_task.label,
                  type: 'GroupTask',
                  tagDict: new Map([
                    ['Task', this.name], 
                    ['MainTask', name], 
                    ['PriorityTask', priority_task.label], 
                    ['GroupTask', group_task.label]
                  ]),
                  children: group_task.cell_tasks.map((cell_task) => {
                    return {
                      label: cell_task.label,
                      type: 'CellTask',
                      tagDict: new Map([
                        ['Task', this.name], 
                        ['MainTask', name], 
                        ['PriorityTask', priority_task.label], 
                        ['GroupTask', group_task.label], 
                        ['CellTask', cell_task.label]
                      ]),
                      children: [
                        {
                          label: 'Start Points',
                          type: 'StartPoints',
                          children: cell_task.start_points.map((point) => {
                            return {
                              label: point,
                              type: 'Point'
                            }
                          })
                        },
                        {
                          label: 'End Points',
                          type: 'EndPoints',
                          children: cell_task.end_points.map((point) => {
                            return {
                              label: point,
                              type: 'Point'
                            }
                          })
                        },
                        {
                          label: 'Radius',
                          type: 'Radius',
                          value: cell_task.radius
                        },
                        {
                          label: 'Run Config',
                          type: 'RunConfig',
                          children: Array.from(cell_task.run_cfg.entries()).map(([key, value]) => {
                            return {
                              label: key,
                              type: 'RunConfigItem',
                              value: value
                            }
                          })
                        },
                        {
                          label: 'Result',
                          type: 'Result',
                          value: cell_task.result_name
                        }
                      ]
                    }
                  })
                }
              })
            }
          })
        })
      });

      tree.children?.push(tasks_tree);

      return tree;
    }
    
    export_json(): any {
      const save_data = {
        name: this.name,
        env_block: this.env_block?.toJson(),
        obstacle_blocks: Array.from(this.obstacle_blocks.entries()).map(([name, block]) => {
          return block.toJson()
        }),
        pipe_blocks: Array.from(this.pipe_blocks.entries()).map(([name, block]) => {
          return block.toJson()
        }),
        main_tasks: Array.from(this.main_tasks.entries()).map(([name, task]) => {
          return {
            name: name,
            priority_tasks: task.priority_tasks.map((priority_task) => {
              return {
                name: priority_task.label,
                group_tasks: priority_task.group_tasks.map((group_task) => {
                  return {
                    name: group_task.label,
                    cell_tasks: group_task.cell_tasks.map((cell_task) => {
                      return {
                        name: cell_task.label,
                        start_points: cell_task.start_points,
                        end_points: cell_task.end_points,
                        radius: cell_task.radius,
                        run_cfg: cell_task.run_cfg,
                        result_name: cell_task.result_name
                      }
                    })
                  }
                })
              }
            })
          }
        }),
      }
      return save_data;
    }

    load_json(data: any, root: BlockRoot) {
      this.name = data.name;

      data.obstacle_blocks.forEach((obstacle_cfg: any) => {
        this.add_obstacle(obstacle_cfg.name, obstacle_cfg, root);
      });
      
      data.pipe_blocks.forEach((pipe_cfg: any) => {
        if (!this.group_cfg.has(pipe_cfg.group_name)) {
          this.add_pipe_group(pipe_cfg.group_name, pipe_cfg.color);
        }
      });

      data.pipe_blocks.forEach((pipe_cfg: any) => {
        console.log('radius:', pipe_cfg.radius)
        this.add_pipe(
          pipe_cfg.name, 
          pipe_cfg.x, pipe_cfg.y, pipe_cfg.z, 
          pipe_cfg.vector_x, pipe_cfg.vector_y, pipe_cfg.vector_z, 
          pipe_cfg.radius, pipe_cfg.group_name, pipe_cfg.is_inlet, 
          root
        );
      });

      data.main_tasks.forEach((task: any) => {
        this.add_main_task(task.name);

        task.priority_tasks.forEach((priority_task: any) => {
          this.add_priority_task(priority_task.name, task.name);

          priority_task.group_tasks.forEach((group_task: any) => {
            this.add_group_task(group_task.name, task.name, priority_task.name);

            group_task.cell_tasks.forEach((cell_task: any) => {
              this.add_cell_task(
                cell_task.name, task.name, priority_task.name, group_task.name, 
                cell_task.start_points, cell_task.end_points, cell_task.radius, cell_task.run_cfg
              );
            });
          });
        });
      });

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
      this.group_cfg.clear();
      this.main_tasks.clear();
    }
    
}