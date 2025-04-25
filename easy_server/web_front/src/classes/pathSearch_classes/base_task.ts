import type { BlockRoot } from "./base_root";
import type { BaseBlock } from "./base_block";
import type { PathSearchTreeNode } from "../../types/data_type";

import { PipeBlock } from "./pipe_block";
import { EnvironmentBlock } from "./environment_block";
import { CylinderObstacleBlock, BoxObstacleBlock } from "./obstacle_block";
import { DiscreteGridBlock, PathBlock } from "./search_block";
import type { ObstacleBlock } from "./obstacle_block";

export class CellTask {
  name: string;
  type: string = 'CellTask';
  start_point: string;
  end_point: string;
  radius: number;
  run_config: Map<string, string|number|boolean> = new Map();
  result: PathBlock| null = null;
  has_result: boolean = false;
  discretize_radius: number;
  resolution: number;

  constructor(
    name: string, start_point: string, end_point: string, 
    radius: number, run_config: Map<string, string|number|boolean> = new Map(), resolution: number
  ) {
    this.name = name;
    this.start_point = start_point;
    this.end_point = end_point;
    this.radius = radius;
    this.discretize_radius = radius / resolution;
    this.run_config = run_config;
    this.resolution = resolution;
  }

  set_result(result: number[][], color: string, root: BlockRoot) {
    if (!this.result) {
      this.result = new PathBlock(this.name, result, this.discretize_radius, color, root);
    }
    else {
      this.result.points = result;
      this.result.update_mesh(root);
    }
    this.has_result = true;
  }

  clear_result(root: BlockRoot) {
    if (this.result) {
      this.result.destory(root);
    }
    this.result = null;
    this.has_result = false;
  }

  hide_result() {
    if (this.result) {
      this.result.hide_mesh();
    }
  }

  show_result() {
    if (this.result) {
      this.result.show_mesh();
    }
  }

  toJson(): any {
    let points = null;
    if (this.result) {
      points = this.result.points.map((point: number[]) => {
        return point.map((value: number) => value * this.resolution);
      });
    }

    return {
      name: this.name,
      start_point: this.start_point,
      end_point: this.end_point,
      radius: this.radius,
      result: {
        points: points,
        radius: this.radius,
      },
    };
  }
}

export class GroupTask {
  name: string;
  type: string = 'GroupTask';
  cell_tasks: CellTask[] = [];

  constructor(name: string) {
    this.name = name;
  }

  add_cell_task(cell_task: CellTask) {
    this.cell_tasks.push(cell_task);
  }

  get_cell_task(priority: number): CellTask|null {
    return this.cell_tasks[priority] ?? null;
  }

  clear_result(root: BlockRoot) {
    this.cell_tasks.forEach((cell_task) => {
      cell_task.clear_result(root);
    });
  }
}

export class PriorityTask {
  name: string;
  type: string = 'PriorityTask';
  group_tasks: Map<string, GroupTask> = new Map();

  constructor(name: string) {
    this.name = name;
  }

  add_group_task(group_task: GroupTask) {
    this.group_tasks.set(group_task.name, group_task);
  }

  get_group_task(group_name: string): GroupTask|null {
    return this.group_tasks.get(group_name) ?? null;
  }
  
  clear_result(root: BlockRoot) {
    this.group_tasks.forEach((group_task) => {
      group_task.clear_result(root);
    });
  }
}

export class MainTask {
  name: string;
  type: string = 'MainTask';
  priority_tasks: PriorityTask[] = [];
  has_preprocess: boolean = false;

  constructor(name: string) {
    this.name = name;
  }

  add_priority_task(priority_task: PriorityTask) {
    this.priority_tasks.push(priority_task);
  }

  get_priority_task(priority: number): PriorityTask|null {
    return this.priority_tasks[priority] ?? null;
  }

  set_has_preprocess() {
    this.has_preprocess = true;
  }

  clear_has_preprocess() {
    this.has_preprocess = false;
  }
}

interface GroupCfg {
  name: string;
  color: string;
  pipes: string[];
}

export class TaskManager {
    name: string;
  
    env_block: EnvironmentBlock|null = null;
    obstacle_blocks: Map<string, BaseBlock> = new Map();
    
    group_cfg: Map<string, GroupCfg> = new Map();
    pipe_blocks: Map<string, PipeBlock> = new Map();

    main_tasks: Map<string, MainTask> = new Map();

    discretized_grid: DiscreteGridBlock|null = null;
  
    constructor(name: string, xlength: number, ylength: number, zlength: number, resolution: number, root: BlockRoot) {
      this.name = name;
      this.env_block = new EnvironmentBlock(name, xlength, ylength, zlength, resolution, root);
    }

    get_xsize(): number {
      return this.env_block!.xsize;
    }

    get_ysize(): number {
      return this.env_block!.ysize;
    }

    get_zsize(): number {
      return this.env_block!.zsize;
    }

    get_resolution(): number {
      return this.env_block!.resolution;
    }
    
    get_environment(): EnvironmentBlock|null {
      return this.env_block;
    }

    // ------ obstacle ------
    add_obstacle(name: string, properties: any, root: BlockRoot): [boolean, string] {
      if (this.obstacle_blocks.get(name)) {
        return [false, 'Obstacle already exists'];
      }

      const grid_resolution = this.env_block!.resolution;
      let obstacle_block: BaseBlock;
      if (properties.type === 'CylinderObstacle') {
        obstacle_block = new CylinderObstacleBlock(
          name, properties.x, properties.y, properties.z, 
          properties.vector_x, properties.vector_y, properties.vector_z, 
          properties.radius, properties.height, properties.sample_resolution, grid_resolution, properties.relax_groups, root
        );
      }
      else if (properties.type === 'BoxObstacle') {
        console.log('Box properties:', properties)

        obstacle_block = new BoxObstacleBlock(
          name, properties.xmin, properties.xmax, properties.ymin, 
          properties.ymax, properties.zmin, properties.zmax, 
          properties.sample_resolution, grid_resolution, properties.relax_groups, root
        );
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

    // ------ pipe group ------
    add_pipe_group(name: string, color: string=''): [boolean, string] {
      if (this.group_cfg.get(name)) {
        return [false, 'Pipe group already exists'];
      }
      const group_cfg: GroupCfg = {
        name: name,
        color: color === '' ? '#' + Math.floor(Math.random()*16777215).toString(16).padStart(6, '0') : color,
        pipes: []
      };
      this.group_cfg.set(name, group_cfg);
      return [true, ''];
    }

    get_pipe_group(name: string): string[]|null {
      return this.group_cfg.get(name)?.pipes as string[] | null;
    }

    get_group_names(): string[] {
      return Array.from(this.group_cfg.keys());
    }
  
    remove_pipe_group(group_name: string, root: BlockRoot) {
      (this.group_cfg.get(group_name)?.pipes as string[]).forEach((pipe_name) => {
        this.remove_pipe(pipe_name, root);
      });
      this.group_cfg.delete(group_name);
    }
  
    // ------ pipe ------
    add_pipe(
      name: string, 
      x: number, y: number, z: number, 
      vec_x: number, vec_y: number, vec_z: number,
      radius: number, group_name: string, is_inlet: boolean, root: BlockRoot
    ): [boolean, string] {
      if ((this.group_cfg.get(group_name)?.pipes as string[]).includes(name)) {
        return [false, 'Pipe already exists in this group'];
      }
      const resolution = this.env_block!.resolution;
      const pipe_block = new PipeBlock(
        name, x, y, z, vec_x, vec_y, vec_z, radius, is_inlet, resolution,
        (this.group_cfg.get(group_name)?.color as string), group_name, root
      );
      this.pipe_blocks.set(name, pipe_block);

      (this.group_cfg.get(group_name)?.pipes as string[]).push(name);
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
      (this.group_cfg.get(pipe_block.group_name)?.pipes as string[]).splice(
        (this.group_cfg.get(pipe_block.group_name)?.pipes as string[]).indexOf(name), 1
      );
    }

    // ------ Main Task -------
    add_main_task(name: string): [boolean, string] {
      if (this.main_tasks.get(name)) {
        return [false, 'Main task already exists'];
      }
      this.main_tasks.set(name, new MainTask(name));
      return [true, ''];
    }

    remove_main_task(name: string) {
      this.main_tasks.delete(name);
    }

    // ------ Priority Task -------
    add_priority_task(name: string, main_task_name: string): [boolean, string] {
      const main_task = this.main_tasks.get(main_task_name);
      if (main_task!.priority_tasks.find((task) => task.name === name)) {
        return [false, 'Priority task already exists'];
      }
      const priority_task = new PriorityTask(name);
      main_task!.priority_tasks.push(priority_task);
      return [true, ''];
    }

    remove_priority_task(name: string, main_task_name: string) {
      const main_task = this.main_tasks.get(main_task_name);
      main_task!.priority_tasks = main_task!.priority_tasks.filter(task => task.name !== name);
    }

    // ------ Group Task -------
    add_group_task(name: string, main_task_name: string, priority_task_name: string): [boolean, string] {
      const main_task = this.main_tasks.get(main_task_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      if (priority_task!.group_tasks.get(name)) {
        return [false, 'Group task already exists'];
      }
      const group_task = new GroupTask(name);
      priority_task!.add_group_task(group_task);
      return [true, ''];
    }

    remove_group_task(name: string, main_task_name: string, priority_task_name: string) {
      const main_task = this.main_tasks.get(main_task_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      priority_task!.group_tasks.delete(name);
    }
    
    // ------ Cell Task -------
    add_cell_task(
      name: string, 
      main_task_name: string, 
      priority_task_name: string, 
      group_task_name: string,
      start_point: string,
      end_point: string,
      radius: number,
      run_cfg: Map<string, string|number|boolean>,
    ): [boolean, string] {
      const main_task = this.main_tasks.get(main_task_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      const group_task = priority_task!.group_tasks.get(group_task_name);
      if (group_task!.cell_tasks.find(task => task.name === name)) {
        return [false, 'Cell task already exists'];
      }
      
      const resolution = this.get_resolution()
      const cell_task = new CellTask(name, start_point, end_point, radius, run_cfg, resolution);
      group_task!.add_cell_task(cell_task);
      return [true, ''];
    }

    remove_cell_task(name: string, main_task_name: string, priority_task_name: string, group_task_name: string) {
      const main_task = this.main_tasks.get(main_task_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      const group_task = priority_task!.group_tasks.get(group_task_name);
      group_task!.cell_tasks = group_task!.cell_tasks.filter(task => task.name !== name);
    }

    // ------ export tree -------
    export_tree(): PathSearchTreeNode {
      const tree: PathSearchTreeNode = {
        id: this.name,
        label: this.name,
        type: 'TaskManager',
        children: [],
      };

      const environment_tree: PathSearchTreeNode = {
        id: 'Environment',
        label: 'Environment' + (this.has_discretized_grid() ? ' (Discretize Mesh)' : ''),
        type: 'TaskEnvironment',
        tagDict: new Map([
          ['Task', this.name],
        ]),
      };
      tree.children?.push(environment_tree);

      const obstacle_tree: PathSearchTreeNode = {
        id: 'Obstacle',
        label: 'Obstacle',
        type: 'TaskObstacle',
        tagDict: new Map([
          ['Task', this.name],
        ]),
        children: Array.from(this.obstacle_blocks.entries()).map(([name, block]) => {
          let name_with_discretize_mesh = name;
          if ((block as ObstacleBlock).has_discretize_mesh) {
            name_with_discretize_mesh = name + ' (Discretize Mesh)';
          }
          return {
              label: name_with_discretize_mesh, 
              tagDict: new Map([['Task', this.name], ['Obstacle', name]]),
              type: 'Obstacle'
          }
        })
      };
      tree.children?.push(obstacle_tree);
  
      const group_tree: PathSearchTreeNode = {
        id: 'PipeGroups',
        label: 'PipeGroups',
        type: 'PipeGroups',
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
          children: (cfg.pipes as string[]).map(pipe_name => {
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
        id: 'TaskTreeRoot',
        label: 'TaskTreeRoot',
        type: 'TaskTreeRoot',
        tagDict: new Map([
          ['Task', this.name],
        ]),
        children: []
      };

      this.main_tasks.forEach((main_task, name) => {
        tasks_tree.children?.push({
          label: main_task.name + (main_task.has_preprocess ? ' (Preprocess)' : ''),
          type: main_task.type,
          tagDict: new Map([
            ['Task', this.name], 
            ['MainTask', main_task.name]
          ]),
          children: main_task.priority_tasks.map((priority_task) => {
            return {
              label: priority_task.name,
              type: priority_task.type,
              tagDict: new Map([
                ['Task', this.name], 
                ['MainTask', name], 
                ['PriorityTask', priority_task.name]
              ]),
              children: Array.from(priority_task.group_tasks.entries()).map(([group_name, group_task]) => {
                return {
                  label: group_task.name,
                  type: group_task.type,
                  tagDict: new Map([
                    ['Task', this.name], 
                    ['MainTask', name], 
                    ['PriorityTask', priority_task.name], 
                    ['GroupTask', group_task.name]
                  ]),
                  children: Array.from(group_task.cell_tasks.entries()).map(([cell_name, cell_task]) => {
                    let cell_task_name = cell_task.name;
                    if (cell_task.has_result){
                      cell_task_name = cell_task_name + ' (Result)';
                    }
                    return {
                      label: cell_task_name,
                      type: cell_task.type,
                      tagDict: new Map([
                        ['Task', this.name], 
                        ['MainTask', name], 
                        ['PriorityTask', priority_task.name], 
                        ['GroupTask', group_task.name], 
                        ['CellTask', cell_task.name]
                      ]),
                      children: [
                        {
                          label: 'Start Point',
                          type: 'StartPoint',
                          value: cell_task.start_point
                        },
                        {
                          label: 'End Point',
                          type: 'EndPoint',
                          value: cell_task.end_point
                        },
                        {
                          label: 'Radius',
                          type: 'Radius',
                          value: cell_task.radius
                        },
                        // {
                        //   label: 'Run Config',
                        //   type: 'RunConfig',
                        //   children: Array.from(cell_task.run_config.entries()).map(([key, value]) => {
                        //     return {
                        //       label: key,
                        //       type: 'RunConfigItem',
                        //       value: value
                        //     }
                        //   })
                        // }
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
        env_block: this.env_block!.toJson(),
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
                name: priority_task.name,
                group_tasks: Array.from(priority_task.group_tasks.entries()).map(([group_name, group_task]) => {
                  return {
                    name: group_name,
                    cell_tasks: Array.from(group_task.cell_tasks.entries()).map(([cell_name, cell_task]) => {
                      return {
                        name: cell_task.name,
                        start_point: cell_task.start_point,
                        end_point: cell_task.end_point,
                        radius: cell_task.radius,
                        run_cfg: cell_task.run_config,
                        result: cell_task.result
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
    
    static load_json(data: any, root: BlockRoot): TaskManager {
      const task_manager = new TaskManager(
        data.name, 
        data.env_block.xlength, data.env_block.ylength, data.env_block.zlength, data.env_block.resolution, root
      );
      data.obstacle_blocks.forEach((obstacle_cfg: any) => {
        task_manager.add_obstacle(obstacle_cfg.name, obstacle_cfg, root);
      });
      
      data.pipe_blocks.forEach((pipe_cfg: any) => {
        if (!task_manager.group_cfg.has(pipe_cfg.group_name)) {
          task_manager.add_pipe_group(pipe_cfg.group_name, pipe_cfg.color);
        }
      });

      data.pipe_blocks.forEach((pipe_cfg: any) => {
        console.log('radius:', pipe_cfg.radius)
        task_manager.add_pipe(
          pipe_cfg.name, 
          pipe_cfg.x, pipe_cfg.y, pipe_cfg.z, 
          pipe_cfg.vector_x, pipe_cfg.vector_y, pipe_cfg.vector_z, 
          pipe_cfg.radius, pipe_cfg.group_name, pipe_cfg.is_inlet,
          root
        );
      });

      data.main_tasks.forEach((task: any) => {
        task_manager.add_main_task(task.name);
        task.priority_tasks.forEach((priority_task: any) => {
          task_manager.add_priority_task(priority_task.name, task.name);
          priority_task.group_tasks.forEach((group_task: any) => {
            task_manager.add_group_task(group_task.name, task.name, priority_task.name);
            group_task.cell_tasks.forEach((cell_task: any) => {
              task_manager.add_cell_task(
                cell_task.name, task.name, priority_task.name, group_task.name, 
                cell_task.start_point, cell_task.end_point, cell_task.radius, cell_task.run_cfg
              );
            });
          });
        });
      });

      return task_manager;
    }

    destory(root: BlockRoot) {
      this.env_block!.destory(root);
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
    
    // ------ Search Parameters -------
    add_discretized_grid(grid_data: number[][], distance_to_obstacle: number[], root: BlockRoot) {
      if (!this.discretized_grid) {
        this.discretized_grid = new DiscreteGridBlock(
          this.name + '_DiscretizedGrid', grid_data, distance_to_obstacle, root
        );
      }
      else {
        this.discretized_grid.points = grid_data;
        this.discretized_grid.distance_to_obstacle = distance_to_obstacle;
        this.discretized_grid.update_mesh(root);
      }
    }

    has_discretized_grid(): boolean {
      return this.discretized_grid !== null;
    }

    show_discretized_grid() {
      if (this.discretized_grid) {
        this.discretized_grid.show_mesh();
      }
    }

    hide_discretized_grid() {
      if (this.discretized_grid) {
        this.discretized_grid.hide_mesh();
      }
    }
    
    // ------ Path Search Result -------
    add_path_set(
      solution_name: string, 
      priority_task_name: string, 
      path_set: Record<string, Record<string, {path: number[][], radius: number}>>, 
      root: BlockRoot
    ) {
      const main_task = this.main_tasks.get(solution_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      
      Object.entries(path_set).forEach(([group_name, group_path_set]) => {
        const group_task = priority_task!.group_tasks.get(group_name);
        const group_cfg = this.group_cfg.get(group_name) as GroupCfg;
        Object.entries(group_path_set).forEach(([cell_task_name, path_data]) => {
          const cell_task = group_task!.cell_tasks.find(task => task.name === cell_task_name);
          cell_task!.set_result(path_data.path, group_cfg!.color, root);
        });
      });
    }

    show_priority_result(solution_name: string, priority_task_name: string) {
      const main_task = this.main_tasks.get(solution_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      priority_task!.group_tasks.forEach((group_task) => {
        group_task.cell_tasks.forEach((cell_task) => {
          cell_task.show_result();
        });
      });
    }

    hide_priority_result(solution_name: string, priority_task_name: string) {
      const main_task = this.main_tasks.get(solution_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      priority_task!.group_tasks.forEach((group_task) => {
        group_task.cell_tasks.forEach((cell_task) => {
          cell_task.hide_result();
        });
      });
    }

    hide_group_result(solution_name: string, priority_task_name: string, group_task_name: string) {
      const main_task = this.main_tasks.get(solution_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      const group_task = priority_task!.group_tasks.get(group_task_name);
      group_task!.cell_tasks.forEach((cell_task) => {
        cell_task.hide_result();
      });
    }

    show_group_result(solution_name: string, priority_task_name: string, group_task_name: string) {
      const main_task = this.main_tasks.get(solution_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      const group_task = priority_task!.group_tasks.get(group_task_name);
      group_task!.cell_tasks.forEach((cell_task) => {
        cell_task.show_result();
      });
    }

    hide_cell_result(solution_name: string, priority_task_name: string, group_task_name: string, cell_task_name: string) {
      const main_task = this.main_tasks.get(solution_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      const group_task = priority_task!.group_tasks.get(group_task_name);
      const cell_task = group_task!.cell_tasks.find(task => task.name === cell_task_name);
      cell_task!.hide_result();
    }

    show_cell_result(solution_name: string, priority_task_name: string, group_task_name: string, cell_task_name: string) {
      const main_task = this.main_tasks.get(solution_name);
      const priority_task = main_task!.priority_tasks.find(task => task.name === priority_task_name);
      const group_task = priority_task!.group_tasks.get(group_task_name);
      const cell_task = group_task!.cell_tasks.find(task => task.name === cell_task_name);
      cell_task!.show_result();
    }
}