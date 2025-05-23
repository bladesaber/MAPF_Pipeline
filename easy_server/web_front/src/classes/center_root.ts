import * as THREE from 'three'

import type { ObstacleMesh } from './path_search/obstacle_mesh';
import { SearchTask, PriorityTask, GroupTask, CellTask } from './path_search/task_runner';
import { GridMesh } from "./path_search/grid";
import { PipeMesh } from './path_search/pipe_mesh';
import type { TreeNode } from '@/types/data_type';

import { BoxObstacle, CylinderObstacle, PointsObstacle, SpheresObstacle } from './path_search/obstacle_mesh';

export class ProjectManager {
  name: string;

  group_color_dict: Map<string, number[]> = new Map();

  // ---------------- Design ----------------
  grid: GridMesh;
  obstacle_dict: Map<string, ObstacleMesh> = new Map();
  pipe_dict: Map<string, Map<string, PipeMesh>> = new Map(); // group_name -> pipe_name -> PipeMesh
  search_task_dict: Map<string, SearchTask> = new Map();

  // ---------------- Smooth ----------------

  // ---------------- CAD ----------------

  // ---------------- Simulate ----------------

  constructor(
    name: string, xlength: number, ylength: number, zlength: number, resolution: number, scene: THREE.Scene
  ) {
    this.name = name;
    this.grid = new GridMesh('grid', xlength, ylength, zlength, resolution, scene);
  }

  export_tree() {
    // ---------------- Design ----------------
    const grid_node: TreeNode = {
      label: 'Grid' + `(${this.grid.discretize_source})`,
      style: 'Grid',
      data: { 'Project': this.name },
    };

    const obstacle_nodes: TreeNode = {
      label: 'Obstacles',
      children: Array.from(this.obstacle_dict.values()).map((obstacle) => {
        return {
          label: obstacle.name + `(${obstacle.discretize_source})`,
          style: 'Obstacle',
          data: { 'Project': this.name, 'Obstacle': obstacle.name },
        };
      }),
      style: 'Obstacles',
      data: { 'Project': this.name },
    };

    const pipe_nodes: TreeNode = {
      label: 'Pipes',
      children: Array.from(this.pipe_dict.entries()).map(([group_name, pipe_dict]) => {
        return {
          label: group_name,
          style: 'PipeGroup',
          data: { 'Project': this.name, 'PipeGroup': group_name },
          children: Array.from(pipe_dict.values()).map((pipe) => {
            return {
              label: pipe.name,
              style: 'Pipe',
              data: { 'Project': this.name, 'PipeGroup': group_name, 'Pipe': pipe.name },
            };
          }),
        };
      }),
      style: 'Pipes',
      data: { 'Project': this.name },
    };
    
    // ------ check finish ------
    this.search_task_dict.forEach((task) => {
      task.check_finish();
    });

    const task_nodes: TreeNode = {
      label: 'Tasks',
      children: Array.from(this.search_task_dict.values()).map((task) => {
        return {
          label: task.name + (task.has_result ? ' (finished)' : ''),
          style: 'SearchTask',
          data: { 'Project': this.name, 'SearchTask': task.name },
          children: task.priority_tasks.map((priority_task) => {
            return {
              label: priority_task.name + (priority_task.has_result ? ' (finished)' : ''),
              style: 'PriorityTask',
              data: { 
                'Project': this.name, 
                'SearchTask': task.name, 
                'PriorityTask': priority_task.name
              },
              children: Array.from(priority_task.group_tasks.entries()).map(([group_name, group_task]) => {
                return {
                  label: group_name + (group_task.has_result ? ' (finished)' : ''),
                  style: 'GroupTask',
                  data: { 
                    'Project': this.name, 
                    'SearchTask': task.name, 
                    'PriorityTask': priority_task.name, 
                    'GroupTask': group_name 
                  },
                  children: group_task.cell_tasks.map((cell_task) => {
                    return {
                      label: cell_task.has_result ? cell_task.name + ' (finished)' : cell_task.name,
                      style: 'CellTask',
                      data: { 
                        'Project': this.name, 
                        'SearchTask': task.name, 
                        'PriorityTask': priority_task.name, 
                        'GroupTask': group_name, 
                        'CellTask': cell_task.name
                      },
                    };
                  }),
                };
              }),
            };
          }),
        };
      }),
      style: 'Tasks',
      data: { 'Project': this.name },
    };

    const search_node: TreeNode = {
      label: 'Design',
      children: [grid_node, obstacle_nodes, pipe_nodes, task_nodes],
      style: 'Design',
      data: { 'Project': this.name },
    };

    // ---------------- Smooth ----------------

    // ---------------- CAD ----------------

    // ---------------- Simulate ----------------

    const parent_node: TreeNode = {
      label: this.name,
      style: 'ProjectRoot',
      data: { 'Project': this.name },
      children: [search_node],
    };

    return parent_node;
  }

  export_json() {
    const save_data = {
      name: this.name,
      grid: this.grid.toJSON(),
      obstacles: Array.from(this.obstacle_dict.values()).map((obstacle) => obstacle.toJSON()),
      pipe_groups: Array.from(this.pipe_dict.entries()).map(([group_name, pipe_dict]) => {
        return {
          group_name: group_name,
          pipes: Array.from(pipe_dict.values()).map((pipe) => pipe.toJSON()),
        };
      }),
      tasks: Array.from(this.search_task_dict.entries()).map(([task_name, task]) => task.toJSON()),
      group_color_dict: Object.fromEntries(this.group_color_dict.entries()),
    }
    return save_data;
  }

  export_pipe_json(is_discrete: boolean = false) {
    const pipe_groups = Array.from(this.pipe_dict.entries()).map(([group_name, pipe_dict]) => {
      return {
        group_name: group_name,
        pipes: Array.from(pipe_dict.values()).map((pipe) => pipe.toJSON(is_discrete, this.grid)),
      };
    });
    return pipe_groups;
  }

  static load_json(json_data: any, scene: THREE.Scene) {
    const proj_manager = new ProjectManager(
      json_data.name, json_data.grid.xlength, json_data.grid.ylength, json_data.grid.zlength, 
      json_data.grid.resolution, scene
    );

    for (const obstacle of json_data.obstacles) {
      if (obstacle.type === 'Box') {
        const box_obstacle = new BoxObstacle(
          obstacle.name, obstacle.color, 
          obstacle.xmin, obstacle.xmax, obstacle.ymin, obstacle.ymax, obstacle.zmin, obstacle.zmax, scene);
        proj_manager.obstacle_dict.set(obstacle.name, box_obstacle);
      } else if (obstacle.type === 'Cylinder') {
        const cylinder_obstacle = new CylinderObstacle(
          obstacle.name, obstacle.color, obstacle.x, obstacle.y, obstacle.z, obstacle.radius, obstacle.height, obstacle.vector_x, obstacle.vector_y, obstacle.vector_z, scene);
        proj_manager.obstacle_dict.set(obstacle.name, cylinder_obstacle);
      } else if (obstacle.type === 'Points') {
        const points_obstacle = new PointsObstacle(obstacle.name, obstacle.color, obstacle.points, scene);
        proj_manager.obstacle_dict.set(obstacle.name, points_obstacle);
      } else if (obstacle.type === 'Spheres') {
        const spheres_obstacle = new SpheresObstacle(
          obstacle.name, obstacle.color, obstacle.points, obstacle.radius, scene);
        proj_manager.obstacle_dict.set(obstacle.name, spheres_obstacle);
      }
    }

    for (const [group_name, color] of Object.entries(json_data.group_color_dict)) {
      proj_manager.group_color_dict.set(group_name, color as number[]);
    }

    json_data.pipe_groups.forEach((pipe_group: any) => {
      const pipe_dict = new Map<string, PipeMesh>();
      pipe_group.pipes.forEach((pipe: any) => {
        const pipe_mesh = new PipeMesh(
          pipe.name, pipe.x, pipe.y, pipe.z, pipe.radius,
          pipe.vector_x, pipe.vector_y, pipe.vector_z, 
          proj_manager.group_color_dict.get(pipe.group_name)!, pipe.group_name, 
          scene
        );
        pipe_dict.set(pipe.name, pipe_mesh);
      });
      proj_manager.pipe_dict.set(pipe_group.group_name, pipe_dict);
    });

    json_data.tasks.forEach((task: any) => {
      const task_manager = new SearchTask(task.name);
      proj_manager.search_task_dict.set(task.name, task_manager);
      task.priority_tasks.forEach((priority_task: any) => {

        const run_config = new Map<string, string | number | boolean>();
        Object.entries(priority_task.run_config).forEach(([key, value]) => {
          run_config.set(key, value as string | number | boolean);
        });

        const priority_task_manager = new PriorityTask(priority_task.name, run_config);
        task_manager.priority_tasks.push(priority_task_manager);
        priority_task.group_tasks.forEach((group_task: any) => {

          const run_config = new Map<string, string | number | boolean>();
          Object.entries(group_task.run_config).forEach(([key, value]) => {
            run_config.set(key, value as string | number | boolean);
          });

          const group_task_manager = new GroupTask(group_task.name, run_config);
          priority_task_manager.group_tasks.set(group_task.name, group_task_manager);
          group_task.cell_tasks.forEach((cell_task: any) => {
            const cell_task_manager = new CellTask(
              cell_task.name, cell_task.start_name, cell_task.end_name, cell_task.radius, 
              proj_manager.group_color_dict.get(group_task.name)!
            );
            group_task_manager.cell_tasks.push(cell_task_manager);
          });
        });
      });
    });

    return proj_manager;
  }
}


export class CenterRoot {
  private scene: THREE.Scene;
  public proj_dict: Map<string, ProjectManager>;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.proj_dict = new Map();
  }

  get_scene() {
    return this.scene;
  }

  // ---------------- Lift Side Tree ----------------
  export_tree() {
    const node_list: TreeNode[] = []
    for (const proj of this.proj_dict.values()) {
      node_list.push(proj.export_tree())
    }
    return node_list
  }

  export_json() {
    const save_data = [
      ...Array.from(this.proj_dict.values()).map((proj) => proj.export_json()),
    ]
    return save_data;
  }

  load_json(json_data: []) {
    for (const data of json_data) {
      const proj_manager = ProjectManager.load_json(data, this.scene);
      this.proj_dict.set(proj_manager.name, proj_manager);
    }
  }
  
} 