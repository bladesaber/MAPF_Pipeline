import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import KDTree
from typing import Union, Any
from queue import Queue
import time
from tqdm import tqdm

from pathSearch_pkg.build import MapfPkg


class CellTaskInfo:
    name: str = None
    start_name: str = None
    end_name: str = None
    radius: float = None
    result: np.ndarray = None

    def __init__(self, name: str, start_name: str, end_name: str, radius: float) -> None:
        self.name = name
        self.start_name = start_name
        self.end_name = end_name
        self.radius = radius


class GroupTaskInfo:
    name: str = None
    cell_infos: dict[str, CellTaskInfo] = {}
    run_config: dict[str, Union[str, float]] = {}

    def __init__(self, name: str, run_config: dict[str, Union[str, float]]) -> None:
        self.name = name
        self.run_config = run_config
        self.cell_infos = {}

class PriorityTaskInfo:
    name: str = None
    group_infos: dict[str, GroupTaskInfo] = {}
    run_cfg: dict[str, Union[str, float]] = {}

    def __init__(self, name: str, run_cfg: dict[str, Union[str, float]]) -> None:
        self.name = name
        self.run_cfg = run_cfg
        self.group_infos = {}


class SearchTaskInfo:
    name: str = None
    priority_infos: dict[str, PriorityTaskInfo] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.graph = nx.Graph()
        self.flag2radius = {}
        self.priority_infos = {}


class ProjectInfo:
    static_obstacle_pcd = []
    relax_group_obstacle_pcd: dict[str, list] = {}
    
    inner_xyzd: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
    x_size: int = None
    y_size: int = None
    z_size: int = None
    resolution: float = None
    grid: MapfPkg.DistanceGrid = None
    pipe_cfg: dict[str, dict[str, list[float]]] = {}
    task_infos: dict[str, SearchTaskInfo] = {}

    graph_dict: dict[str, nx.Graph] = {}
    flag2radius: dict[int, float] = {}


class PathServerUtils:
    project_map: dict[str, ProjectInfo] = {}

    @staticmethod
    def xyz2flag(x: int, y: int, z: int, x_size: int, y_size: int, z_size: int):
        return z * x_size * y_size + y * x_size + x

    @staticmethod
    def flag2xyz(flag: int, x_size: int, y_size: int, z_size: int):
        z = flag // (x_size * y_size)
        y = (flag - z * x_size * y_size) // x_size
        x = flag - z * x_size * y_size - y * x_size
        return x, y, z

    @staticmethod
    def compute_inner_xyzd(x_size: int, y_size: int, z_size: int, resolution: float, obstacle_pcd: np.ndarray):
        # print('[Debug]: Grid initialized:', x_size, y_size, z_size, resolution)
        obstacle_xyzs = obstacle_pcd[:, :3]
        obstacle_radius = obstacle_pcd[:, 3]
        xs, ys, zs = np.meshgrid(
            np.arange(0, x_size, 1),
            np.arange(0, y_size, 1),
            np.arange(0, z_size, 1)
        )
        positions = np.concatenate([
            xs[..., np.newaxis], 
            ys[..., np.newaxis], 
            zs[..., np.newaxis]
        ], axis=-1).reshape(-1, 3)
        
        pos_data, dist_data = positions, []
        if len(obstacle_xyzs) > 0:
            tree = KDTree(obstacle_xyzs)
            dists, inds = tree.query(positions, k=1, return_distance=True)
            dist_data = np.concatenate(dists, axis=-1)

            inds = np.concatenate(inds, axis=-1)
            radius = obstacle_radius[inds]
            dist_data = dist_data - radius

        else:
            dist_data = np.full(len(positions), 500.0)
        
        pos_data = pos_data.astype(np.int32)
        dist_data = dist_data.astype(np.float32)

        return pos_data, dist_data
    
    @staticmethod
    def get_planner(
        planner_style: str, radius: float, max_iterations: int, with_constraint_table: bool,
        run_config: dict[str, Union[str, float]]
    ):
        if planner_style == 'AstarSearchBoost':
            return MapfPkg.AstarSearchBoost(radius, max_iterations, with_constraint_table)
        elif planner_style == 'AstarSearchBucket':
            return MapfPkg.AstarSearchBucket(radius, max_iterations, with_constraint_table)
        elif planner_style == 'WeightedAstarSearchBoost':
            return MapfPkg.WeightedAstarSearchBoost(
                radius, max_iterations, run_config['g_weight'], run_config['h_weight'], with_constraint_table
            )
        elif planner_style == 'WeightedAstarSearchBucket':
            return MapfPkg.WeightedAstarSearchBucket(
                radius, max_iterations, run_config['g_weight'], run_config['h_weight'], with_constraint_table
            )
        elif planner_style == 'ScaledWeightedAstarSearchBoost':
            return MapfPkg.ScaledWeightedAstarSearchBoost(
                radius, max_iterations, run_config['g_weight'], run_config['h_weight'], 
                run_config['threshold1'], run_config['scale_factor1'], with_constraint_table
            )
        elif planner_style == 'ScaledWeightedAstarSearchBucket':
            return MapfPkg.ScaledWeightedAstarSearchBucket(
                radius, max_iterations, run_config['g_weight'], run_config['h_weight'], 
                run_config['threshold1'], run_config['scale_factor1'], with_constraint_table
            )
        elif planner_style == 'DoubleScaledWeightedAstarSearchBoost':
            return MapfPkg.DoubleScaledWeightedAstarSearchBoost(
                radius, max_iterations, run_config['g_weight'], run_config['h_weight'], 
                run_config['threshold1'], run_config['scale_factor1'], 
                run_config['threshold2'], run_config['scale_factor2'], with_constraint_table
            )
        elif planner_style == 'DoubleScaledWeightedAstarSearchBucket':
            return MapfPkg.DoubleScaledWeightedAstarSearchBucket(
                radius, max_iterations, run_config['g_weight'], run_config['h_weight'], 
                run_config['threshold1'], run_config['scale_factor1'], 
                run_config['threshold2'], run_config['scale_factor2'], with_constraint_table
            )
        elif planner_style == 'BiDirectionalAstarSearchBoost':
            return MapfPkg.BiDirectionalAstarSearchBoost(
                radius, max_iterations, run_config['g_weight'], run_config['h_weight'], with_constraint_table
            )
        elif planner_style == 'BiDirectionalAstarSearchBucket':
            return MapfPkg.BiDirectionalAstarSearchBucket(
                radius, max_iterations, run_config['g_weight'], run_config['h_weight'], with_constraint_table
            )
        else:
            raise ValueError(f'Invalid planner name: {planner_style}')

    @staticmethod
    def init_env(project_id: str, pipe_data: dict):
        project_info = PathServerUtils.project_map[project_id]
        project_info.grid = MapfPkg.DistanceGrid(
            project_info.x_size, 
            project_info.y_size, 
            project_info.z_size, 
            project_info.resolution
        )

        pos_data, dist_data = project_info.inner_xyzd
        # for (x, y, z), dist in tqdm(zip(pos_data, dist_data), total=len(pos_data), desc='Initializing inner grid'):
        #     if dist < 500.0:
        #         project_info.grid.set_obstacle(MapfPkg.LocXyz(int(x), int(y), int(z)), dist)
        valid_bool = dist_data < 500.0
        project_info.grid.set_obstacles(pos_data[valid_bool], dist_data[valid_bool])
        
        for group_pipes_cfg in pipe_data:
            group_name = group_pipes_cfg['group_name']
            project_info.graph_dict[group_name] = nx.Graph()

            project_info.pipe_cfg[group_name] = {}
            for pipe_cfg in group_pipes_cfg['pipes']:
                pipe_name = pipe_cfg['name']
                project_info.pipe_cfg[group_name][pipe_name] = [
                    pipe_cfg['x'], pipe_cfg['y'], pipe_cfg['z'], pipe_cfg['radius']
                ]
        
        x_size = project_info.x_size
        y_size = project_info.y_size
        z_size = project_info.z_size
        for group_name, pipes_cfg in project_info.pipe_cfg.items():
            for pipe_name, pipe_xyzr in pipes_cfg.items():
                flag = PathServerUtils.xyz2flag(
                    pipe_xyzr[0], pipe_xyzr[1], pipe_xyzr[2], x_size, y_size, z_size
                )
                project_info.flag2radius[flag] = pipe_xyzr[3]

    @staticmethod
    def update_task_cache(project_id: str, cache_data: dict):
        project_info = PathServerUtils.project_map[project_id]
        x_size = project_info.x_size
        y_size = project_info.y_size
        z_size = project_info.z_size

        for group_name, group_graph in project_info.graph_dict.items():
            group_graph.clear()
            for pipe_name, pipe_xyzr in project_info.pipe_cfg[group_name].items():
                flag = PathServerUtils.xyz2flag(
                    pipe_xyzr[0], pipe_xyzr[1], pipe_xyzr[2], x_size, y_size, z_size
                )
                group_graph.add_node(flag)

        search_task = SearchTaskInfo(cache_data['name'])
        for priority_data in cache_data['priority_tasks']:
            priority_task = PriorityTaskInfo(priority_data['name'], priority_data['run_config'])
            for group_data in priority_data['group_tasks']:
                group_name = group_data['name']
                group_task = GroupTaskInfo(group_name, group_data['run_config'])
                graph = project_info.graph_dict[group_name]

                for cell_data in group_data['cell_tasks']:
                    cell_task = CellTaskInfo(
                        cell_data['name'], cell_data['start_name'], cell_data['end_name'], cell_data['radius']
                    )
                    if len(cell_data['result']) > 0:
                        cell_task.result = np.array(cell_data['result'])
                        for i in range(cell_task.result.shape[0] - 1):
                            flag0 = PathServerUtils.xyz2flag(
                                cell_task.result[i, 0], 
                                cell_task.result[i, 1], 
                                cell_task.result[i, 2], 
                                x_size, y_size, z_size
                            )
                            flag1 = PathServerUtils.xyz2flag(
                                cell_task.result[i + 1, 0], 
                                cell_task.result[i + 1, 1], 
                                cell_task.result[i + 1, 2], 
                                x_size, y_size, z_size
                            )
                            graph.add_edge(flag0, flag1)
                            if flag0 not in project_info.flag2radius:
                                project_info.flag2radius[flag0] = cell_task.radius
                            else:
                                project_info.flag2radius[flag0] = max(
                                    project_info.flag2radius[flag0], cell_task.radius
                                )

                            if flag1 not in project_info.flag2radius:
                                project_info.flag2radius[flag1] = cell_task.radius
                            else:
                                project_info.flag2radius[flag1] = max(
                                    project_info.flag2radius[flag1], cell_task.radius
                                )

                    group_task.cell_infos[cell_task.name] = cell_task
                priority_task.group_infos[group_name] = group_task
            search_task.priority_infos[priority_task.name] = priority_task
        project_info.task_infos[search_task.name] = search_task

        # print level info
        # PathServerUtils.print_level_info(project_id, search_task.name)

    @staticmethod
    def run_cell_task(
        project_id: str, task_name: str, priority_name: str, group_name: str, cell_name: str,
        
    ):
        project_info = PathServerUtils.project_map[project_id]
        search_task = project_info.task_infos[task_name]
        priority_task = search_task.priority_infos[priority_name]
        group_task = priority_task.group_infos[group_name]
        cell_task = group_task.cell_infos[cell_name]

        x_size = project_info.x_size
        y_size = project_info.y_size
        z_size = project_info.z_size
        start_loc = project_info.pipe_cfg[group_name][cell_task.start_name][:3]
        end_loc = project_info.pipe_cfg[group_name][cell_task.end_name][:3]
        graph = project_info.graph_dict[group_name]

        start_flag = PathServerUtils.xyz2flag(
            start_loc[0], start_loc[1], start_loc[2], x_size, y_size, z_size
        )
        end_flag = PathServerUtils.xyz2flag(
            end_loc[0], end_loc[1], end_loc[2], x_size, y_size, z_size
        )

        # if start_flag and end_flag are not in the graph, return
        if start_flag not in graph.nodes() or end_flag not in graph.nodes():
            return {'is_success': False, 'error': 'Start or end flag not in the graph'}
        
        # if start_flag already has path to end_flag, return
        if nx.has_path(graph, start_flag, end_flag):
            return {
                'is_success': False, 'error': 'Start and end are already connected',
                'path': nx.shortest_path(graph, start_flag, end_flag)
            }
        
        planner = PathServerUtils.get_planner(
                group_task.run_config['planner_method'], 
                cell_task.radius, 
                group_task.run_config['max_iterations'], 
                group_task.run_config['with_constraint_table'], 
                group_task.run_config
            )
        runner = MapfPkg.PyWrapAstarRunner(
            cell_name, 
            project_info.grid, 
            planner
        )

        # add dynamic obstacles to the grid
        for other_group_name, other_group_graph in project_info.graph_dict.items():
            if other_group_name != group_name:
                for flag in other_group_graph.nodes():
                    x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
                    runner.add_dynamic_obstacle(
                        MapfPkg.ObstacleData(MapfPkg.LocXyz(x, y, z), project_info.flag2radius[flag])
                    )
                    # print(f'[Debug]: Added pipe obstacle for group: {other_group_name} and x: {x} y: {y} z: {z} radius: {project_info.flag2radius[flag]}')
        
        # add relax group obstacles to the grid
        for relax_group_name, obstacle_pcd in project_info.relax_group_obstacle_pcd.items():
            if relax_group_name != group_name:
                for x, y, z, r in obstacle_pcd:
                    runner.add_dynamic_obstacle(
                        MapfPkg.ObstacleData(MapfPkg.LocXyz(int(x), int(y), int(z)), float(r))
                    )
                    # print(f'[Debug]: Added dynamic obstacle for group: {relax_group_name} and x: {x} y: {y} z: {z} radius: {r}')
        
        start_locs = []
        connected_nodes = nx.node_connected_component(graph, start_flag)
        for flag in connected_nodes:
            x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
            start_locs.append(MapfPkg.LocXyz(x, y, z))

        end_locs = []
        connected_nodes = nx.node_connected_component(graph, end_flag)
        for flag in connected_nodes:
            x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
            end_locs.append(MapfPkg.LocXyz(x, y, z))
            
        # # ------------------------------------------
        # for start_loc in start_locs:
        #     print(f'[Debug]: Start loc: {start_loc.x}, {start_loc.y}, {start_loc.z}')
        # for end_loc in end_locs:
        #     print(f'[Debug]: End loc: {end_loc.x}, {end_loc.y}, {end_loc.z}')
        # # print(f'[Debug]: Is obstacle: {project_info.grid.is_obstacle(MapfPkg.LocXyz(50, 38, 23))}')
        # # ------------------------------------------
        
        is_success = runner.run(start_locs, end_locs)
        if is_success:
            result: MapfPkg.PathResult = runner.result
            del runner
            return {
                'is_success': True, 
                'path': np.array([[loc.x, loc.y, loc.z] for loc in result.path]).tolist(), 
                'radius': result.radius, 
                'time_cost': result.time_cost
                }
        else:
            del runner
            return {'is_success': False, 'error': 'Failed to find path'}

    @staticmethod
    def run_group_task(project_id: str, task_name: str, priority_name: str, group_name: str):
        project_info = PathServerUtils.project_map[project_id]
        search_task = project_info.task_infos[task_name]
        priority_task = search_task.priority_infos[priority_name]
        group_task = priority_task.group_infos[group_name]
        graph = project_info.graph_dict[group_name]

        x_size = project_info.x_size
        y_size = project_info.y_size
        z_size = project_info.z_size

        planner = PathServerUtils.get_planner(
            group_task.run_config['planner_method'], 
            radius=0.0, 
            max_iterations=group_task.run_config['max_iterations'], 
            with_constraint_table=group_task.run_config['with_constraint_table'], 
            run_config=group_task.run_config
        )
        runner = MapfPkg.PyWrapGroupAstarRunner(
            group_task.name, 
            project_info.grid, 
            planner
        )

        # add dynamic obstacles to the grid
        for other_group_name, other_group_graph in project_info.graph_dict.items():
            if other_group_name != group_name:
                for flag in other_group_graph.nodes():
                    x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
                    runner.add_dynamic_obstacle(
                        MapfPkg.ObstacleData(MapfPkg.LocXyz(x, y, z), project_info.flag2radius[flag])
                    )
                    # print(f'[Debug]: Added pipe obstacle for group: {other_group_name} and x: {x} y: {y} z: {z} radius: {project_info.flag2radius[flag]}')
        
        # add relax group obstacles to the grid
        for relax_group_name, obstacle_pcd in project_info.relax_group_obstacle_pcd.items():
            if relax_group_name != group_name:
                for x, y, z, r in obstacle_pcd:
                    runner.add_dynamic_obstacle(
                        MapfPkg.ObstacleData(MapfPkg.LocXyz(int(x), int(y), int(z)), float(r))
                    )
                    # print(f'[Debug]: Added dynamic obstacle for group 222: {relax_group_name} and x: {x} y: {y} z: {z} radius: {r}')
        
        for cell_name, cell_info in group_task.cell_infos.items():
            start_loc = project_info.pipe_cfg[group_name][cell_info.start_name][:3]
            end_loc = project_info.pipe_cfg[group_name][cell_info.end_name][:3]

            start_flag = PathServerUtils.xyz2flag(
                start_loc[0], start_loc[1], start_loc[2], x_size, y_size, z_size
            )
            end_flag = PathServerUtils.xyz2flag(
                end_loc[0], end_loc[1], end_loc[2], x_size, y_size, z_size
            )

            start_locs = []
            connected_nodes = nx.node_connected_component(graph, start_flag)
            for flag in connected_nodes:
                x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
                start_locs.append(MapfPkg.LocXyz(x, y, z))

            end_locs = []
            connected_nodes = nx.node_connected_component(graph, end_flag)
            for flag in connected_nodes:
                x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
                end_locs.append(MapfPkg.LocXyz(x, y, z))

            runner.add_task(
                cell_name,
                cell_info.start_name,
                cell_info.end_name,
                start_locs,
                end_locs,
                cell_info.radius
            )

        is_success = runner.run()
        if is_success:
            result: MapfPkg.PathSetData = runner.result
            path_set = {}
            for cell_name, path_data in zip(list(group_task.cell_infos.keys()), result.paths):
                path_set[cell_name] = {
                    'path': np.array([
                        [loc.x, loc.y, loc.z] for loc in path_data.path
                    ]).tolist(),
                    'radius': path_data.radius,
                    'time_cost': path_data.time_cost
                }
            del runner
            return {
                'is_success': True, 'path_set': path_set
            }
        else:
            del runner
            return {'is_success': False, 'error': 'Failed to find path'}

    @staticmethod
    def run_priority_task(
        project_id: str, task_name: str, priority_name: str, 
        mode: str = 'single', data_queue: Queue = None
    ):
        project_info = PathServerUtils.project_map[project_id]
        search_task = project_info.task_infos[task_name]
        priority_task = search_task.priority_infos[priority_name]

        x_size = project_info.x_size
        y_size = project_info.y_size
        z_size = project_info.z_size

        planner_store = {}
        runner = MapfPkg.PyWrapCbsRunner(priority_name, project_info.grid)
        for group_name, group_task in priority_task.group_infos.items():
            planner = PathServerUtils.get_planner(
                group_task.run_config['planner_method'], 
                radius=0.0, 
                max_iterations=group_task.run_config['max_iterations'], 
                with_constraint_table=group_task.run_config['with_constraint_table'], 
                run_config=group_task.run_config
            )
            planner_store[group_name] = planner
            runner.add_set(group_name, planner)
        
        # PathServerUtils.print_level_info(project_id, task_name)

        # add dynamic obstacles to the grid
        for group_name, group_task in priority_task.group_infos.items():
            for other_group_name, other_group_task in project_info.graph_dict.items():
                if other_group_name != group_name:
                    for flag in other_group_task.nodes():
                        x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
                        runner.add_dynamic_obstacle(
                            group_name,
                            MapfPkg.ObstacleData(
                                MapfPkg.LocXyz(x, y, z), project_info.flag2radius[flag]
                            )
                        )
                        # print(f'[Debug]: Added dynamic obstacle for group: {group_name} and x: {x} y: {y} z: {z} radius: {project_info.flag2radius[flag]}')

            # add relax group obstacles to the grid
            for relax_group_name, obstacle_pcd in project_info.relax_group_obstacle_pcd.items():
                if relax_group_name != group_name:
                    for x, y, z, r in obstacle_pcd:
                        runner.add_dynamic_obstacle(
                            group_name,
                            MapfPkg.ObstacleData(MapfPkg.LocXyz(int(x), int(y), int(z)), float(r))
                        )

        for group_name, group_task in priority_task.group_infos.items():
            group_graph = project_info.graph_dict[group_name]

            for cell_name, cell_info in group_task.cell_infos.items():
                start_loc = project_info.pipe_cfg[group_name][cell_info.start_name]
                end_loc = project_info.pipe_cfg[group_name][cell_info.end_name]

                start_flag = PathServerUtils.xyz2flag(
                    start_loc[0], start_loc[1], start_loc[2], x_size, y_size, z_size
                )
                end_flag = PathServerUtils.xyz2flag(
                    end_loc[0], end_loc[1], end_loc[2], x_size, y_size, z_size
                )

                connected_nodes = nx.node_connected_component(group_graph, start_flag)
                start_locs = []
                for flag in connected_nodes:
                    x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
                    start_locs.append(MapfPkg.LocXyz(x, y, z))

                connected_nodes = nx.node_connected_component(group_graph, end_flag)
                end_locs = []
                for flag in connected_nodes:
                    x, y, z = PathServerUtils.flag2xyz(flag, x_size, y_size, z_size)
                    end_locs.append(MapfPkg.LocXyz(x, y, z))

                runner.add_task(
                    group_name,
                    cell_name,
                    cell_info.start_name,
                    cell_info.end_name,
                    start_locs,
                    end_locs,
                    cell_info.radius
                )
        
        cbs_planner = MapfPkg.CbsSearchNearEnd(
            project_info.grid, 
            priority_task.run_cfg['g_weight'], 
            priority_task.run_cfg['h_weight'], 
            priority_task.run_cfg['max_iterations']
        )
        runner.set_cbs_planner(cbs_planner)

        if mode == 'single':
            is_success = runner.init_root_node()
            if not is_success:
                return {'is_success': False, 'error': 'Failed to initialize root node'}

            for _ in range(priority_task.run_cfg['max_iterations']):
                is_success = runner.single_step_search()
                if is_success:
                    result: MapfPkg.CbsResult = runner.result
                    path_res = {}
                    for group_name, path_set in result.paths.items():
                        path_res[group_name] = {}
                        group_task = priority_task.group_infos[group_name]
                        for cell_name, path in zip(list(group_task.cell_infos.keys()), path_set.paths):
                            path_res[group_name][cell_name] = {
                                'path': np.array([[loc.x, loc.y, loc.z] for loc in path.path]).tolist(),
                                'radius': path.radius,
                                'time_cost': path.time_cost
                            }
                    return {'is_success': True, 'path_res': path_res}

            return {'is_success': False, 'error': 'Failed to find path'}
        
        elif mode == 'stream':
            start_time = time.time()
            is_success = runner.init_root_node()
            print(f'[Debug]: init_root_node time: {time.time() - start_time}')

            if not is_success:
                data_queue.put({
                    'is_finished': True,
                    'result': {
                        'is_success': False, 
                        'error': 'Failed to initialize root node'
                    }
                })
                return
            else:
                result: MapfPkg.CbsResult = runner.result
                path_res = {}
                for group_name, path_set in result.paths.items():
                    path_res[group_name] = {}
                    group_task = priority_task.group_infos[group_name]
                    for cell_name, path in zip(list(group_task.cell_infos.keys()), path_set.paths):
                        path_res[group_name][cell_name] = {
                            'path': np.array([[loc.x, loc.y, loc.z] for loc in path.path]).tolist(),
                            'radius': path.radius,
                            'time_cost': path.time_cost
                        }
                data_queue.put({
                    'is_finished': False,
                    'result': {'is_success': False, 'result': path_res}
                })
                # print('debug: put the first result at: ', time.time())
            
            for iteration in range(priority_task.run_cfg['max_iterations']):
                start_time = time.time()
                is_success = runner.single_step_search()
                result: MapfPkg.CbsResult = runner.result
                time_cost = np.round(time.time() - start_time, 2)
                print(f'[Debug iteration {iteration}]: conflict_count: {result.m_conflict_size}, single_step_search time: {time_cost}')

                path_res = {}
                for group_name, path_set in result.paths.items():
                    path_res[group_name] = {}
                    group_task = priority_task.group_infos[group_name]
                    for cell_name, path in zip(list(group_task.cell_infos.keys()), path_set.paths):
                        path_res[group_name][cell_name] = {
                            'path': np.array([[loc.x, loc.y, loc.z] for loc in path.path]).tolist(),
                            'radius': path.radius,
                            'time_cost': path.time_cost
                        }

                if is_success:
                    data_queue.put({
                        'is_finished': True,
                        'result': {'is_success': True, 'result': path_res}
                    })
                    break
                else:
                    data_queue.put({
                        'is_finished': False,
                        'result': {'is_success': False, 'result': path_res},
                        'msg': f'[Debug iteration {iteration}]: conflict_count: {result.m_conflict_size}, single_step_search time: {time_cost}'
                    })
                    # print(f'debug: put the iteration {iteration} result at: {time.time()}')

            data_queue.put({
                'is_finished': True,
                'result': {'is_success': False, 'error': 'Failed to find path'}
            })
            del runner
            return

    @staticmethod
    def print_level_info(project_id: str, task_name: str):
        project_info = PathServerUtils.project_map[project_id]
        search_task = project_info.task_infos[task_name]
        print(f'[Debug]: Search task: {search_task.name}')
        for priority_name, priority_task in search_task.priority_infos.items():
            print(f'  Priority task: {priority_task.name}')
            print(f'    Run config: {priority_task.run_cfg}')
            for group_name, group_task in priority_task.group_infos.items():
                print(f'    Group task: {group_task.name}')
                print(f'      Run config: {group_task.run_config}')
                for cell_name, cell_task in group_task.cell_infos.items():
                    print(f'      Cell task: {cell_task.name}')
                    print(f'        Start name: {cell_task.start_name}')
                    print(f'        End name: {cell_task.end_name}')
                    print(f'        Radius: {cell_task.radius}')
                    print(f'        Result: {cell_task.result}')
        
        # print level pipe config
        for group_name, pipes_cfg in project_info.pipe_cfg.items():
            print(f'[Debug]: Group Pipe Config: {group_name}')
            for pipe_name, pipe_cfg in pipes_cfg.items():
                print(f'    Pipe: {pipe_name} x: {pipe_cfg[0]} y: {pipe_cfg[1]} z: {pipe_cfg[2]} radius: {pipe_cfg[3]}')
