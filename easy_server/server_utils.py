import threading
import json
from typing import Optional, Callable
import time
from typing import Dict, List, Tuple, Union
import queue
from functools import partial

from sklearn.neighbors import KDTree
from pathSearch_pkg.build import MapfPkg
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

class StreamDataResponse:
    def __init__(
        self, name: str, 
        get_data_size_callback: Callable, 
        get_chunk_data_callback: Callable,
        chunk_size: int = 100, time_interval: float = 0.1
    ):
        """
        get_data_size_callback must be a function that returns int
        get_chunk_data_callback must be a function including data_index and chunk_size
        """
        self.name = name
        self.chunk_size = chunk_size
        self.get_data_size_callback = get_data_size_callback
        self.get_chunk_data_callback = get_chunk_data_callback
        self.last_data_index = 0
        self.time_interval = time_interval
 
    def generate(self):
        try:
            while True:
                chunk_data = self.update_progress()
                yield json.dumps(chunk_data)
                time.sleep(self.time_interval)
                if chunk_data['is_finished']:
                    break
        except Exception as e:
            self.error_process(e)
            raise e
        finally:
            self.final_process()

    def update_progress(self) -> dict:
        data_size = self.get_data_size_callback()
        if self.last_data_index + self.chunk_size >= data_size:
            chunk_data = self.get_progress(
                data=self.get_chunk_data_callback(self.last_data_index, self.chunk_size),
                chunk_index=self.last_data_index,
                is_finished=True,
                progress=100,
                msg=f'Processing Finished',
            )
        else:
            chunk_data = self.get_progress(
                data=self.get_chunk_data_callback(self.last_data_index, self.chunk_size),
                chunk_index=self.last_data_index,
                is_finished=False,
                progress=int(float(self.last_data_index) / float(data_size) * 100),
                msg=f'Processing... {self.last_data_index}/{data_size}'
            )
            self.last_data_index += self.chunk_size
        return chunk_data

    def error_process(self, e: Exception):
        print('[Error]: ', e)

    def final_process(self):
        print(f'[Debug]: Task {self.name} finished')

    def get_progress(self, data, chunk_index: int, is_finished: bool, progress: int, msg: str) -> dict:
        return {
            'name': self.name,
            'data': data,
            'chunk_index': chunk_index,
            'is_finished': is_finished,
            'progress': progress,
            'msg': msg,
            'total_finished': False,
        }


class StreamTaskResponse:
    def __init__(self, name: str, callback_func: Callable, interval_time: float = 0.05):
        """
        callback_func must be a function that returns dict which has 'is_finished' key and 
        must include data_queue and last_result in its arguments
        """
        self.name = name
        self.interval_time = interval_time
        self.last_update_time = 0.0
        self.data_queue = queue.Queue()
        self.last_result = None
        self.callback_func = partial(callback_func, data_queue=self.data_queue)
        self.thread = None
        self.run_count = 0
    
    def start(self):
        self.thread = threading.Thread(target=self.callback_func)
        self.thread.start()
        while True:
            time.sleep(self.interval_time)
            progress_data = self.update_progress()
            yield json.dumps(progress_data)
            if progress_data['is_finished']:
                break
        self.thread.join()

    def update_progress(self) -> dict:
        if not self.data_queue.empty():
            run_data = self.data_queue.get()
            self.run_count += 1
            if run_data['is_finished']:
                self.last_result = run_data['data']
                return self.get_progress(
                    is_finished=True,
                    progress=self.run_count,
                    msg=f'[Debug]: Task {self.name} finished',
                )
            else:
                return self.get_progress(
                    is_finished=False,
                    progress=self.run_count,
                    msg=run_data['msg'],
                )
        else:
            return self.get_progress(
                is_finished=False,
                progress=self.run_count,
                msg=f'[Debug]: Waiting for task {self.name}...',
            )

    def get_progress(self, is_finished: bool, progress: int, msg: str) -> dict:
        return {
            'name': self.name,
            'is_finished': is_finished,
            'progress': progress,
            'msg': msg,
            'total_finished': False,
        }
    

class SocketManager:
    socket_map = {}

    def __init__(self, ws):
        self.id = id(ws)
        self.ws = ws
        SocketManager.socket_map[self.id] = self
    
    def send_msg(self, msg: dict):
        self.ws.send(json.dumps(msg))

    def close(self):
        self.ws.close()

    @staticmethod
    def remove_socket(manager):
        manager.close()
        del SocketManager.socket_map[manager.id]


class SocketDataResponse:
    def __init__(
            self, manager: SocketManager, 
            run_type: str, name: str, 
            get_data_size_callback: Callable, 
            get_chunk_data_callback: Callable,
            chunk_size: int = 100, time_interval: float = 0.1
    ):
        self.manager = manager
        self.run_type = run_type
        self.name = name
        self.chunk_size = chunk_size
        self.get_data_size_callback = get_data_size_callback
        self.get_chunk_data_callback = get_chunk_data_callback
        self.last_data_index = 0
        self.time_interval = time_interval

    def generate(self):
        try:
            while True:
                chunk_data = self.update_progress()
                self.manager.send_msg(chunk_data)
                time.sleep(self.time_interval)
                if chunk_data['is_finished']:
                    break
        except Exception as e:
            self.error_process(e)
            raise e
        finally:
            self.final_process()

    def update_progress(self) -> dict:
        data_size = self.get_data_size_callback()
        if self.last_data_index + self.chunk_size >= data_size:
            chunk_data = self.get_progress(
                data=self.get_chunk_data_callback(self.last_data_index, self.chunk_size),
                chunk_index=self.last_data_index,
                is_finished=True,
                progress=100,
                msg=f'Processing Finished',
            )
        else:
            chunk_data = self.get_progress(
                data=self.get_chunk_data_callback(self.last_data_index, self.chunk_size),
                chunk_index=self.last_data_index,
                is_finished=False,
                progress=int(float(self.last_data_index) / float(data_size) * 100),
                msg=f'Processing... {self.last_data_index}/{data_size}',
            )
            self.last_data_index += self.chunk_size
        return chunk_data
    
    def error_process(self, e: Exception):
        print('[Error]: ', e)
    
    def final_process(self):
        # print(f'[Debug]: Task {self.name} finished')
        pass

    def get_progress(self, data, chunk_index: int, is_finished: bool, progress: int, msg: str) -> dict:
        return {
            'run_type': self.run_type,
            'name': self.name,
            'data': data,
            'chunk_index': chunk_index,
            'is_finished': is_finished,
            'progress': progress,
            'msg': msg,
            'total_finished': False,
        }


class SocketTaskResponse:
    def __init__(self, manager: SocketManager, run_type: str, name: str, callback_func: Callable, interval_time: float = 0.05):
        self.manager = manager
        self.run_type = run_type
        self.name = name
        self.interval_time = interval_time
        self.last_update_time = 0.0
        self.data_queue = queue.Queue()
        self.last_result = None
        self.callback_func = partial(callback_func, data_queue=self.data_queue)
        self.thread = None
        self.run_count = 0

    def start(self):
        self.thread = threading.Thread(target=self.callback_func)
        self.thread.start()
        while True:
            time.sleep(self.interval_time)
            progress_data = self.update_progress()
            self.manager.send_msg(progress_data)
            if progress_data['is_finished']:
                break
        self.thread.join()
    
    def update_progress(self) -> dict:
        if not self.data_queue.empty():
            run_data = self.data_queue.get()
            self.run_count += 1
            if run_data['is_finished']:
                self.last_result = run_data['data']
                return self.get_progress(
                    is_finished=True,
                    progress=self.run_count,
                    msg=f'[Debug]: Task {self.run_type}/{self.name} finished',
                )
            else:
                return self.get_progress(
                    is_finished=False,
                    progress=self.run_count,
                    msg=run_data['msg'],
                    progress_data=run_data.get('progress_data', {}),
                )
        else:
            return self.get_progress(
                is_finished=False,
                progress=self.run_count,
                msg=f'[Debug]: Waiting for task {self.run_type}/{self.name}...',
            )
    
    def get_progress(self, is_finished: bool, progress: int, msg: str, progress_data: dict = None) -> dict:
        return {
            'run_type': self.run_type,
            'name': self.name,
            'is_finished': is_finished,
            'progress': progress,
            'progress_data': progress_data,
            'msg': msg,
            'total_finished': False,
        }


class PathSearchRunner(object):
    def __init__(self):
        self.grid = None
        self.running_cache = {}
        self.running_graph = {}
    
    def preprocess(self, data: dict):
        for group_name, group_task in data.items():
            self.running_graph[group_name] = nx.Graph()
            for pipe_name, pipe_data in group_task.items():
                self.running_cache[pipe_name] = np.array([
                    [pipe_data['x'], pipe_data['y'], pipe_data['z'], pipe_data['radius']]
                ])

    @staticmethod
    def init_grid(
        data_queue: queue.Queue, 
        x_size: int, y_size: int, z_size: int, resolution: float, obstacle_points: np.ndarray
    ):
        # print('[Debug]: Grid initialized:', x_size, y_size, z_size, resolution)
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
        
        validate_data = []
        grid = MapfPkg.DistanceGrid(x_size, y_size, z_size, resolution)

        if len(obstacle_points) > 0:
            tree = KDTree(obstacle_points)
            dists, inds = tree.query(positions, k=1, return_distance=True)
            for pos, dist in zip(positions, dists):
                grid.set_obstacle(MapfPkg.LocXyz(pos[0], pos[1], pos[2]), dist[0])
                validate_data.append([pos[0], pos[1], pos[2], dist[0]])
        else:
            for pos in positions:
                validate_data.append([pos[0], pos[1], pos[2], 100.0])
        
        data_queue.put({
            'is_finished': True,
            'data': {
                'validate_data': np.array(validate_data).tolist(),
                'grid': grid,
            },
            'msg': 'Grid prepared successfully.'
        })

    @staticmethod
    def run_cell_task(
        data_queue: queue.Queue, 
        grid: MapfPkg.DistanceGrid, 
        obstacle_points: np.ndarray,
        starts: np.ndarray, 
        goals: np.ndarray, 
        radius: float, 
        max_iter: int
    ):
        grid_set = MapfPkg.SearchGridSet(grid)
        for (x, y, z, radius) in obstacle_points:
            grid_set.set_obstacle(MapfPkg.LocXyz(x, y, z), radius)

        starts_locxyz, goals_locxyz = [], []
        for start, goal in zip(starts, goals):
            starts_locxyz.append(MapfPkg.LocXyz(start[0], start[1], start[2]))
            goals_locxyz.append(MapfPkg.LocXyz(goal[0], goal[1], goal[2]))
            
        planner = MapfPkg.AstarSearchBoost(radius, max_iter, False)
        path_store = MapfPkg.PyWrapAstarStruct(grid_set, planner)
        is_success = path_store.run(starts_locxyz, goals_locxyz)

        path = []
        if is_success:
            for loc in path_store.path:
                path.append([loc.x, loc.y, loc.z])

        data_queue.put({
            'is_finished': True,
            'data': {
                'is_success': is_success,
                'path': path,
            },
            'msg': 'Path found successfully.' if len(path) > 0 else 'No path found.'
        })
    
    @staticmethod
    def run_priority_task(
        data_queue: queue.Queue,
        grid: MapfPkg.DistanceGrid,
        task_set: dict[str, list[dict[str, str|float]]],
        group_pipes: dict[str, list[str]],
        running_cache: dict[str, np.ndarray],
        running_graph: dict[str, nx.Graph],
    ):
        cbs = MapfPkg.CbsSearchNearEnd(grid, 1.0, 1.0, 1000)

        task_manager_map = {} # Store task managers to prevent garbage collection
        astar_planners = {}   # Store planners to prevent garbage collection
        planners = {}      # Store planners to prevent garbage collection

        dynamic_obstacle_map = MapfPkg.DynamicObstacleMap()

        for group_name, cell_tasks in task_set.items():
            astar = MapfPkg.AstarSearchBoost(0.5, 100000, False)
            astar_planners[group_name] = astar

            planner = MapfPkg.SetPairNodePlanner(astar)
            planners[group_name] = planner

            task_manager = MapfPkg.PairNodeTaskSet(group_name)
            task_manager_map[group_name] = task_manager

            for cell_task in cell_tasks:
                start_name: str = cell_task['start_name']
                end_name: str = cell_task['end_name']
                
                start_points = []
                for point in running_cache[start_name]:
                    start_points.append(MapfPkg.LocXyz(point[0], point[1], point[2]))

                end_points = []
                for point in running_cache[end_name]:
                    end_points.append(MapfPkg.LocXyz(point[0], point[1], point[2]))

                task_manager.add_task(
                    cell_task['task_name'], start_name, end_name,
                    start_locs=start_points, end_locs=end_points,
                    radius=cell_task['radius']
                )
            
            for other_group_name, other_pipes in group_pipes.items():
                if other_group_name == group_name:
                    continue

                dynamic_constraints = []
                for other_pipe in other_pipes:
                    for (x, y, z, radius) in running_cache[other_pipe]:
                        dynamic_constraints.append([x, y, z, radius])
                dynamic_constraints = pd.DataFrame(dynamic_constraints, columns=['x', 'y', 'z', 'radius'])
                dynamic_constraints.drop_duplicates(subset=['x', 'y', 'z'], inplace=True, keep='first')
                dynamic_constraints = dynamic_constraints.to_numpy()

                for (x, y, z, radius) in dynamic_constraints:
                    dynamic_obstacle_map.add_obstacle(
                        group_name=group_name, 
                        obstacle=MapfPkg.ObstacleData(MapfPkg.LocXyz(x, y, z), radius)
                    )

            cbs.add_task_set(task_manager, planner)

        cbs_store = MapfPkg.PyWrapCbsStruct(cbs, dynamic_obstacle_map)

        # # ------ auto mode
        # print('[Debug]: Running CBS...')
        # is_success = cbs_store.run()
        # cbs_result: dict[str, MapfPkg.PathSet] = cbs_store.result.paths

        # ------ single step mode
        is_success = False
        for iteration in range(1000):
            is_success, result = cbs_store.single_step_run()
            if is_success:
                break
            else:
                cbs_result: dict[str, MapfPkg.PathSet] = result.paths
                path_set_tmp = {}
                for group_name, path_set in cbs_result.items():
                    path_set_tmp[group_name] = {}
                    for path in path_set.paths:
                        path_set_tmp[group_name][path.name] = {
                            'path': np.array([[loc.x, loc.y, loc.z] for loc in path.path]).tolist(),
                            'radius': path.radius
                        }
                data_queue.put({
                    'is_finished': False,
                    'progress_data': {
                        'is_success': is_success,
                        'path_set_group': path_set_tmp,
                    },
                    'msg': f'Priority search failed. Iteration {iteration + 1}.'
                })

        path_set_group = {}
        if is_success:
            cbs_result: dict[str, MapfPkg.PathSet] = result.paths
            group_names = list(cbs_result.keys())

            # ------ assemble result ------
            for group_name, path_set in cbs_result.items():
                path_set_group[group_name] = {}
                for path in path_set.paths:
                    path_set_group[group_name][path.name] = {
                        'path': np.array([[loc.x, loc.y, loc.z] for loc in path.path]).tolist(),
                        'radius': path.radius
                    }
            
            for group_name in group_names:
                group_graph = running_graph[group_name]

                # ------ flat result ------
                flat_locs = []
                for path in cbs_result[group_name].paths:
                    for loc in path.path:
                        flat_locs.append([loc.x, loc.y, loc.z, path.radius])

                # ------ update running graph ------
                current_task_manager = task_manager_map[group_name]
                new_related_components = []

                for priority in range(current_task_manager.get_task_size()):
                    task_cfg = current_task_manager.get_task(priority)
                    group_graph.add_edge(task_cfg.start_name, task_cfg.end_name)
                    new_related_components.extend([task_cfg.start_name, task_cfg.end_name])
                
                new_related_components = list(set(new_related_components))
                
                # ------ find connected components which related to current search task ------
                connected_components = nx.node_connected_component(group_graph, new_related_components[0])

                # ------ update last connective domain ------
                last_connective_domain = []
                for component in connected_components:
                    if component in new_related_components:
                        continue
                    for (x, y, z, radius) in running_cache[component]:
                        last_connective_domain.append([x, y, z, radius])
                    break
                
                for (x, y, z, radius) in flat_locs:
                    last_connective_domain.append([x, y, z, radius])
                
                # ------ sort by radius and drop duplicates ------
                last_connective_domain = pd.DataFrame(last_connective_domain, columns=['x', 'y', 'z', 'radius'])
                last_connective_domain.sort_values(by=['radius'], inplace=True, ascending=False)
                last_connective_domain.drop_duplicates(subset=['x', 'y', 'z'], inplace=True, keep='first')
                last_connective_domain = last_connective_domain.to_numpy().tolist()

                # ------ update running cache ------
                for component in connected_components:
                    running_cache[component] = last_connective_domain
        
        # # ------ debug ------
        # for group_name, group_graph in running_graph.items():
        #     PathSearchRunner.draw_graph(group_graph)
        # # ------ debug end ------

        data_queue.put({
            'is_finished': True,
            'data': {
                'is_success': is_success,
                'path_set_group': path_set_group,
            },
            'msg': 'Priority search success.' if is_success else 'Priority search failed.'
        })

    @staticmethod
    def draw_graph(graph:nx.Graph):
        nx.draw(graph, with_labels=True)
        plt.show()
