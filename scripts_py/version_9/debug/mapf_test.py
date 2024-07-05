import os
import h5py
import numpy as np
import pandas as pd
import pyvista
import matplotlib.pyplot as plt
import json

from build import mapf_pipeline
from scripts_py.version_9.mapf_pkg.cbs_utils import CbsSolver
from scripts_py.version_9.mapf_pkg.smooth_optimizer import PathOptimizer
from scripts_py.version_9.mapf_pkg.visual_utils import VisUtils
from scripts_py.version_9.mapf_pkg.pcd2mesh_utils import Pcd2MeshConverter


def debug_kdtree():
    pcd = np.array([
        [1., 0., 0.],
        [0., 2., 0.],
        [0., 0., 3.]
    ])

    tree = mapf_pipeline.KDtree_pcl()
    tree.update_data(pcd)
    tree.create_tree()
    tree.nearestKSearch(x=1., y=0., z=0., k=1)
    found_point = tree.get_point_from_data(tree.result_idxs_1D[0])
    print(f"Found point:({found_point.x}, {found_point.y}, {found_point.z}) distance:{tree.result_distance_1D[0]}")

    tree.clear_data()
    pcd = np.array([
        [0., 2., 0.],
        [0., 0., 3.],
        [10., 0., 0.]
    ])
    tree.update_data(pcd)
    tree.create_tree()
    tree.nearestKSearch(1., 0., 0., 1)
    found_point = tree.get_point_from_data(tree.result_idxs_1D[0])
    print(f"Found point:({found_point.x}, {found_point.y}, {found_point.z}) distance:{tree.result_distance_1D[0]}")


def debug_grid():
    grid_env = mapf_pipeline.DiscreteGridEnv(
        size_of_x=100, size_of_y=100, size_of_z=50,
        x_init=-10.0, y_init=-10.0, z_init=-25.0,
        x_grid_length=1.0, y_grid_length=1.0, z_grid_length=0.5
    )

    xyz = np.array([-5.0, -10.0, -24.5])
    grid0 = grid_env.xyz2grid(xyz[0], xyz[1], xyz[2])
    flag0 = grid_env.xyz2flag(xyz[0], xyz[1], xyz[2])
    rec_grid = grid_env.flag2grid(flag0)
    rec_xyz = grid_env.grid2xyz(rec_grid[0], rec_grid[1], rec_grid[2])
    print(f"xyz:{xyz} grid:{grid0} flag:{flag0} remap_grid:{rec_grid} remap_xyz:{rec_xyz}")

    candidate = []
    candidate.extend(mapf_pipeline.candidate_1D)
    candidate.extend(mapf_pipeline.candidate_2D)
    candidate.extend(mapf_pipeline.candidate_3D)

    edge_flag0 = grid_env.grid2flag(0, 0, 10)
    for flag in grid_env.get_valid_neighbors(edge_flag0, 2, candidate):
        print(f"valid neighbour: {grid_env.flag2grid(flag)}")


def debug_state_detector():
    grid_env = mapf_pipeline.DiscreteGridEnv(
        size_of_x=100, size_of_y=100, size_of_z=50,
        x_init=-10.0, y_init=-10.0, z_init=-20.0,
        x_grid_length=0.5, y_grid_length=0.5, z_grid_length=1.0
    )
    detector = mapf_pipeline.DynamicStepStateDetector(grid=grid_env)
    detector.update_dynamic_info(shrink_distance=2.0, scale=1)

    start_pipe = [
        {'grid': [5, 5, 10], 'direction': [1.0, 0.0, 0.0]},
        {'grid': [10, 10, 20], 'direction': [0.0, 1.0, 0.0]},
        {'grid': [5, 5, 15], 'direction': [0.0, 0.0, 1.0]},
    ]
    target_pipe = [
        {'grid': [30, 30, 25], 'direction': [1.0, 0.0, 0.0]},
        {'grid': [40, 40, 5], 'direction': [0.0, 1.0, 0.0]},
        {'grid': [5, 5, 45], 'direction': [0.0, 0.0, 1.0]},
        {'grid': [10, 10, 45], 'direction': [0.0, 0.0, 1.0]},
    ]
    for info in start_pipe:
        grid, direction = info['grid'], info['direction']
        flag = grid_env.grid2flag(x_grid=grid[0], y_grid=grid[1], z_grid=grid[2])
        detector.insert_start_flags(flag, direction[0], direction[1], direction[2], False)
    for info in target_pipe:
        grid, direction = info['grid'], info['direction']
        flag = grid_env.grid2flag(x_grid=grid[0], y_grid=grid[1], z_grid=grid[2])
        detector.insert_target_flags(flag, direction[0], direction[1], direction[2], False)

    print(f"start_pipe flags:{detector.get_start_pos_flags()} target_pipe flags:{detector.get_target_pos_flags()}")
    for flag in detector.get_start_pos_flags():
        direction = detector.get_start_info(flag)
        print(f"statr loc_flag:{flag} "
              f"grid:{grid_env.flag2grid(flag)} direction:({direction[0]}, {direction[1]}, {direction[2]}) "
              f"is_target:{detector.is_target(flag)}/false")

    target_xyzs = []
    for flag in detector.get_target_pos_flags():
        direction = detector.get_target_info(flag)
        target_xyzs.append(grid_env.flag2grid(flag))
        print(f"target loc_flag:{flag} "
              f"grid:{grid_env.flag2grid(flag)} direction:({direction[0]}, {direction[1]}, {direction[2]}) "
              f"is_target:{detector.is_target(flag)}/true")
    target_xyzs = np.array(target_xyzs)

    for flag in detector.get_target_pos_flags():
        grid_xyz = np.array(grid_env.flag2grid(flag), dtype=int)
        vec = np.random.randint(0, 2, size=(3,), dtype=int)
        scale = np.random.randint(0, 5, size=(1,), dtype=int)
        xyz0 = grid_xyz + vec * scale
        xyz0_flag = grid_env.grid2flag(xyz0[0], xyz0[1], xyz0[2])
        adjust_scale = detector.adjust_scale(xyz0_flag, 5)
        min_dist = np.min(np.linalg.norm(target_xyzs - np.array(xyz0), axis=1, ord=2))
        print(f"adjust_scale:{adjust_scale} xyz0:{xyz0} grid_xyz:{grid_xyz} dist:{min_dist}")


def debug_collision_detector():
    detector = mapf_pipeline.CollisionDetector()
    obstacle_pcd = np.array([
        [0., 0., 0., 1.],
        [5., 5., 0., 1.],
        [10., 10., 0., 1.],
        [20., 20., 0., 3.],
        [18., 20., 0., 0.1],
    ])
    detector.update_data(obstacle_pcd)
    detector.create_tree()

    print(f"should be false: {detector.is_valid(0., 0., 0.1, 1.)}")
    print(f"should be true: {detector.is_valid(2., 2., 0.0, 1.)}")
    print(f"should be false: {detector.is_valid(16.1, 20., 0., 1.)}")
    print(f"should be true: {detector.is_valid(15.999, 20., 0., 1.)}")
    print(f"should be false: {detector.is_valid(8, 9, 0, 10, 11, 0, 1.)}")


def debug_constraint_table():
    avoid_table = mapf_pipeline.ConflictAvoidTable()
    avoid_table.insert(1)
    avoid_table.insert(2)
    avoid_table.insert(3)
    avoid_table.insert(1)
    avoid_table.insert(1)
    avoid_table.insert(2)
    print(f"loc 1:{avoid_table.get_num_of_conflict(1)} "
          f"loc 2:{avoid_table.get_num_of_conflict(2)} "
          f"loc 3:{avoid_table.get_num_of_conflict(3)} "
          f"loc 4:{avoid_table.get_num_of_conflict(4)}")
    print(avoid_table.get_data())


def debug_astar_2D():
    grid_env = mapf_pipeline.DiscreteGridEnv(
        size_of_x=11, size_of_y=11, size_of_z=1,
        x_init=0.0, y_init=0.0, z_init=0.0,
        x_grid_length=1.0, y_grid_length=1.0, z_grid_length=1.0
    )
    obstacle_detector = mapf_pipeline.CollisionDetector()
    obstacle_detector.create_tree()

    dynamic_detector = mapf_pipeline.CollisionDetector()
    dynamic_detector.create_tree()

    state_detector = mapf_pipeline.detector = mapf_pipeline.DynamicStepStateDetector(grid=grid_env)
    state_detector.update_dynamic_info(shrink_distance=2.0, scale=1)

    avoid_table = mapf_pipeline.ConflictAvoidTable()

    start_pipe = [
        {'grid': [0, 0, 0], 'direction': [1.0, 0.0, 0.0]},
    ]
    target_pipe = [
        {'grid': [10, 3, 0], 'direction': [1.0, 0.0, 0.0]},
    ]
    for info in start_pipe:
        grid, direction = info['grid'], info['direction']
        flag = grid_env.grid2flag(x_grid=grid[0], y_grid=grid[1], z_grid=grid[2])
        state_detector.insert_start_flags(flag, direction[0], direction[1], direction[2], False)
    for info in target_pipe:
        grid, direction = info['grid'], info['direction']
        flag = grid_env.grid2flag(x_grid=grid[0], y_grid=grid[1], z_grid=grid[2])
        state_detector.insert_target_flags(flag, direction[0], direction[1], direction[2], False)

    candidate = []
    candidate.extend(mapf_pipeline.candidate_1D)
    candidate.extend(mapf_pipeline.candidate_2D)

    res_path = mapf_pipeline.PathResult(grid_env)
    solver = mapf_pipeline.StandardAStarSolver(grid_env, obstacle_detector, dynamic_detector, state_detector)
    solver.update_configuration(
        pipe_radius=1.0,
        search_step_scale=3,
        grid_expand_candidates=candidate,
        use_curvature_cost=True,
        curvature_cost_weight=5.0,
        use_avoid_table=True,
        use_theta_star=False,
    )
    is_success = solver.find_path(res_path, 200, avoid_table)
    print(f"state:{is_success} num_generate:{solver.num_generated} num_expand:{solver.num_expanded}")

    if is_success:
        print(f"info: radius:{res_path.get_radius()} "
              f"path_length:{res_path.get_length()} "
              f"timeCost:{solver.search_time_cost}")

        print('flags:', res_path.get_path_flags())
        print('radius:', res_path.get_radius())
        print('length:', res_path.get_length())
        print('step_length:', res_path.get_step_length())

        xyzr = np.array(res_path.get_path())
        xyz_list = xyzr[:, :3]
        print(xyzr)

        fig, ax = plt.subplots()
        ax.plot(xyz_list[:, 0], xyz_list[:, 1], '*-')
        ax.set_aspect('equal')
        plt.show()


def debug_orient_astar_2D():
    # 这个例子可以看出方法的不完备性，只能缓解问题

    grid_env = mapf_pipeline.DiscreteGridEnv(
        size_of_x=11, size_of_y=11, size_of_z=0,
        x_init=0.0, y_init=0.0, z_init=0.0,
        x_grid_length=1.0, y_grid_length=1.0, z_grid_length=1.0
    )
    obstacle_detector = mapf_pipeline.CollisionDetector()
    obstacle_detector.create_tree()

    dynamic_detector = mapf_pipeline.CollisionDetector()
    dynamic_detector.create_tree()

    state_detector = mapf_pipeline.detector = mapf_pipeline.DynamicStepStateDetector(grid=grid_env)
    state_detector.update_dynamic_info(shrink_distance=1.0, scale=1)

    avoid_table = mapf_pipeline.ConflictAvoidTable()

    start_pipe = [
        {'grid': [0, 0, 0], 'direction': [1.0, 0.0, 0.0]},
    ]
    target_pipe = [
        {'grid': [9, 3, 0], 'direction': [0.0, 1.0, 0.0]},
    ]
    for info in start_pipe:
        grid, direction = info['grid'], info['direction']
        flag = grid_env.grid2flag(x_grid=grid[0], y_grid=grid[1], z_grid=grid[2])
        state_detector.insert_start_flags(flag, direction[0], direction[1], direction[2], False)
    for info in target_pipe:
        grid, direction = info['grid'], info['direction']
        flag = grid_env.grid2flag(x_grid=grid[0], y_grid=grid[1], z_grid=grid[2])
        state_detector.insert_target_flags(flag, direction[0], direction[1], direction[2], False)

    candidate = []
    candidate.extend(mapf_pipeline.candidate_1D)
    candidate.extend(mapf_pipeline.candidate_2D)

    res_path = mapf_pipeline.PathResult(grid_env)
    solver = mapf_pipeline.StandardAStarSolver(grid_env, obstacle_detector, dynamic_detector, state_detector)
    solver.update_configuration(
        pipe_radius=1.0,
        search_step_scale=1,
        grid_expand_candidates=candidate,
        use_curvature_cost=True,
        curvature_cost_weight=1000.0,
        use_avoid_table=True,
        use_theta_star=False,
    )
    is_success = solver.find_path(res_path, 200, avoid_table)
    print(f"state:{is_success} num_generate:{solver.num_generated} num_expand:{solver.num_expanded}")

    if is_success:
        print(
            f"info: radius:{res_path.get_radius()} path_length:{res_path.get_length()} timeCost:{solver.search_time_cost}")

        xyz_list = []
        for flag in res_path.get_path_flags():
            xyz_list.append(grid_env.flag2xyz(flag))
        xyz_list = np.array(xyz_list)

        fig, ax = plt.subplots()
        ax.plot(xyz_list[:, 0], xyz_list[:, 1], '*-')
        ax.set_aspect('equal')
        plt.show()


def debug_group_astar_2D():
    grid_env = mapf_pipeline.DiscreteGridEnv(
        size_of_x=11, size_of_y=11, size_of_z=1,
        x_init=0.0, y_init=0.0, z_init=0.0,
        x_grid_length=1.0, y_grid_length=1.0, z_grid_length=1.0
    )
    obstacle_detector = mapf_pipeline.CollisionDetector()
    obstacle_detector.create_tree()
    dynamic_obstacles = []

    avoid_table = mapf_pipeline.ConflictAvoidTable()

    candidate = []
    candidate.extend(mapf_pipeline.candidate_1D)
    candidate.extend(mapf_pipeline.candidate_2D)

    task_list = [
        mapf_pipeline.TaskInfo(
            task_name='path_1',
            begin_tag='tag0', final_tag='tag1',
            begin_marks=[
                (grid_env.grid2flag(x_grid=0, y_grid=0, z_grid=1), 1, 0, 0)
            ],
            final_marks=[
                (grid_env.grid2flag(x_grid=10, y_grid=3, z_grid=1), 1, 0, 0)
            ],
            search_radius=0.5, step_scale=1, shrink_distance=1.0, shrink_scale=1, expand_grid_cell=candidate,
            with_curvature_cost=True, curvature_cost_weight=10.0,
            use_constraint_avoid_table=True, with_theta_star=False
        ),
        mapf_pipeline.TaskInfo(
            task_name='path_2',
            begin_tag='tag2', final_tag='tag3',
            begin_marks=[
                (grid_env.grid2flag(x_grid=0, y_grid=9, z_grid=1), 1, 0, 0)
            ],
            final_marks=[
                (grid_env.grid2flag(x_grid=10, y_grid=6, z_grid=1), 1, 0, 0)
            ],
            search_radius=0.5, step_scale=1, shrink_distance=1.0, shrink_scale=1, expand_grid_cell=candidate,
            with_curvature_cost=True, curvature_cost_weight=10.0,
            use_constraint_avoid_table=True, with_theta_star=False
        ),
        mapf_pipeline.TaskInfo(
            task_name='path_3',
            begin_tag='tag0', final_tag='tag2',
            begin_marks=[
                (grid_env.grid2flag(x_grid=0, y_grid=0, z_grid=1), 1, 0, 0)
            ],
            final_marks=[
                (grid_env.grid2flag(x_grid=0, y_grid=9, z_grid=1), 1, 0, 0)
            ],
            search_radius=0.5, step_scale=1, shrink_distance=1.0, shrink_scale=1, expand_grid_cell=candidate,
            with_curvature_cost=True, curvature_cost_weight=10.0,
            use_constraint_avoid_table=True, with_theta_star=False
        )
    ]

    solver = mapf_pipeline.GroupAstar(grid_env, obstacle_detector)
    solver.update_task_tree(task_list)

    for task_info in solver.get_task_tree():
        print(f"{task_info.task_name}")
        print(f"search_radius:{task_info.search_radius} "
              f"step_scale:{task_info.step_scale} "
              f"shrink_distance:{task_info.shrink_distance} "
              f"shrink_scale:{task_info.shrink_scale} "
              f"with_theta_star:{task_info.with_theta_star} "
              f"use_constraint_avoid_table:{task_info.use_constraint_avoid_table} "
              f"with_curvature_cost:{task_info.with_curvature_cost} "
              f"curvature_cost_weight:{task_info.curvature_cost_weight}\n")

    is_success = solver.find_path(dynamic_obstacles, max_iter=200, avoid_table=avoid_table)
    if is_success:
        path_res = solver.get_res()

        fig, ax = plt.subplots()
        for task_info in solver.get_task_tree():
            path = path_res[task_info.task_name]

            xyz_list = []
            for flag in path.get_path_flags():
                xyz_list.append(grid_env.flag2xyz(flag))
            xyz_list = np.array(xyz_list)

            ax.plot(xyz_list[:, 0], xyz_list[:, 1], '*-')
        ax.set_aspect('equal')
        plt.show()


def debug_cbs_node():
    obstacle_detector = mapf_pipeline.CollisionDetector()
    obstacle_detector.create_tree()

    candidate = []
    candidate.extend(mapf_pipeline.candidate_1D)
    candidate.extend(mapf_pipeline.candidate_2D)

    group_idx0 = 0
    grid_env0 = mapf_pipeline.DiscreteGridEnv(
        size_of_x=11, size_of_y=11, size_of_z=1,
        x_init=0.0, y_init=0.0, z_init=0.0,
        x_grid_length=1.0, y_grid_length=1.0, z_grid_length=1.0
    )
    dynamic_obstacles0 = []
    task_list0 = [
        mapf_pipeline.TaskInfo(
            task_name='path_1',
            begin_tag='tag0', final_tag='tag1',
            begin_marks=[
                (grid_env0.grid2flag(x_grid=0, y_grid=0, z_grid=1), 1, 0, 0)
            ],
            final_marks=[
                (grid_env0.grid2flag(x_grid=10, y_grid=3, z_grid=1), 1, 0, 0)
            ],
            search_radius=0.5, step_scale=1, shrink_distance=1.0, shrink_scale=1, expand_grid_cell=candidate,
            with_curvature_cost=True, curvature_cost_weight=10.0,
            use_constraint_avoid_table=True, with_theta_star=False
        ),
        mapf_pipeline.TaskInfo(
            task_name='path_2',
            begin_tag='tag0', final_tag='tag2',
            begin_marks=[
                (grid_env0.grid2flag(x_grid=0, y_grid=0, z_grid=1), 1, 0, 0)
            ],
            final_marks=[
                (grid_env0.grid2flag(x_grid=7, y_grid=10, z_grid=1), 1, 0, 0)
            ],
            search_radius=1.0, step_scale=1, shrink_distance=1.0, shrink_scale=1, expand_grid_cell=candidate,
            with_curvature_cost=True, curvature_cost_weight=10.0,
            use_constraint_avoid_table=True, with_theta_star=False
        )
    ]

    group_idx1 = 1
    grid_env1 = mapf_pipeline.DiscreteGridEnv(
        size_of_x=21, size_of_y=21, size_of_z=1,
        x_init=0.0, y_init=0.0, z_init=0.0,
        x_grid_length=0.5, y_grid_length=0.5, z_grid_length=1.0
    )
    dynamic_obstacles1 = []
    task_list1 = [
        mapf_pipeline.TaskInfo(
            task_name='path_1',
            begin_tag='tag0', final_tag='tag1',
            begin_marks=[
                (grid_env1.grid2flag(x_grid=0, y_grid=18, z_grid=1), 1, 0, 0)
            ],
            final_marks=[
                (grid_env1.grid2flag(x_grid=20, y_grid=12, z_grid=1), 1, 0, 0)
            ],
            search_radius=0.5, step_scale=1, shrink_distance=1.0, shrink_scale=1, expand_grid_cell=candidate,
            with_curvature_cost=True, curvature_cost_weight=10.0,
            use_constraint_avoid_table=True, with_theta_star=False,
        )
    ]

    cbs_node = mapf_pipeline.CbsNode(node_id=0)
    cbs_node.update_group_cell(
        group_idx=group_idx0, group_task_tree=task_list0, group_grid=grid_env0,
        obstacle_detector=obstacle_detector, group_dynamic_obstacles=dynamic_obstacles0
    )
    cbs_node.update_group_cell(
        group_idx=group_idx1, group_task_tree=task_list1, group_grid=grid_env1,
        obstacle_detector=obstacle_detector, group_dynamic_obstacles=dynamic_obstacles1
    )

    is_success0 = cbs_node.update_group_path(group_idx=group_idx0, max_iter=200)
    is_success1 = cbs_node.update_group_path(group_idx=group_idx1, max_iter=200)
    print(f"{group_idx0} is success:{is_success0}, {group_idx1} is success:{is_success1}")

    if is_success0 and is_success1:
        is_conflict = cbs_node.find_inner_conflict_point2point()
        print(f"is_conflict: {is_conflict}")

        for group_idx in [group_idx0, group_idx1]:
            print(f"{group_idx} conflict_length:{cbs_node.get_conflict_length(group_idx=group_idx)}")

        for conflict_cell in cbs_node.get_conflict_cells():
            print(f"idx0:{conflict_cell.idx0} x0:{conflict_cell.x0} y0:{conflict_cell.y0} "
                  f"z0:{conflict_cell.z0} radius0:{conflict_cell.radius0} "
                  f"idx1:{conflict_cell.idx1} x1:{conflict_cell.x1} y1:{conflict_cell.y1} "
                  f"z1:{conflict_cell.z1} radius1:{conflict_cell.radius1}")

        fig, ax = plt.subplots()
        for group_idx, task_list, grid_env in [
            (group_idx0, task_list0, grid_env0),
            (group_idx1, task_list1, grid_env1)
        ]:
            for task_info in task_list:
                path = cbs_node.get_group_path(group_idx=group_idx, name=task_info.task_name)

                xyz_list = []
                for flag in path.get_path_flags():
                    xyz_list.append(grid_env.flag2xyz(flag))
                xyz_list = np.array(xyz_list)
                ax.plot(xyz_list[:, 0], xyz_list[:, 1], '*-')

        ax.set_aspect('equal')
        plt.show()


def debug_cbs_solver(algorithm_json: str):
    with open(algorithm_json, 'r') as f:
        setup_json = json.load(f)

    grid_cfg = setup_json['grid_env']
    pipe_cfg = setup_json['pipes']

    aux_constrain_df = None
    if grid_cfg.get('dynamic_constrain_file', False):
        aux_constrain_df = pd.read_csv(grid_cfg['dynamic_constrain_file'], index_col=0)

    block_res_list = []
    last_leaf_info = {}

    for block_info in setup_json['search_tree']['block_sequence']:
        solver = CbsSolver(grid_cfg, pipe_cfg)
        root = solver.init_block_root(
            block_info['groups'], last_leafs_info=last_leaf_info, aux_constrain_df=aux_constrain_df
        )

        group_idxs = list(solver.task_infos.keys())
        is_success, res_node = solver.solve_block(root, group_idxs, max_iter=100000, max_node_limit=2000)

        if is_success:
            CbsSolver.draw_node_3D(
                res_node, group_idxs=group_idxs, task_infos=solver.task_infos, obstacle_df=solver.obstacle_df,
                pipe_cfg=solver.pipe_cfg
            )

            last_leaf_info = solver.convert_node_to_leaf_info(res_node, group_idxs, last_leaf_info)
            block_res_list.append(solver.save_path(res_node, group_idxs, solver.task_infos))

        else:
            print(f"[INFO]: False at solving {block_info['block_id']}")
            break


def debug_sequence_solve(algorithm_json: str):
    with open(algorithm_json, 'r') as f:
        setup_json = json.load(f)

    grid_cfg = setup_json['grid_env']
    pipe_cfg = setup_json['pipes']

    aux_constrain_df = None
    if grid_cfg.get('dynamic_constrain_file', False):
        aux_constrain_df = pd.read_csv(grid_cfg['dynamic_constrain_file'], index_col=0)

    block_res_list = []
    last_leaf_info = {}

    for block_info in setup_json['search_tree']['block_sequence']:
        solver = CbsSolver(grid_cfg, pipe_cfg)
        root = solver.init_block_root(
            block_info['groups'], last_leafs_info=last_leaf_info, aux_constrain_df=aux_constrain_df
        )

        # ------ debug vis
        high_group_idx = block_info['groups'][0]['group_idx']
        CbsSolver.draw_node_3D(
            root, group_idxs=[], task_infos=solver.task_infos, obstacle_df=solver.obstacle_df,
            pipe_cfg=solver.pipe_cfg, highlight_group_idx=high_group_idx
        )
        # ------

        group_idxs = list(solver.task_infos.keys())
        is_success, res_node = solver.solve_block(root, group_idxs, max_iter=100000, max_node_limit=2000)

        if is_success:
            last_leaf_info = solver.convert_node_to_leaf_info(res_node, group_idxs, last_leaf_info)
            block_res_list.append(solver.save_path(res_node, group_idxs, solver.task_infos))

        else:
            print(f"[INFO]: False at solving {block_info['block_id']}")
            break


def debug_cbs_first_check(algorithm_json: str):
    with open(algorithm_json, 'r') as f:
        setup_json = json.load(f)

    grid_cfg = setup_json['grid_env']
    pipe_cfg = setup_json['pipes']

    aux_constrain_df = None
    if grid_cfg.get('dynamic_constrain_file', False):
        aux_constrain_df = pd.read_csv(grid_cfg['dynamic_constrain_file'], index_col=0)

    last_leaf_info_list = [{}]
    for block_info in setup_json['search_tree']['block_sequence']:
        solver = CbsSolver(grid_cfg, pipe_cfg)
        root = solver.init_block_root(
            block_info['groups'], last_leafs_info=last_leaf_info_list[-1], aux_constrain_df=aux_constrain_df
        )
        group_idxs = list(solver.task_infos.keys())
        solver.first_check(root, group_idxs, max_iter=100000)


def debug_smooth_optimizer(algo_file: str, smooth_file: str, res_file: str):
    with open(smooth_file, 'r') as f:
        smooth_json = json.load(f)
    with open(algo_file, 'r') as f:
        algo_json = json.load(f)

    path_res = {}
    for _, res_cell in h5py.File(res_file).items():
        group_idx = int(res_cell['group_idx'][0])
        path_res[group_idx] = {}
        for name, xyzr in res_cell['path_result'].items():
            path_res[group_idx][name] = xyzr

    grid_cfg = algo_json['grid_env']
    pipe_cfg = algo_json['pipes']
    obs_xyzr = pd.read_csv(grid_cfg['obstacle_file'], index_col=0)[['x', 'y', 'z', 'radius']].values

    opt = PathOptimizer(grid_cfg, path_res, obs_xyzr, smooth_json['conflict_setting'])

    # ------ step 1: add path
    for path_info in smooth_json['path_list']:
        begin_name, end_name = path_info['begin_name'], path_info['end_name']
        group_idx = pipe_cfg[begin_name]['group_idx']
        opt.network_cells[group_idx].add_path(
            name=path_info['name'],
            begin_xyz=np.array(pipe_cfg[begin_name]['discrete_position']),
            begin_vec=np.array(pipe_cfg[begin_name]['direction']),
            end_xyz=np.array(pipe_cfg[end_name]['discrete_position']),
            end_vec=np.array(pipe_cfg[end_name]['direction'])
        )

    # ------ step 2: refit graph
    for setting in smooth_json['setting']:
        opt.network_cells[setting['group_idx']].refit_graph(setting['segment_degree'])

    # ------ step 3: update optimization info
    for setting in smooth_json['setting']:
        for key in list(setting['segments'].keys()):
            cell = setting['segments'].pop(key)
            setting['segments'][cell['seg_idx']] = cell

        group_idx = setting['group_idx']
        opt.network_cells[group_idx].update_optimization_info(
            segment_infos=setting['segments'],
            path_infos=setting['paths_cost']
        )

    # vis = VisUtils()
    # for group_idx, net_cell in opt.network_cells.items():
    #     net_cell.draw_segment(with_spline=True, with_control=True, vis=vis)
    # vis.show()

    # ------ step 4: prepare tensor
    for group_id, net_cell in opt.network_cells.items():
        net_cell.prepare_tensor()

    # for group_idx, net_cell in opt.network_cells.items():
    #     net_cell.draw_path_tensor()

    # max_lr = np.linalg.norm(np.array([0.1, 0.1, 0.1]))
    # opt.find_obstacle_conflict_cells(max_lr)
    # opt.find_path_conflict_cells(max_lr)
    # opt.draw_conflict_graph(with_obstacle=True, with_path_conflict=True)

    opt.run(max_iter=10, lr=0.1)
    vis = VisUtils()
    for group_idx, net_cell in opt.network_cells.items():
        net_cell.draw_segment(with_spline=False, with_control=True, vis=vis)
    vis.show()


def debug_pcd2mesh(algo_file, smooth_res_file):
    with open(algo_file, 'r') as f:
        algo_json = json.load(f)
    pipe_cfg = algo_json['pipes']

    reso = 0.2
    group_mesher = {}

    for _, group_cell in h5py.File(smooth_res_file).items():
        group_idx = group_cell['group_idx'][0]
        group_mesher[group_idx] = Pcd2MeshConverter()

        for i, (seg_name, xyzr) in enumerate(group_cell['spline'].items()):
            xyzr = np.array(xyzr)
            path = xyzr[:, :3]
            radius = xyzr[0, -1]
            left_direction, right_direction = None, None
            with_left_clamp, with_right_clamp = False, False

            for pipe_name, pipe_info in pipe_cfg.items():
                pipe_xyz = np.array(pipe_info['discrete_position'])
                pipe_direction = np.array(pipe_info['direction'])

                if np.all(np.isclose(path[0] - pipe_xyz, 0.0)):
                    if pipe_info['is_input']:
                        path = np.concatenate([(path[0] - pipe_direction).reshape((1, -1)), path], axis=0)
                    else:
                        path = np.concatenate([(path[0] + pipe_direction).reshape((1, -1)), path], axis=0)
                    left_direction = pipe_direction
                    with_left_clamp = True

                if np.all(np.isclose(path[-1] - pipe_xyz, 0.0)):
                    if pipe_info['is_input']:
                        path = np.concatenate([path, (path[-1] - pipe_direction).reshape((1, -1))], axis=0)
                    else:
                        path = np.concatenate([path, (path[-1] + pipe_direction).reshape((1, -1))], axis=0)
                    right_direction = pipe_direction
                    with_right_clamp = True

            group_mesher[group_idx].add_segment(
                seg_name, path, radius,
                left_direction=left_direction, right_direction=right_direction,
                with_left_clamp=with_left_clamp, with_right_clamp=with_right_clamp,
                reso_info={
                    'length_reso': 0.2,
                    'sphere_reso': 0.15,
                    'relax_factor': 1e-3
                }
            )

        # # ------ debug vis
        # for seg_name, info in group_mesher[group_idx].segment_cell.items():
        #     info['surface_cell'].generate_pcd_by_sphere(
        #         length_reso=info['reso_info']['length_reso'],
        #         sphere_reso=info['reso_info']['sphere_reso'],
        #         relax_factor=info['reso_info']['relax_factor']
        #     )
        #     info['surface_cell'].draw()
        # # ------

        group_mesher[group_idx].generate_pcd_data()
        group_mesher[group_idx].remove_inner_pcd()
        pcd_data = group_mesher[group_idx].get_pcd_data()

        # ------ debug point cloud
        debug_dir = '/home/admin123456/Desktop/work/path_examples/s5/debug'
        pcd_ply = pyvista.PolyData(pcd_data)
        pcd_ply.plot()
        Pcd2MeshConverter.save_ply(pcd_ply, file=os.path.join(debug_dir, f"group_{group_idx}.ply"))
        # ------


if __name__ == '__main__':
    # debug_kdtree()
    # debug_grid()
    # debug_state_detector()
    # debug_collision_detector()
    # debug_constraint_table()
    # debug_astar_2D()
    # debug_orient_astar_2D()
    # debug_group_astar_2D()
    # debug_cbs_node()

    # debug_cbs_solver('/home/admin123456/Desktop/work/path_examples/s5/algorithm_setup_orig.json')
    # debug_cbs_first_check('/home/admin123456/Desktop/work/path_examples/s5/algorithm_setup_orig.json')
    # debug_sequence_solve('/home/admin123456/Desktop/work/path_examples/s5/algorithm_setup_orig.json')

    # debug_smooth_optimizer(
    #     algo_file='/home/admin123456/Desktop/work/path_examples/s5/algorithm_setup.json',
    #     smooth_file='/home/admin123456/Desktop/work/path_examples/s5/smooth_setup.json',
    #     res_file='/home/admin123456/Desktop/work/path_examples/s5/search_result.hdf5'
    # )

    debug_pcd2mesh(
        algo_file='/home/admin123456/Desktop/work/path_examples/s5/algorithm_setup.json',
        smooth_res_file='/home/admin123456/Desktop/work/path_examples/s5/smooth_result.hdf5'
    )

    pass
