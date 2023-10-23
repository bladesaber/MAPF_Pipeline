import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from scripts_py.version_9.springer.spring_smoother_v3 import Shape_Utils
from scripts_py.version_9.springer.spring_smoother_v3 import EnvParser
from scripts_py.version_9.springer.spring_smoother_v3 import OptimizerScipy
from scripts_py.version_9.springer.spring_smoother_v3 import BSpline_utils
from scripts_py.version_9.springer.spring_smoother_v3 import VarTree
from scripts_py.version_9.springer.smoother_utils import RecordAnysisVisulizer


def env1_test():
    radius = 0.3

    # ------
    p0_pose = np.array([1.5, 1.5, 0.5])
    p0_shape_pcd = Shape_Utils.create_BoxPcd(1.0, 1.0, 0.0, 2.0, 2.0, 1.0, reso=0.2)
    connector0_pose = np.array([2.0, 1.5, 0.5])
    p0_shape_pcd = Shape_Utils.removePointInSphereShell(
        p0_shape_pcd, center=connector0_pose, radius=radius, with_bound=True,
        direction=np.array([1., 0., 0.]), reso=0.2, scale_dist=0.15
    )
    p0_shape_pcd = p0_shape_pcd - p0_pose

    # ------
    p1_pose = np.array([5.5, 4.5, 0.5])
    connector1_pose = np.array([5.0, 4.5, 0.5])
    p1_shape_pcd = Shape_Utils.create_BoxPcd(5.0, 4.0, 0.0, 6.0, 5.0, 1.0, reso=0.2)
    p1_shape_pcd = Shape_Utils.removePointInSphereShell(
        p1_shape_pcd, center=connector1_pose, radius=radius, with_bound=True,
        direction=np.array([1., 0., 0.]), reso=0.2, scale_dist=0.15
    )
    p1_shape_pcd = p1_shape_pcd - p1_pose

    pipes_cfg = [
        {
            'node_type': 'connector', 'name': 'connector0', 'position': connector0_pose,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
            'exclude_edges': []
        },
        {
            'node_type': 'connector', 'name': 'connector1', 'position': connector1_pose,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'p1', 'value': -0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'p1', 'value': 0.0},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'p1', 'value': 0.0},
            'exclude_edges': []
        }
    ]

    structors_cfg = [
        {
            'node_type': 'structor', 'name': 'p0', 'position': p0_pose, 'shape_pcd': p0_shape_pcd, 'reso': 0.2,
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'},
            'exclude_edges': {
                'plane_min_conflict': ['x', 'y', 'z'],
                'plane_max_conflict': ['x', 'y', 'z']
            }
        },
        {
            'node_type': 'structor', 'name': 'p1', 'position': p1_pose, 'shape_pcd': p1_shape_pcd, 'reso': 0.2,
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'},
            'exclude_edges': {
                'plane_min_conflict': ['x', 'y', 'z'],
                'plane_max_conflict': ['x', 'y', 'z']
            }
        }
    ]

    bgs_cfg = [
        {
            'node_type': 'cell', 'name': 'planeMin', 'position': np.array([-2.0, -2.0, -2.0]),
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'}
        },
        {
            'node_type': 'cell', 'name': 'planeMax', 'position': np.array([7.0, 7.0, 1.0]),
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'}
        },
    ]

    paths_cfg = {
        0: {
            'name': 'path_01',
            'src_name': 'connector0',
            'end_name': 'connector1',
            'start_vec': np.array([1., 0., 0.]),
            'end_vec': np.array([1., 0., 0.]),
            'radius': radius,
            'xyzs': np.array([
                [2.0, 1.5, 0.5],

                [2.5, 1.5, 0.5],
                [3.0, 1.5, 0.5],
                [3.0, 1.0, 0.5],
                [3.0, 0.5, 0.5],
                [3.5, 0.5, 0.5],
                [4.0, 0.5, 0.5],
                [4.0, 1.0, 0.5],
                [4.0, 1.5, 0.5],
                [4.0, 2.0, 0.5],
                [4.0, 2.5, 0.5],
                [4.0, 3.0, 0.5],
                [3.5, 3.0, 0.5],
                [3.0, 3.0, 0.5],
                [2.5, 3.0, 0.5],
                [2.0, 3.0, 0.5],
                [1.5, 3.0, 0.5],
                [1.0, 3.0, 0.5],
                [0.5, 3.0, 0.5],
                [0.5, 3.5, 0.5],
                [0.5, 4.0, 0.5],
                [0.5, 4.5, 0.5],
                [0.5, 5.0, 0.5],
                [0.5, 5.5, 0.5],
                [0.5, 6.0, 0.5],
                [1.0, 6.0, 0.5],
                [1.0, 5.5, 0.5],
                [1.0, 5.0, 0.5],
                [1.0, 4.5, 0.5],
                [1.5, 4.5, 0.5],
                [2.0, 4.5, 0.5],
                [2.5, 4.5, 0.5],
                [3.0, 4.5, 0.5],
                [3.5, 4.5, 0.5],
                [4.0, 4.5, 0.5],
                [4.5, 4.5, 0.5],

                [5.0, 4.5, 0.5]
            ])
        }
    }

    parser = EnvParser()

    """ # ------ debug
    var_tree, paths_cfg = parser.create_graph(pipes_cfg, structors_cfg, bgs_cfg, paths_cfg, k=2)
    # parser.plot_env(
    #     xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=None, paths_cfg=paths_cfg, with_path=True,
    #     with_structor=True, with_bound=True, with_constraint=False
    # )

    constraints = parser.get_constraints(
        xs=var_tree.get_xs_init(), var_tree=var_tree, paths_cfg=paths_cfg, opt_step=0.1
    )
    parser.plot_env(
        xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=constraints, paths_cfg=paths_cfg,
        with_path=True, with_structor=True, with_bound=True, with_constraint=True
    )
    """

    """ # ------ debug
    path_cfg = paths_cfg[0]
    cmat = path_cfg['xyzs']
    path = BSpline_utils.sample_uniform(cmat, k=3, sample_num=80, point_num=400)
    real_res, fake_res = BSpline_utils.compare_curvature(
        path, path_cfg['start_vec'].reshape((1, -1)), path_cfg['end_vec'].reshape((1, -1))
    )
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0.scatter(path[:, 0], path[:, 1], s=real_res * 5.0 + 1.0)
    ax1.scatter(path[:, 0], path[:, 1], s=fake_res * 5.0 + 1.0)
    plt.show()
    """

    """
    path_cfg = paths_cfg[0]
    cmat = path_cfg['xyzs']
    path = BSpline_utils.sample_uniform(cmat, k=3, sample_num=80, point_num=300)
    dists = np.linalg.norm(path[1:, :] - path[:-1, :], ord=2, axis=1)
    print(f"max_dist:{np.max(dists)} min_dist:{np.min(dists)} "
          f"max_dif:{np.max(dists) - np.min(dists)} percent:{(np.max(dists) - np.min(dists)) / np.mean(dists)}")
    plt.plot(path[:, 0], path[:, 1], '-*')
    plt.show()
    """

    opt_step = 0.1
    k = 3

    while True:
        var_tree, paths_cfg = parser.create_graph(pipes_cfg, structors_cfg, bgs_cfg, paths_cfg, k=k)

        constraints = parser.get_constraints(
            xs=var_tree.get_xs_init(), var_tree=var_tree, paths_cfg=paths_cfg, opt_step=0.1
        )
        parser.plot_env(
            xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=constraints, paths_cfg=paths_cfg,
            with_path=True, with_structor=True, with_bound=True, with_constraint=True, with_control_points=True
        )
        # for path_idx in paths_cfg:
        #     path_cfg = paths_cfg[path_idx]
        #     path = path_cfg['b_mat'].dot(path_cfg['xyzs'])
        #     real_res, fake_res = BSpline_utils.compare_curvature(
        #         path, path_cfg['start_vec'], path_cfg['end_vec']
        #     )
        #     fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        #     ax0.scatter(path[:, 0], path[:, 1], s=real_res * 5.0 + 1.0)
        #     ax1.scatter(path[:, 0], path[:, 1], s=fake_res * 5.0 + 1.0)
        #     plt.show()

        optimizer = OptimizerScipy()

        xs_init = var_tree.get_xs_init()
        opt_xs, cur_cost = optimizer.solve_problem(xs_init, var_tree, paths_cfg, constraints, opt_step)

        xs_init += opt_xs

        nodes_info = var_tree.nodes_info
        name_to_nodeIdx = var_tree.name_to_nodeIdx
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]
            src_node = nodes_info[path_cfg['src_node_idx']]
            src_x, src_y, src_z = VarTree.get_xyz_from_node(src_node, xs_init, name_to_nodeIdx, nodes_info, 1)
            end_node = nodes_info[path_cfg['end_node_idx']]
            end_x, end_y, end_z = VarTree.get_xyz_from_node(end_node, xs_init, name_to_nodeIdx, nodes_info, 1)
            cell_xyzs = xs_init[path_cfg['path_cols']].reshape((-1, 3))
            c_mat = np.concatenate(([[src_x, src_y, src_z]], cell_xyzs, [[end_x, end_y, end_z]]), axis=0)
            new_cmat = BSpline_utils.sample_uniform(c_mat, k=k, sample_num=path_cfg['b_mat'].shape[0], point_num=200)
            path_cfg.update({'xyzs': new_cmat})


def env2_test():
    radius = 0.3

    # ------
    p0_pose = np.array([1.5, 1.5, 0.5])
    p0_shape_pcd = Shape_Utils.create_BoxPcd(1.0, 1.0, 0.0, 2.0, 2.0, 1.0, reso=0.2)
    connector0_pose = np.array([2.0, 1.5, 0.5])
    p0_shape_pcd = Shape_Utils.removePointInSphereShell(
        p0_shape_pcd, center=connector0_pose, radius=radius, with_bound=True,
        direction=np.array([1., 0., 0.]), reso=0.2, scale_dist=0.15
    )
    p0_shape_pcd = p0_shape_pcd - p0_pose

    # ------
    p1_pose = np.array([5.5, 4.5, 0.5])
    connector1_pose = np.array([5.0, 4.5, 0.5])
    p1_shape_pcd = Shape_Utils.create_BoxPcd(5.0, 4.0, 0.0, 6.0, 5.0, 1.0, reso=0.2)
    p1_shape_pcd = Shape_Utils.removePointInSphereShell(
        p1_shape_pcd, center=connector1_pose, radius=radius, with_bound=True,
        direction=np.array([1., 0., 0.]), reso=0.2, scale_dist=0.15
    )
    p1_shape_pcd = p1_shape_pcd - p1_pose

    pipes_cfg = [
        {
            'node_type': 'connector', 'name': 'connector0', 'position': connector0_pose,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'p0', 'value': 0.0},
            'exclude_edges': []
        },
        {
            'node_type': 'connector', 'name': 'connector1', 'position': connector1_pose,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'p1', 'value': -0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'p1', 'value': 0.0},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'p1', 'value': 0.0},
            'exclude_edges': []
        }
    ]

    structors_cfg = [
        {
            'node_type': 'structor', 'name': 'p0', 'position': p0_pose, 'shape_pcd': p0_shape_pcd, 'shape_reso': 0.2,
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'},
            'exclude_edges': {
                'plane_min_conflict': ['x', 'y', 'z'],
                'plane_max_conflict': ['z']
            }
        },
        {
            'node_type': 'structor', 'name': 'p1', 'position': p1_pose, 'shape_pcd': p1_shape_pcd, 'shape_reso': 0.2,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'planeMax', 'value': -0.5},
            'pose_edge_y': {'type': 'var_value'},
            'pose_edge_z': {'type': 'fix_value'},
            'exclude_edges': {
                'plane_min_conflict': ['x', 'y', 'z'],
                'plane_max_conflict': ['x', 'z']
            }
        }
    ]

    bgs_cfg = [
        {
            'node_type': 'cell', 'name': 'planeMin', 'position': np.array([0.0, 0.0, 0.0]),
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'}
        },
        {
            'node_type': 'cell', 'name': 'planeMax', 'position': np.array([6.0, 7.0, 1.0]),
            'pose_edge_x': {'type': 'var_value'},
            'pose_edge_y': {'type': 'var_value'},
            'pose_edge_z': {'type': 'fix_value'}
        },
    ]

    paths_cfg = {
        0: {
            'name': 'path_01',
            'src_name': 'connector0',
            'end_name': 'connector1',
            'start_vec': np.array([1., 0., 0.]),
            'end_vec': np.array([1., 0., 0.]),
            'radius': radius,
            'xyzs': np.array([
                [2.0, 1.5, 0.5],

                [2.5, 1.5, 0.5],
                [3.0, 1.5, 0.5],
                [3.0, 1.0, 0.5],
                [3.0, 0.5, 0.5],
                [3.5, 0.5, 0.5],
                [4.0, 0.5, 0.5],
                [4.0, 1.0, 0.5],
                [4.0, 1.5, 0.5],
                [4.0, 2.0, 0.5],
                [4.0, 2.5, 0.5],
                [4.0, 3.0, 0.5],
                [3.5, 3.0, 0.5],
                [3.0, 3.0, 0.5],
                [2.5, 3.0, 0.5],
                [2.0, 3.0, 0.5],
                [1.5, 3.0, 0.5],
                [1.0, 3.0, 0.5],
                [0.5, 3.0, 0.5],
                [0.5, 3.5, 0.5],
                [0.5, 4.0, 0.5],
                [0.5, 4.5, 0.5],
                [0.5, 5.0, 0.5],
                [0.5, 5.5, 0.5],
                [0.5, 6.0, 0.5],
                [1.0, 6.0, 0.5],
                [1.0, 5.5, 0.5],
                [1.0, 5.0, 0.5],
                [1.0, 4.5, 0.5],
                [1.5, 4.5, 0.5],
                [2.0, 4.5, 0.5],
                [2.5, 4.5, 0.5],
                [3.0, 4.5, 0.5],
                [3.5, 4.5, 0.5],
                [4.0, 4.5, 0.5],
                [4.5, 4.5, 0.5],

                [5.0, 4.5, 0.5]
            ])
        }
    }

    parser = EnvParser()

    """ # ------ debug
    var_tree, paths_cfg = parser.create_graph(pipes_cfg, structors_cfg, bgs_cfg, paths_cfg, k=2)
    # parser.plot_env(
    #     xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=None, paths_cfg=paths_cfg, with_path=True,
    #     with_structor=True, with_bound=True, with_constraint=False
    # )

    constraints = parser.get_constraints(
        xs=var_tree.get_xs_init(), var_tree=var_tree, paths_cfg=paths_cfg, opt_step=0.1
    )
    parser.plot_env(
        xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=constraints, paths_cfg=paths_cfg,
        with_path=True, with_structor=True, with_bound=True, with_constraint=True
    )
    """

    """ # ------ debug
    path_cfg = paths_cfg[0]
    cmat = path_cfg['xyzs']
    path = BSpline_utils.sample_uniform(cmat, k=3, sample_num=80, point_num=400)
    real_res, fake_res = BSpline_utils.compare_curvature(
        path, path_cfg['start_vec'].reshape((1, -1)), path_cfg['end_vec'].reshape((1, -1))
    )
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    ax0.scatter(path[:, 0], path[:, 1], s=real_res * 5.0 + 1.0)
    ax1.scatter(path[:, 0], path[:, 1], s=fake_res * 5.0 + 1.0)
    plt.show()
    """

    """
    path_cfg = paths_cfg[0]
    cmat = path_cfg['xyzs']
    path = BSpline_utils.sample_uniform(cmat, k=3, sample_num=80, point_num=300)
    dists = np.linalg.norm(path[1:, :] - path[:-1, :], ord=2, axis=1)
    print(f"max_dist:{np.max(dists)} min_dist:{np.min(dists)} "
          f"max_dif:{np.max(dists) - np.min(dists)} percent:{(np.max(dists) - np.min(dists)) / np.mean(dists)}")
    plt.plot(path[:, 0], path[:, 1], '-*')
    plt.show()
    """

    opt_step = 0.1
    k = 3

    record_vis_up = RecordAnysisVisulizer()
    record_vis_up.record_video_init(file='/home/admin123456/Desktop/work/test_dir/up_test.mp4')

    run_steps = 200
    run_angle = np.pi * 3.0
    step_angle = run_angle / run_steps

    for step in range(run_steps):
        var_tree, paths_cfg = parser.create_graph(
            pipes_cfg, structors_cfg, bgs_cfg, paths_cfg, k=k, length_tol=0.02, sample_tol=0.02
        )

        constraints = parser.get_constraints(
            xs=var_tree.get_xs_init(), var_tree=var_tree, paths_cfg=paths_cfg, opt_step=0.1
        )
        # parser.plot_env(
        #     xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=constraints, paths_cfg=paths_cfg,
        #     with_path=True, with_structor=True, with_bound=True, with_constraint=True, with_control_points=True,
        #     with_tube=True
        # )

        # for path_idx in paths_cfg:
        #     path_cfg = paths_cfg[path_idx]
        #     path = path_cfg['b_mat'].dot(path_cfg['xyzs'])
        #     real_res, fake_res = BSpline_utils.compare_curvature(
        #         path, path_cfg['start_vec'], path_cfg['end_vec']
        #     )
        #     fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        #     ax0.scatter(path[:, 0], path[:, 1], s=real_res * 5.0 + 1.0)
        #     ax1.scatter(path[:, 0], path[:, 1], s=fake_res * 5.0 + 1.0)
        #     plt.show()

        # -------
        optimizer = OptimizerScipy()

        xs_init = var_tree.get_xs_init()
        opt_xs, cur_cost = optimizer.solve_problem(xs_init, var_tree, paths_cfg, constraints, opt_step)
        xs_init += opt_xs

        nodes_info = var_tree.nodes_info
        name_to_nodeIdx = var_tree.name_to_nodeIdx
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]
            src_node = nodes_info[path_cfg['src_node_idx']]
            src_x, src_y, src_z = VarTree.get_xyz_from_node(src_node, xs_init, name_to_nodeIdx, nodes_info, 1)
            end_node = nodes_info[path_cfg['end_node_idx']]
            end_x, end_y, end_z = VarTree.get_xyz_from_node(end_node, xs_init, name_to_nodeIdx, nodes_info, 1)
            cell_xyzs = xs_init[path_cfg['path_cols']].reshape((-1, 3))
            c_mat = np.concatenate(([[src_x, src_y, src_z]], cell_xyzs, [[end_x, end_y, end_z]]), axis=0)
            new_cmat = BSpline_utils.sample_uniform(c_mat, k=k, sample_num=path_cfg['b_mat'].shape[0], point_num=200)
            path_cfg.update({'xyzs': new_cmat})

        for cfg in pipes_cfg:
            parser.update_node_info(cfg, xs_init, nodes_info, name_to_nodeIdx)
        for cfg in structors_cfg:
            parser.update_node_info(cfg, xs_init, nodes_info, name_to_nodeIdx)
        for cfg in bgs_cfg:
            parser.update_node_info(cfg, xs_init, nodes_info, name_to_nodeIdx)

        node0_tem = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
        xyz_max = VarTree.get_xyz_from_node(node0_tem, xs_init, name_to_nodeIdx, nodes_info, 1)
        node1_tem = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
        xyz_min = VarTree.get_xyz_from_node(node1_tem, xs_init, name_to_nodeIdx, nodes_info, 1)

        parser.record_video(
            record_vis_up, xs=xs_init, var_tree=var_tree, paths_cfg=paths_cfg,
            with_path=False, with_structor=True, with_bound=True, with_control_points=True, with_tube=True
        )
        obj_center = (np.array(xyz_max) + np.array(xyz_min)) / 2.0
        camera_pose, focal_pose = record_vis_up.get_camera_pose(
            obj_center, focal_length=8.0, radius=12.0, angle=step * step_angle, height=10.0
        )
        record_vis_up.set_camera(camera_pose, focal_pose, 1.0)
        record_vis_up.write_frame()
        # record_vis_up.show(auto_close=False)

        print(f"[Info]: Run Step:{step} Boundary: {list(xyz_min)} --- {list(xyz_max)}")
        print()


def env3_test():
    import json

    path = '/home/admin123456/Desktop/work/example6/grid_springer_env_cfg.json'
    with open(path, 'r') as f:
        env_cfg = json.load(f)

    pipes_cfg = []
    pipes = env_cfg['pipe_cfgs']
    pipe_name_to_info = {}
    for group_idx_str in pipes:
        group_pipes = pipes[group_idx_str]
        for name in group_pipes.keys():
            pipe_cfg = group_pipes[name]
            pipes_cfg.append(pipe_cfg)
            pipe_name_to_info[name] = pipe_cfg

    obstacle_df = pd.read_csv(env_cfg['obstacle_path'], index_col=0)
    obstacle_df.drop_duplicates(inplace=True)

    structors_cfg = []
    structors = env_cfg['obstacle_cfgs']
    for name in structors.keys():
        if name == 'wall':
            continue
        cfg = structors[name]

        pcd_world = obstacle_df[obstacle_df['tag'] == name][['x', 'y', 'z']].values
        position = np.array(cfg['position'])
        shape_pcd = pcd_world - position
        cfg.update({
            "shape_pcd": shape_pcd
        })
        structors_cfg.append(cfg)

    bgs_cfg = [
        {
            'node_type': 'cell', 'name': 'planeMin', 'position': np.array([0.0, 0.0, 0.0]),
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'}
        },
        {
            'node_type': 'cell', 'name': 'planeMax', 'position': np.array([
                env_cfg['global_params']['envScaleX'],
                env_cfg['global_params']['envScaleY'],
                env_cfg['global_params']['envScaleZ']
            ]),
            'pose_edge_x': {'type': 'var_value'},
            'pose_edge_y': {'type': 'var_value'},
            'pose_edge_z': {'type': 'var_value'}
        },
    ]

    paths_npy = np.load('/home/admin123456/Desktop/work/example6/result.npy', allow_pickle=True).item()
    paths_cfg = {}
    for group_idx in paths_npy.keys():
        path_info = paths_npy[group_idx][0]
        pipe0_info = pipe_name_to_info[path_info['name0']]
        pipe1_info = pipe_name_to_info[path_info['name1']]
        paths_cfg[group_idx] = {
            'name': f"path_{group_idx}",
            'src_name': path_info['name0'],
            'end_name': path_info['name1'],
            'start_vec': np.array(pipe0_info['direction']),
            'end_vec': np.array(pipe1_info['direction']),
            'radius': path_info['radius'],
            'xyzs': path_info['path_xyzrl'][:, :3]
        }

    parser = EnvParser()

    is_record_video = True
    if is_record_video:
        record_vis_up = RecordAnysisVisulizer()
        record_vis_up.record_video_init(file='/home/admin123456/Desktop/work/example6/up_test.mp4')

    opt_step = 0.2
    k = 3
    run_steps = 300
    run_angle = np.pi * 6.0
    step_angle = run_angle / run_steps

    accumulate_volume_cost = 0.0
    tensor_writer = SummaryWriter(logdir='/home/admin123456/Desktop/work/example6/debug')

    for step in range(run_steps):
        var_tree, paths_cfg = parser.create_graph(
            pipes_cfg, structors_cfg, bgs_cfg, paths_cfg, k=k, length_tol=0.1, sample_tol=0.1
        )

        constraints = parser.get_constraints(
            xs=var_tree.get_xs_init(), var_tree=var_tree, paths_cfg=paths_cfg, opt_step=0.1
        )
        print('constraints num:', len(constraints))
        # for cfg in constraints:
        #     if cfg['type'] == 'shape_conflict':
        #         info_0, info_1, threshold = cfg['info_0'], cfg['info_1'], cfg['threshold']
        #         xyz0, xyz1 = info_0['debug_xyz'], info_1['debug_xyz']
        #         cost = OptimizerScipy.penalty_shape_conflict(xyz0, xyz1, threshold, method='larger')
        #         print(f"{info_0['name']} pose{list(xyz0)} {info_1['name']} pose:{list(xyz1)} "
        #               f"dist:{np.linalg.norm(xyz0-xyz1)} threshold:{threshold}, cost:{cost}, method:larger")

        # parser.plot_env(
        #     xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=constraints, paths_cfg=paths_cfg,
        #     with_path=True, with_structor=True, with_bound=True, with_constraint=True, with_control_points=False,
        #     with_tube=True
        # )

        # for path_idx in paths_cfg:
        #     path_cfg = paths_cfg[path_idx]
        #     path = path_cfg['b_mat'].dot(path_cfg['xyzs'])
        #     real_res, fake_res = BSpline_utils.compare_curvature(
        #         path, path_cfg['start_vec'], path_cfg['end_vec']
        #     )
        #     fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
        #     ax0.scatter(path[:, 0], path[:, 1], s=real_res * 5.0 + 1.0)
        #     ax1.scatter(path[:, 0], path[:, 1], s=fake_res * 5.0 + 1.0)
        #     plt.show()

        # ------
        optimizer = OptimizerScipy()

        xs_init = var_tree.get_xs_init()
        opt_xs, costs_tuple = optimizer.solve_problem(
            xs_init, var_tree, paths_cfg, constraints, opt_step,
            penalty_weight=500.0, volume_cost_weight=0.1
        )
        xs_init += opt_xs

        # ------
        length_cost, penalty_cost, volume_cost = costs_tuple
        tensor_writer.add_scalar(tag='length_cost', scalar_value=length_cost, global_step=step)
        tensor_writer.add_scalar(tag='pentalty_cost', scalar_value=penalty_cost, global_step=step)
        accumulate_volume_cost += volume_cost
        tensor_writer.add_scalar(tag='volume_cost', scalar_value=accumulate_volume_cost, global_step=step)
        # ------

        nodes_info = var_tree.nodes_info
        name_to_nodeIdx = var_tree.name_to_nodeIdx
        for path_idx in paths_cfg.keys():
            path_cfg = paths_cfg[path_idx]
            src_node = nodes_info[path_cfg['src_node_idx']]
            src_x, src_y, src_z = VarTree.get_xyz_from_node(src_node, xs_init, name_to_nodeIdx, nodes_info, 1)
            end_node = nodes_info[path_cfg['end_node_idx']]
            end_x, end_y, end_z = VarTree.get_xyz_from_node(end_node, xs_init, name_to_nodeIdx, nodes_info, 1)
            cell_xyzs = xs_init[path_cfg['path_cols']].reshape((-1, 3))
            c_mat = np.concatenate(([[src_x, src_y, src_z]], cell_xyzs, [[end_x, end_y, end_z]]), axis=0)
            new_cmat = BSpline_utils.sample_uniform(c_mat, k=k, sample_num=path_cfg['b_mat'].shape[0], point_num=200)
            path_cfg.update({'xyzs': new_cmat})

        for cfg in pipes_cfg:
            parser.update_node_info(cfg, xs_init, nodes_info, name_to_nodeIdx)
        for cfg in structors_cfg:
            parser.update_node_info(cfg, xs_init, nodes_info, name_to_nodeIdx)
        for cfg in bgs_cfg:
            parser.update_node_info(cfg, xs_init, nodes_info, name_to_nodeIdx)

        node0_tem = VarTree.get_node_from_name('planeMax', name_to_nodeIdx, nodes_info)
        xyz_max = VarTree.get_xyz_from_node(node0_tem, xs_init, name_to_nodeIdx, nodes_info, 1)
        node1_tem = VarTree.get_node_from_name('planeMin', name_to_nodeIdx, nodes_info)
        xyz_min = VarTree.get_xyz_from_node(node1_tem, xs_init, name_to_nodeIdx, nodes_info, 1)

        if is_record_video:
            parser.record_video(
                record_vis_up, xs=xs_init, var_tree=var_tree, paths_cfg=paths_cfg,
                with_path=False, with_structor=True, with_bound=True, with_control_points=True, with_tube=True
            )
            obj_center = (np.array(xyz_max) + np.array(xyz_min)) / 2.0
            camera_pose, focal_pose = record_vis_up.get_camera_pose(
                obj_center, focal_length=20.0, radius=60.0, angle=step * step_angle, height=50.0
            )
            record_vis_up.set_camera(camera_pose, focal_pose, 1.0)
            record_vis_up.write_frame()

        print(f"[Info]: Run Step:{step} Boundary: {list(xyz_min)} --- {list(xyz_max)}")
        print()

    # ------
    var_tree, paths_cfg = parser.create_graph(
        pipes_cfg, structors_cfg, bgs_cfg, paths_cfg, k=k, length_tol=0.1, sample_tol=0.1
    )
    constraints = parser.get_constraints(
        xs=var_tree.get_xs_init(), var_tree=var_tree, paths_cfg=paths_cfg, opt_step=0.1
    )
    # parser.plot_env(
    #     xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=constraints, paths_cfg=paths_cfg,
    #     with_path=True, with_structor=True, with_bound=True, with_constraint=True, with_control_points=False,
    #     with_tube=True
    # )
    parser.save_to_npy(
        xs=var_tree.get_xs_init(), var_tree=var_tree, constraints=constraints, paths_cfg=paths_cfg,
        file_path='/home/admin123456/Desktop/work/example6/record.npy'
    )
    parser.plot_npy_record(file_path='/home/admin123456/Desktop/work/example6/record.npy')


if __name__ == '__main__':
    # env2_test()
    # env3_test()

    parser = EnvParser()
    parser.plot_npy_record(file_path='/home/admin123456/Desktop/work/example6/record.npy')