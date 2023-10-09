import numpy as np

from scripts.version_9.springer.spring_smoother_v2 import VarTree, OptimizerScipy, Shape_Utils, EnvParser

def env1_test():
    radius = 0.3

    # ------
    p0_pose = np.array([0.5, 0.5, 0.5])
    p0_shape_pcd = Shape_Utils.create_BoxPcd(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, reso=0.2)
    connector0_pose = np.array([1.0, 0.5, 0.5])
    p0_shape_pcd = Shape_Utils.removePointInBoxShell(
        p0_shape_pcd, shellRange=[
            connector0_pose[0] - radius, connector0_pose[0] + radius,
            connector0_pose[1] - radius, connector0_pose[1] + radius,
            connector0_pose[2] - radius, connector0_pose[2] + radius
        ]
    )
    p0_shape_pcd = p0_shape_pcd - p0_pose

    # ------
    p1_pose = np.array([3.5, 0.5, 0.5])
    connector1_pose = np.array([3.0, 0.5, 0.5])
    p1_shape_pcd = Shape_Utils.create_BoxPcd(3.0, 0.0, 0.0, 4.0, 1.0, 1.0, reso=0.2)
    p1_shape_pcd = Shape_Utils.removePointInBoxShell(
        p1_shape_pcd, shellRange=[
            connector1_pose[0] - radius, connector1_pose[0] + radius,
            connector1_pose[1] - radius, connector1_pose[1] + radius,
            connector1_pose[2] - radius, connector1_pose[2] + radius
        ]
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
            'node_type': 'structor', 'name': 'p0', 'position': p0_pose, 'shape_pcd': p0_shape_pcd,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'exclude_edges': {
                'plane_min_conflict': ['x', 'y', 'z']
            }
        },
        {
            'node_type': 'structor', 'name': 'p1', 'position': p1_pose, 'shape_pcd': p1_shape_pcd,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'planeMax', 'value': -0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'exclude_edges': {
                'plane_min_conflict': ['y', 'z'],
                'plane_max_conflict': ['x']
            }
        }
    ]

    bgs_cfg = [
        {
            'node_type': 'cell', 'name': 'planeMin', 'position': np.array([0., 0., 0.]),
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'}
        },
        {
            'node_type': 'cell', 'name': 'planeMax', 'position': np.array([4., 1.2, 1.2]),
            'pose_edge_x': {'type': 'var_value'},
            'pose_edge_y': {'type': 'var_value'},
            'pose_edge_z': {'type': 'var_value'}
        },
    ]

    paths_cfg = {
        0: {
            'name': 'path_01',
            'src_name': 'connector0',
            'end_name': 'connector1',
            'radius': radius,
            'xyzs': np.array([
                [1.0, 0.5, 0.5],
                [1.5, 0.5, 0.5],
                [2.0, 0.5, 0.5],
                [2.5, 0.5, 0.5],
                [3.0, 0.5, 0.5]
            ])
        }
    }

    parser = EnvParser()
    parser.create_vars_tree(pipes_cfg, structors_cfg, bgs_cfg)
    paths_cfg = parser.define_path(paths_cfg, k=3)

    # parser.plot_env(
    #     xs=parser.var_tree.get_xs_init(), constraints=None, paths_cfg=paths_cfg, with_path=True,
    #     with_structor=True, with_bound=True, with_constraint=False
    # )

    xs_init, paths_cfg = parser.solve(paths_cfg, opt_step=0.1)

    parser.plot_env(
        xs=xs_init, constraints=None, paths_cfg=paths_cfg, with_path=True,
        with_structor=True, with_bound=True, with_constraint=False
    )

def env2_test():
    radius = 0.3

    # ------
    p0_pose = np.array([0.5, 0.5, 0.5])
    p0_shape_pcd = Shape_Utils.create_BoxPcd(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, reso=0.2)
    connector0_pose = np.array([1.0, 0.5, 0.5])
    p0_shape_pcd = Shape_Utils.removePointInBoxShell(
        p0_shape_pcd, shellRange=[
            connector0_pose[0] - radius, connector0_pose[0] + radius,
            connector0_pose[1] - radius, connector0_pose[1] + radius,
            connector0_pose[2] - radius, connector0_pose[2] + radius
        ]
    )
    p0_shape_pcd = p0_shape_pcd - p0_pose

    # ------
    p1_pose = np.array([3.5, 3.5, 0.5])
    connector1_pose = np.array([3.0, 3.5, 0.5])
    p1_shape_pcd = Shape_Utils.create_BoxPcd(3.0, 3.0, 0.0, 4.0, 4.0, 1.0, reso=0.2)
    p1_shape_pcd = Shape_Utils.removePointInBoxShell(
        p1_shape_pcd, shellRange=[
            connector1_pose[0] - radius, connector1_pose[0] + radius,
            connector1_pose[1] - radius, connector1_pose[1] + radius,
            connector1_pose[2] - radius, connector1_pose[2] + radius
        ]
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
            'node_type': 'structor', 'name': 'p0', 'position': p0_pose, 'shape_pcd': p0_shape_pcd,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'exclude_edges': {
                'plane_min_conflict': ['x', 'y', 'z'],
                'plane_max_conflict': ['z']
            }
        },
        {
            'node_type': 'structor', 'name': 'p1', 'position': p1_pose, 'shape_pcd': p1_shape_pcd,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'planeMax', 'value': -0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'planeMax', 'value': -0.5},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'exclude_edges': {
                'plane_min_conflict': ['z'],
                'plane_max_conflict': ['x', 'y', 'z']
            }
        }
    ]

    bgs_cfg = [
        {
            'node_type': 'cell', 'name': 'planeMin', 'position': np.array([0., 0., 0.]),
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'}
        },
        {
            'node_type': 'cell', 'name': 'planeMax', 'position': np.array([4., 4., 1.]),
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
            'radius': radius,
            'xyzs': np.array([
                [1.0, 0.5, 0.5],
                [1.5, 0.5, 0.5],
                [2.0, 0.5, 0.5],
                [2.0, 1.0, 0.5],
                [2.0, 1.5, 0.5],
                [2.0, 2.0, 0.5],
                [2.0, 2.5, 0.5],
                [2.0, 3.0, 0.5],
                [2.0, 3.5, 0.5],
                [2.5, 3.5, 0.5],
                [3.0, 3.5, 0.5],
            ])
        }
    }

    parser = EnvParser()
    parser.create_vars_tree(pipes_cfg, structors_cfg, bgs_cfg)
    paths_cfg = parser.define_path(paths_cfg, k=2)

    parser.plot_env(
        xs=parser.var_tree.get_xs_init(), constraints=None, paths_cfg=paths_cfg, with_path=True,
        with_structor=True, with_bound=True, with_constraint=False
    )

    xs_init, paths_cfg = parser.solve(paths_cfg, opt_step=0.1)

    parser.plot_env(
        xs=xs_init, constraints=None, paths_cfg=paths_cfg, with_path=True,
        with_structor=True, with_bound=True, with_constraint=False
    )

def env3_test():
    radius = 0.3

    # ------
    p0_pose = np.array([0.5, 0.5, 0.5])
    p0_shape_pcd = Shape_Utils.create_BoxPcd(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, reso=0.2)
    connector0_pose = np.array([1.0, 0.5, 0.5])
    p0_shape_pcd = Shape_Utils.removePointInBoxShell(
        p0_shape_pcd, shellRange=[
            connector0_pose[0] - radius, connector0_pose[0] + radius,
            connector0_pose[1] - radius, connector0_pose[1] + radius,
            connector0_pose[2] - radius, connector0_pose[2] + radius
        ]
    )
    p0_shape_pcd = p0_shape_pcd - p0_pose

    # ------
    p1_pose = np.array([2.5, 2.5, 2.5])
    connector1_pose = np.array([2.0, 2.5, 2.5])
    p1_shape_pcd = Shape_Utils.create_BoxPcd(2.0, 2.0, 2.0, 3.0, 3.0, 3.0, reso=0.2)
    p1_shape_pcd = Shape_Utils.removePointInBoxShell(
        p1_shape_pcd, shellRange=[
            connector1_pose[0] - radius, connector1_pose[0] + radius,
            connector1_pose[1] - radius, connector1_pose[1] + radius,
            connector1_pose[2] - radius, connector1_pose[2] + radius
        ]
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
            'node_type': 'structor', 'name': 'p0', 'position': p0_pose, 'shape_pcd': p0_shape_pcd,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'planeMin', 'value': 0.5},
            'exclude_edges': {
                'plane_min_conflict': ['x', 'y', 'z']
            }
        },
        {
            'node_type': 'structor', 'name': 'p1', 'position': p1_pose, 'shape_pcd': p1_shape_pcd,
            'pose_edge_x': {'type': 'value_shift', 'ref_obj': 'planeMax', 'value': -0.5},
            'pose_edge_y': {'type': 'value_shift', 'ref_obj': 'planeMax', 'value': -0.5},
            'pose_edge_z': {'type': 'value_shift', 'ref_obj': 'planeMax', 'value': -0.5},
            'exclude_edges': {
                'plane_max_conflict': ['x', 'y', 'z']
            }
        }
    ]

    bgs_cfg = [
        {
            'node_type': 'cell', 'name': 'planeMin', 'position': np.array([0., 0., 0.]),
            'pose_edge_x': {'type': 'fix_value'},
            'pose_edge_y': {'type': 'fix_value'},
            'pose_edge_z': {'type': 'fix_value'}
        },
        {
            'node_type': 'cell', 'name': 'planeMax', 'position': np.array([3., 3., 3.]),
            'pose_edge_x': {'type': 'var_value'},
            'pose_edge_y': {'type': 'var_value'},
            'pose_edge_z': {'type': 'var_value'}
        },
    ]

    paths_cfg = {
        0: {
            'name': 'path_01',
            'src_name': 'connector0',
            'end_name': 'connector1',
            'radius': radius,
            'xyzs': np.array([
                [1.0, 0.5, 0.5],
                [1.5, 0.5, 0.5],
                [1.5, 1.0, 0.5],
                [1.5, 2.0, 0.5],
                [1.5, 2.5, 0.5],
                [1.5, 2.5, 1.0],
                [1.5, 2.5, 1.5],
                [1.5, 2.5, 2.0],
                [1.5, 2.5, 2.5],
                [2.0, 2.5, 2.5],
            ])
        }
    }

    parser = EnvParser()
    parser.create_vars_tree(pipes_cfg, structors_cfg, bgs_cfg)
    paths_cfg = parser.define_path(paths_cfg, k=2)

    parser.plot_env(
        xs=parser.var_tree.get_xs_init(), constraints=None, paths_cfg=paths_cfg, with_path=True,
        with_structor=True, with_bound=True, with_constraint=False
    )

    xs_init, paths_cfg = parser.solve(paths_cfg, opt_step=0.1)

    parser.plot_env(
        xs=xs_init, constraints=None, paths_cfg=paths_cfg, with_path=True,
        with_structor=True, with_bound=True, with_constraint=False
    )


if __name__ == '__main__':
    # env1_test()
    env3_test()
