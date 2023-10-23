import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from scripts_py.vis_utils import plot_Arc2D, plot_Arrow2D, plot_Path2D
from scripts_py.vis_utils import create_Graph3D, plot_Arrow3D, plot_Path3D
from scripts_py.utils import compute_dubinsAuxAngel, polar3D2vec, vec2polar3D

from build import mapf_pipeline

erroeCode_dict = {
    mapf_pipeline.DubinsErrorCodes.EDUBOK: 'success',
    mapf_pipeline.DubinsErrorCodes.EDUBCOCONFIGS: 'colocated configurations',
    mapf_pipeline.DubinsErrorCodes.EDUBPARAM: 'path parameterisitation error',
    mapf_pipeline.DubinsErrorCodes.EDUBBADRHO: 'the rho value is invalid',
    mapf_pipeline.DubinsErrorCodes.EDUBNOPATH: 'no connection between configurations with this word',
}

Dubins_SegmentType = {
    mapf_pipeline.DubinsPathType.LSL: (mapf_pipeline.SegmentType.L_SEG, mapf_pipeline.SegmentType.S_SEG, mapf_pipeline.SegmentType.L_SEG),
    mapf_pipeline.DubinsPathType.LSR: (mapf_pipeline.SegmentType.L_SEG, mapf_pipeline.SegmentType.S_SEG, mapf_pipeline.SegmentType.R_SEG),
    mapf_pipeline.DubinsPathType.RSL: (mapf_pipeline.SegmentType.R_SEG, mapf_pipeline.SegmentType.S_SEG, mapf_pipeline.SegmentType.L_SEG),
    mapf_pipeline.DubinsPathType.RSR: (mapf_pipeline.SegmentType.R_SEG, mapf_pipeline.SegmentType.S_SEG, mapf_pipeline.SegmentType.R_SEG),
    # mapf_pipeline.DubinsPathType.RLR: (mapf_pipeline.SegmentType.R_SEG, mapf_pipeline.SegmentType.L_SEG, mapf_pipeline.SegmentType.R_SEG),
    # mapf_pipeline.DubinsPathType.LRL: (mapf_pipeline.SegmentType.L_SEG, mapf_pipeline.SegmentType.R_SEG, mapf_pipeline.SegmentType.L_SEG)
}

Dubins_Direction = {
    mapf_pipeline.SegmentType.L_SEG: 'left',
    mapf_pipeline.SegmentType.S_SEG: 'straight',
    mapf_pipeline.SegmentType.R_SEG: 'right'
}

Dubins_Dire = {
    mapf_pipeline.SegmentType.L_SEG: False,
    mapf_pipeline.SegmentType.R_SEG: True
}

def debug_dubinsPath_2D(p0, p1, radius):
    for method in Dubins_SegmentType.keys():
        res = mapf_pipeline.DubinsPath()
        errorCode = mapf_pipeline.compute_dubins_path(
            res, p0, p1, radius, 
            # mapf_pipeline.DubinsPathType.LSL
            # mapf_pipeline.DubinsPathType.RSR
            method
        )
        mapf_pipeline.compute_dubins_info(res)

        print('Circle 1st: move: %f from %f -> %f direction: %s' % (
            np.rad2deg(res.param[0]), 
            np.rad2deg(res.start_range[0]),
            np.rad2deg(res.start_range[1]),
            Dubins_Direction[Dubins_SegmentType[res.type][0]]
        ))
        print('Circle 3st: move: %f from %f -> %f direction: %s' % (
            np.rad2deg(res.param[2]), 
            np.rad2deg(res.final_range[0]),
            np.rad2deg(res.final_range[1]),
            Dubins_Direction[Dubins_SegmentType[res.type][-1]]
        ))

        path_wayPoints = mapf_pipeline.sample_dubins_path(res, 30)
        path_wayPoints = np.array(path_wayPoints)

        fig, ax = plt.subplots()
        plot_Arrow2D(res.q0[0], res.q0[1], res.q0[2], ax=ax)
        plot_Arrow2D(res.q1[0], res.q1[1], res.q1[2], ax=ax)

        ax.plot(res.start_center[0], res.start_center[1], 'or')
        ax.plot(res.final_center[0], res.final_center[1], 'or')
        ax.plot(res.line_sxy[0], res.line_sxy[1], '^r')
        ax.plot(res.line_fxy[0], res.line_fxy[1], '^r')

        # plot_Arc2D(
        #     ax,
        #     center=np.array([res.start_center[0], res.start_center[1]]),
        #     radius=radius, 
        #     start_angel=res.start_range[0],
        #     end_angel=res.start_range[1],
        #     right_dire=Dubins_Dire[Dubins_SegmentType[res.type][0]]
        # )
        # plot_Arc2D(
        #     ax,
        #     center=np.array([res.final_center[0], res.final_center[1]]),
        #     radius=radius, 
        #     start_angel=res.final_range[0],
        #     end_angel=res.final_range[1],
        #     right_dire=Dubins_Dire[Dubins_SegmentType[res.type][0]]
        # )

        plot_Path2D(ax, path_wayPoints)

        ax.legend()
        ax.grid(True)
        ax.axis("equal")
        plt.show()

def dubinsCruve3D_compute(
    xyz_d, theta_d, 
    method1=mapf_pipeline.DubinsPathType.LSL, 
    method2=mapf_pipeline.DubinsPathType.LSL, 
    radius=1.0
):
    best_reses, best_cost = None, np.inf
    p0 = (0., 0., 0.)
    p1 = (xyz_d[0], xyz_d[1], theta_d[0])

    for method0 in Dubins_SegmentType.keys():
        res0 = mapf_pipeline.DubinsPath()
        errorCode = mapf_pipeline.compute_dubins_path(res0, p0, p1, radius, method0)
        if errorCode != mapf_pipeline.DubinsErrorCodes.EDUBOK:
            continue

        cost0 = res0.total_length
        
        hs_length = res0.total_length
        p2 = (hs_length, xyz_d[2], theta_d[1])
        for method1 in Dubins_SegmentType.keys():
            res1 = mapf_pipeline.DubinsPath()
            errorCode = mapf_pipeline.compute_dubins_path(res1, p0, p2, radius, method1)
            if errorCode != mapf_pipeline.DubinsErrorCodes.EDUBOK:
                continue
            
            cost1 = res1.total_length

            cost = cost0 + cost1
            if cost < best_cost:
                best_cost = cost
                best_reses = (res0, res1)

    res0, res1 = best_reses

    ### ------ extract_path
    mapf_pipeline.compute_dubins_info(res0)
    path_xys = mapf_pipeline.sample_dubins_path(res0, 30)
    path_xys = np.array(path_xys)

    mapf_pipeline.compute_dubins_info(res1)
    path_xzs = mapf_pipeline.sample_dubins_path(res1, 30)
    path_xzs = np.array(path_xzs)

    ### ------ debug
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    plot_Arrow2D(res0.q0[0], res0.q0[1], res0.q0[2], ax=ax0)
    plot_Arrow2D(res0.q1[0], res0.q1[1], res0.q1[2], ax=ax0)
    ax0.plot(res0.start_center[0], res0.start_center[1], 'or')
    ax0.plot(res0.final_center[0], res0.final_center[1], 'or')
    ax0.plot(res0.line_sxy[0], res0.line_sxy[1], '^r')
    ax0.plot(res0.line_fxy[0], res0.line_fxy[1], '^r')
    plot_Path2D(ax0, path_xys)
    ax0.legend()
    ax0.grid(True)
    ax0.axis("equal")

    plot_Arrow2D(res1.q0[0], res1.q0[1], res1.q0[2], ax=ax1)
    plot_Arrow2D(res1.q1[0], res1.q1[1], res1.q1[2], ax=ax1)
    ax1.plot(res1.start_center[0], res1.start_center[1], 'or')
    ax1.plot(res1.final_center[0], res1.final_center[1], 'or')
    ax1.plot(res1.line_sxy[0], res1.line_sxy[1], '^r')
    ax1.plot(res1.line_fxy[0], res1.line_fxy[1], '^r')
    plot_Path2D(ax1, path_xzs)
    ax1.legend()
    ax1.grid(True)
    ax1.axis("equal")
    # plt.show()
    ### --------------------------------

    path_xyzs = np.concatenate([
        path_xys, 
        path_xzs[:, 1:2]
        # np.zeros((path_xys.shape[0], 1))
    ], axis=1)
    
    return path_xyzs

def dubinsCruve3D_debug(
        xyz0, theta0, 
        xyz1, theta1, 
        method1=mapf_pipeline.DubinsPathType.LSL, 
        method2=mapf_pipeline.DubinsPathType.LSL, 
        radius=1.0
    ):
    xyz_d = xyz1 - xyz0
    theta_d = theta1 - theta0

    assert theta_d[1] >= -math.pi / 2.0 and theta_d[1] <= math.pi / 2.0

    if abs(theta_d[1]) <= math.pi / 4.0:
        path_xyzs = dubinsCruve3D_compute(xyz_d, theta_d, method1=method1, method2=method2, radius=radius)

    else:
        xyz_d = np.array([xyz_d[0], xyz_d[2], xyz_d[1]])

        vec = polar3D2vec(theta_d)
        vec = np.array([vec[0], vec[2], vec[1]])
        theta_d = vec2polar3D(vec)

        path_xyzs = dubinsCruve3D_compute(xyz_d, theta_d, method1=method1, method2=method2, radius=radius)
        path_xyzs = np.concatenate([
            path_xyzs[:, 0:1], path_xyzs[:, 2:3], path_xyzs[:, 1:2]
        ], axis=1)

    return path_xyzs

if __name__ == '__main__':
    # ### ------ 2D dubins path debug
    # p0_theta = np.deg2rad(45.0)
    # p0_xy = np.array([1.0, 1.0])
    # p0 = (p0_xy[0], p0_xy[1], p0_theta)

    # p1_theta = np.deg2rad(-45.0)
    # p1_xy = np.array([-3.0, -3.0])
    # p1 = (p1_xy[0], p1_xy[1], p1_theta)

    # radius = 0.66

    # debug_dubinsPath_2D(p0, p1, radius)
    # ### ---------------------------------------------

    ### ------ 3D dubins path debug
    xyz0 = np.array([0.0, 0.0, 0.0])
    theta0 = np.array([np.deg2rad(0.0), np.deg2rad(0.0)])
    xyz1 = np.array([4.0, 4.0, 4.0])
    theta1 = np.array([np.deg2rad(30.0), np.deg2rad(30.0)])

    path_xyzs = dubinsCruve3D_debug(xyz0, theta0, xyz1, theta1, radius=1.0)

    ### ------ draw solution
    vec0 = polar3D2vec(theta0)
    vec1 = polar3D2vec(theta1)

    ax = create_Graph3D(xmax=6.0, ymax=6.0, zmax=6.0)
    plot_Arrow3D(ax, xyz0, vec0)
    plot_Arrow3D(ax, xyz1, vec1)
    plot_Path3D(ax, path_xyzs)

    plt.show()
    ### ---------------------------------------------

    pass
