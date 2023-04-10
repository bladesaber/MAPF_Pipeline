import numpy as np
import math
from scipy.spatial import transform

import matplotlib.pyplot as plt
from scripts.debug.dubinsCruve_debug import plot_arrow, plot_path

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

def plot_3dArrow(ax, xyz, vec, length=1.0):
    ax.quiver(
        xyz[0], xyz[1], xyz[2], 
        vec[0], vec[1], vec[2], 
        length=length, normalize=True, color='r'
    )

def plot_3dPath(ax, xyzs):
    ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2])
    # ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2], '*-')
    # ax.scatter(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2])

def create_3dGraph(
    xmax, ymax, zmax, xmin=0., ymin=0., zmin=0.
):
    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)
    ax.grid(True)

    return ax

def polar3d_to_vec(theta, length=1.0):
    dz = length * math.sin(theta[1])
    dl = length * math.cos(theta[1])
    dx = dl * math.cos(theta[0])
    dy = dl * math.sin(theta[0])
    return np.array([dx, dy, dz])

def vec_to_polar(vec):
    theta0 = math.atan2(vec[1], vec[0])
    theta1 = math.atan2(
        vec[2],
        math.sqrt(vec[0]**2 + vec[1]**2)
    )
    return np.array([theta0, theta1])

def compute_aux_angel(target_theta):
    '''
                         C
                        **
                     * * |
                   *  *  |
                 *   *   |
               *    *    |
             *     *     |   y
           *      *      |
         *       *       |
       *        *        |
     *         *         |
    *---------*----------|
    A  (1.0)        x    B

    angel(AC, AB) = target_angel (1)
    x^2 + y^2 = 1.0              (2)

    =>
    x^ + y^2 = 1.0
    y / (1.0 + x) = tan(target_theta)

    solve by sympy in math_infer.ipynb
    '''
    tan_theta = math.tan(target_theta)
    x = (1.0 - tan_theta ** 2) / (1.0 + tan_theta**2)
    y = 2.0 * tan_theta / (1.0 + tan_theta**2)
    # assert x >= 0. and y >= 0.

    radian = math.atan2(y, x)

    return radian

def dubinsCruve3D_debug(
    xyz_d, theta_d, 
    method1=mapf_pipeline.DubinsPathType.LSL, 
    method2=mapf_pipeline.DubinsPathType.LSL, 
    radius=1.0
):
    ### ------ first compute
    p0 = (0., 0., 0.)
    p1 = (xyz_d[0], xyz_d[1], theta_d[0])

    res0 = mapf_pipeline.DubinsPath()
    errorCode = mapf_pipeline.compute_dubins_path(res0, p0, p1, radius, method1)
    if errorCode != mapf_pipeline.DubinsErrorCodes.EDUBOK:
        raise ValueError('Error')
    
    ### ------ second compute
    p0 = (0., 0., 0.)
    aux_theta = compute_aux_angel(theta_d[1])
    p1 = (res0.total_length, xyz_d[2], aux_theta)
    # p1 = (0., xyz_d[2], theta_d[1])

    res1 = mapf_pipeline.DubinsPath()
    errorCode = mapf_pipeline.compute_dubins_path(res1, p0, p1, radius, method2)
    if errorCode != mapf_pipeline.DubinsErrorCodes.EDUBOK:
        raise ValueError('Error')

    ### ------ extract_path
    mapf_pipeline.compute_dubins_info(res0)
    path_xys = mapf_pipeline.sample_dubins_path(res0, 50)
    path_xys = np.array(path_xys)

    mapf_pipeline.compute_dubins_info(res1)
    path_xzs = mapf_pipeline.sample_dubins_path(res1, 50)
    path_xzs = np.array(path_xzs)

    ### ------ debug
    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1)

    plot_arrow(res0.q0[0], res0.q0[1], res0.q0[2], ax=ax0)
    plot_arrow(res0.q1[0], res0.q1[1], res0.q1[2], ax=ax0)
    ax0.plot(res0.start_center[0], res0.start_center[1], 'or')
    ax0.plot(res0.final_center[0], res0.final_center[1], 'or')
    ax0.plot(res0.line_sxy[0], res0.line_sxy[1], '^r')
    ax0.plot(res0.line_fxy[0], res0.line_fxy[1], '^r')
    plot_path(ax0, path_xys)
    ax0.legend()
    ax0.grid(True)
    ax0.axis("equal")

    plot_arrow(res1.q0[0], res1.q0[1], res1.q0[2], ax=ax1)
    plot_arrow(res1.q1[0], res1.q1[1], res1.q1[2], ax=ax1)
    ax1.plot(res1.start_center[0], res1.start_center[1], 'or')
    ax1.plot(res1.final_center[0], res1.final_center[1], 'or')
    ax1.plot(res1.line_sxy[0], res1.line_sxy[1], '^r')
    ax1.plot(res1.line_fxy[0], res1.line_fxy[1], '^r')
    plot_path(ax1, path_xzs)
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

def dubinsCruve3D_full_debug(
    xyz0, theta0, xyz1, theta1,
    radius=1.0
):
    xyz_d = xyz1 - xyz0
    theta_d = theta1 - theta0

    assert theta_d[1] >= -math.pi / 2.0 and theta_d[1] <= math.pi / 2.0

    if abs(theta_d[1]) <= math.pi / 4.0:
        path_xyzs = dubinsCruve3D_debug(xyz_d, theta_d, radius=radius)

    else:
        xyz_d = np.array([xyz_d[0], xyz_d[2], xyz_d[1]])

        vec = polar3d_to_vec(theta_d)
        vec = np.array([vec[0], vec[2], vec[1]])
        theta_d = vec_to_polar(vec)

        path_xyzs = dubinsCruve3D_debug(xyz_d, theta_d, radius=radius)
        path_xyzs = np.concatenate([
            path_xyzs[:, 0:1], path_xyzs[:, 2:3], path_xyzs[:, 1:2]
        ], axis=1)

    return path_xyzs

def dubinsCruve3D_compute(xyz_d, theta_d, sample_size, radius=1.0):
    best_reses, best_cost = None, np.inf
    p0 = (0., 0., 0.)
    p1 = (xyz_d[0], xyz_d[1], theta_d[0])
    aux_theta = compute_aux_angel(theta_d[1])
    
    for method1 in Dubins_SegmentType.keys():
        
        res0 = mapf_pipeline.DubinsPath()
        errorCode = mapf_pipeline.compute_dubins_path(res0, p0, p1, radius, method1)
        if errorCode != mapf_pipeline.DubinsErrorCodes.EDUBOK:
            continue
            
        cost1 = res0.total_length
        
        p2 = (res0.total_length, xyz_d[2], aux_theta)
        for method2 in Dubins_SegmentType.keys():
            res1 = mapf_pipeline.DubinsPath()
            errorCode = mapf_pipeline.compute_dubins_path(res1, p0, p2, radius, method2)
            if errorCode != mapf_pipeline.DubinsErrorCodes.EDUBOK:
                continue
            
            cost2 = res1.total_length
            cost = cost1 + cost2

            if cost < best_cost:
                best_cost = cost
                best_reses = (res0, res1)
    
    if best_reses is None:
        return None
    
    res0, res1 = best_reses

    ### ------ extract_path
    mapf_pipeline.compute_dubins_info(res0)
    path_xys = mapf_pipeline.sample_dubins_path(res0, sample_size)
    path_xys = np.array(path_xys)

    mapf_pipeline.compute_dubins_info(res1)
    path_xzs = mapf_pipeline.sample_dubins_path(res1, sample_size)
    path_xzs = np.array(path_xzs)

    path_xyzs = np.concatenate([
        path_xys, 
        path_xzs[:, 1:2]
    ], axis=1)
    
    return path_xyzs

def dubinsCruve3D_full_compute(xyz0, theta0, xyz1, theta1, sample_size, radius=1.0):
    xyz_d = xyz1 - xyz0
    theta_d = theta1 - theta0

    assert theta_d[1] >= -math.pi / 2.0 and theta_d[1] <= math.pi / 2.0

    if abs(theta_d[1]) <= math.pi / 4.0:
        path_xyzs = dubinsCruve3D_compute(xyz_d, theta_d, sample_size, radius=radius)

    else:
        xyz_d = np.array([xyz_d[0], xyz_d[2], xyz_d[1]])

        vec = polar3d_to_vec(theta_d)
        vec = np.array([vec[0], vec[2], vec[1]])
        theta_d = vec_to_polar(vec)

        path_xyzs = dubinsCruve3D_compute(xyz_d, theta_d, sample_size, radius=radius)
        path_xyzs = np.concatenate([
            path_xyzs[:, 0:1], path_xyzs[:, 2:3], path_xyzs[:, 1:2]
        ], axis=1)

    return path_xyzs

def main():
    xyz0 = np.array([0.0, 0.0, 0.0])
    theta0 = np.array([np.deg2rad(0.0), np.deg2rad(0.0)])
    xyz1 = np.array([3.0, 4.0, 2.5])
    theta1 = np.array([np.deg2rad(240.0), np.deg2rad(-65)])

    # path_xyzs = dubinsCruve3D_fullSolution(xyz0, theta0, xyz1, theta1)
    path_xyzs = dubinsCruve3D_full_compute(xyz0, theta0, xyz1, theta1, sample_size=40, radius=1.0)

    ### ------ draw solution
    vec0 = polar3d_to_vec(theta0)
    vec1 = polar3d_to_vec(theta1)

    ax = create_3dGraph(xmax=6.0, ymax=6.0, zmax=6.0)
    plot_3dArrow(ax, xyz0, vec0)
    plot_3dArrow(ax, xyz1, vec1)
    plot_3dPath(ax, path_xyzs)

    plt.show()

if __name__ == '__main__':
    main()
