import numpy as np
import math

from build import mapf_pipeline
from scripts.utils import compute_aux_angel

Dubins_SegmentType = {
    mapf_pipeline.DubinsPathType.LSL: (mapf_pipeline.SegmentType.L_SEG, mapf_pipeline.SegmentType.S_SEG, mapf_pipeline.SegmentType.L_SEG),
    mapf_pipeline.DubinsPathType.LSR: (mapf_pipeline.SegmentType.L_SEG, mapf_pipeline.SegmentType.S_SEG, mapf_pipeline.SegmentType.R_SEG),
    mapf_pipeline.DubinsPathType.RSL: (mapf_pipeline.SegmentType.R_SEG, mapf_pipeline.SegmentType.S_SEG, mapf_pipeline.SegmentType.L_SEG),
    mapf_pipeline.DubinsPathType.RSR: (mapf_pipeline.SegmentType.R_SEG, mapf_pipeline.SegmentType.S_SEG, mapf_pipeline.SegmentType.R_SEG),
    # mapf_pipeline.DubinsPathType.RLR: (mapf_pipeline.SegmentType.R_SEG, mapf_pipeline.SegmentType.L_SEG, mapf_pipeline.SegmentType.R_SEG),
    # mapf_pipeline.DubinsPathType.LRL: (mapf_pipeline.SegmentType.L_SEG, mapf_pipeline.SegmentType.R_SEG, mapf_pipeline.SegmentType.L_SEG)
}

def compute_shorestDubinsPath3D(xyz_d, theta_d, radius):
    best_reses, best_cost = None, np.inf
    p0 = (0., 0., 0.)
    p1 = (xyz_d[0], xyz_d[1], theta_d[0])
    aux_theta = compute_aux_angel(theta_d[1])
    xy_length = np.linalg.norm(xyz_d[:2], ord=2)

    for method0 in Dubins_SegmentType.keys():
        res0 = mapf_pipeline.DubinsPath()
        errorCode = mapf_pipeline.compute_dubins_path(res0, p0, p1, radius, method0)
        if errorCode != mapf_pipeline.DubinsErrorCodes.EDUBOK:
            continue

        cost0 = res0.total_length
        
        hs_length = xy_length * 0.3 + res0.total_length * 0.7
        p2 = (hs_length, xyz_d[2], aux_theta)
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
    
    return best_reses, best_cost

def compute_dubinsPath3D(xyz_AlphaBeta0, xyz_AlphaBeta1, radius):
    xyz0 = xyz_AlphaBeta0[:3]
    theta0 = xyz_AlphaBeta0[3:]

    xyz1 = xyz_AlphaBeta1[:3]
    theta1 = xyz_AlphaBeta1[3:]

    xyz_d = xyz1 - xyz0
    theta_d = theta1 - theta0

    ### beta <= 45.0 or beta >= -45.0
    if abs(theta_d[1]) <= math.pi / 4.0:
        invert_yz = False
        best_solution, best_cost = compute_shorestDubinsPath3D(xyz_d, theta_d, radius)

    else:
        invert_yz = True

        xyz_d = (xyz_d[0], xyz_d[2], xyz_d[1])
        vec = mapf_pipeline.polar3D_to_vec3D(theta_d)
        vec = (vec[0], vec[2], vec[1])
        theta_d = mapf_pipeline.vec3D_to_polar3D(vec)

        best_solution, best_cost = compute_shorestDubinsPath3D(xyz_d, theta_d, radius)
    
    return (best_solution, best_cost, invert_yz)