import numpy as np
import pandas as pd
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from build import mapf_pipeline

def plot_arrow(x, y, yaw, arrow_length=1.0, head_width=0.1, ax=None):
    if ax is not None:
        ax.arrow(x, y, arrow_length * math.cos(yaw), arrow_length * math.sin(yaw), head_width=head_width, fc='r', ec='k')
        ax.plot(x, y, 'xr')
    else:
        plt.arrow(x, y, arrow_length * math.cos(yaw), arrow_length * math.sin(yaw), head_width=head_width, fc='r', ec='k')
        plt.plot(x, y, 'xr')

def plot_ARC(ax, center, radius, start_angel, end_angel, segmentType, shift_angel=0.0):
    start_angel = np.rad2deg(start_angel)
    end_angel = np.rad2deg(end_angel)
    shift_angel = np.rad2deg(shift_angel)

    if segmentType == mapf_pipeline.SegmentType.R_SEG:
        tem = start_angel
        start_angel = end_angel
        end_angel = tem

    ax.add_patch(mpatches.Arc(
        center,
        radius * 2.0, radius * 2.0,
        angle = shift_angel,
        theta1=start_angel, theta2=end_angel,
    ))

def plot_path(ax, wayPoints):
    ax.scatter(wayPoints[:, 0], wayPoints[:, 1])

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

def compute_LeftCenter(vec, radius, p):
    vec_l = np.array([-vec[1], vec[0]])
    center_left = p + vec_l * radius
    return center_left

def main():
    ### x, y, theta
    p0_theta = np.deg2rad(45.0)
    p0_xy = np.array([1.0, 1.0])
    p0 = (p0_xy[0], p0_xy[1], p0_theta)

    p1_theta = np.deg2rad(-45.0)
    p1_xy = np.array([-3.0, -3.0])
    p1 = (p1_xy[0], p1_xy[1], p1_theta)

    radius = 0.66
    
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

        print(res.lengths)

        path_wayPoints = mapf_pipeline.sample_dubins_path(res, 30)
        path_wayPoints = np.array(path_wayPoints)

        fig, ax = plt.subplots()
        plot_arrow(res.q0[0], res.q0[1], res.q0[2], ax=ax)
        plot_arrow(res.q1[0], res.q1[1], res.q1[2], ax=ax)

        ax.plot(res.start_center[0], res.start_center[1], 'or')
        ax.plot(res.final_center[0], res.final_center[1], 'or')
        ax.plot(res.line_sxy[0], res.line_sxy[1], '^r')
        ax.plot(res.line_fxy[0], res.line_fxy[1], '^r')

        # plot_ARC(
        #     ax,
        #     center=np.array([res.start_center[0], res.start_center[1]]),
        #     radius=radius, 
        #     start_angel=res.start_range[0],
        #     end_angel=res.start_range[1],
        #     segmentType=Dubins_SegmentType[res.type][0]
        # )
        # plot_ARC(
        #     ax,
        #     center=np.array([res.final_center[0], res.final_center[1]]),
        #     radius=radius, 
        #     start_angel=res.final_range[0],
        #     end_angel=res.final_range[1],
        #     segmentType=Dubins_SegmentType[res.type][-1]
        # )

        plot_path(ax, path_wayPoints)

        ax.legend()
        ax.grid(True)
        ax.axis("equal")
        plt.show()

if __name__ == '__main__':
    main()
