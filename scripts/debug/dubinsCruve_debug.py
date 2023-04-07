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

methods = [
    mapf_pipeline.DubinsPathType.LSL,
    mapf_pipeline.DubinsPathType.LSR,
    mapf_pipeline.DubinsPathType.RSL,
    mapf_pipeline.DubinsPathType.RSR,
    mapf_pipeline.DubinsPathType.RLR,
    mapf_pipeline.DubinsPathType.LRL
]

erroeCode_dict = {
    mapf_pipeline.DubinsErrorCodes.EDUBOK: 'success',
    mapf_pipeline.DubinsErrorCodes.EDUBCOCONFIGS: 'colocated configurations',
    mapf_pipeline.DubinsErrorCodes.EDUBPARAM: 'path parameterisitation error',
    mapf_pipeline.DubinsErrorCodes.EDUBBADRHO: 'the rho value is invalid',
    mapf_pipeline.DubinsErrorCodes.EDUBNOPATH: 'no connection between configurations with this word',
}

def compute_LeftCenter(vec, radius, p):
    vec_l = np.array([-vec[1], vec[0]])
    center_left = p + vec_l * radius
    return center_left

def mod2pi(x, zero_2_2pi=True, degree=False):
    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    return mod_angle

def main():
    ### x, y, theta
    p0_theta = np.deg2rad(45.0)
    p0_xy = np.array([1.0, 1.0])
    p0 = (p0_xy[0], p0_xy[1], p0_theta)
    p0_vec = np.array([np.cos(p0_theta), np.sin(p0_theta)])

    p1_theta = np.deg2rad(-45.0)
    p1_xy = np.array([-3.0, -3.0])
    p1 = (p1_xy[0], p1_xy[1], p1_theta)
    p1_vec = np.array([np.cos(p1_theta), np.sin(p1_theta)])

    radius = 0.5

    # res = mapf_pipeline.DubinsPath()
    # for method in methods:
    #     res = mapf_pipeline.DubinsPath()
    #     errorCode = mapf_pipeline.compute_dubins_path(res, p0, p1, radius, method)
        
    #     if errorCode == mapf_pipeline.DubinsErrorCodes.EDUBOK:
    #         print(res.param)
    #         break
    
    res = mapf_pipeline.DubinsPath()
    errorCode = mapf_pipeline.compute_dubins_path(
        res, p0, p1, radius, mapf_pipeline.DubinsPathType.LSL
    )
    p0_move_theta = res.param[0]
    p1_move_theta = res.param[2]

    p0_left_center = compute_LeftCenter(p0_vec, radius, p0_xy)
    p1_left_center = compute_LeftCenter(p1_vec, radius, p1_xy)

    fig, ax = plt.subplots()
    plot_arrow(p0[0], p0[1], p0[2], ax=ax)
    plot_arrow(p1[0], p1[1], p1[2], ax=ax)
    ax.plot(p0_left_center[0], p0_left_center[1], 'or')
    ax.plot(p1_left_center[0], p1_left_center[1], 'or')

    # ax.add_patch(mpatches.Arc(
    #     p0_left_center,
    #     radius * 2.0, radius * 2.0,
    #     angle = 0,
    #     theta1=np.rad2deg(p0_theta) - 90.0, theta2=np.rad2deg(p0_theta) - 90.0 + np.rad2deg(p0_move_theta),
    # ))
    theta_0 = np.deg2rad(np.rad2deg(p0_theta) - 90.0 + np.rad2deg(p0_move_theta))
    inter_0 = p0_left_center + radius * np.array([math.cos(theta_0), math.sin(theta_0)])
    ax.plot(inter_0[0], inter_0[1], '^')
    ax.add_patch(mpatches.Circle(p0_left_center, radius=radius, fill=False))

    end_angel = 360.0 - 45.0 - 90.0
    theta_1 = end_angel-np.rad2deg(p1_move_theta)
    inter_1 = p1_left_center + radius * np.array([math.cos(theta_1), math.sin(theta_1)])
    ax.plot(inter_1[0], inter_1[1], '^')
    ax.add_patch(mpatches.Circle(p1_left_center, radius=radius, fill=False))
    # ax.add_patch(mpatches.Arc(
    #     p1_left_center,
    #     radius * 2.0, radius * 2.0,
    #     angle = 0.,
    #     theta1=end_angel-np.rad2deg(p1_move_theta), theta2=end_angel,
    # ))

    ax.plot([inter_0[0], inter_1[0]], [inter_0[1], inter_1[1]])

    print(res.param)
    print(np.linalg.norm(inter_0-inter_1, ord=2))

    ax.legend()
    ax.grid(True)
    ax.axis("equal")
    plt.show()

def test():
    fig,ax=plt.subplots()
    
    #使用一个横线和一个弧线组成头的边框
    head=mpatches.Arc(
        [0, 0],
        3,
        3,
        angle=0.0,
        theta1=0.0,
        theta2=90,
    )
    
    ax.add_patch(head)
    ax.axis("equal")

    plt.show()#展示图像

if __name__ == '__main__':
    main()
    # test()