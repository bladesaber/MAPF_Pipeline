import numpy as np
import pandas as pd

def point2line(xyz:np.array, plane_xyz:np.array, plane_vec:np.array):
    x = plane_xyz - xyz
    a = np.linalg.norm(x, ord=2)
    b = np.dot(x, plane_vec.T) / np.linalg.norm(plane_vec, ord=2)
    dist = np.square(a**2 - b**2)
    return dist

def points_define_plane(xyz0:np.array, xyz1:np.array, xyz2:np.array):
    vec1 = xyz0 - xyz1
    vec2 = xyz0 - xyz2
    plane_vec = np.cross(vec1, vec2)
    d = -np.dot(xyz0, plane_vec.T)
    plane_vec = np.array([
        plane_vec[0],
        plane_vec[1],
        plane_vec[2],
        d
    ])
    return plane_vec

def planes_intersect_line(plane1_vec:np.array, plane2_vec:np.array):
    line_vec = np.cross(plane1_vec[:3], plane2_vec[:3])

    xy_cor = np.array([
        [plane1_vec[0], plane1_vec[1]],
        [plane2_vec[0], plane2_vec[1]]
    ])
    ### assume z = 1
    res = np.array([
        [-plane1_vec[2]-plane1_vec[3]],
        [-plane2_vec[2]-plane2_vec[3]]
    ])
    xy = (np.linalg.inv(xy_cor) @ res).T

    return line_vec, np.array([xy[0], xy[1], 1.])
