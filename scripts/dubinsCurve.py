import numpy as np
import math
import sympy
from sympy.vector import CoordSys3D
from sympy import MatrixSymbol
from sympy.vector import Vector

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

def test():
    from sympy import Point, Line

    p1 = Point(1, 2, 3)
    p2 = Point(4, 5, 6)
    l1 = Line(p1, p2)
    # print(l1)

    print(type(l1.direction))

    # xyz3 = np.array([])
    #
    # vec1_s = xyz3 - xyz1
    # vec2_s = xyz3 - xyz2
    # plane1_vec = np.cross(vec1_s, vec1)
    # plane2_vec = np.cross(vec2_s, vec2)
    # vec3 = np.cross(plane1_vec, plane2_vec)
    #
    # ### dist1 = r
    # a1 = np.linalg.norm(xyz3 - xyz1, ord=2)
    # b1 = np.dot(xyz3 - xyz1, vec3.T) / np.linalg.norm(vec3, ord=2)
    # dist1 = np.square(a1 ** 2 - b1 ** 2)
    #
    # ### dist2 = r
    # a2 = np.linalg.norm(xyz3 - xyz2, ord=2)
    # b2 = np.dot(xyz3 - xyz2, vec3.T) / np.linalg.norm(vec3, ord=2)
    # dist2 = np.square(a2 ** 2 - b2 ** 2)

if __name__ == '__main__':
    # x = sympy.Symbol('x')
    # y = sympy.Symbol('y')
    #
    # expr = sympy.Equality(x**2 + y**2, 1)
    # print(expr)
    #
    # r = sympy.solve(expr, y, dict=True)
    # print(r)

    test()