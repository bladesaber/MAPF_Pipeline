import numpy as np
import math

import sympy
### similar to mathematica

from sympy import Symbol
from sympy.vector import CoordSys3D
from sympy import MatrixSymbol
from sympy.vector import Vector
from sympy import Point3D, Line3D
from sympy import symbols
from sympy import separatevars

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

def compute_perfect_DubinsCruve():
    ### too complex to solve

    ### define environment
    coodr = CoordSys3D('coodr')
    ps_x, ps_y, ps_z = symbols('ps_x ps_y ps_z')
    vs_x, vs_y, vs_z = symbols('vs_x vs_y vs_z')
    point_s = Point3D(ps_x, ps_y, ps_z)
    vec_s = vs_x * coodr.i + vs_y * coodr.j + vs_z * coodr.k

    pt_x, pt_y, pt_z = symbols('pt_x pt_y pt_z')
    vt_x, vt_y, vt_z = symbols('vt_x vt_y vt_z')
    point_t = Point3D(pt_x, pt_y, pt_z)
    vec_t = vt_x * coodr.i + vt_y * coodr.j + vt_z * coodr.k

    radius = sympy.Symbol('radius')

    ### define denpedent variables
    pm_x, pm_y, pm_z = symbols('pm_x pm_y pm_z')
    point_m = Point3D(pm_x, pm_y, pm_z)

    ### compute two plane's intersect line vector
    ### Step 1: compute norm vector vector_sm of point_s->point_m and vector_s
    ### Step 2: compute norm vector vector_tm of point_t->point_m and vector_t
    ### Step 3: two plane's intersect line vector norm_tm
    vec_sm = (point_m.x - point_s.x) * coodr.i + \
             (point_m.y - point_s.y) * coodr.j + \
             (point_m.z - point_s.z) * coodr.k
    norm_sm = vec_sm.cross(vec_s)

    vec_tm = (point_m.x - point_t.x) * coodr.i + \
             (point_m.y - point_t.y) * coodr.j + \
             (point_m.z - point_t.z) * coodr.k
    norm_tm = vec_tm.cross(vec_t)

    vec_m = (norm_sm.cross(norm_tm)).normalize()

    ### define equation
    ### Eq1: distance between point_s and line(point_m, norm_tm) is radius
    ### Eq1: distance between point_t and line(point_m, norm_tm) is radius
    length_ms = sympy.sqrt(vec_sm.dot(vec_sm) - (-vec_sm).dot(vec_m))
    length_mt = sympy.sqrt(vec_tm.dot(vec_tm) - (-vec_tm).dot(vec_m))

    expr1 = sympy.Equality(length_ms, radius)
    expr2 = sympy.Equality(length_mt, radius)

    r = sympy.solve([expr1, expr2], [pm_x, pm_y, pm_z], dict=True)

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