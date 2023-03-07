import numpy as np
from scipy.special import comb

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

def bezier_smooth(xyzs: np.array, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = xyzs.shape[0]
    xs = xyzs[:, 0]
    ys = xyzs[:, 1]
    zs = xyzs[:, 2]

    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xs_new = np.dot(xs, polynomial_array)
    ys_new = np.dot(ys, polynomial_array)
    zs_new = np.dot(zs, polynomial_array)

    smooth_path = np.concatenate([
        xs_new.reshape((-1, 1)),
        ys_new.reshape((-1, 1)),
        zs_new.reshape((-1, 1)),
    ], axis=1)

    return smooth_path
