import numpy as np
import dolfinx
from sklearn.neighbors import KDTree

"""
Gmsh -> Ansys Fluent: bdf格式
Ansys Fluent -> Gmsh: cgns格式
"""


class CrossSimulatorUtil(object):
    @staticmethod
    def convert_simple_base_function(domain: dolfinx.mesh.Mesh, coords: np.ndarray, values: np.ndarray, r=1e-4):
        if values.shape[0] == 1:
            function_space = dolfinx.fem.FunctionSpace(domain, ('CG', 1))
        elif values.shape[0] > 1:
            function_space = dolfinx.fem.VectorFunctionSpace(domain, ('CG', 1))
        else:
            raise ValueError('[ERROR]: Non-Valid Function Space')

        tree = KDTree(coords)
        idxs_list, dists_list = tree.query_radius(domain.geometry.x, r=r, return_distance=True)
        coord_idxs = []
        for idxs, dists in zip(idxs_list, dists_list):
            if (len(idxs) != 1) or (not np.isclose(np.min(dists), 0.0)):
                raise ValueError('[ERROR]: Finite Element Mesh is not complete match')
            coord_idxs.append(idxs[0])

        # coords = coords[coord_idxs]
        values = values[coord_idxs, :]

        u = dolfinx.fem.Function(function_space)
        u.x.array[:] = values.reshape(-1)

        return u
