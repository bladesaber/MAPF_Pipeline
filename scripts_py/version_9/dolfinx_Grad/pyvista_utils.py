import numpy as np
import pyvista


class PyVistaUtils(object):
    @staticmethod
    def save_vtk(grid: pyvista.DataSet, file: str):
        assert file.endswith('.vtk')
        pyvista.save_meshio(file, grid)

    @staticmethod
    def create_grid(coordinates: np.ndarray, connectivity: np.ndarray[np.ndarray[int]]):
        """
        Cell Type:
        * ``EMPTY_CELL = 0``
        * ``VERTEX = 1``
        * ``POLY_VERTEX = 2``
        * ``LINE = 3``
        * ``POLY_LINE = 4``
        * ``TRIANGLE = 5``
        * ``TRIANGLE_STRIP = 6``
        * ``POLYGON = 7``
        * ``PIXEL = 8``
        * ``QUAD = 9``
        * ``TETRA = 10``
        * ``VOXEL = 11``
        * ``HEXAHEDRON = 12``
        * ``WEDGE = 13``
        * ``PYRAMID = 14``
        * ``PENTAGONAL_PRISM = 15``
        * ``HEXAGONAL_PRISM = 16``
        * ``QUADRATIC_EDGE = 21``
        * ``QUADRATIC_TRIANGLE = 22``
        * ``QUADRATIC_QUAD = 23``
        * ``QUADRATIC_POLYGON = 36``
        * ``QUADRATIC_TETRA = 24``
        * ``QUADRATIC_HEXAHEDRON = 25``
        * ``QUADRATIC_WEDGE = 26``
        * ``QUADRATIC_PYRAMID = 27``
        * ``BIQUADRATIC_QUAD = 28``
        * ``TRIQUADRATIC_HEXAHEDRON = 29``
        * ``QUADRATIC_LINEAR_QUAD = 30``
        * ``QUADRATIC_LINEAR_WEDGE = 31``
        * ``BIQUADRATIC_QUADRATIC_WEDGE = 32``
        * ``BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33``
        * ``BIQUADRATIC_TRIANGLE = 34``
        """

        if coordinates.shape[1] == 2:
            coordinates_xyz = np.zeros((coordinates.shape[0], 3))
            coordinates_xyz[:, :2] = coordinates
        else:
            coordinates_xyz = coordinates

        cells, cell_types = [], []
        for connectivity_tuple in connectivity:
            if connectivity_tuple.shape[0] == 3:
                cell_type = 5
            else:
                raise NotImplementedError("Unexpected Cell Type")

            cells.append([connectivity_tuple.shape[0]] + list(connectivity_tuple))
            cell_types.append(cell_type)

        cells = np.array(cells).reshape(-1)
        cell_types = np.array(cell_types)

        grid = pyvista.UnstructuredGrid(cells, cell_types, coordinates_xyz)

        return grid
