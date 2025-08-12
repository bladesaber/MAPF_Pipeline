import os
import json

import numpy as np
import pyvista
import meshio

from cad_server_utils import GmshStlReader
from scripts_py.simulator.su2_simulator import Su2Simulator

class SimulationTask:
    def __init__(self):
        self.vertices = []
        self.cells = []
        self.terminal_cfg = []
        self.mesh: pyvista.UnstructuredGrid
        self.su2_reader = None

        self.is_vertices_ready = False
        self.is_cells_ready = False

    @staticmethod    
    def process(
            vertices: list, cells: list, terminal_cfg: dict,
            min_size: float, max_size: float, 
            stl_file: str, su2_file: str,
            thresold: float = 0.1
        ):
        vertices = np.array(vertices)
        cells = np.array(cells)

        assert cells.shape[1] == 4, f"cells.shape: {cells.shape} is not 4"
        assert np.all(cells[:, 0] == 3), f"cells[:, 0]: is not 3"
        mesh = pyvista.UnstructuredGrid(
            {pyvista.CellType.TRIANGLE: cells[:, 1:].reshape(-1)}, vertices
        )
        # mesh.plot(style='wireframe')

        mesh.extract_surface().save(stl_file)
        # meshio.write(stl_file, mesh)

        gmsh_reader = GmshStlReader(stl_file)
        gmsh_reader.read_stl_file()
        gmsh_reader.mesh_stl()
        gmsh_reader.generate_mesh(dim=3, max_size=max_size, min_size=min_size)
        gmsh_reader.export_su2(su2_file)
        gmsh_reader.close()

        su2_reader = Su2Simulator(su2_file)

        mesh_pcd = su2_reader.points
        wall_cells = np.array(su2_reader.marker_dict['wall']['marker_cell'])
        wall_cell_centers = np.mean(mesh_pcd[wall_cells[:, 1:]], axis=1)

        wall_assign_group = np.zeros(wall_cells.shape[0], dtype=int)
        terminal2name = {0: 'wall'}

        for i, terminal_cfg in enumerate(terminal_cfg):
            name = terminal_cfg['name']
            vector = np.array(terminal_cfg['vector']).reshape(1, -1)
            norm_vector = vector / np.linalg.norm(vector)
            circel_center = np.array([terminal_cfg['x'], terminal_cfg['y'], terminal_cfg['z']])
            terminal2name[i + 1] = name.replace('_', '').replace('/', '')

            if terminal_cfg['type'] == 'CircleDisk':
                circle2wall_vector = wall_cell_centers - circel_center
                circle2wall_dist = np.abs(np.sum(circle2wall_vector * norm_vector, axis=1))
                wall_assign_group[circle2wall_dist < thresold] = i + 1
            else:
                raise NotImplementedError(f"terminal_cfg['type']: {terminal_cfg['type']} is not implemented")
        
        new_market_dict = {}
        for market_id, name in terminal2name.items():
            marker_cell = wall_cells[wall_assign_group == market_id]
            new_market_dict[name] = {
                'marker_cell': marker_cell.tolist(),
                'n_elems': marker_cell.shape[0]
            }
        su2_reader.marker_dict = new_market_dict
        su2_reader.encode_su2(su2_file)

        # plotter = pyvista.Plotter()
        # for name, marker_cell in new_market_dict.items():
        #     tmp_centers = np.mean(mesh_pcd[np.array(marker_cell['marker_cell'])[:, 1:]], axis=1)
        #     tmp_mesh = pyvista.PointSet(tmp_centers)
        #     plotter.add_mesh(tmp_mesh, color=np.random.rand(3))
        # plotter.show()

    def run_simulation(self):
        pass


class SimulationServerUtils:
    simulation_map: dict[str, SimulationTask] = {}

