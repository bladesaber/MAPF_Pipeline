import dolfinx
import ufl

import numpy as np
import pyvista
from sklearn.neighbors import KDTree
from itertools import combinations

from scripts_py.mesh_utils import MeshUtils


class ReinitializationSolver(object):
    '''reinitialization level set function to a signed distance function without moving the zero-level set'''    
    @staticmethod
    def solve(domain: dolfinx.mesh.Mesh, phi: dolfinx.fem.Function) -> dolfinx.fem.Function:
        grid: pyvista.UnstructuredGrid = MeshUtils.mesh_to_vtk(domain)
        phi_np = phi.x.array.copy()
        cell_conns =grid.cell_connectivity.reshape((-1, 3))

        cell_phi = phi_np[cell_conns]
        cell_phi_min = np.min(cell_phi, axis=1)
        cell_phi_max = np.max(cell_phi, axis=1)
        cell_surface_bool = np.bitwise_and(cell_phi_min <= 0.0, cell_phi_max >= 0.0)
        
        surface_points = []
        surface_cell_conns = cell_conns[cell_surface_bool]
        for conns in surface_cell_conns:
            local_surface_points = []
            for p_idx in conns:
                p_phi = phi_np[p_idx]
                if p_phi == 0.0:
                    local_surface_points.append(grid.points[p_idx])
            
            for p1, p2 in combinations(conns, 2):
                p1_point = grid.points[p1]
                p2_point = grid.points[p2]
                p1_phi = phi_np[p1]
                p2_phi = phi_np[p2]
                if p1_phi * p2_phi < 0.0:
                    weight = np.abs(p1_phi) / (np.abs(p1_phi) + np.abs(p2_phi))
                    mid_p = p1_point + weight * (p2_point - p1_point)
                    local_surface_points.append(mid_p)
            
            local_surface_points_np = np.array(local_surface_points)
            local_surface_points.append(np.mean(local_surface_points_np, axis=0))
            surface_points.extend(local_surface_points)
        
        surface_points = np.array(surface_points)
        
        # # ------ visualize the surface points ------
        # plotter = pyvista.Plotter()
        # plotter.add_mesh(grid, color='black', style='wireframe')
        # # plotter.add_mesh(pyvista.PointSet(grid.cell_centers().points[cell_surface_bool]), color='green', point_size=8)
        # plotter.add_mesh(pyvista.PointSet(grid.points[phi_np==1.0]), color='red', point_size=8)
        # plotter.add_mesh(pyvista.PointSet(grid.points[phi_np==-1.0]), color='blue', point_size=8)
        # plotter.add_mesh(pyvista.PointSet(surface_points), color='orange', point_size=8)
        # plotter.show()
        # # ------
        
        # reconstruct the signed distance function on the surface
        surface_tree = KDTree(surface_points)
        dists_list, idxs_list = surface_tree.query(grid.points, k=1, return_distance=True)
        abs_sdf = np.concatenate(dists_list, axis=-1)
        phi.x.array[:] = np.sign(phi_np) * abs_sdf
        
        return phi
