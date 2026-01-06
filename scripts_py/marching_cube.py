import numpy as np
import pyvista
from sklearn.neighbors import KDTree

import mcubes
from skimage import measure


class MarchingCubeUtils(object):
    @staticmethod
    def marching_cubes_from_3D_field(
        xmin: float, xmax: float,
        ymin: float, ymax: float,
        zmin: float, zmax: float,
        resolution: int,
        field: np.ndarray,
        level: float = 0.1
    ) -> dict:
        assert field.ndim == 3, "field must be 3D"
        vertices, triangles = mcubes.marching_cubes(field, level)  # should larger than min mesh size
        vertices[:, 0] = xmin + vertices[:, 0] / (resolution - 1) * (xmax - xmin)
        vertices[:, 1] = ymin + vertices[:, 1] / (resolution - 1) * (ymax - ymin)
        vertices[:, 2] = zmin + vertices[:, 2] / (resolution - 1) * (zmax - zmin)
        return {
            "vertices": vertices,
            "connectivity": triangles
        }
    
    @staticmethod
    def marching_cubes_from_2D_field(
        field: np.ndarray,
        level: float = 0.1
    ) -> dict:
        assert field.ndim == 2, "field must be 2D"
        raw_contours = measure.find_contours(field, level=level)
                
        all_vertices = []
        all_edges = []
        vertex_offset = 0
        for contour in raw_contours:
            if len(contour) < 3:
                raise ValueError("Contour is not a valid contour")
            
            if np.allclose(contour[0], contour[-1]):                
                vertices = np.array(contour[:-1])
                edges = np.hstack([
                    np.arange(vertices.shape[0]).reshape(-1, 1),
                    np.array(list(range(1, vertices.shape[0])) + [0]).reshape(-1, 1)
                ]) + vertex_offset
                
            else:
                vertices = np.array(contour)
                edges = np.hstack([
                    np.arange(vertices.shape[0] - 1).reshape(-1, 1),
                    np.arange(1, vertices.shape[0]).reshape(-1, 1)
                ]) + vertex_offset
                                    
            all_vertices.append(vertices)
            all_edges.append(edges)
            vertex_offset += vertices.shape[0]
        
        if len(all_vertices) > 1:
            vertices = np.concatenate(all_vertices, axis=0)
            edges = np.concatenate(all_edges, axis=0)
        
        return {
            "num_contours": len(raw_contours),
            "vertices": vertices,
            "connectivity": edges
        }
    
    @staticmethod
    def marching_cubes_from_source_points(
        source_positions: np.ndarray, 
        resolution: int,
        level: float = 0.1,
        use_binary_field: bool = False,
        with_pyvista: bool = False,
        **kwargs
    ) -> dict:
        """ level should larger than min mesh size """
        tdim = source_positions.shape[1]
        
        if tdim == 3:
            x = np.linspace(kwargs['xmin'], kwargs['xmax'], resolution)
            y = np.linspace(kwargs['ymin'], kwargs['ymax'], resolution)
            z = np.linspace(kwargs['zmin'], kwargs['zmax'], resolution)
            xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')  # Critical: Use indexing='ij' and build grid in (z,y,x) order for PyMCubes
            xyzs = np.concatenate([xg[..., np.newaxis], yg[..., np.newaxis], zg[..., np.newaxis]], axis=-1)
            cube_points = xyzs.reshape((-1, 3))
                
            tree = KDTree(source_positions)
            dists_list, idxs_list = tree.query(cube_points, k=1, return_distance=True)
            dists = np.concatenate(dists_list, axis=-1)
            
            field = np.zeros(shape=(resolution * resolution * resolution, ))
            if use_binary_field:
                field = dists > level
                level = 0.0
            else:
                field = dists
            field = field.reshape((resolution, resolution, resolution))
            
            res_dict = MarchingCubeUtils.marching_cubes_from_3D_field(
                xmin=kwargs['xmin'], xmax=kwargs['xmax'],
                ymin=kwargs['ymin'], ymax=kwargs['ymax'],
                zmin=kwargs['zmin'], zmax=kwargs['zmax'],
                resolution=resolution,
                field=field, level=level
            )
            
            if with_pyvista:
                res_dict.update({
                    'mesh': pyvista.UnstructuredGrid({pyvista.CellType.TRIANGLE: res_dict['connectivity']}, res_dict['vertices'])
                })
                
        elif tdim == 2:
            x = np.linspace(kwargs['xmin'], kwargs['xmax'], resolution)
            y = np.linspace(kwargs['ymin'], kwargs['ymax'], resolution)
            xg, yg = np.meshgrid(x, y)
            xys = np.concatenate([xg[..., np.newaxis], yg[..., np.newaxis]], axis=-1)
            cube_points = xys.reshape((-1, 2))
            
            tree = KDTree(source_positions)
            dists_list, idxs_list = tree.query(cube_points, k=1, return_distance=True)
            dists = np.concatenate(dists_list, axis=-1)
            
            field = np.zeros(shape=(resolution * resolution, ))
            if use_binary_field:
                field = dists > level
                level = 0.0
            else:
                field = dists
            field = field.reshape((resolution, resolution))
            
            res_dict = MarchingCubeUtils.marching_cubes_from_2D_field(field=field, level=level)
            
            if with_pyvista:
                vertices = res_dict['vertices']
                vertices = np.concatenate([vertices, np.zeros((vertices.shape[0], 1))], axis=1)
                res_dict.update({
                    'mesh': pyvista.UnstructuredGrid({pyvista.CellType.LINE: res_dict['connectivity']}, vertices)
                })
                
        else:
            raise ValueError(f"source_positions must be 2D or 3D, but got {tdim}D")

        return res_dict
        

def debug_3D_case():
    path = np.array([
        [0.2, 0.0, 0.2],
        [0.2, 0.5, 0.2],
        [0.2, 0.5, 0.8],
        [0.8, 0.5, 0.8],
        [0.8, 1.0, 0.8],
    ])
    
    detail_path = []
    for i in range(path.shape[0] - 1):
        step = int(np.linalg.norm(path[i + 1] - path[i]) / 0.025)
        if i == 0:
            detail_path.append(np.linspace(path[i], path[i + 1], step))
        else:
            detail_path.append(np.linspace(path[i], path[i + 1], step)[1:])
    detail_path = np.concatenate(detail_path, axis=0)
    
    # ------ debug sdf case ------
    res_dict = MarchingCubeUtils.marching_cubes_from_source_points(
        xmin=0.0, xmax=1.0,
        ymin=0.0, ymax=1.0,
        zmin=0.0, zmax=1.0,
        source_positions=detail_path,
        resolution=20,
        level=0.1,
        with_pyvista=True
    )
    plotter = pyvista.Plotter()
    plotter.add_mesh(res_dict['mesh'], color='cyan', opacity=0.8, style='wireframe')
    plotter.show()
    # ------ debug sdf case ------
    
    # ------ debug binary case ------
    res_dict = MarchingCubeUtils.marching_cubes_from_source_points(
        xmin=-0.2, xmax=1.2,
        ymin=-0.2, ymax=1.2,
        zmin=-0.2, zmax=1.2,
        source_positions=detail_path,
        resolution=80,
        level=0.1,
        use_binary_field=True,
        with_pyvista=True
    )
    plotter = pyvista.Plotter()
    plotter.add_mesh(res_dict['mesh'], color='cyan', opacity=0.8, style='wireframe')
    plotter.show()
    # ------ debug binary case ------


def debug_2D_case():
    path = np.array([
        [0.2, 0.0],
        [0.2, 0.5],
        [0.8, 0.5],
        [0.8, 1.0],
    ])
    
    detail_path = []
    for i in range(path.shape[0] - 1):
        step = int(np.linalg.norm(path[i + 1] - path[i]) / 0.02)
        if i == 0:
            detail_path.append(np.linspace(path[i], path[i + 1], step))
        else:
            detail_path.append(np.linspace(path[i], path[i + 1], step)[1:])
    detail_path = np.concatenate(detail_path, axis=0)
    
    # ------ debug sdf case ------
    res_dict = MarchingCubeUtils.marching_cubes_from_source_points(
        xmin=-0.0, xmax=1.0,
        ymin=-0.0, ymax=1.0,
        source_positions=detail_path,
        resolution=20,
        level=0.1,
        use_binary_field=False,
        with_pyvista=True
    )
    assert res_dict['num_contours'] == 2, "should be two contours"
    plotter = pyvista.Plotter()
    plotter.add_mesh(res_dict['mesh'], color='cyan', opacity=0.8, style='wireframe')
    plotter.add_mesh(res_dict['mesh'], color='red', style='points')
    plotter.show()
    # ------ debug sdf case ------
    
    # ------ debug binary case ------
    res_dict = MarchingCubeUtils.marching_cubes_from_source_points(
        xmin=-0.2, xmax=1.2,
        ymin=-0.2, ymax=1.2,
        source_positions=detail_path,
        resolution=50,
        level=0.1,
        use_binary_field=True,
        with_pyvista=True
    )
    plotter = pyvista.Plotter()
    plotter.add_mesh(res_dict['mesh'], color='cyan', opacity=0.8, style='wireframe')
    plotter.add_mesh(res_dict['mesh'], color='red', style='points')
    plotter.show()
    # ------ debug binary case ------


if __name__ == "__main__":
    debug_3D_case()
    # debug_2D_case()