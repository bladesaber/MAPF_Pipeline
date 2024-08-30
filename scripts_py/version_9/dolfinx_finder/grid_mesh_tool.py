import os
import shutil
import numpy as np
import pyvista
from tqdm import tqdm
from typing import Dict, Callable, List, Union
from sklearn.neighbors import KDTree
import dolfinx

# from ..simulator.dolfin_simulators import FluidSimulator
# from ..dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.simulator.dolfin_simulators import FluidSimulator
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

"""
由于边界过于粗糙，仿真极容易失败，故不采用
"""


class GridMesh(object):
    def __init__(self, mesh: pyvista.UnstructuredGrid, xyz_num: np.ndarray):
        self.mesh = mesh
        res = self.generate_mesh(self.mesh, xyz_num=xyz_num)
        self.pcd = res['pcd_data']
        self.faces = res['faces']
        self.cells = res['cells']

    @staticmethod
    def generate_mesh(mesh: pyvista.UnstructuredGrid, xyz_num: np.ndarray):
        points = np.asarray(mesh.points)
        xyz_num = xyz_num.astype(int)

        xyz_min = np.min(points, axis=0)
        xyz_max = np.max(points, axis=0)
        xs = np.linspace(xyz_min[0], xyz_max[0], num=xyz_num[0])
        ys = np.linspace(xyz_min[1], xyz_max[1], num=xyz_num[1])
        zs = np.linspace(xyz_min[2], xyz_max[2], num=xyz_num[2])

        xs, ys, zs = np.meshgrid(xs, ys, zs)
        pcd_data = np.concatenate([
            xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]
        ], axis=-1).reshape((-1, 3))
        pcd_ids = np.arange(0, pcd_data.shape[0], 1).reshape((-1, 1))
        pcd_data = np.concatenate([pcd_data, pcd_ids], axis=1)
        pcd_data = pcd_data.reshape((xyz_num[0], xyz_num[1], xyz_num[2], 4))

        face_xy_group = np.concatenate([
            pcd_data[:-1, :-1, :, 3:4],
            pcd_data[1:, :-1, :, 3:4],
            pcd_data[1:, 1:, :, 3:4],
            pcd_data[:-1, 1:, :, 3:4]
        ], axis=-1).astype(int)
        faces0 = face_xy_group[:, :, :-1, :].reshape(-1, 4)  # XY Bottom
        faces1 = face_xy_group[:, :, 1:, :].reshape(-1, 4)  # XY Upper

        face_xz_group = np.concatenate([
            pcd_data[:-1, :, :-1, 3:4],
            pcd_data[1:, :, :-1, 3:4],
            pcd_data[1:, :, 1:, 3:4],
            pcd_data[:-1, :, 1:, 3:4]
        ], axis=-1).astype(int)
        faces2 = face_xz_group[:, :-1, :, :].reshape(-1, 4)  # YZ Left  # todo why is not XZ ??
        faces3 = face_xz_group[:, 1:, :, :].reshape(-1, 4)  # YZ Right

        face_yz_group = np.concatenate([
            pcd_data[:, :-1, :-1, 3:4],
            pcd_data[:, 1:, :-1, 3:4],
            pcd_data[:, 1:, 1:, 3:4],
            pcd_data[:, :-1, 1:, 3:4],
        ], axis=-1).astype(int)
        faces4 = face_yz_group[:-1, :, :, :].reshape(-1, 4)  # XZ Front
        faces5 = face_yz_group[1:, :, :, :].reshape(-1, 4)  # XZ Back

        cells = np.concatenate([
            pcd_data[:-1, :-1, :-1, 3:4],
            pcd_data[1:, :-1, :-1, 3:4],
            pcd_data[1:, 1:, :-1, 3:4],
            pcd_data[:-1, 1:, :-1, 3:4],
            pcd_data[:-1, :-1, 1:, 3:4],
            pcd_data[1:, :-1, 1:, 3:4],
            pcd_data[1:, 1:, 1:, 3:4],
            pcd_data[:-1, 1:, 1:, 3:4],
        ], axis=-1).reshape((-1, 8)).astype(int)

        pcd_data = pcd_data.reshape((-1, 4))[:, :3]

        # ------ Filter valid cell
        cell_centers = np.mean(pcd_data[cells], axis=1)
        in_mesh_cell_idxs = mesh.find_containing_cell(cell_centers)
        cells_valid = in_mesh_cell_idxs >= 0

        cells = cells[cells_valid, :]
        faces0 = faces0[cells_valid, :]
        faces1 = faces1[cells_valid, :]
        faces2 = faces2[cells_valid, :]
        faces3 = faces3[cells_valid, :]
        faces4 = faces4[cells_valid, :]
        faces5 = faces5[cells_valid, :]

        points_valid = np.zeros(pcd_data.shape[0], dtype=bool)
        points_valid[cells.reshape(-1)] = True
        points_re_idx = np.cumsum(points_valid) - 1

        pcd_data = pcd_data[points_valid, :]
        cells = points_re_idx[cells]
        faces0 = points_re_idx[faces0]
        faces1 = points_re_idx[faces1]
        faces2 = points_re_idx[faces2]
        faces3 = points_re_idx[faces3]
        faces4 = points_re_idx[faces4]
        faces5 = points_re_idx[faces5]

        # ------ Extract Surfaces
        box_length = (xyz_max - xyz_min) / (xyz_num - 1.0)
        faces_surface = GridMesh.extract_surface(
            pcd_data, cells, faces0, faces1, faces2, faces3, faces4, faces5, box_length
        )
        faces0_surface, faces1_surface, faces2_surface, faces3_surface, faces4_surface, faces5_surface = faces_surface

        faces0 = faces0[faces0_surface]  # XY Bottom Surface
        faces1 = faces1[faces1_surface]  # XY Upper Surface
        faces2 = faces2[faces2_surface]  # XZ Front Surface
        faces3 = faces3[faces3_surface]  # XZ Back Surface
        faces4 = faces4[faces4_surface]  # YZ Left Surface
        faces5 = faces5[faces5_surface]  # YZ Right Surface

        return {
            'pcd_data': pcd_data,
            'faces': (faces0, faces1, faces2, faces3, faces4, faces5),
            'cells': cells,
        }

    @staticmethod
    def extract_surface(
            pcd_data: np.ndarray, cells: np.ndarray,
            faces0: np.ndarray, faces1: np.ndarray, faces2: np.ndarray,
            faces3: np.ndarray, faces4: np.ndarray, faces5: np.ndarray,
            box_length: np.ndarray
    ):
        cell_centers = np.mean(pcd_data[cells], axis=1)
        step_norm = np.linalg.norm(box_length) * 1.05
        x_length = np.array([box_length[0], 0., 0.])
        y_length = np.array([0., box_length[1], 0.])
        z_length = np.array([0., 0., box_length[2]])

        faces0_surface = np.zeros(faces0.shape[0], dtype=bool)
        faces1_surface = np.zeros(faces1.shape[0], dtype=bool)
        faces2_surface = np.zeros(faces2.shape[0], dtype=bool)
        faces3_surface = np.zeros(faces3.shape[0], dtype=bool)
        faces4_surface = np.zeros(faces4.shape[0], dtype=bool)
        faces5_surface = np.zeros(faces5.shape[0], dtype=bool)

        tree = KDTree(cell_centers)
        idxs_list = tree.query_radius(cell_centers, r=step_norm)
        for i, (center_xyz, idxs) in enumerate(zip(cell_centers, idxs_list)):
            neg_centers = cell_centers[idxs, :]

            has_bottom = (np.sum(np.abs(neg_centers - (center_xyz - z_length)), axis=1) < 1e-4).sum() > 0
            if not has_bottom:
                faces0_surface[i] = True

            has_upper = (np.sum(np.abs(neg_centers - (center_xyz + z_length)), axis=1) < 1e-4).sum() > 0
            if not has_upper:
                faces1_surface[i] = True

            has_back = (np.sum(np.abs(neg_centers - (center_xyz + y_length)), axis=1) < 1e-4).sum() > 0
            if not has_back:
                faces5_surface[i] = True

            has_front = (np.sum(np.abs(neg_centers - (center_xyz - y_length)), axis=1) < 1e-4).sum() > 0
            if not has_front:
                faces4_surface[i] = True

            has_left = (np.sum(np.abs(neg_centers - (center_xyz - x_length)), axis=1) < 1e-4).sum() > 0
            if not has_left:
                faces2_surface[i] = True

            has_right = (np.sum(np.abs(neg_centers - (center_xyz + x_length)), axis=1) < 1e-4).sum() > 0
            if not has_right:
                faces3_surface[i] = True

        return faces0_surface, faces1_surface, faces2_surface, faces3_surface, faces4_surface, faces5_surface

    @staticmethod
    def generate_unstructured_vtk(pcd: np.ndarray, faces_idx: np.ndarray, cells_idx: np.ndarray):
        if faces_idx.shape[1] == 3:
            faces_idx = np.concatenate([np.full((faces_idx.shape[0], 1), fill_value=3), faces_idx], axis=1)
            faces_type = np.full(faces_idx.shape[0], fill_value=pyvista.CellType.TRIANGLE)
        else:
            faces_idx = np.concatenate([np.full((faces_idx.shape[0], 1), fill_value=4), faces_idx], axis=1)
            faces_type = np.full(faces_idx.shape[0], fill_value=pyvista.CellType.QUAD)

        if cells_idx.shape[1] == 4:
            cells_idx = np.concatenate([np.full((cells_idx.shape[0], 1), fill_value=4), cells_idx], axis=1)
            cells_type = np.full(cells_idx.shape[0], fill_value=pyvista.CellType.TETRA)
        else:
            cells_idx = np.concatenate([np.full((cells_idx.shape[0], 1), fill_value=8), cells_idx], axis=1)
            cells_type = np.full(cells_idx.shape[0], fill_value=pyvista.CellType.HEXAHEDRON)

        elements = np.concatenate([cells_idx.reshape(-1), faces_idx.reshape(-1)], axis=0)
        elements_type = np.concatenate([cells_type, faces_type], axis=0)

        mesh = pyvista.UnstructuredGrid(elements, elements_type, pcd)
        return mesh

    @staticmethod
    def export_mesh_file(
            file: str, pcd: np.ndarray, faces_idx: np.ndarray, cells_idx: np.ndarray,
            is_idx_from_zero=True,
            pcd_group: np.ndarray = None, faces_group: np.ndarray = None, cells_group: np.ndarray = None,
    ):
        assert file.endswith('.mesh')
        dim = pcd.shape[1]

        if is_idx_from_zero:
            faces_idx = faces_idx + 1
            cells_idx = cells_idx + 1

        if pcd_group is None:
            pcd_group = np.ones(pcd.shape[0], dtype=int)
        if faces_group is None:
            faces_group = np.ones(faces_idx.shape[0], dtype=int)
        if cells_group is None:
            cells_group = np.ones(cells_idx.shape[0], dtype=int)

        with open(file, 'w') as f:
            f.write('MeshVersionFormatted 2\n')
            f.write('Dimension\n')
            f.write(f"{dim}\n")

            f.write('Vertices\n')
            f.write(f"{pcd.shape[0]}\n")
            for (x, y, z), group_id in zip(pcd, pcd_group):
                f.write(f"    {x}    {y}    {z}    {group_id}\n")

            face_state = None
            if faces_idx.shape[1] == 3:
                f.write('Triangles\n')
                face_state = 'Triangles'
            elif faces_idx.shape[1] == 4:
                f.write('Quadrilaterals\n')
                face_state = 'Quadrilaterals'

            f.write(f"{faces_idx.shape[0]}\n")

            for face_idx, group_id in zip(faces_idx, faces_group):
                if face_state == 'Triangles':
                    f.write(f"{face_idx[0]} {face_idx[1]} {face_idx[2]} {group_id}\n")
                elif face_state == 'Quadrilaterals':
                    f.write(f"{face_idx[0]} {face_idx[1]} {face_idx[2]} {face_idx[3]} {group_id}\n")

            cell_state = None
            if cells_idx.shape[1] == 4:
                f.write('Tetrahedra\n')
                cell_state = 'Tetrahedra'
            elif cells_idx.shape[1] == 8:
                f.write('Hexahedra\n')
                cell_state = 'Hexahedra'
            f.write(f"{cells_idx.shape[0]}\n")

            for cell_idx, group_id in zip(cells_idx, cells_group):
                if cell_state == 'Tetrahedra':
                    f.write(f"{cell_idx[0]} {cell_idx[1]} {cell_idx[2]} {cell_idx[3]} {group_id}\n")
                elif cell_state == 'Hexahedra':
                    f.write(f"{cell_idx[0]} {cell_idx[1]} {cell_idx[2]} {cell_idx[3]} "
                            f"{cell_idx[4]} {cell_idx[5]} {cell_idx[6]} {cell_idx[7]} "
                            f"{group_id}\n")

            f.write('End')

    @staticmethod
    def define_faces_group(face_group_infos: List[Dict], faces: np.ndarray, pcd: np.ndarray):
        faces_center = np.mean(pcd[faces], axis=1)
        faces_group = np.ones(faces_center.shape[0], dtype=int)
        faces_center_mesh = pyvista.PolyData(faces_center)

        for info in face_group_infos:
            if info['type'] == 'cylinder':
                mesh = pyvista.Cylinder(
                    center=info['center'], direction=info['direction'], radius=info['radius'], height=info['height']
                )
            elif info['type'] == 'box':
                mesh = pyvista.Box(bounds=(
                    info['xmin'], info['xmax'], info['ymin'], info['ymax'], info['zmin'], info['zmax']
                ))
            elif info['type'] == 'sphere':
                mesh = pyvista.Sphere(center=info['center'], radius=info['radius'])
            else:
                raise ValueError('Non Type')

            enclosed_points = faces_center_mesh.select_enclosed_points(mesh, check_surface=False)
            valid = np.asarray(enclosed_points.point_data['SelectedPoints']).astype(bool)
            faces_group[valid] = info['group_id']

            assert info['group_id'] != 1

        return faces_group


class GridSmoothRecorder(object):
    def __init__(
            self, origin_mesh: pyvista.UnstructuredGrid, xyz_num: np.ndarray, record_dir
    ):
        self.mesh_vtk = self.generate_mesh(origin_mesh, xyz_num)

        if os.path.exists(record_dir):
            shutil.rmtree(record_dir)
        os.mkdir(record_dir)
        self.record_dir = record_dir

    @staticmethod
    def generate_mesh(mesh: pyvista.UnstructuredGrid, xyz_num: np.ndarray):
        assert xyz_num[0] == xyz_num[1]  # todo why ?

        points = np.asarray(mesh.points)
        xyz_num = xyz_num.astype(int)

        xyz_min = np.min(points, axis=0)
        xyz_max = np.max(points, axis=0)
        xs = np.linspace(xyz_min[0], xyz_max[0], num=xyz_num[0])
        ys = np.linspace(xyz_min[1], xyz_max[1], num=xyz_num[1])
        zs = np.linspace(xyz_min[2], xyz_max[2], num=xyz_num[2])

        xs, ys, zs = np.meshgrid(xs, ys, zs)
        pcd_data = np.concatenate([
            xs[..., np.newaxis], ys[..., np.newaxis], zs[..., np.newaxis]
        ], axis=-1).reshape((-1, 3))
        pcd_ids = np.arange(0, pcd_data.shape[0], 1).reshape((-1, 1))
        pcd_data = np.concatenate([pcd_data, pcd_ids], axis=1)
        pcd_data = pcd_data.reshape((xyz_num[0], xyz_num[1], xyz_num[2], 4))

        cells = np.concatenate([
            pcd_data[:-1, :-1, :-1, 3:4],
            pcd_data[1:, :-1, :-1, 3:4],
            pcd_data[1:, 1:, :-1, 3:4],
            pcd_data[:-1, 1:, :-1, 3:4],
            pcd_data[:-1, :-1, 1:, 3:4],
            pcd_data[1:, :-1, 1:, 3:4],
            pcd_data[1:, 1:, 1:, 3:4],
            pcd_data[:-1, 1:, 1:, 3:4],
        ], axis=-1).reshape((-1, 8)).astype(int)

        pcd_data = pcd_data.reshape((-1, 4))[:, :3]

        # ------ Filter valid cell
        cell_centers = np.mean(pcd_data[cells], axis=1)
        in_mesh_cell_idxs = mesh.find_containing_cell(cell_centers)
        cells_valid = in_mesh_cell_idxs >= 0

        cells = cells[cells_valid, :]
        points_valid = np.zeros(pcd_data.shape[0], dtype=bool)
        points_valid[cells.reshape(-1)] = True
        points_re_idx = np.cumsum(points_valid) - 1
        pcd_data = pcd_data[points_valid, :]
        cells = points_re_idx[cells]

        return GridSmoothRecorder.generate_unstructured_vtk(pcd_data, cells)

    @staticmethod
    def generate_unstructured_vtk(pcd: np.ndarray, cells_idx: np.ndarray):
        if cells_idx.shape[1] == 4:
            cells_idx = np.concatenate([np.full((cells_idx.shape[0], 1), fill_value=4), cells_idx], axis=1)
            cells_type = np.full(cells_idx.shape[0], fill_value=pyvista.CellType.TETRA)
        else:
            cells_idx = np.concatenate([np.full((cells_idx.shape[0], 1), fill_value=8), cells_idx], axis=1)
            cells_type = np.full(cells_idx.shape[0], fill_value=pyvista.CellType.HEXAHEDRON)

        mesh = pyvista.UnstructuredGrid(cells_idx, cells_type, pcd)
        return mesh

    def mapping_scalar(
            self, name, uh: dolfinx.fem.Function, domain: dolfinx.mesh.Mesh,
            default_value: float, is_point_field=True
    ):
        if is_point_field:
            grid = VisUtils.convert_to_grid(domain)
            point_field = self.mesh_vtk.points
            point_value = np.full(point_field.shape[0], fill_value=default_value)

            is_valid = grid.find_containing_cell(point_field) >= 0
            data = FluidSimulator.eval_point_by_pyvista(uh, point_field[is_valid], domain, grid)
            point_value[is_valid] = data.reshape(-1)
            self.mesh_vtk.point_data[name] = point_value

        else:
            cell_field = self.mesh_vtk.cell_centers().points
            data = FluidSimulator.eval_point_by_pyvista(uh, cell_field, domain, None)
            self.mesh_vtk.cell_data[name] = data.reshape(-1)

    def write(self, step: int):
        self.mesh_vtk.save(os.path.join(self.record_dir, "res_u%.5d.vtu" % step))


def debug_mesh_file_1():
    points = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 1.],
        [1., 1., 1.],
        [0., 1., 1.],
        [0., 0., 2.],
        [1., 0., 2.],
        [1., 1., 2.],
        [0., 1., 2.],
    ])

    faces_idx = np.array([
        [0, 1, 2, 3],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [5, 6, 10, 9],
        [4, 5, 9, 8],
        [4, 7, 11, 8],
        [7, 6, 10, 11],
        [8, 9, 10, 11],
    ])

    cells_idx = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8, 9, 10, 11]
    ])

    GridMesh.export_mesh_file(
        '/home/admin123456/Desktop/work/path_examples/grid_mesher/test.mesh',
        pcd=points, faces_idx=faces_idx, cells_idx=cells_idx,
        is_idx_from_zero=True
    )


def debug_fit_vtk(vtk_file, output_file, xyz_num, face_group_infos=None):
    mesh: pyvista.UnstructuredGrid = pyvista.read(vtk_file)
    res = GridMesh.generate_mesh(mesh, xyz_num=xyz_num)

    pcd, faces_idx, cells_idx = res['pcd_data'], res['faces'], res['cells']
    faces_idx = np.concatenate(list(faces_idx), axis=0)

    if face_group_infos is None:
        faces_group = None
    else:
        faces_group = GridMesh.define_faces_group(face_group_infos, faces_idx, pcd)

    GridMesh.export_mesh_file(
        output_file, res['pcd_data'], faces_idx, res['cells'],
        faces_group=faces_group,
        is_idx_from_zero=True
    )


if __name__ == '__main__':
    # debug_mesh_file_1()

    # debug_fit_vtk(
    #     vtk_file='/home/admin123456/Desktop/work/path_examples/s3_05/model.vtk',
    #     output_file='/home/admin123456/Desktop/work/path_examples/s3_05/grid_model.mesh',
    #     xyz_num=np.array([81, 71, 61])
    # )

    mesh_vtk = pyvista.read('/home/admin123456/Desktop/work/path_examples/s3_05/model.vtk')
    mesh_grid = GridSmoothRecorder.generate_mesh(mesh_vtk, xyz_num=np.array([111, 111, 71]))
    print(mesh_grid.n_cells)
    mesh_grid.plot()
