import numpy as np
import pyvista
import dolfinx
from typing import Union, Callable, List, Sequence, Literal, Dict

from scripts_py.version_9.dolfinx_grad.mesh_utils import MeshObj, MeshUtils
from scripts_py.version_9.dolfinx_grad.deform_geometry import VolumeBspline, AreaBspline


class FFDMeshObj(MeshObj):
    def __init__(
            self,
            ffd_num: List[int], ffd_degree: List[int], fit_nIter=100,
            mesh_file: str = None, mesh_name='model', dim=None,
            domain: dolfinx.mesh.Mesh = None,
            cell_tags: dolfinx.mesh.MeshTags = None,
            facet_tags: dolfinx.mesh.MeshTags = None,
    ):
        super().__init__(
            mesh_file=mesh_file, mesh_name=mesh_name, dim=dim,
            domain=domain, cell_tags=cell_tags, facet_tags=facet_tags
        )

        if self.topology_dim == 2:
            assert len(ffd_num) == 2 and len(ffd_degree) == 2
            mesh_pcd = np.asarray(self.points)[:, :2]
            mesh_min = np.min(mesh_pcd, axis=0)
            mesh_max = np.max(mesh_pcd, axis=0)
            mesh_range = np.maximum(mesh_max - mesh_min, 1.0)
            uv_range = [
                [mesh_min[0] - mesh_range[0], mesh_max[0] + mesh_range[0]],
                [mesh_min[1] - mesh_range[1], mesh_max[1] + mesh_range[1]],
            ]
            self.ffd_obj = AreaBspline(
                uv_range=uv_range, uv_num=ffd_num, uv_degree=ffd_degree, pcd_area=mesh_pcd,
                fit_nIter=fit_nIter, with_torch=True
            )
        elif self.topology_dim == 3:
            raise NotImplementedError
        else:
            raise ValueError('[Error]: Non-valid mesh topology dimension.')

        self.fix_dofs: np.ndarray = None
        self.fix_var: np.ndarray = None

    def deform(self, ffd_cof_dif: np.ndarray):
        self.ffd_obj.deform_coef(ffd_cof_dif)
        pcd_new = self.ffd_obj.get_pcd()
        # if self.fix_dofs is not None:
        #     pcd_new[self.fix_dofs, :] = self.fix_pcd
        self.domain.geometry.x[:] = pcd_new

    def set_fix_boundary(self, fix_dofs: np.ndarray):
        self.fix_dofs = fix_dofs
        self.fix_var = np.asarray(self.points[self.fix_dofs, :])

    def compute_cof_grad(self, pcd_grad: np.ndarray):
        _, cof_dif = self.ffd_obj.compute_coef_gradient(pcd_grad)
        return cof_dif

    def draw_mesh(self):
        plotter = pyvista.Plotter()
        plotter.add_mesh(self.grid, style='wireframe')
        if self.fix_dofs is not None:
            plotter.add_mesh(pyvista.PointSet(self.points[self.fix_dofs, :]), color=(0., 0., 1.))
        plotter.show()
