import dolfinx

import numpy as np
import pyvista

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.deform_geometry import VolumeBspline, SurfaceBspline


class FFDMeshObj(MeshObj):
    '''
    本意是通过FFD变形来优化Shape Optimization的变形,但由于未能处理Fixed边界的问题,所以暂时无法使用
    '''
    
    def __init__(
        self,
        ffd_num: list[int], ffd_degree: list[int], fit_nIter=100,
        xdmf_file: str| None = None,
        mesh_name="model",
        domain: dolfinx.mesh.Mesh | None = None,
        cell_tags: dolfinx.mesh.MeshTags | None = None,
        facet_tags: dolfinx.mesh.MeshTags | None = None
    ):
        super().__init__(
            xdmf_file=xdmf_file, mesh_name=mesh_name,
            domain=domain, cell_tags=cell_tags, facet_tags=facet_tags
        )

        if self.tdim == 2:
            assert len(ffd_num) == 2 and len(ffd_degree) == 2
            mesh_pcd = np.asarray(self.points)[:, :2]
            mesh_min = np.min(mesh_pcd, axis=0)
            mesh_max = np.max(mesh_pcd, axis=0)
            mesh_range = np.maximum(mesh_max - mesh_min, 1.0)
            uv_range = [
                [mesh_min[0] - mesh_range[0], mesh_max[0] + mesh_range[0]],
                [mesh_min[1] - mesh_range[1], mesh_max[1] + mesh_range[1]],
            ]
            self.ffd_obj = SurfaceBspline(
                pcd_surface=mesh_pcd, uv_num=ffd_num, uv_degree=ffd_degree, fit_nIter=fit_nIter
            )
        elif self.tdim == 3:
            raise NotImplementedError
        else:
            raise ValueError('[Error]: Non-valid mesh topology dimension.')

