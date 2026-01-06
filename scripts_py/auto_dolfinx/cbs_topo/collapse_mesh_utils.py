import dolfinx

import numpy as np

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.marching_cube import MarchingCubeUtils
from scripts_py.dolfinx_utils import DolfinUtils


class CollapseMeshObj(MeshObj):
    def __init__(
        self,
        xdmf_file: str| None = None,
        mesh_name="model",
        domain: dolfinx.mesh.Mesh | None = None,
        cell_tags: dolfinx.mesh.MeshTags | None = None,
        facet_tags: dolfinx.mesh.MeshTags | None = None,
    ):
        super().__init__(xdmf_file, mesh_name, domain, cell_tags, facet_tags)
    
    def collapse_mesh(self, cell_func: dolfinx.fem.Function):
        
        
        cell_indices = np.array(MeshUtils.cell_indices_of_points(self.grid, self.points))
        
        inside_bool = cell_indices != -1
        outside_bool = cell_indices == -1
        
        
        # DolfinUtils.function_eval(
        #     uh=cell_func,
        #     pcd_xyz=self.points,
        #     cell_indices=self.cells_to_points
        # )
    