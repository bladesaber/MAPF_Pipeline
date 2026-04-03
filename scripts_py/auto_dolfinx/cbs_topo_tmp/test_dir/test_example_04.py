import os

import numpy as np
import pyvista

from scripts_py.auto_dolfinx.cbs_topo.cbs_base import CbsMeshObj
from scripts_py.auto_dolfinx.cbs_topo.utils_shape import ShapeUtils


base_dir = '/home/admin123456/Desktop/work/gmsh_test/t37'

mesh = CbsMeshObj(
    msh_file=os.path.join(base_dir, 'model.msh'),
    cell_mode=False,
    xdmf_file=os.path.join(base_dir, 'model.xdmf'),
    is_standard_mesh=True
)
mesh.create_neighbors_info()

cell_weights = pyvista.read(os.path.join(base_dir, 'test.vtu')).cell_data['topo_weight']
cell_weights = (cell_weights > 0.5).astype(np.float32)
new_cell_weight = ShapeUtils.convert_weight_to_ConvexHull(mesh, cell_weights)

mesh.grid.cell_data['new_cell_weight'] = new_cell_weight
mesh.grid.save(os.path.join(base_dir, 'test_1.vtu'))

new_cell_weight = mesh.to_standard_ConvexHull_cell(new_cell_weight, mode='inflate')
mesh.grid.cell_data['new_cell_weight'] = new_cell_weight
mesh.grid.save(os.path.join(base_dir, 'test_2.vtu'))