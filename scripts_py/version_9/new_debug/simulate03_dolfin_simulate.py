import numpy as np
import dolfinx
import os
from functools import partial

from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.simulator.dolfin_simulator import StokeSimulator
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils

proj_dir = '/home/admin123456/Desktop/work/path_examples/topo_path_03'

model_name = 'model'
msh_name = f"{model_name}.msh"
xdmf_name = f"{model_name}.xdmf"
inlet_marker = 20
outlet_marker = 21
boundary_marker = 22

# model_name = 'grid_model'
# msh_name = f"{model_name}.msh"
# xdmf_name = f"{model_name}.xdmf"
# inlet_marker = 5
# outlet_marker = 6
# boundary_marker = 7

MeshUtils.msh_to_XDMF(
    msh_file=os.path.join(proj_dir, msh_name),
    output_file=os.path.join(proj_dir, xdmf_name),
    dim=3
)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, xdmf_name),
)


def inlet_velocity_expression(x, tdim: int, velocity_vector: np.ndarray):
    assert velocity_vector.shape[0] == tdim
    num = x.shape[1]
    values = np.tile(velocity_vector[np.newaxis, :], (num, 1)).T
    return values


model = StokeSimulator('stoke', domain, cell_tags, facet_tags)
model.add_boundary(
    'inlet', value=partial(inlet_velocity_expression, tdim=model.tdim, velocity_vector=np.array([1., 0., 0.])),
    marker=inlet_marker, is_velocity=True
)
model.add_boundary('outlet', value=0.0, marker=outlet_marker, is_velocity=False)
model.add_boundary('boundary', value=0.0, marker=boundary_marker, is_velocity=True)

model.run(
    result_dir='/home/admin123456/Desktop/work/path_examples/topo_path_03',
    name='stoke'
)
