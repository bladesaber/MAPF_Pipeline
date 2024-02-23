"""
Ref: https://github.com/dolfin-adjoint/dolfin-adjoint/tree/main/examples/shape-optimization-pipe
"""

import os
import shutil

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.remesh_helper import ReMesher

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/remesh_01'
remesh_dir = os.path.join(proj_dir, 'remesh_debug')
if os.path.exists(remesh_dir):
    shutil.rmtree(remesh_dir)
os.mkdir(remesh_dir)

# MeshUtils.msh_to_XDMF(
#     name='model',
#     msh_file=os.path.join(proj_dir, 'model.msh'),
#     output_file=os.path.join(proj_dir, 'model.xdmf'),
#     dim=2
# )

orig_xdmf_file = os.path.join(proj_dir, 'model.xdmf')
orig_msh_file = os.path.join(proj_dir, 'model.msh')
save_dir = remesh_dir
maxSize = 0.8
model_name = 'model'

for i in range(5):
    domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
        file=orig_xdmf_file,
        mesh_name=model_name,
        cellTag_name=f'{model_name}_cells',
        facetTag_name=f'{model_name}_facets'
    )

    vertex_indices = ReMesher.reconstruct_vertex_indices(
        orig_msh_file=orig_msh_file,
        domain=domain,
    )

    msh_file, xdmf_file = ReMesher.remesh_run(
        domain=domain,
        vertex_indices=vertex_indices,
        orig_msh_file=orig_msh_file,
        minSize=0.0, maxSize=maxSize, dim=2,
        save_dir=save_dir,
        model_name=f"remesh_{i}",
        tmp_dir=None
    )

    maxSize = maxSize / 2.0
    orig_msh_file = msh_file
    orig_xdmf_file = xdmf_file
    model_name = f"remesh_{i}"

    domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
        file=orig_xdmf_file,
        mesh_name=model_name,
        cellTag_name=f'{model_name}_cells',
        facetTag_name=f'{model_name}_facets'
    )
    print('[Num]: ', MeshUtils.num_of_entities(domain, 2), '\n')


