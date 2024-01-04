import numpy as np
import ufl
from basix.ufl import element
import dolfinx
from ufl import inner, grad, div
# import matplotlib.pyplot as plt

from Thirdparty.pyadjoint.pyadjoint import *

from scripts_py.version_9.AD_dolfinx.type_Function import Function
from scripts_py.version_9.AD_dolfinx.type_Mesh import Mesh
from scripts_py.version_9.AD_dolfinx.block_solve import solve
from scripts_py.version_9.AD_dolfinx.block_assemble import assemble
from scripts_py.version_9.AD_dolfinx.type_utils import start_annotation
from scripts_py.version_9.AD_dolfinx.type_DirichletBC import dirichletbc
from scripts_py.version_9.AD_dolfinx.backend_dolfinx import (
    VisUtils, MeshUtils, XDMFRecorder, SolverUtils, TensorBoardRecorder
)

from scripts_py.version_9.AD_dolfinx.type_Function import Function

# ------ create xdmf
msh_file = '/home/admin123456/Desktop/work/topopt_exps/fluid_shape1/pipe.msh'
MeshUtils.msh_to_XDMF(
    name='fluid_2D',
    msh_file=msh_file,
    output_file='/home/admin123456/Desktop/work/topopt_test/fluid_top1/fluid_2D.xdmf',
    dim=2
)
# -------------------
