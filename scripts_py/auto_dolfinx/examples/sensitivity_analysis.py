import dolfinx
import ufl
from dolfinx.io.utils import XDMFFile
from mpi4py import MPI

import numpy as np
import pyvista

from thirdparty.pyadjoint.pyadjoint import Tape, ReducedFunctional, Control, compute_gradient
from thirdparty.pyadjoint.pyadjoint.tape import set_working_tape

from scripts_py.mesh_utils import MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils

from scripts_py.auto_dolfinx.types.type_mesh import DolfinxMesh
from scripts_py.auto_dolfinx.types.type_function import DolfinxFunction
from scripts_py.auto_dolfinx.blocks.block_solve import equation_solve
from scripts_py.auto_dolfinx.blocks.block_assemble import assemble_scalar
from scripts_py.auto_dolfinx.blocks.block_assign import assign

from scripts_py.mesh_utils import MeshObj
from scripts_py.vtu_recorder import VtuRecorder


tape = Tape()
set_working_tape(tape)

xdmf_file='/home/admin123456/Desktop/work/gmsh_test/t11/klein.xdmf'
with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as f:
    domain = f.read_mesh(name='mesh')
domain = DolfinxMesh(domain)
mesh = MeshObj(domain=domain)

recorder = VtuRecorder(mesh, save_dir='/home/admin123456/Desktop/work/gmsh_test/t11/cache')

point_scalar_func_space = DolfinUtils.point_scalar_function_space(domain)
# V2 = DolfinUtils.point_scalar_function_space(domain, degree=2)

u = DolfinxFunction(point_scalar_func_space, name='u')  # state
v = DolfinxFunction(point_scalar_func_space, name='v')  # adjoint

g = DolfinxFunction(point_scalar_func_space, name='g')
g.interpolate(lambda x: np.sin(x[2]) * np.cos(x[1]))  # type: ignore

u_old = DolfinxFunction(point_scalar_func_space, name='u_old')  # Solution at the previous time level
assign(func=u_old, other=g, tape=tape)  # type: ignore

nu = 1.0  # Thermal diffusivity
T = 1.
t = 0.0
step = 0.1

F_form = (
    u * v * ufl.dx   # type: ignore
    - u_old * v * ufl.dx  # type: ignore
    + step * nu * ufl.inner(ufl.grad(v), ufl.grad(u)) * ufl.dx  # type: ignore
)

count_num = 0
while t <= T:
    t += step
    
    u = equation_solve(
        domain=domain,
        F_form=F_form,
        state=u,  # type: ignore
        adjoint=v,  # type: ignore
        bcs=[],
        is_linear=False,
        tape=tape,
        adj_kwargs={
            'mat_assemble_method': 'lift',
            'with_residual': False
        }
    )
    # u_old.interpolate(u)  # type: ignore
    assign(func=u_old, other=u, tape=tape)  # type: ignore
    
    recorder.update_point_data(
        key='u', value=u.x.array.copy()  # type: ignore
    )
    recorder.save(step=count_num)
    print(f'step/{count_num}')
    
    count_num += 1

J = assemble_scalar(
    ufl.inner(u, u) * ufl.dx,  # type: ignore
    domain=domain,
    tape=tape,
)
m = Control(
    g, # type: ignore
    riesz_map={'riesz_representation': 'L2', 'comm': domain.comm}
)

dJdm: np.ndarray = compute_gradient(J, m, apply_riesz=True)  # type: ignore
# print(np.min(dJdm), np.max(dJdm))

recorder.update_point_data(key='dJdm', value=dJdm)
recorder.save(step=count_num - 1)

