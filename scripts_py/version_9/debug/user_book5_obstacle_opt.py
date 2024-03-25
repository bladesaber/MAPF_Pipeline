import os
import shutil
import numpy as np
import dolfinx
import ufl
from ufl import grad, dot, inner, div
from functools import partial
import pyvista

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_shapeOptimizer_Obstacle import FluidShapeOptObstacle
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.recorder_utils import TensorBoardRecorder
from scripts_py.version_9.dolfinx_Grad.collision_objs import ObstacleCollisionObj

from scripts_py.version_9.app.env_utils import Shape_Utils

"""
1. 实际上MeshQuality的检测很大可能使变形不充分
2. 粘连管壁方法也会造成变形不充分(有可能切换边界条件是可以解决的，例如改为纽曼边界,或者更换为定距Dire边界)
   粘连方法在实际中还是不可用，尤其是多管道中，必须考虑其他方法，产生锯齿优化
3. 分层优化：a.dolfenx产生目标空间 b.将整个网格考虑为弹性体，再做二次优化
4. 场优化，将排斥区域考虑为场，参考possion优化。利用可能碰撞的局部点构造一个简略的不完整场即可，方便表达式表述，
   不要构造全空间的场
"""

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book7'
MeshUtils.msh_to_XDMF(
    name='model', dim=3,
    msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, 'model.xdmf'),
    mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

input_marker = 56
output_markers = [57]
bry_markers = [58, 59]
bry_fixed_markers = [56, 57, 58]
bry_free_markers = [59]
isStokeEqu = False
beta_rho = 3.0 / 4.0
deformation_lower = 1e-2
load_initiation = True
u_pickle_file = '/home/admin123456/Desktop/work/topopt_exps/user_book7/init_step_100/navier_stoke_u.pkl'
p_pickle_file = '/home/admin123456/Desktop/work/topopt_exps/user_book7/init_step_100/navier_stoke_p.pkl'


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    dist = 0.5 - np.sqrt(np.power(x[1] - 0.5, 2) + np.power(x[2] - 0.5, 2))
    values[0] = np.power(dist, 2.0) * 6.0
    return values


obs_pcd_0 = Shape_Utils.create_CylinderPcd(
    xyz=np.array([4.25, 1.5, 2.0]), radius=0.75, height=4.0, direction=np.array([0, 0, 1]), reso=0.1
)
obs_pcd_1 = Shape_Utils.create_CylinderPcd(
    xyz=np.array([1.7, 1.8, 2.0]), radius=0.75, height=4.0, direction=np.array([0, 0, 1]), reso=0.1
)
obs_coords = np.concatenate([obs_pcd_0, obs_pcd_1], axis=0)
obs_obj = ObstacleCollisionObj(obs_coords)
obs_obj.save_vtu(os.path.join(proj_dir, 'obstacles', 'obs1.vtu'))
obs_objs = [obs_obj]
# obs_objs = []

# ------ debug vis environment
# grid = VisUtils.convert_to_grid(domain)
# plt = pyvista.Plotter()
# plt.add_mesh(grid, style='wireframe')
# plt.add_mesh(pyvista.PointSet(obs_obj.get_coords()), style='wireframe')
# plt.show()

# ------------------------------------------------------------------------------------------------
state_ksp_option = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
adjoint_ksp_option = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
gradient_ksp_option = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}

Re = 100.

deformation_cfg = {
    'volume_change': 0.15,
    'quality_measures': {
        # max_angle is not support for 3D
        # 'max_angle': {
        #     'measure_type': 'max',
        #     'tol_upper': 165.0,
        #     'tol_lower': 0.0
        # },
        # 'min_angle': {
        #     'measure_type': 'min',
        #     'tol_upper': 180.0,
        #     'tol_lower': 10.0
        # }
    }
}

opt = FluidShapeOptObstacle(
    name='player1',
    domain=domain, cell_tags=cell_tags, facet_tags=facet_tags, bry_markers=bry_markers,
    conflict_radius=0.1, conflict_offset=0.05, freeze_radius=0.05,
    Re=Re, isStokeEqu=isStokeEqu, deformation_cfg=deformation_cfg,
    beta_rho=beta_rho, deformation_lower=deformation_lower
)
if load_initiation:
    opt.load_initiation_pickle(u_pickle_file=u_pickle_file, p_pickle_file=p_pickle_file)

for marker in bry_markers:
    bc_value = dolfinx.fem.Function(opt.V, name=f"bry_u{marker}")
    opt.add_state_boundary(bc_value, marker, is_velocity=True)

bc_in_value = dolfinx.fem.Function(opt.V, name='inflow_u')
bc_in_value.interpolate(partial(inflow_velocity_exp, tdim=opt.tdim))
opt.add_state_boundary(bc_in_value, input_marker, is_velocity=True)

for marker in output_markers:
    bc_out_value = dolfinx.fem.Function(opt.Q, name=f"outflow_p_{marker}")
    opt.add_state_boundary(bc_out_value, marker, is_velocity=False)

for marker in bry_fixed_markers:
    bc_value = dolfinx.fem.Function(opt.V_S, name=f"fix_bry_shape_{marker}")
    opt.add_control_boundary(bc_value, marker)

# --------------------------------------
opt.state_initiation(
    state_ksp_option=state_ksp_option, adjoint_ksp_option=adjoint_ksp_option, gradient_ksp_option=gradient_ksp_option,
)

# -----------------------------------------------------
cost_functional_list = [
    IntegralFunction(
        domain=opt.domain,
        form=inner(grad(opt.u), grad(opt.u)) * ufl.dx,
        name=f"minium_energy"
    )
]
cost_weight = {
    'minium_energy': 1.0
}

shape_regularization = ShapeRegularization([
    VolumeRegularization(
        opt.control_problem,
        mu=1.0,
        target_volume_rho=0.5,
        method='percentage_div'
    )
])

scalar_product_method = {
    'method': 'Poincare-Steklov operator',
    'lambda_lame': 1.0,  # it is very important here
    'damping_factor': 0.2,  # it is very important here
    'cell_tags': cell_tags,
    'facet_tags': facet_tags,
    'bry_free_markers': bry_free_markers,
    'bry_fixed_markers': bry_fixed_markers,
    'mu_fix': 1.0,
    'mu_free': 1.0,
    'use_inhomogeneous': True,
    'inhomogeneous_exponent': 1.0,
    'update_inhomogeneous': True
}
# scalar_product_method = {
#     'method': 'default'
# }
opt.optimization_initiation(
    cost_functional_list=cost_functional_list,
    cost_weight=cost_weight,
    shapeRegularization=shape_regularization,
    scalar_product_method=scalar_product_method
)

# ------ Check Whether PETSc Setting Valid
# print(f"[Info]: Check PETSc Setting:")
# opt.opt_problem.compute_gradient(
#     opt.domain.comm,
#     state_kwargs={'with_debug': True},
#     adjoint_kwargs={'with_debug': True},
#     gradient_kwargs={'with_debug': True, 'A_assemble_method': 'Identity_row'}
# )
#
# print(f"[Info]: Check Mesh Deformation Setting:")
# is_intersections = opt.deformation_handler.detect_collision(opt.domain)
# print(f"[Info]: is_intersections:{is_intersections}")
# for measure_method in opt.deformation_handler.quality_measures.keys():
#     qualitys = MeshQuality.estimate_mesh_quality(domain, measure_method)
#     print(f"[Info]: {measure_method}: {np.min(qualitys)} -> {np.max(qualitys)}")

# ----------------------------------------------------
logger_dicts = {
    'volume': {'volume': dolfinx.fem.form(dolfinx.fem.Constant(opt.domain, 1.0) * ufl.dx)}
}

tensorBoard_dir = os.path.join(proj_dir, 'log')
if os.path.exists(tensorBoard_dir):
    shutil.rmtree(tensorBoard_dir)
os.mkdir(tensorBoard_dir)
log_recorder = TensorBoardRecorder(tensorBoard_dir)

log_dict = opt.init_solve_cfg(
    record_dir=proj_dir,
    logger_dicts=logger_dicts,
    with_debug=False
)
FluidShapeOptObstacle.log_dict(log_recorder, [log_dict], step=0)

step = 0
while True:
    step += 1
    res_dict = opt.single_solve(obs_objs=obs_objs, mesh_objs=[], step=step, with_debug=False)

    if not res_dict['state']:
        break

    log_dict = res_dict['log_dict']
    FluidShapeOptObstacle.log_dict(log_recorder, [log_dict], step=step)

    if res_dict['is_converge']:
        break

    if step > 100:
        break

opt.opt_problem.state_system.solve(domain.comm, with_debug=False)
MeshUtils.save_XDMF(os.path.join(proj_dir, 'last_model.xdmf'), domain, cell_tags, facet_tags)
