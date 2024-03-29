import os
import numpy as np
import dolfinx
import ufl
from ufl import grad, dot, inner, div
from functools import partial
import pyvista

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_shapeOpt_simple import FluidShapeOptSimple
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional, IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.remesh_helper import MeshQuality
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

# ------ Example 1
# proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book3'
# MeshUtils.msh_to_XDMF(
#     name='model', dim=2,
#     msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
# )
# domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
#     file=os.path.join(proj_dir, 'model.xdmf'),
#     mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
# )
# input_marker = 13
# output_markers = [14]
# bry_markers = [15, 16]
# bry_fixed_markers = [13, 14, 15]
# bry_free_markers = [16]
# isStokeEqu = False
# load_initiation = False
#
# def inflow_velocity_exp(x, tdim):
#     num = x.shape[1]
#     values = np.zeros((tdim, num))
#     values[0] = 6.0 * (1. - x[1]) * x[1]
#     return values


# ------ Example 2
# proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book4'
# MeshUtils.msh_to_XDMF(
#     name='model', dim=2,
#     msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
# )
# domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
#     file=os.path.join(proj_dir, 'model.xdmf'),
#     mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
# )
# input_marker = 18
# output_markers = [19]
# bry_markers = [20, 21]
# bry_fixed_markers = [18, 19, 20]
# bry_free_markers = [21]
# isStokeEqu = True
# load_initiation = False
#
#
# def inflow_velocity_exp(x, tdim):
#     num = x.shape[1]
#     values = np.zeros((tdim, num))
#     values[0] = 6.0 * (1. - x[1]) * x[1]
#     return values


# ------ Example 3
# proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book2'
# MeshUtils.msh_to_XDMF(
#     name='model', dim=3,
#     msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
# )
# domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
#     file=os.path.join(proj_dir, 'model.xdmf'),
#     mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
# )
# # grid = VisUtils.convert_to_grid(domain)
# # plt = pyvista.Plotter()
# # plt.add_mesh(grid, style='wireframe')
# # plt.show()
#
# input_marker = 13
# output_markers = [14]
# bry_markers = [15]
# bry_fixed_markers = [13, 14]
# bry_free_markers = [15]
# isStokeEqu = False
# load_initiation = False
#
#
# def inflow_velocity_exp(x, tdim):
#     num = x.shape[1]
#     values = np.zeros((tdim, num))
#
#     # values[0] = 1.0
#
#     dist = 0.5 - np.sqrt(np.power(x[1] - 0.5, 2) + np.power(x[2] - 0.5, 2))
#     values[0] = np.power(dist, 2.0) * 6.0
#
#     return values

# ------ Example 4
# proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book5'
# # MeshUtils.msh_to_XDMF(
# #     name='model', dim=3,
# #     msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
# # )
# domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
#     file=os.path.join(proj_dir, 'model.xdmf'),
#     mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
# )
# # grid = VisUtils.convert_to_grid(domain)
# # plt = pyvista.Plotter()
# # plt.add_mesh(grid, style='wireframe')
# # plt.show()
#
# input_marker = 17
# output_markers = [18]
# bry_markers = [19, 20]
# bry_fixed_markers = [17, 18, 19]
# bry_free_markers = [20]
# isStokeEqu = False
# load_initiation = True
#
#
# def inflow_velocity_exp(x, tdim):
#     num = x.shape[1]
#     values = np.zeros((tdim, num))
#     dist = 0.7 - np.sqrt(np.power(x[1] - 0.5, 2) + np.power(x[2] - 0.5, 2))
#     values[0] = np.power(dist, 2.0) * 6.0
#     return values

# ------ Example 5
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
        'min_angle': {
            'measure_type': 'min',
            'tol_upper': 180.0,
            'tol_lower': 10.0
        }
    }
}

opt = FluidShapeOptSimple(
    domain=domain, cell_tags=cell_tags, facet_tags=facet_tags, Re=Re, isStokeEqu=isStokeEqu,
    deformation_cfg=deformation_cfg
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
print(f"[Info]: Check PETSc Setting:")
opt.opt_problem.compute_gradient(
    opt.domain.comm,
    state_kwargs={'with_debug': True},
    adjoint_kwargs={'with_debug': True},
    gradient_kwargs={
        'with_debug': True, 'A_assemble_method': 'Identity_row',
    }
)

print(f"[Info]: Check Mesh Deformation Setting:")
is_intersections = opt.deformation_handler.detect_collision(opt.domain)
print(f"[Info]: is_intersections:{is_intersections}")
for measure_method in opt.deformation_handler.quality_measures.keys():
    qualitys = MeshQuality.estimate_mesh_quality(domain, measure_method)
    print(f"[Info]: {measure_method}: {np.min(qualitys)} -> {np.max(qualitys)}")

# ----------------------------------------------------
out_dict = {}
for marker in output_markers:
    out_dict[f"marker_{marker}"] = dolfinx.fem.form(dot(opt.u, opt.n_vec) * opt.ds(marker))
logger_dicts = {
    'outflow': out_dict,
    'energy': {'energy': dolfinx.fem.form(inner(opt.u, opt.u) * ufl.dx)},
    'energy_loss': {'energy_loss': dolfinx.fem.form(inner(grad(opt.u), grad(opt.u)) * ufl.dx)},
    'volume': {'volume': dolfinx.fem.form(dolfinx.fem.Constant(opt.domain, 1.0) * ufl.dx)}
}

opt.solve(record_dir=proj_dir, logger_dicts=logger_dicts, max_iter=150, with_debug=False)

opt.opt_problem.state_system.solve(domain.comm, with_debug=False)
for name in logger_dicts['outflow'].keys():
    print(f"{name}: {AssembleUtils.assemble_scalar(logger_dicts['outflow'][name])}")

MeshUtils.save_XDMF(os.path.join(proj_dir, 'last_model.xdmf'), domain, cell_tags, facet_tags)
