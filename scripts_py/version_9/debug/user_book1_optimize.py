import os
import numpy as np
import dolfinx
import ufl
from ufl import grad, dot, inner, div
from functools import partial

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_shapeOptimizer_simple import FluidShapeOptSimple
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional, IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.remesh_helper import MeshQuality

# ------ Example 1
proj_dir = '/home/admin123456/Desktop/work/topopt_exps/user_book1'
MeshUtils.msh_to_XDMF(
    name='model', dim=2,
    msh_file=os.path.join(proj_dir, 'model.msh'), output_file=os.path.join(proj_dir, 'model.xdmf'),
)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, 'model.xdmf'),
    mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

input_marker = 1
output_markers = [5, 6, 7]
bry_markers = [2, 3, 4]
bry_fixed_markers = [1, 4, 5, 6, 7]
bry_free_markers = [2, 3]
state_ksp_option = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
adjoint_ksp_option = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}
gradient_ksp_option = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'}


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
    return values


# ----------------------------------------
Re = 100
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
            'tol_lower': 15.0
        }
    }
}
opt = FluidShapeOptSimple(
    domain=domain, cell_tags=cell_tags, facet_tags=facet_tags, Re=Re, isStokeEqu=False,
    deformation_cfg=deformation_cfg
)

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
cost_functional_list = []

# ------ Cost: Tracking Goal
tracking_goal = -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(
    ufl.dot(bc_in_value, opt.n_vec) * opt.ds(input_marker)
)) / 3.0
for marker in output_markers:
    cost_functional_list.append(ScalarTrackingFunctional(
        domain=opt.domain,
        integrand_form=ufl.dot(opt.u, opt.n_vec) * opt.ds(marker),
        tracking_goal=tracking_goal,
        name='outflow_track'
    ))

# ------ Cost: Minium Energy
cost_functional_list.append(
    IntegralFunction(
        domain=opt.domain,
        form=inner(grad(opt.u), grad(opt.u)) * ufl.dx,
        name=f"minium_energy"
    )
)

cost_weight = {
    'outflow_track': 1.0,
    'minium_energy': 1.0,
}

shape_regularization = ShapeRegularization([
    VolumeRegularization(opt.control_problem, mu=0.2, target_volume_rho=0.8, method='percentage_div')
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
    'update_inhomogeneous': False
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
#     gradient_kwargs={
#         'with_debug': True,
#         'A_assemble_method': 'Identity_row',
#     }
# )
#
# print(f"[Info]: Check Mesh Deformation Setting:")
# is_intersections = opt.deformation_handler.detect_collision(opt.domain)
# print(f"[Info]: is_intersections:{is_intersections}")
# for measure_method in opt.deformation_handler.quality_measures.keys():
#     qualitys = MeshQuality.estimate_mesh_quality(domain, measure_method)
#     print(f"[Info]: {measure_method}: {np.min(qualitys)} -> {np.max(qualitys)}")

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
