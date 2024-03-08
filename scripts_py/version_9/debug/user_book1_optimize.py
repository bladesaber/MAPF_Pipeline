import os
import numpy as np
import dolfinx
import ufl
from ufl import grad, dot, inner, div
from functools import partial

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.navier_stoke_shape_optimizer import FluidShapeOptSimple
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional, IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.fluid_tools.navier_stoke_simulators import FluidSimulatorIPCS

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

Re = 100


def inflow_velocity_exp(x, tdim):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[0] = 12.0 * (0.0 - x[1]) * (x[1] + 1.0)
    return values


# ----------------------------------------
opt = FluidShapeOptSimple(
    domain=domain, cell_tags=cell_tags, facet_tags=facet_tags, Re=Re, isStokeEqu=False
)

bcs_info_state = []
for marker in bry_markers:
    bc_value = dolfinx.fem.Function(opt.V, name=f"bry_u{marker}")
    bc_dofs = MeshUtils.extract_entity_dofs(
        (opt.W0, opt.V), opt.fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    )
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, opt.W0)
    bcs_info_state.append((bc, opt.W0, bc_dofs, bc_value))

bc_in_value = dolfinx.fem.Function(opt.V, name='inflow_u')
bc_in_value.interpolate(partial(inflow_velocity_exp, tdim=opt.tdim))
bc_in_dofs = MeshUtils.extract_entity_dofs(
    (opt.W0, opt.V), opt.fdim, MeshUtils.extract_facet_entities(domain, facet_tags, input_marker)
)
bc_in1 = dolfinx.fem.dirichletbc(bc_in_value, bc_in_dofs, opt.W0)
bcs_info_state.append((bc_in1, opt.W0, bc_in_dofs, bc_in_value))

for marker in output_markers:
    bc_out_value = dolfinx.fem.Function(opt.Q, name=f"outflow_p_{marker}")
    bc_out_dofs = MeshUtils.extract_entity_dofs(
        (opt.W1, opt.Q), opt.fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    )
    bc_out = dolfinx.fem.dirichletbc(bc_out_value, bc_out_dofs, opt.W1)
    bcs_info_state.append((bc_out, opt.W1, bc_out_dofs, bc_out_value))

# -------
bcs_info_control = []
for marker in bry_fixed_markers:
    bc_value = dolfinx.fem.Function(opt.V_S, name=f"fix_bry_shape_{marker}")
    bc_dofs = MeshUtils.extract_entity_dofs(
        opt.V_S, opt.fdim, MeshUtils.extract_facet_entities(domain, facet_tags, marker)
    )
    bc = dolfinx.fem.dirichletbc(bc_value, bc_dofs, None)
    bcs_info_control.append((bc, opt.V_S, bc_dofs, bc_value))

opt.init_state_problem(
    bcs_info_state=bcs_info_state,
    bcs_info_control=bcs_info_control,
    state_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    adjoint_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
    gradient_ksp_option={'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
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

# ------ Cost: Balance Goal
# cost_functional_list.append(
#     IntegralFunction(
#         domain=opt.domain,
#         form=0.5 * inner(
#             dot(opt.u, opt.n_vec) * opt.ds(5) - dot(opt.u, opt.n_vec) * opt.ds(6),
#             dot(opt.u, opt.n_vec) * opt.ds(5) - dot(opt.u, opt.n_vec) * opt.ds(6)
#         ),
#         name=f"outflow_balance"
#     )
# )
# cost_functional_list.append(
#     IntegralFunction(
#         domain=opt.domain,
#         form=0.5 * inner(
#             dot(opt.u, opt.n_vec) * opt.ds(5) - dot(opt.u, opt.n_vec) * opt.ds(7),
#             dot(opt.u, opt.n_vec) * opt.ds(5) - dot(opt.u, opt.n_vec) * opt.ds(7)
#         ),
#         name=f"outflow_balance"
#     )
# )

# ------ Cost: Minium Energy
cost_functional_list.append(
    IntegralFunction(
        domain=opt.domain,
        form=inner(grad(opt.u), grad(opt.u)) * ufl.dx,
        name=f"minium_energy"
    )
)

cost_weights = {
    'outflow_track': 1.0,
    'minium_energy': 1.0,
}

shape_regularization = ShapeRegularization([
    VolumeRegularization(opt.control_problem, mu=1.0, target_volume_rho=0.8, method='percentage_div')
])

opt.init_opt_problem(
    cost_functional_list=cost_functional_list,
    cost_weights=cost_weights,
    shapeRegularization=shape_regularization,
    bry_free_markers=bry_free_markers,
    bry_fixed_markers=bry_fixed_markers,
)

# ----------------------------------------------------
logger_dicts = {
    'outflow': {
        'marker_5': dolfinx.fem.form(dot(opt.u, opt.n_vec) * opt.ds(5)),
        'marker_6': dolfinx.fem.form(dot(opt.u, opt.n_vec) * opt.ds(6)),
        'marker_7': dolfinx.fem.form(dot(opt.u, opt.n_vec) * opt.ds(7)),
    },
    'energy': {
        'energy': dolfinx.fem.form(inner(opt.u, opt.u) * ufl.dx),
    },
    'energy_loss': {
        'energy_loss': dolfinx.fem.form(inner(grad(opt.u), grad(opt.u)) * ufl.dx),
    },
    'volume': {
        'volume': dolfinx.fem.form(dolfinx.fem.Constant(opt.domain, 1.0) * ufl.dx)
    }
}
opt.set_logger_dicts(logger_dicts)

opt.solve(record_dir=proj_dir, max_iter=150, with_debug=False)

opt.opt_problem.state_system.solve(domain.comm, with_debug=True)
for name in logger_dicts['outflow'].keys():
    print(f"{name}: {AssembleUtils.assemble_scalar(logger_dicts['outflow'][name])}")

MeshUtils.save_XDMF(os.path.join(proj_dir, 'last_model.xdmf'), domain, cell_tags, facet_tags)
