import os
import numpy as np
import dolfinx
import ufl
from ufl import grad, dot, inner, div
from functools import partial
import argparse
import json

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils, AssembleUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_shapeOptimizer_simple import FluidShapeOptSimple
from scripts_py.version_9.dolfinx_Grad.lagrange_method.cost_functions import ScalarTrackingFunctional, IntegralFunction
from scripts_py.version_9.dolfinx_Grad.lagrange_method.shape_regularization import ShapeRegularization, \
    VolumeRegularization
from scripts_py.version_9.dolfinx_Grad.user_book.step1_project_tool import ImportTool


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Simulation Tool")
    parser.add_argument('--json_file', type=str, default=None)
    parser.add_argument('--init_mesh', type=int, default=0)
    parser.add_argument('--with_debug', type=int, default=0)
    parser.add_argument('--load_guess_res', type=int, default=0)
    args = parser.parse_args()
    return args


args = parse_args()
assert args.json_file is not None

with open(args.json_file, 'r') as f:
    cfg = json.load(f)

if args.init_mesh:
    MeshUtils.msh_to_XDMF(
        name='model', dim=cfg['dim'],
        msh_file=os.path.join(cfg['proj_dir'], 'model.msh'), output_file=os.path.join(cfg['proj_dir'], 'model.xdmf'),
    )
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(cfg['proj_dir'], 'model.xdmf'),
    mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
)

optimize_cfg = cfg['optimize_cfg']
input_markers = [int(marker) for marker in cfg['input_markers'].keys()]
output_markers = cfg['output_markers']
bry_markers = cfg['bry_free_markers'] + cfg['bry_fix_markers']
bry_fixed_markers = cfg['bry_fix_markers'] + input_markers + output_markers
bry_free_markers = cfg['bry_free_markers']

condition_module = ImportTool.import_module(cfg['proj_dir'], cfg['condition_package_name'])
condition_inflow_dict = {}
for marker in input_markers:
    marker_str = str(marker)
    marker_fun_name = cfg['input_markers'][marker_str]
    inflow_fun = ImportTool.get_module_function(condition_module, marker_fun_name)
    condition_inflow_dict[marker] = partial(inflow_fun, tdim=cfg['dim'])

opt = FluidShapeOptSimple(
    domain=domain, cell_tags=cell_tags, facet_tags=facet_tags,
    Re=optimize_cfg['Re'], isStokeEqu=optimize_cfg['isStokeEqu'],
    deformation_cfg=optimize_cfg['deformation_cfg']
)
if args.load_guess_res:
    opt.load_initiation_pickle(
        u_pickle_file=os.path.join(cfg['proj_dir'], optimize_cfg['u_initation_pickle']),
        p_pickle_file=os.path.join(cfg['proj_dir'], optimize_cfg['p_initation_pickle']),
    )

for marker in bry_markers:
    bc_value = dolfinx.fem.Function(opt.V, name=f"bry_u{marker}")
    opt.add_state_boundary(bc_value, marker, is_velocity=True)

for marker in condition_inflow_dict.keys():
    inflow_value = dolfinx.fem.Function(opt.V, name='inflow_u')
    inflow_value.interpolate(condition_inflow_dict[marker])
    opt.add_state_boundary(value=inflow_value, marker=marker, is_velocity=True)

for marker in output_markers:
    bc_out_value = dolfinx.fem.Function(opt.Q, name=f"outflow_p_{marker}")
    opt.add_state_boundary(bc_out_value, marker, is_velocity=False)

for marker in bry_fixed_markers:
    bc_value = dolfinx.fem.Function(opt.V_S, name=f"fix_bry_shape_{marker}")
    opt.add_control_boundary(bc_value, marker)

opt.state_initiation(
    state_ksp_option=optimize_cfg['state_ksp_option'],
    adjoint_ksp_option=optimize_cfg['adjoint_ksp_option'],
    gradient_ksp_option=optimize_cfg['gradient_ksp_option'],
)

# ------ Custom Adjust Here
cost_functional_list = []

# ---------------------------- Example 1: Tracking Goal
# inflow_sum = 0.0
# for marker in condition_inflow_dict.keys():
#     inflow_value = dolfinx.fem.Function(opt.V, name='inflow_u')
#     inflow_value.interpolate(condition_inflow_dict[marker])
#     inflow_sum += -1.0 * AssembleUtils.assemble_scalar(dolfinx.fem.form(
#         ufl.dot(inflow_value, opt.n_vec) * opt.ds(marker)
#     ))
# tracking_goal = inflow_sum / len(output_markers)
# for marker in output_markers:
#     cost_functional_list.append(ScalarTrackingFunctional(
#         domain=opt.domain,
#         integrand_form=ufl.dot(opt.u, opt.n_vec) * opt.ds(marker),
#         tracking_goal=tracking_goal,
#         name='outflow_track'
#     ))

# ---------------------------- Example 2:  Minium Energy
cost_functional_list.append(
    IntegralFunction(
        domain=opt.domain,
        form=inner(grad(opt.u), grad(opt.u)) * ufl.dx,
        name=f"minium_energy"
    )
)

cost_weight = {
    # 'outflow_track': 1.0,
    'minium_energy': 1.0,
}

shape_regularization = ShapeRegularization([
    VolumeRegularization(opt.control_problem, mu=0.5, target_volume_rho=0.6, method='percentage_div')
])

scalar_product_method: dict = optimize_cfg['scalar_product_method']
if scalar_product_method['method'] == 'Poincare-Steklov operator':
    scalar_product_method.update({
        'cell_tags': cell_tags,
        'facet_tags': facet_tags,
        'bry_free_markers': bry_free_markers,
        'bry_fixed_markers': bry_fixed_markers,
    })

opt.optimization_initiation(
    cost_functional_list=cost_functional_list,
    cost_weight=cost_weight,
    shapeRegularization=shape_regularization,
    scalar_product_method=scalar_product_method
)

out_dict = {}
for marker in output_markers:
    out_dict[f"marker_{marker}"] = dolfinx.fem.form(dot(opt.u, opt.n_vec) * opt.ds(marker))
logger_dicts = {
    'outflow': out_dict,
    'energy': {'energy': dolfinx.fem.form(inner(opt.u, opt.u) * ufl.dx)},
    'energy_loss': {'energy_loss': dolfinx.fem.form(inner(grad(opt.u), grad(opt.u)) * ufl.dx)},
    'volume': {'volume': dolfinx.fem.form(dolfinx.fem.Constant(opt.domain, 1.0) * ufl.dx)}
}

opt.solve(
    record_dir=cfg['proj_dir'],
    logger_dicts=logger_dicts,
    max_iter=optimize_cfg['max_iter'],
    max_step_limit=optimize_cfg['max_step_limit'],
    opt_tol_rho=optimize_cfg['opt_tol_rho'],
    with_debug=args.with_debug
)

opt.opt_problem.state_system.solve(domain.comm, with_debug=False)
for name in logger_dicts['outflow'].keys():
    print(f"{name}: {AssembleUtils.assemble_scalar(logger_dicts['outflow'][name])}")

MeshUtils.save_XDMF(os.path.join(cfg['proj_dir'], 'last_model.xdmf'), domain, cell_tags, facet_tags)
