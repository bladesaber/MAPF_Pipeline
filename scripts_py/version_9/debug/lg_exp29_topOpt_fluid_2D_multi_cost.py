import numpy as np
import ufl
import dolfinx
from functools import partial
import os
from mpi4py import MPI

from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_topology_opt import FluidTopoLogyModel
from scripts_py.version_9.dolfinx_finder.topology_optimize_utils import DistanceBandFilter

"""
1. 先规范写法
2. 再检验流量均衡
3. 再检验符合工况（简单将每个工况视为1个state problem即可）
4. 先考虑斯托克方程
"""

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/multi_fluid_02'
model_xdmf = os.path.join(proj_dir, 'model.xdmf')

res_dir = os.path.join(proj_dir, 'res')
if not os.path.exists(res_dir):
    os.mkdir(res_dir)

MeshUtils.msh_to_XDMF(
    name='model',
    msh_file=os.path.join(proj_dir, 'model.msh'),
    output_file=model_xdmf,
    dim=2
)

domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=model_xdmf,
    mesh_name='model',
    cellTag_name='model_cells',
    facetTag_name='model_facets'
)
grid = VisUtils.convert_to_grid(domain)

grad_filter = DistanceBandFilter(grid, radius_filter=0.05)
# grad_filter = DensityBandFilter(grid, radius_filter=0.06)

model = FluidTopoLogyModel(
    domain, cell_tags, facet_tags, grad_filter=grad_filter,
    alpha_function_info={
        'alpha_min': 0.01,
        'alpha_max': 5000.0,
        'p': 2.0
    },
    learn_info={
        'max_step': 0.3,
        'min_step': 0.01,
        'max_iteration': 400,
        'log_step': 10,
        'record_step': 5
    },
    res_dir=res_dir
)


def inlet_velocity_exp(x, dire_axis: int, tdim: int):
    num = x.shape[1]
    values = np.zeros((tdim, num))
    values[dire_axis] = 1.0
    return values


condition_list = [
    {
        'name': '1',
        'input_dict': {
            42: partial(inlet_velocity_exp, dire_axis=0, tdim=model.tdim)
        },
        'output_dict': {
            43: None,
            44: None
        },
        'boundary_markers': [45],
        'regularization_volume_ratio': 0.25,
        'regularization_volume_weight': 200.0,
        'control_init_value': 0.1,
        'cost_dict': {
            'mini_energy': {
                'type': 'minimum_dissipation',
                'weight': 1.0
            },
            'average_flux': {
                'type': 'average_flux',
                'weight': 100.0
            },
        },
        'control_name': 'rho_1'
    }
]

control_dict = {}
for condition in condition_list:
    if condition['control_name'] in control_dict.keys():
        condition.update({'control': control_dict[condition['control_name']]})

    else:
        control = model.generate_control(f"rho_{condition['name']}", init_value=condition['control_init_value'])
        condition.update({'control': control})
        control_dict.update({condition['control_name']: control})

for condition in condition_list:
    other_rhos = []
    for other_condition in condition_list:
        if other_condition['control_name'] != condition['control_name']:
            other_rhos.append(other_condition['control'])

    model.add_condition(
        name=condition['name'],
        input_dict=condition['input_dict'],
        output_dict=condition['output_dict'],
        boundary_markers=condition['boundary_markers'],
        rho=condition['control'],
        other_rhos=other_rhos,
        regularization_volume_ratio=condition['regularization_volume_ratio'],
        regularization_volume_weight=condition['regularization_volume_weight'],
        cost_dict=condition['cost_dict']
    )

model.generate_problem()

# ------ debug for solving state problem
print('[INFO]: Init Flux')
model.state_system.solve(comm=MPI.COMM_WORLD, with_debug=True)
state_u = model.state_system.state_problems[0].state.sub(0)
for condition in condition_list:
    for outlet_marker, _ in condition['output_dict'].items():
        flux = AssembleUtils.assemble_scalar(dolfinx.fem.form(ufl.dot(state_u, model.n_vec) * model.ds(outlet_marker)))
        print(f"outlet {outlet_marker} flux: {flux}")
# ------

model.run_optimize()

# ------ debug for solving state problem
print('[INFO]: Last Flux')
model.state_system.solve(comm=MPI.COMM_WORLD, with_debug=True)
state_u = model.state_system.state_problems[0].state.sub(0)
for condition in condition_list:
    for outlet_marker, _ in condition['output_dict'].items():
        flux = AssembleUtils.assemble_scalar(dolfinx.fem.form(ufl.dot(state_u, model.n_vec) * model.ds(outlet_marker)))
        print(f"outlet {outlet_marker} flux: {flux}")
# ------
