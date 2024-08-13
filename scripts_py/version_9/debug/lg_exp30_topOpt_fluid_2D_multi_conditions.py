import shutil
import numpy as np
import ufl
import dolfinx
from typing import Dict, Union, List
from functools import partial
from sklearn.neighbors import KDTree
import os
from ufl import div, inner, grad
from scipy import sparse
from tqdm import tqdm
from mpi4py import MPI
import scipy

from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import AssembleUtils
from scripts_py.version_9.dolfinx_Grad.fluid_tools.fluid_topology_opt import FluidTopoLogyModel
from scripts_py.version_9.dolfinx_Grad.topology_optimize_utils import DistanceBandFilter, DensityBandFilter
from scripts_py.version_9.dolfinx_Grad.recorder_utils import VTKRecorder

"""
1. 先规范写法
2. 再检验流量均衡
3. 再检验符合工况（简单将每个工况视为1个state problem即可）
4. 先考虑斯托克方程
"""

proj_dir = '/home/admin123456/Desktop/work/topopt_exps/multi_fluid_01'
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
        'alpha_max': 3000.0,
        'p': 2.0
    },
    learn_info={
        'max_step': 0.3,
        'min_step': 0.01,
        'max_iteration': 1000,
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
            36: partial(inlet_velocity_exp, dire_axis=0, tdim=model.tdim)
        },
        'output_dict': {
            39: None,
        },
        'boundary_markers': [37, 38, 40, 41, 42],
        'regularization_volume_ratio': 0.15,
        'regularization_volume_weight': 500.0,
        'control_init_value': 0.1,
        'cost_dict': {
            'mini_energy': {
                'type': 'minimum_dissipation',
                'weight': 1.0
            }
        },
        'control_name': 'rho_1'
    },
    {
        'name': '2',
        'input_dict': {
            37: partial(inlet_velocity_exp, dire_axis=0, tdim=model.tdim)
        },
        'output_dict': {
            40: None,
        },
        'boundary_markers': [36, 38, 39, 41, 42],
        'regularization_volume_ratio': -1.0,  # 不需要重复regularization
        'regularization_volume_weight': 200.0,
        'control_init_value': 0.1,
        'cost_dict': {
            'mini_energy': {
                'type': 'minimum_dissipation',
                'weight': 3.0
            }
        },
        'control_name': 'rho_1'
    },
    {
        'name': '3',
        'input_dict': {
            38: partial(inlet_velocity_exp, dire_axis=1, tdim=model.tdim)
        },
        'output_dict': {
            41: None,
        },
        'boundary_markers': [36, 37, 39, 40, 42],
        'regularization_volume_ratio': -1.0,  # 不需要重复regularization
        'regularization_volume_weight': 200.0,
        'control_init_value': 0.1,
        'cost_dict': {
            'mini_energy': {
                'type': 'minimum_dissipation',
                'weight': 1.0
            }
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

model.run_optimize()
