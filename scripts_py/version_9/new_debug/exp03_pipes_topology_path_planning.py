"""
Fail，失败案例，目前Topology Optimisation无法应用于多管道路径生成。同时密窄空间下，可能由于雷诺数过大，导致生成形状无任何规律
"""
import shutil
import numpy as np
import os
import json

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.topology_path_planner.grad_runner import RasGradRunner, StokeGradRunner
from scripts_py.version_9.topology_path_planner.optimize_runner import TopologyPathSolver

proj_dir = '/home/admin123456/Desktop/work/path_examples/s3_05'

MeshUtils.msh_to_XDMF(
    msh_file=os.path.join(proj_dir, 'model.msh'),
    output_file=os.path.join(proj_dir, 'model.xdmf'),
    dim=3
)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(file=os.path.join(proj_dir, 'model.xdmf'))

is_Stoke_runner = False
condition_json_names = [
    'cond_A01.json',
    'cond_B01.json'
]

if is_Stoke_runner:
    grad_runner = StokeGradRunner(domain, cell_tags, facet_tags, grad_cfg={
        'equ_info': {
            'velocity_order': 2,
            'pressure_order': 1,
            'is_stoke': True
        },
        'grad_filter_radius': 0.002 * 2.1,
        'max_step_size': 0.3,
        'min_step_size': 0.2,
    })
else:
    grad_runner = RasGradRunner(domain, cell_tags, facet_tags, grad_cfg={
        'equ_info': {
            'velocity_order': 2,
            'pressure_order': 1,
            'is_stoke': True
        },
        'grad_filter_radius': 0.002 * 2.1,
        'max_step_size': 0.3,
        'min_step_size': 0.2,
    })

optimizer = TopologyPathSolver(
    grad_runner,
    learn_cfg={
        'max_iteration': 500,
        'stat_num': 50,
        'log_step': 10,
        'record_step': 2,
        'grid_nums': np.array([81, 81, 41]),
        'grid_smooth_dir': os.path.join(proj_dir, 'grid_smooth_res'),
        'tensorboard_dir': os.path.join(proj_dir, 'tensor_log')
    }
)

condition_cfgs = []
for name in condition_json_names:
    with open(os.path.join(proj_dir, name), 'r') as f:
        cond_cfg = json.load(f)
    condition_cfgs.append(cond_cfg)

rho_dir = os.path.join(proj_dir, 'res_rho')
if os.path.exists(rho_dir):
    shutil.rmtree(rho_dir)
os.mkdir(rho_dir)
for cfg in condition_cfgs:
    if cfg['control_name'] not in grad_runner.controls_dicts.keys():
        grad_runner.generate_control(
            name=cfg['control_name'],
            result_dir=rho_dir,
            init_value=cfg['control_init_value']
        )

# ------
res_dir = os.path.join(proj_dir, 'res')
if os.path.exists(res_dir):
    shutil.rmtree(res_dir)
os.mkdir(res_dir)
for cfg in condition_cfgs:
    other_rhos = []
    for name, other_rho in grad_runner.controls_dicts.items():
        if name != cfg['control_name']:
            other_rhos.append(other_rho)

    input_dict = {}
    for marker, vector in cfg['input_dict'].items():
        input_dict[int(marker)] = np.array(vector)
    output_dict = {}
    for marker, value in cfg['output_dict'].items():
        output_dict[int(marker)] = value

    if not is_Stoke_runner:
        grad_runner.add_condition(
            cond_name=cfg['name'],
            input_dict=input_dict,
            output_dict=output_dict,
            boundary_markers=cfg['boundary_markers'],
            rho=grad_runner.controls_dicts[cfg['control_name']],
            rho_openfoam_name=cfg['control_openfoam_name'],
            other_rhos=other_rhos,
            alpha_cfg=cfg['alpha_cfg'],
            cost_cfgs=cfg['cost_cfgs'],
            regularization_cfgs=cfg['regularization_cfgs'],
            openfoam_cfg=cfg['openfoam_cfg'],
            result_dir=res_dir,
        )
    else:
        grad_runner.add_condition(
            cond_name=cfg['name'],
            input_dict=input_dict,
            output_dict=output_dict,
            boundary_markers=cfg['boundary_markers'],
            rho=grad_runner.controls_dicts[cfg['control_name']],
            other_rhos=other_rhos,
            alpha_cfg=cfg['alpha_cfg'],
            cost_cfgs=cfg['cost_cfgs'],
            regularization_cfgs=cfg['regularization_cfgs'],
            result_dir=res_dir,
        )

grad_runner.generate_problem()

# ------ debug
# print('[INFO]: Start Simulating.')
# grad_runner.simulate()
# grad_runner.log_info(step=1)
# ------

optimizer.run_optimize()
