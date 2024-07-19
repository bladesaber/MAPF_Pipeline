import os
import shutil
import argparse
from importlib import util as import_util
import json
import sys
from typing import List

from scripts_py.version_9.dolfinx_Grad.simulator_convert import OpenFoamUtils


def create_project(
        proj_dir: str,
        simulate_methods: List[str],
        with_recombine_cfg: bool,
):
    if proj_dir is None:
        print('[Info]: Param(Proj_dir) Is Not A Valid Path')
        return
    if not os.path.exists(proj_dir):
        os.mkdir(proj_dir)

    open(os.path.join(proj_dir, 'model.geo'), 'w')  # create gmsh file
    open(os.path.join(proj_dir, 'condition.py'), 'w')

    simulate_cfg = {}
    if 'ipcs' in simulate_methods:
        simulate_cfg['ipcs'] = {
            'dt': 1 / 400.0,
            'dynamic_viscosity': 0.01,
            'density': 1.0,
            'body_force': None,
            'max_iter': 5000,
            'log_iter': 100,
            'tol': 5e-6,
            'is_channel_fluid': True,
            'trial_iter': 100,
        }

    if 'navier_stoke' in simulate_methods:
        simulate_cfg['navier_stoke'] = {
            'kinematic_viscosity_nu': 0.01,
            'ksp_option': {
                'ksp_type': 'preonly',
                'pc_type': 'lu',
                'pc_factor_mat_solver_type': 'mumps'
            },
            'snes_option': {
                'snes_type': 'newtonls',
                'snes_linesearch_type': 'bt',
                'snes_linesearch_order': 1,
                'snes_linesearch_maxstep': 1e4,
                'snes_linesearch_damping': 0.5,
                'snes_linesearch_minlambda': 1e-2
            },
            'criterion': {
                'rtol': 1e-6,
                'atol': 1e-6,
                'max_it': 1e3
            }
        }

    if 'stoke' in simulate_methods:
        simulate_cfg['stoke'] = {
            'ksp_option': {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        }

    if 'openfoam' in simulate_methods:
        simulate_cfg['openfoam'] = {
            'configuration': {
                'controlDict': {
                    'location': 'system',
                    'class_name': 'dictionary',
                    'args': OpenFoamUtils.default_controlDict
                },
                'fvSchemes': {
                    'location': 'system',
                    'class_name': 'dictionary',
                    'args': OpenFoamUtils.default_fvSchemes
                },
                'fvSolution': {
                    'location': 'system',
                    'class_name': 'dictionary',
                    'args': OpenFoamUtils.default_fvSolution
                },
                'decomposeParDict': {
                    'location': 'system',
                    'class_name': 'dictionary',
                    'args': OpenFoamUtils.default_decomposeParDict
                },
                'physicalProperties': {
                    'location': 'constant',
                    'class_name': 'dictionary',
                    'args': OpenFoamUtils.default_physicalProperties
                },
                'momentumTransport': {
                    'location': 'constant',
                    'class_name': 'dictionary',
                    'args': OpenFoamUtils.default_momentumTransport
                },
                'U': {
                    'location': 0,
                    'class_name': 'volVectorField',
                    'args': OpenFoamUtils.example_U_property
                },
                'p': {
                    'location': 0,
                    'class_name': 'volScalarField',
                    'args': OpenFoamUtils.example_p_property
                },
                'nut': {
                    'location': 0,
                    'class_name': 'volScalarField',
                    'args': OpenFoamUtils.example_nut_property
                },
                'k': {
                    'location': 0,
                    'class_name': 'volScalarField',
                    'args': OpenFoamUtils.example_k_property
                },
                'f': {
                    'location': 0,
                    'class_name': 'volScalarField',
                    'args': OpenFoamUtils.example_f_property
                },
                'epsilon': {
                    'location': 0,
                    'class_name': 'volScalarField',
                    'args': OpenFoamUtils.example_epsilon_property
                }
            },
            'k-Epsilon_args': {
                'abs_velocity_U': None,
                'characteristic_length_L': None,
                'kinematic_viscosity': None,
            },
            'modify_type_dict': None,
            'unit_scale': None,
            'cache_dir': None,
            'geo_tag': None
        }

    optimize_cfg = {
        'remesh_dir': None,
        'kinematic_viscosity': 0.01,
        'snes_option': {
            'snes_type': 'newtonls',
            'snes_linesearch_type': 'bt',
            'snes_linesearch_order': 1,
            'snes_linesearch_maxstep': 1e4,
            'snes_linesearch_damping': 0.5,
            'snes_linesearch_minlambda': 1e-2
        },
        'snes_criterion': {
            'rtol': 1e-6,
            'atol': 1e-6,
            'max_it': 1e3
        },
        'state_ksp_option': {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        'adjoint_ksp_option': {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        'gradient_ksp_option': {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type': 'mumps'},
        'deformation_cfg': {
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
        },
        'scalar_product_method': {
            'method': 'Poincare-Steklov operator',
            'lambda_lame': 1.0,  # it is very important here
            'damping_factor': 0.2,  # it is very important here
            'mu_fix': 1.0,
            'mu_free': 1.0,
            'use_inhomogeneous': False,
            'inhomogeneous_exponent': 1.0,
            'update_inhomogeneous': False
        },
        # 'scalar_product_method' : {'method': 'default'}
        'point_radius': 0.1,
        'run_strategy_cfg': {
            'max_iter': 100,
            'max_step_limit': 0.1,
            'init_stepSize': 1.0,
            'stepSize_lower': 1e-4,
            'deformation_lower': 1e-2,
            'loss_tol_rho': 0.02,
            'beta_rho': 0.75,
        },
        'obs_avoid_cfg': {
            # 'method': 'sigmoid_v1',  # for sigmoid_v1
            # 'c': 100,                # for sigmoid_v1
            # 'break_range': 1e-02,    # for sigmoid_v1
            # 'reso': 1e-03,           # for sigmoid_v1

            'method': 'relu_v1',  # for relu_v1
            'c': 5,  # for relu_v1
            'lower': 1e-02,  # for relu_v1

            'bbox_rho': 0.85,
            'bbox_w_lower': 0.1,
            'weight': 5.0
        },
        'cost_functions': [
            {
                'name': 'MiniumEnergy',
                'weight': 1.0
            }
        ],
        'regularization_functions': [
            {
                'name': 'VolumeRegularization',
                'mu': 0.2,
                'target_volume_rho': 0.6,
                'method': 'percentage_div'
            }
        ],
    }
    proj_cfg = {
        'name': None,
        'proj_dir': proj_dir,
        "geo_file": "model.geo",
        "msh_file": "model.msh",
        "xdmf_file": "model.xdmf",
        'dim': None,
        'input_markers': {},  # marker: function_name
        'output_markers': [],
        'bry_fix_markers': [],
        'bry_free_markers': [],
        'condition_package_name': 'condition.py',
        'simulate_cfg': simulate_cfg,
        'optimize_cfg': optimize_cfg,
        'velocity_init_pkl': None,
        'pressure_init_pkl': None,
        "velocity_opt_pkl": None,
        "pressure_opt_pkl": None,
        "opt_xdmf_file": None,
        'obstacle_dir': None,
        'obstacle_names': []
    }
    with open(os.path.join(proj_dir, 'fea_cfg.json'), 'w') as f:
        json.dump(proj_cfg, f, indent=4)

    if with_recombine_cfg:
        recombine_cfg = {
            'tag_name': None,
            'proj_dir': proj_dir,
            'recombine_cfgs': []
        }

        with open(os.path.join(proj_dir, 'recombine_cfg.json'), 'w') as f:
            json.dump(recombine_cfg, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Tool")
    parser.add_argument('--proj_dir', type=str, default=None)
    parser.add_argument('--create_cfg', type=int, default=0)
    parser.add_argument('--with_recombine_cfg', type=int, default=0)
    parser.add_argument('--simulate_methods', type=str, nargs='+', default=['ipcs', 'navier_stoke'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    create_project(
        proj_dir=args.proj_dir,
        simulate_methods=args.simulate_methods,
        with_recombine_cfg=args.with_recombine_cfg
    )


if __name__ == '__main__':
    main()