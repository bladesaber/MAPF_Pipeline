import shutil
import numpy as np
import os

from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.topology_path_planner.grad_runner import RasGradRunner
from scripts_py.version_9.topology_path_planner.optimize_runner import TopologyPathSolver

proj_dir = '/home/admin123456/Desktop/work/path_examples/topo_path_03'

openfoam_cfg = {
    'configuration': {
        'controlDict': {
            'location': 'system',
            'class_name': 'dictionary',
            'args': {
                'application': 'rhoMrfFoam',
                'startFrom': 'latestTime',
                'startTime': 0,
                'stopAt': 'endTime',
                'endTime': 5.0,
                'deltaT': 0.005,
                'writeControl': 'timeStep',
                'writeInterval': 100,
                'purgeWrite': 0,
                'writeFormat': 'ascii',
                'writePrecision': 6,
                'writeCompression': 'off',
                'timeFormat': 'general',
                'runTimeModifiable': 'true'
            }
        },
        'fvSchemes': {
            'location': 'system',
            'class_name': 'dictionary',
            'args': {
                'ddtSchemes': {
                    'default': 'steadyState',
                },
                'gradSchemes': {
                    'default': 'Gauss linear',
                },
                'divSchemes': {
                    'default': 'Gauss linear',
                    'div(phi,U)': 'bounded Gauss linearUpwind grad(U)',
                    'div(phi,k)': 'bounded Gauss limitedLinear 1',
                    'div(phi,epsilon)': 'bounded Gauss limitedLinear 1',
                    'div(nonlinearStress)': 'Gauss linear'
                },
                'laplacianSchemes': {
                    'default': 'Gauss linear corrected'
                },
                'interpolationSchemes': {
                    'default': 'linear'
                },
                'snGradSchemes': {
                    'default': 'corrected'
                },
                'wallDist': {
                    'method': 'meshWave'
                }
            }
        },
        'fvSolution': {
            'location': 'system',
            'class_name': 'dictionary',
            'args': {
                'solvers': {
                    'p': {
                        'solver': 'GAMG',
                        'tolerance': 1e-06,
                        'relTol': 0.1,
                        'smoother': 'GaussSeidel'
                    },
                    'pFinal': {
                        'single': '$p',
                        'relTol': 0.0,
                    },
                    '"(U|k|epsilon)"': {
                        'solver': 'smoothSolver',
                        'smoother': 'symGaussSeidel',
                        'tolerance': 1e-05,
                        'relTol': 0.1
                    }
                },
                'SIMPLE': {
                    'nNonOrthogonalCorrectors': 0,
                    'consistent': 'no',
                    'residualControl': {
                        'p': 0.01,
                        'U': 0.001,
                        '"(k|epsilon)"': 0.001
                    }
                },
                'relaxationFactors': {
                    'fields': {
                        'p': 0.3
                    },
                    'equations': {
                        'U': 0.9,
                        '"(k|epsilon)"': 0.9
                    }
                }
            }
        },
        'transportProperties': {
            'location': 'constant',
            'class_name': 'dictionary',
            'args': {
                'transportModel': 'Newtonian',
                'nu': '[0 2 -1 0 0 0 0] 0.01'
            }
        },
        'turbulenceProperties': {
            'location': 'constant',
            'class_name': 'dictionary',
            'args': {
                'simulationType': 'RAS',  # laminar
                'RAS': {
                    'RASModel': 'kEpsilon',
                    'turbulence': 'on',
                    'printCoeffs': 'on'
                }
            }
        },
        'U': {
            'location': 0,
            'class_name': 'volVectorField',
            'args': {
                'dimensions': '[0 1 -1 0 0 0 0]',
                'internalField': 'uniform (0 0 0)',
                'boundaryField': {
                    'inlet': {
                        'type': 'fixedValue',
                        'value': 'uniform (1 0 0)'
                    },
                    'outlet': {
                        'type': 'zeroGradient'
                    },
                    'boundary': {
                        'type': 'noSlip'
                    }
                }
            }
        },
        'p': {
            'location': 0,
            'class_name': 'volScalarField',
            'args': {
                'dimensions': '[0 2 -2 0 0 0 0]',
                'internalField': 'uniform 0',
                'boundaryField': {
                    'inlet': {
                        'type': 'zeroGradient'
                    },
                    'outlet': {
                        'type': 'fixedValue',
                        'value': 'uniform 0'
                    },
                    'boundary': {
                        'type': 'zeroGradient'
                    }
                }
            }
        },
        'rho': {
            'location': 0,
            'class_name': 'volScalarField',
            'args': {
                'dimensions': '[0 0 -1 0 0 0 0]',
                'internalField': 'uniform 0',
                'boundaryField': {
                    'inlet': {
                        'type': 'fixedValue',
                        'value': 'uniform 0.0'
                    },
                    'outlet': {
                        'type': 'fixedValue',
                        'value': 'uniform 0.0'
                    },
                    'boundary': {
                        'type': 'fixedValue',
                        'value': 'uniform 0.0'
                    }
                }
            }
        },
        'nut': {
            'location': 0,
            'class_name': 'volScalarField',
            'args': {
                'dimensions': '[0 2 -1 0 0 0 0]',
                'internalField': 'uniform 0',
                'boundaryField': {
                    'inlet': {
                        'type': 'calculated',
                        'value': 'uniform 0'
                    },
                    'outlet': {
                        'type': 'calculated',
                        'value': 'uniform 0'
                    },
                    'boundary': {
                        'type': 'nutkWallFunction',
                        'value': 'uniform 0'
                    }
                }
            }
        },
        'k': {
            'location': 0,
            'class_name': 'volScalarField',
            'args': {
                'dimensions': '[0 2 -2 0 0 0 0]',
                'internalField': 'uniform 0.01',
                'boundaryField': {
                    'inlet': {
                        'type': 'fixedValue',
                        'value': 'uniform 0.01'
                    },
                    'outlet': {
                        'type': 'zeroGradient'
                    },
                    'boundary': {
                        'type': 'kqRWallFunction',
                        'value': 'uniform 0.01'
                    }
                }
            }
        },
        'epsilon': {
            'location': 0,
            'class_name': 'volScalarField',
            'args': {
                'dimensions': '[0 2 -3 0 0 0 0]',
                'internalField': 'uniform 0.01',
                'boundaryField': {
                    'inlet': {
                        'type': 'fixedValue',
                        'value': 'uniform 0.01'
                    },
                    'outlet': {
                        'type': 'zeroGradient'
                    },
                    'boundary': {
                        'type': 'epsilonWallFunction',
                        'value': 'uniform 0.01'
                    }
                }
            }
        }
    },
    'modify_type_dict': {
        'boundary': 'wall'
    },
    'cache_dir': os.path.join(proj_dir, 'cache'),
    'run_cfg': {
        'msh_version': 'msh2',
        'run_code': 'rhoMrfFoam'
    },
    'source_msh_file': os.path.join(proj_dir, 'model.msh'),
    'kinematic_viscosity': 0.01
}

grad_cfg = {
    'equ_info': {
        'velocity_order': 2,
        'pressure_order': 1,
        'is_stoke': True     # 这里使用了Stoke方程作为代理，NavierStoke方程失败，原因不明
    },
    'alpha_info': {
        'alpha_min': 0.01,
        'alpha_max': 2000.0,
        'p': 2.0
    },
    'grad_filter_radius': 0.035 * 2.1,
    'max_step_size': 0.2,
    'min_step_size': 0.1,
}

MeshUtils.msh_to_XDMF(
    msh_file=os.path.join(proj_dir, 'model.msh'),
    output_file=os.path.join(proj_dir, 'model.xdmf'),
    dim=3
)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(file=os.path.join(proj_dir, 'model.xdmf'))

grad_runner = RasGradRunner(domain, cell_tags, facet_tags, grad_cfg=grad_cfg)

optimizer = TopologyPathSolver(
    grad_runner,
    learn_cfg={
        'max_iteration': 500,
        'stat_num': 50,
        'log_step': 10,
        'record_step': 2,
        'grid_nums': np.array([51, 51, 51]),
        'grid_smooth_dir': os.path.join(proj_dir, 'grid_smooth_res')
    }
)

condition_cfgs = [
    {
        'name': 'cond1',
        'input_dict': {
            20: np.array([1., 0., 0.])
        },
        'output_dict': {
            21: 0.0,
        },
        'boundary_markers': [22],

        'control_name': 'rho1',
        'control_openfoam_name': 'rho',
        'control_init_value': 0.1,

        'cost_cfgs': {
            # 'minimum_dissipation': {
            #     'weight': 1.0
            # },
            'minimum_pressure_loss': {
                'weight': 1.0,
                'marker1': 20,
                'marker2': 21,
            }
        },
        'regularization_cfgs': {
            'volume_regularization': {
                'ratio': 0.2,
                'weight': 800.0,
            }
        },
        'openfoam_cfg': openfoam_cfg
    },
]

# ------
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

    grad_runner.add_condition(
        cond_name=cfg['name'],
        input_dict=cfg['input_dict'],
        output_dict=cfg['output_dict'],
        boundary_markers=cfg['boundary_markers'],
        rho=grad_runner.controls_dicts[cfg['control_name']],
        rho_openfoam_name=cfg['control_openfoam_name'],
        other_rhos=other_rhos,
        cost_cfgs=cfg['cost_cfgs'],
        regularization_cfgs=cfg['regularization_cfgs'],
        openfoam_cfg=cfg['openfoam_cfg'],
        result_dir=res_dir,
    )

grad_runner.generate_problem()

# ------ debug
# grad_runner.simulate()
# grad_runner.log_info(step=1)
# ------

optimizer.run_optimize()
