import os
import numpy as np

from scripts_py.version_9.simulator.openfoam_simulator import OpenFoamOrgSimulator
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils
from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils

proj_dir = '/home/admin123456/Desktop/work/path_examples/topo_path_01'

openfoam_cfg = {
    'configuration': {
        'controlDict': {
            'location': 'system',
            'class_name': 'dictionary',
            'args': {
                'application': 'simpleFoam',
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
                    'div((nuEff*dev2(T(grad(U)))))': 'Gauss linear',
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
                'internalField': 'uniform 0.04',
                'boundaryField': {
                    'inlet': {
                        'type': 'fixedValue',
                        'value': 'uniform 0.04'
                    },
                    'outlet': {
                        'type': 'zeroGradient'
                    },
                    'boundary': {
                        'type': 'kqRWallFunction',
                        'value': 'uniform 0.04'
                    }
                }
            }
        },
        'epsilon': {
            'location': 0,
            'class_name': 'volScalarField',
            'args': {
                'dimensions': '[0 2 -3 0 0 0 0]',
                'internalField': 'uniform 2.4',
                'boundaryField': {
                    'inlet': {
                        'type': 'fixedValue',
                        'value': 'uniform 2.4'
                    },
                    'outlet': {
                        'type': 'zeroGradient'
                    },
                    'boundary': {
                        'type': 'epsilonWallFunction',
                        'value': 'uniform 2.4'
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
        'run_code': 'simpleFoam'
    }
}

MeshUtils.msh_to_XDMF(
    msh_file=os.path.join(proj_dir, 'model.msh'),
    output_file=os.path.join(proj_dir, 'model.xdmf'),
    dim=3
)
domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
    file=os.path.join(proj_dir, 'model.xdmf'),
)
grid = VisUtils.convert_to_grid(domain)

simulator = OpenFoamOrgSimulator('test', domain, cell_tags, facet_tags, openfoam_cfg)
res_info = simulator.run(
    run_name='test0',
    source_msh_file=os.path.join(proj_dir, 'model.msh'),
    with_preprocess=True, with_simulate=True, with_postprocess=True
)

