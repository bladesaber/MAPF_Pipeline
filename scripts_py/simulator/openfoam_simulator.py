"""
Note:
referece: https://openfoamwiki.net/index.php/2D_Mesh_Tutorial_using_GMSH

sudo apt-get install openfoam11
Add source /opt/openfoam11/etc/bashrc to ~/.bashrc and update

1. Please use Openfoam tool gmshToFoam *.msh create the geometry
2. Openfoam only support *.msh(version 2), So when you save the .msh file in GMSH,
   "File->Export->*.msh" please select (version 2 ASCII)
3. gmsh convert msh4 to msh2: gmsh *.msh -save -format msh2 -o *.msh
4. gmsh convert msh2 to msh4: gmsh *.msh -save -format msh4 -o *.msh
5. gmsh --help for help refo
6. please modify /constant/polyMesh/boundary:  wall boundary to wall type
7. in GMSH, please use Netgen to optimize the mesh, original mesh can usually be divergence
"""

import numpy as np
import os
import shutil
import time
import subprocess
from typing import Union, Dict, List
import dolfinx
import pyvista
from tqdm import tqdm
from sklearn.neighbors import KDTree

from scripts_py.version_9.dolfinx_Grad.vis_mesh_utils import VisUtils


class PredefineContent(object):
    default_Com_controlDict = {
        'application': 'foamRun',
        'solver': 'incompressibleFluid',
        'startFrom': 'latestTime',
        'startTime': 0,
        'stopAt': 'endTime',
        'endTime': 1000,
        'deltaT': 1,
        'writeControl': 'timeStep',
        'writeInterval': 100,
        'purgeWrite': 0,
        'writeFormat': 'ascii',
        'writePrecision': 6,
        'writeCompression': 'off',
        'timeFormat': 'general',
        'timePrecision': 6,
        'runTimeModifiable': 'true'
    }

    default_Com_fvSchemes = {
        'ddtSchemes': {
            'default': 'steadyState',
        },
        'gradSchemes': {
            'default': 'Gauss linear'
        },
        'divSchemes': {
            'default': 'none',
            'div(phi,U)': 'bounded Gauss linearUpwind grad(U)',
            'div(phi,k)': 'bounded Gauss limitedLinear 1',
            'div(phi,epsilon)': 'bounded Gauss limitedLinear 1',
            'div((nuEff*dev2(T(grad(U)))))': 'Gauss linear',
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

    default_Com_fvSolution = {
        'solvers': {
            'p': {
                'solver': 'GAMG',
                'tolerance': 1e-06,
                'relTol': 0.1,
                'smoother': 'GaussSeidel'
            },
            'pcorr': {
                'solver': 'GAMG',
                'tolerance': 1e-06,
                'relTol': 0,
                'moother': 'GaussSeidel'
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
                'p': 1e-2,
                'U': 1e-3,
                '"(k|epsilon)"': 1e-3
            }
        },
        'relaxationFactors': {
            'fields': {
                'p': 0.3,
            },
            'equations': {
                'U': 0.9,
                '"(k|epsilon)"': 0.9
            }
        }
    }

    default_Com_physicalProperties = {
        'viscosityModel': 'constant',
        'nu': '[0 2 -1 0 0 0 0] 0.01'
    }

    default_Com_momentumTransport = {
        'simulationType': 'RAS',
        'RAS': {
            'model': 'kEpsilon',
            'turbulence': 'on',
            'printCoeffs': 'on',
            'viscosityModel': 'Newtonian'
        }
    }

    default_Com_decomposeParDict = {
        'numberOfSubdomains': 8,
        'method': 'hierarchical',
        'simpleCoeffs': {
            'n': '(4 2 1)',
            'delta': 0.001
        },
        'hierarchicalCoeffs': {
            'n': '(4 2 1)',
            'delta': 0.001,
            'order': 'xyz'
        },
        'manualCoeffs': {
            'dataFile': '\"\"'
        },
        'distributed': 'no',
        'roots': '( )'
    }

    example_Com_U_property = {
        'dimensions': '[0 1 -1 0 0 0 0]',
        'internalField': 'uniform (0 0 0)',
        'boundaryField': {
            'inlet': {
                'type': 'flowRateInletVelocity',
                'meanVelocity': 1.,
                'profile': 'turbulentBL',
                'value': 'uniform (1 0 0)'
            },
            'outlet': {
                'type': 'zeroGradient'
            },
            'wall': {
                'type': 'noSlip'
            },
        }
    }

    example_Com_p_property = {
        'dimensions': '[0 2 -2 0 0 0 0]',
        'internalField': 'uniform 0',
        'boundaryField': {
            'inlet': {
                'type': 'zeroGradient',
            },
            'outlet': {
                'type': 'fixedValue',
                'value': 'uniform 0'
            },
            'wall': {
                'type': 'zeroGradient'
            },
        }
    }

    example_Com_nut_property = {
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
            'wall': {
                'type': 'nutkWallFunction',
                'value': 'uniform 0'
            },
        }
    }

    example_Com_k_property = {
        'dimensions': '[0 2 -2 0 0 0 0]',
        'internalField': 'uniform 1.0',
        'boundaryField': {
            'inlet': {
                'type': 'fixedValue',
                'value': 'uniform 1.0'
            },
            'outlet': {
                'type': 'zeroGradient'
            },
            'wall': {
                'type': 'kqRWallFunction',
                'value': 'uniform 1.0'
            },
        }
    }

    example_Com_epsilon_property = {
        'dimensions': '[0 2 -3 0 0 0 0]',
        'internalField': 'uniform 1.0',
        'boundaryField': {
            'inlet': {
                'type': 'fixedValue',
                'value': 'uniform 1.0'
            },
            'outlet': {
                'type': 'zeroGradient'
            },
            'wall': {
                'type': 'epsilonWallFunction',
                'value': 'uniform 1.0'
            },
        }
    }

    example_Com_f_property = {
        'dimensions': '[0 0 -1 0 0 0 0]',
        'internalField': 'uniform 0',
        'boundaryField': {
            'inlet': {
                'type': 'zeroGradient'
            },
            'outlet': {
                'type': 'zeroGradient'
            },
            'wall': {
                'type': 'fWallFunction',
                'value': 'uniform 0'
            },
        }
    }

    example_Org_controlDict = {
        'application': 'icoFoam',
        'startFrom': 'latestTime',
        'startTime': 0,
        'stopAt': 'endTime',
        'endTime': 0.5,
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

    default_Org_fvSchemes = {
        'ddtSchemes': {
            'default': 'steadyState',
        },
        'gradSchemes': {
            'default': 'Gauss linear',
            'grad(p)': 'Gauss linear'
        },
        'divSchemes': {
            'default': 'Gauss linear',
            'div(phi,U)': 'Gauss linear',
        },
        'laplacianSchemes': {
            'default': 'Gauss linear orthogonal'
        },
        'interpolationSchemes': {
            'default': 'linear'
        },
        'snGradSchemes': {
            'default': 'orthogonal'
        }
    }

    default_Org_fvSolution = {
        'solvers': {
            'p': {
                'solver': 'PCG',
                'preconditioner': 'DIC',
                'tolerance': 1e-06,
                'relTol': 0.05,
            },
            'pFinal': {
                'single': '$p',
                'relTol': 0.001,
            },
            'U': {
                'solver': 'smoothSolver',
                'smoother': 'symGaussSeidel',
                'tolerance': 1e-05,
                'relTol': 0.1
            }
        },
        'PISO': {
            'nCorrectors': 2,
            'nNonOrthogonalCorrectors': 0,
            'pRefCell': 0,
            'pRefValue': 0
        }
    }

    default_Org_transportProperties = {
        'nu': 0.01
    }

    example_Org_U_property = {
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
            'wall': {
                'type': 'noSlip'
            },
        }
    }

    example_Org_p_property = {
        'dimensions': '[0 2 -2 0 0 0 0]',
        'internalField': 'uniform 0',
        'boundaryField': {
            'inlet': {
                'type': 'zeroGradient',
            },
            'outlet': {
                'type': 'fixedValue',
                'value': 'uniform 0'
            },
            'wall': {
                'type': 'zeroGradient'
            },
        }
    }

    example_Org_decomposeParDict = {
        'numberOfSubdomains': 8,
        'method': 'hierarchical',
        'simpleCoeffs': {
            'n': '(4 2 1)',
            'delta': 0.001
        },
        'hierarchicalCoeffs': {
            'n': '(4 2 1)',
            'delta': 0.001,
            'order': 'xyz'
        },
        'distributed': 'no',
        'roots': '( )'
    }


class OpenFoamComSimulator(object):
    @staticmethod
    def exec_cmd(code: str):
        res = subprocess.run(code, shell=True, check=True)
        return res

    @staticmethod
    def exec_bash(file: str):
        assert file.endswith('.sh')
        subprocess.run(f"bash {file}", shell=True, check=False)

    @staticmethod
    def msh_version_change_code(orig_msh: str, target_msh: str, version_format='msh2'):
        code_lines = [f"gmsh {orig_msh} -save -format {version_format} -o {target_msh} >> gmsh_ver_change.log;"]
        return code_lines

    @staticmethod
    def msh2vtk_code(orig_msh: str, target_vtk: str):
        code_lines = [f"gmsh {orig_msh} -save -format vtk -o {target_vtk} >> msh2vtk.log;"]
        return code_lines

    @staticmethod
    def gmsh2foam_code(msh_file):
        code_lines = [f"gmshToFoam {msh_file} >> gmsh2foam.log;"]
        return code_lines

    @staticmethod
    def vtk2foam_code(vtk_file):
        code_lines = [f"vtkUnstructuredToFoam {vtk_file} >> vtk2foam.log;"]
        return code_lines

    @staticmethod
    def foam2vtk_code():
        code_lines = [f"foamToVTK -ascii -latestTime -allPatches -overwrite >> foam2vtk.log;"]
        return code_lines

    @staticmethod
    def simulate_code(run_code='foamRun'):
        code_lines = [f"{run_code} >> simulate.log;"]
        return code_lines

    @staticmethod
    def simulate_parallel_code(num_of_process: int, run_code='foamRun', remove_conda_env=False, conda_sh: str = None):
        """
        Please deactivate conda environment
        """
        code_lines = ['decomposePar >> decomposePar.log;']

        # if remove_conda_env:
        #     assert conda_sh is not None
        #     # In linux bash, source has been replaced by .
        #     # code += f"source {conda_sh}; conda deactivate; echo $CONDA_DEFAULT_ENV; "
        #     code_lines.append(f". {conda_sh}; conda deactivate; echo $CONDA_DEFAULT_ENV >> debug_conda.log;")

        code_lines.append(f'mpiexec -np {num_of_process} {run_code} -parallel >> parallel_simulate.log;')
        code_lines.append('reconstructPar -latestTime >> reconstructPar.log;')
        return code_lines

    @staticmethod
    def dict2content(arg_dict: dict, layer=0):
        content = []
        tab_count = " " * layer * 4
        for key, value in arg_dict.items():
            if key == 'single':
                content.append(f"{tab_count}{value};")
                continue

            if isinstance(value, dict):
                content.append('')
                content.append(f"{tab_count}{key}\n{tab_count}{{")
                sub_content = OpenFoamComSimulator.dict2content(value, layer + 1)
                content.extend(sub_content)
                content.append(f"{tab_count}}}")
                content.append('')

            elif isinstance(value, (str, int, float)):
                content.append(f"{tab_count}{key} {value};")
            else:
                raise ValueError("[ERROR] Non-Valid dict value")
        return content

    @staticmethod
    def create_foam_file(
            openfoam_dir: str, location: Union[str, int, float], class_name: str, object_name: str, arg_dict: dict
    ):
        """
        object_name: such as controlDict
        """
        if isinstance(location, str):
            assert location in ['system', 'constant']

        location_dir = os.path.join(openfoam_dir, str(location))
        if not os.path.exists(location_dir):
            os.mkdir(location_dir)

        content = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "  =========                 |",
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox",
            "   \\\\    /   O peration     | Website:  https://openfoam.org",
            "    \\\\  /    A nd           | Version:  dev",
            "     \\\\/     M anipulation  |",
            "\*---------------------------------------------------------------------------*/",
            "FoamFile\n{",
            "    format      ascii;",
            f"    class       {class_name};",
            f"    location    \"{location}\";",
            f"    object      {object_name};",
            "}",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        ]
        content.extend(OpenFoamComSimulator.dict2content(arg_dict))
        with open(os.path.join(location_dir, object_name), 'w') as f:
            content = '\n'.join(content)
            f.write(content)

    @staticmethod
    def modify_boundary_type(openfoam_dir: str, modify_boundary_types: Dict[str, str]):
        boundary_file = os.path.join(openfoam_dir, 'constant/polyMesh', 'boundary')
        content = []
        modify_flag, modify_type = False, None
        with open(boundary_file, 'r') as f:
            lines = f.readlines()
            for line_txt in lines:
                line_strip = line_txt.strip()
                if (line_strip in modify_boundary_types.keys()) and (not modify_flag):
                    modify_flag = True
                    modify_type = modify_boundary_types[line_strip]
                    content.append(line_txt)
                    continue

                if modify_flag:
                    words = line_strip.split(' ')
                    if words[0] == 'type':
                        line_txt = line_txt.replace(words[-1].replace(';', ''), modify_type)
                        modify_flag = False

                content.append(line_txt)

        with open(boundary_file, 'w') as f:
            f.write(''.join(content))

    @staticmethod
    def create_snappy_blockMesh(
            openfoam_dir, xmin, ymin, zmin, xmax, ymax, zmax, grid_x, grid_y, grid_z,
            scale=1.0, padding=0.1
    ):
        content = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "  =========                 |",
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox",
            "   \\\\    /   O peration     | Website:  https://openfoam.org",
            "    \\\\  /    A nd           | Version:  dev",
            "     \\\\/     M anipulation  |",
            "\*---------------------------------------------------------------------------*/",
            "FoamFile\n{",
            "    format      ascii;",
            f"    class       dictionary;",
            f"    location    \"system\";",
            f"    object      blockMeshDict;",
            "}",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        ]
        xmin, ymin, zmin = xmin - padding, ymin - padding, zmin - padding
        xmax, ymax, zmax = xmax + padding, ymax + padding, zmax + padding

        content.extend([
            f"scale {scale};\n",
            'vertices\n(',
            f"  ({xmin} {ymin} {zmin})",
            f"  ({xmax} {ymin} {zmin})",
            f"  ({xmax} {ymax} {zmin})",
            f"  ({xmin} {ymax} {zmin})",
            f"  ({xmin} {ymin} {zmax})",
            f"  ({xmax} {ymin} {zmax})",
            f"  ({xmax} {ymax} {zmax})",
            f"  ({xmin} {ymax} {zmax})",
            ');\n',
            'blocks\n(',
            '   hex(0 1 2 3 4 5 6 7)'
            f"  ({grid_x} {grid_y} {grid_z})  simpleGrading  (1 1 1)",
            ');\n',
            'boundary\n(',
            '    xMin{\n        type patch;\n        faces ( (0 3 7 4) );\n    }\n',
            '    xMax{\n        type patch;\n        faces ( (1 2 6 5) );\n    }\n',
            '    yMin{\n        type patch;\n        faces ( (0 1 5 4) );\n    }\n',
            '    yMax{\n        type patch;\n        faces ( (3 7 6 2) );\n    }\n',
            '    zMin{\n        type patch;\n        faces ( (0 1 2 3) );\n    }\n',
            '    zMax{\n        type patch;\n        faces ( (4 5 6 7) );\n    }',
            ');'
        ])

        with open(os.path.join(openfoam_dir, 'system/blockMeshDict'), 'w') as f:
            content = '\n'.join(content)
            f.write(content)

    @staticmethod
    def create_snappyHexMeshDict(castellated_dict: dict, snap_dict: dict, add_layers_dict: dict):
        content = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "  =========                 |",
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox",
            "   \\\\    /   O peration     | Website:  https://openfoam.org",
            "    \\\\  /    A nd           | Version:  dev",
            "     \\\\/     M anipulation  |",
            "\*---------------------------------------------------------------------------*/",
            "FoamFile\n{",
            "    format      ascii;",
            f"    class       dictionary;",
            f"    location    \"system\";",
            f"    object      snappyHexMeshDict;",
            "}",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        ]
        raise NotImplementedError

    @staticmethod
    def create_sample_file(proj_dir: str, sample_points: np.ndarray, sample_fields: List[str]):
        location_dir = os.path.join(proj_dir, 'postProcessing')
        if not os.path.exists(location_dir):
            os.mkdir(location_dir)

        content = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "  =========                 |",
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox",
            "   \\\\    /   O peration     | Website:  https://openfoam.org",
            "    \\\\  /    A nd           | Version:  dev",
            "     \\\\/     M anipulation  |",
            "\*---------------------------------------------------------------------------*/",
            "Description",
            "    Writes out values of fields from cells nearest to specified locations.",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        ]

        content.extend([
            f"fields ({' '.join(sample_fields)})\n",
            '#includeEtc "caseDicts/postProcessing/probes/probes.cfg"'
        ])

        raise NotImplementedError

    def __init__(
            self,
            name,
            domain: dolfinx.mesh.Mesh,
            cell_tags: dolfinx.mesh.MeshTags,
            facet_tags: dolfinx.mesh.MeshTags,
            openfoam_cfg: Dict,
    ):
        self.name = name
        self.domain = domain
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1
        self.grid = VisUtils.convert_to_grid(self.domain)

        self.openfoam_cfg = openfoam_cfg
        self.cache_dir = self.openfoam_cfg['cache_dir']
        assert self.cache_dir is not None
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        if self.openfoam_cfg['configuration'].get('decomposeParDict'):
            self.run_parallel = True
            self.num_of_threats = int(
                self.openfoam_cfg['configuration']['decomposeParDict']['args']['numberOfSubdomains']
            )
        else:
            self.run_parallel = False

        # ------ mesh structure variables
        self.points: np.ndarray = None
        self.cell_centers: np.ndarray = None
        self.n_cells: int = -1

        self.openfoamVtk2dolfinMesh_point_idxs: np.ndarray = None
        self.openfoamVtk2dolfinMesh_cell_center_idxs: np.ndarray = None

    def generate_openfoam2dolfin_idxs(self, run_dir, tol=1e-4):
        code_lines = [f"cd {run_dir}"]
        code_lines.extend(self.foam2vtk_code())
        self.exec_cmd(' && '.join(code_lines))
        tmp_vtk = self.extract_vtk(run_dir)

        self.points = tmp_vtk.points
        self.cell_centers = tmp_vtk.cell_centers().points
        self.n_cells = tmp_vtk.n_cells

        # ------ generate map from openfoam points to dolfin mesh points
        tree = KDTree(tmp_vtk.points)
        dist_list, idxs_list = tree.query(self.grid.points, k=1, return_distance=True)
        dist_list = np.concatenate(dist_list)
        if np.max(dist_list) > tol:
            return False
        self.openfoamVtk2dolfinMesh_point_idxs = np.concatenate(idxs_list)

        # ------ generate map from openfoam cell centers to dolfin mesh cell centers
        tree = KDTree(tmp_vtk.cell_centers().points)
        dist_list, idxs_list = tree.query(self.grid.cell_centers().points, k=1, return_distance=True)
        dist_list = np.concatenate(dist_list)
        if np.max(dist_list) > tol:
            return False
        self.openfoamVtk2dolfinMesh_cell_center_idxs = np.concatenate(idxs_list)

        return True

    def run(
            self, run_name, source_msh_file=None, run_cfg: Dict = None,
            with_preprocess=True, with_simulate=True, with_postprocess=True
    ):
        run_dir = os.path.join(self.cache_dir, run_name)
        return_cfg = {'state': True}
        if run_cfg is None:
            run_cfg = self.openfoam_cfg['run_cfg']

        if with_preprocess:
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
            os.mkdir(run_dir)
            return_cfg.update({'run_dir': run_dir})

            msh_file = os.path.join(run_dir, 'model.msh')
            if source_msh_file != msh_file:
                shutil.copy(source_msh_file, msh_file)

            for object_name, object_info in self.openfoam_cfg['configuration'].items():
                self.create_foam_file(
                    run_dir, object_info['location'], object_info['class_name'],
                    object_name=object_name, arg_dict=object_info['args']
                )

            preprocess_bash_file = os.path.join(run_dir, 'preprocess.sh')
            with open(preprocess_bash_file, 'w') as f:
                code_lines = [f"cd {run_dir};"]
                if run_cfg.get('msh_version', False):
                    code = self.msh_version_change_code(msh_file, msh_file, version_format=run_cfg['msh_version'])
                    code_lines.extend(code)
                code_lines.extend(self.gmsh2foam_code(msh_file))
                f.write('\n'.join(code_lines))

            self.exec_bash(preprocess_bash_file)
            self.modify_boundary_type(run_dir, self.openfoam_cfg['modify_type_dict'])

            if not self.generate_openfoam2dolfin_idxs(run_dir):
                return_cfg.update({'state': False})

        if with_simulate:
            simulate_bash_file = os.path.join(run_dir, 'simulate.sh')
            with open(simulate_bash_file, 'w') as f:
                code_lines = [f"cd {run_dir};", "foamListTimes -rm;"]
                if self.run_parallel:
                    code_lines.extend(self.simulate_parallel_code(self.num_of_threats, run_cfg['run_code']))
                else:
                    code_lines.extend(self.simulate_code(run_cfg['run_code']))
                f.write('\n'.join(code_lines))
            self.exec_bash(simulate_bash_file)

        if with_postprocess:
            postprocess_bash_file = os.path.join(run_dir, 'postprocess.sh')
            with open(postprocess_bash_file, 'w') as f:
                code_lines = [f"cd {run_dir};", 'touch para.foam;']
                code_lines.extend(self.foam2vtk_code())
                f.write('\n'.join(code_lines))
            self.exec_bash(postprocess_bash_file)

            res_vtk = self.extract_vtk(run_dir)
            success_flag = res_vtk is not None
            return_cfg.update({'state': success_flag, 'res_vtk': res_vtk})

        return return_cfg

    @staticmethod
    def extract_vtk(run_dir) -> pyvista.UnstructuredGrid:
        vtk_dir = os.path.join(run_dir, 'VTK')
        for name in os.listdir(vtk_dir):
            if name.endswith('.vtk'):
                res_vtk = pyvista.read(os.path.join(vtk_dir, name))
                return res_vtk
        return None

    @staticmethod
    def modify_internal_field(file: str, content: np.ndarray):
        name = os.path.basename(file).split('.')[0]
        temp_file = file.replace(name, f"{name}_tmp")
        is_vector = content.ndim > 1
        if is_vector:
            assert content.shape[1] == 3

        # print(f"[DEBUG]: Modify internal field {name}")
        is_internal_field_tag = False
        with open(temp_file, 'w') as target_f:
            with open(file, 'r') as source_f:
                for line in source_f.readlines():
                    if line.strip().startswith('internalField'):
                        is_internal_field_tag = True
                        if is_vector:
                            target_f.write(f"internalField nonuniform List<vector>\n{content.shape[0]}\n")
                        else:
                            target_f.write(f"internalField nonuniform List<scalar>\n{content.shape[0]}\n")

                        target_f.write('(\n')
                        if is_vector:
                            for data in content:
                                target_f.write(f"({data[0]} {data[1]} {data[2]})\n")
                        else:
                            for data in content:
                                target_f.write(f"{data}\n")
                        target_f.write(');\n')

                    else:
                        if not is_internal_field_tag:
                            target_f.write(line)

                    if is_internal_field_tag and line.strip().endswith(';'):
                        is_internal_field_tag = False

        os.remove(file)
        os.rename(temp_file, file)


class OpenFoamOrgSimulator(OpenFoamComSimulator):
    @staticmethod
    def create_foam_file(
            openfoam_dir: str, location: Union[str, int, float], class_name: str,
            object_name: str, arg_dict: dict, version=2.0
    ):
        """
        object_name: such as controlDict
        """
        if isinstance(location, str):
            assert location in ['system', 'constant']

        location_dir = os.path.join(openfoam_dir, str(location))
        if not os.path.exists(location_dir):
            os.mkdir(location_dir)

        content = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "  =========                 |",
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox",
            "   \\\\    /   O peration     | Website:  https://openfoam.org",
            "    \\\\  /    A nd           | Version:  dev",
            "     \\\\/     M anipulation  |",
            "\*---------------------------------------------------------------------------*/",
            "FoamFile\n{",
            f"    version     {version};",
            "    format      ascii;",
            f"    class       {class_name};",
            f"    location    \"{location}\";",
            f"    object      {object_name};",
            "}",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
        ]
        content.extend(OpenFoamComSimulator.dict2content(arg_dict))
        with open(os.path.join(location_dir, object_name), 'w') as f:
            content = '\n'.join(content)
            f.write(content)

    @staticmethod
    def foam2vtk_code():
        code_lines = [f"foamToVTK -ascii -latestTime -one-boundary -overwrite >> foam2vtk.log;"]
        return code_lines

    @staticmethod
    def extract_vtk(run_dir) -> pyvista.UnstructuredGrid:
        vtk_dir = os.path.join(run_dir, 'VTK')
        for name in os.listdir(vtk_dir):
            if name.endswith('.vtm'):
                vtk_block = pyvista.read(os.path.join(vtk_dir, name))
                res_vtk = vtk_block['internal']
                return res_vtk
        return None
