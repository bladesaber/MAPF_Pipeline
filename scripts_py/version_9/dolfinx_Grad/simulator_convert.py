import numpy as np
import dolfinx
import pyvista
from sklearn.neighbors import KDTree
import h5py
from typing import List, Union
import pandas as pd
import subprocess
import os

"""
Gmsh -> Ansys Fluent: bdf格式
Ansys Fluent -> Gmsh: cgns格式
"""


class CrossSimulatorUtil(object):
    @staticmethod
    def convert_to_simple_function(
            domain: dolfinx.mesh.Mesh, coords: np.ndarray, value_list: List[np.ndarray], r=1e-3
    ) -> List[dolfinx.fem.Function]:
        tree = KDTree(coords)
        dists_list, idxs_list = tree.query(domain.geometry.x, k=1, return_distance=True)
        coord_idxs = []
        for idxs, dists in zip(idxs_list, dists_list):
            if dists[0] > r:
                raise ValueError('[ERROR]: Finite Element Mesh is not complete match')
            coord_idxs.append(idxs[0])
        coord_idxs = np.array(coord_idxs)

        # coords = coords[coord_idxs]
        fun_list = []
        for value in value_list:
            if value.ndim == 1:
                function_space = dolfinx.fem.FunctionSpace(domain, ("CG", 1))
                value = value[coord_idxs].reshape(-1)
            elif value.ndim == 2:
                function_space = dolfinx.fem.VectorFunctionSpace(domain, ("CG", 1))
                value = value[coord_idxs, :].reshape(-1)
            else:
                raise ValueError("[ERROR]: Non-Valid Shape")

            fun = dolfinx.fem.Function(function_space)
            fun.x.array[:] = value
            fun_list.append(fun)

        return fun_list


class FluentUtils(object):
    @staticmethod
    def read_hdf5(file: str) -> h5py.File:
        assert file.split('.')[-1] in ['cgns', 'hdf5']
        return h5py.File(file)

    @staticmethod
    def read_data(data_h5: h5py.File, tag_sequence: List[str]):
        data = data_h5
        for tag in tag_sequence:
            data = data[tag]
        return data

    @staticmethod
    def print_h5df_tree(data_h5: h5py.File, pre=''):
        items = len(data_h5)
        for key, val in data_h5.items():
            items -= 1
            if items == 0:
                # the last item
                if type(val) == h5py._hl.group.Group:
                    print(pre + '└── ' + key)
                    CrossSimulatorUtil.print_h5df_tree(val, pre + '    ')
                else:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
            else:
                if type(val) == h5py._hl.group.Group:
                    print(pre + '├── ' + key)
                    CrossSimulatorUtil.print_h5df_tree(val, pre + '│   ')
                else:
                    print(pre + '├── ' + key + ' (%d)' % len(val))

    @staticmethod
    def read_ansys_fluent_data(
            data: h5py.File, read_tags: List[str], tdim: int, coord_scale=1.0, save_csv_file: str = None
    ):
        assert tdim in [2, 3]

        res_dict = {}
        for tag in read_tags:
            if tag == 'coord':
                coord_list = [
                    np.array(data['Base']['Zone']['GridCoordinates']['CoordinateX'][' data']).reshape((-1, 1)),
                    np.array(data['Base']['Zone']['GridCoordinates']['CoordinateY'][' data']).reshape((-1, 1))
                ]
                if tdim == 3:
                    coord_list.append(
                        np.array(data['Base']['Zone']['GridCoordinates']['CoordinateZ'][' data']).reshape((-1, 1))
                    )
                xyz_coord = np.concatenate(coord_list, axis=1)
                res_dict[tag] = xyz_coord * coord_scale

            elif tag == 'velocity':
                vel_list = [
                    np.array(data['Base']['Zone']['FlowSolution.N:1']['VelocityX'][' data']).reshape((-1, 1)),
                    np.array(data['Base']['Zone']['FlowSolution.N:1']['VelocityY'][' data']).reshape((-1, 1)),
                ]
                if tdim == 3:
                    vel_list.append(
                        np.array(data['Base']['Zone']['FlowSolution.N:1']['VelocityZ'][' data']).reshape((-1, 1))
                    )
                vel = np.concatenate(vel_list, axis=1)
                res_dict[tag] = vel

            elif tag == 'pressure':
                pressure = np.array(data['Base']['Zone']['FlowSolution.N:1']['PressureStagnation'][' data']).reshape(
                    (-1, 1))
                res_dict[tag] = pressure

            else:
                raise ValueError("[ERROR]: Non-Valid Tag")

        if save_csv_file is not None:
            assert save_csv_file.endswith('.csv')
            df = pd.DataFrame()
            for tag in res_dict.keys():
                data = res_dict[tag]
                if tag == 'coord':
                    if tdim == 2:
                        df[['coord_x', 'coord_y']] = data
                    else:
                        df[['coord_x', 'coord_y', 'coord_z']] = data
                elif tag == 'velocity':
                    if tdim == 2:
                        df[['vel_x', 'vel_y']] = data
                    else:
                        df[['vel_x', 'vel_y', 'vel_z']] = data
                elif tag == 'pressure':
                    df['pressure'] = data

            df.to_csv(save_csv_file)

        return res_dict


class OpenFoamUtils(object):
    @staticmethod
    def exec_cmd(code: str):
        res = subprocess.run(code, shell=True, check=True)
        return res

    @staticmethod
    def get_msh_version_change_code(proj_dir: str, orig_msh: str, target_msh: str, version_format='msh2'):
        code = f"cd {proj_dir}; "
        code += f"gmsh {orig_msh} -save -format {version_format} -o {target_msh} >> version_convert.log;"
        return code

    @staticmethod
    def get_msh2vtk_code(orig_msh: str, target_vtk: str):
        code = f"gmsh {orig_msh} -save -format vtk -o {target_vtk} >> vtk_convert.log;"
        return code

    @staticmethod
    def get_gmsh2foam_code(proj_dir: str, msh_file):
        assert os.path.exists(proj_dir) and os.path.isdir(proj_dir)
        system_dir = os.path.join(proj_dir, 'system')
        assert os.path.exists(os.path.join(system_dir, 'controlDict'))
        code = f"cd {proj_dir}; "
        code += f"gmshToFoam {msh_file} >> convertMesh.log;"
        return code

    @staticmethod
    def get_vtk2foam_code(proj_dir: str, vtk_file):
        assert os.path.exists(proj_dir) and os.path.isdir(proj_dir)
        system_dir = os.path.join(proj_dir, 'system')
        assert os.path.exists(os.path.join(system_dir, 'controlDict'))
        code = f"cd {proj_dir}; "
        code += f"vtkUnstructuredToFoam {vtk_file} >> convertMesh.log;"
        return code

    @staticmethod
    def get_simulation_code(proj_dir: str, run_code='foamRun'):
        assert os.path.exists(proj_dir) and os.path.isdir(proj_dir)
        assert os.path.exists(os.path.join(proj_dir, 'system'))
        assert os.path.exists(os.path.join(proj_dir, 'constant'))
        code = f"cd {proj_dir}; "
        code += f"{run_code} >> simulate.log;"
        return code

    @staticmethod
    def get_simulation_parallel_code(
            proj_dir: str, num_of_process: int, run_code='foamRun',
            remove_conda_env=False, conda_sh: str = None
    ):
        """
        Please deactivate conda environment
        """
        assert os.path.exists(proj_dir) and os.path.isdir(proj_dir)
        assert os.path.exists(os.path.join(proj_dir, 'system'))
        assert os.path.exists(os.path.join(proj_dir, 'constant'))

        code = f"cd {proj_dir}; "
        code += 'decomposePar >> decomposePar.log; '

        if remove_conda_env:
            assert conda_sh is not None
            # In linux bash, source has been replaced by .
            # code += f"source {conda_sh}; conda deactivate; echo $CONDA_DEFAULT_ENV; "
            code += f". {conda_sh}; conda deactivate; echo $CONDA_DEFAULT_ENV >> debug_conda.log; "

        code += f'mpiexec -np {num_of_process} {run_code} -parallel >> simulate.log; '

        code += 'reconstructPar -latestTime >> reconstructPar.log;'

        return code

    @staticmethod
    def get_foam2vtk_code(proj_dir: str):
        code = f"cd {proj_dir}; "
        code += f"foamToVTK -ascii -latestTime -allPatches >> output_result.log;"
        return code

    @staticmethod
    def convert_dict2content(arg_dict: dict, layer=0, with_wrap=True):
        content = []
        tab_count = " " * layer * 4
        for key, value in arg_dict.items():
            if isinstance(value, (str, int, float)):
                if with_wrap:
                    content.append(f"{tab_count}{key} {value};\n")
                else:
                    content.append(f"{tab_count}{key} {value};")

            elif isinstance(value, dict):
                content.append(f"{tab_count}{key}\n{tab_count}{{")
                sub_content = OpenFoamUtils.convert_dict2content(value, layer + 1, with_wrap=False)
                content.extend(sub_content)
                content.append(f"{tab_count}}}\n")
            else:
                raise ValueError("[ERROR] Non-Valid dict value")
        return content

    default_controlDict = {
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

    default_fvSchemes = {
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

    default_fvSolution = {
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

    default_physicalProperties = {
        'viscosityModel': 'constant',
        'nu': '[0 2 -1 0 0 0 0] 0.01'
    }

    default_momentumTransport = {
        'simulationType': 'RAS',
        'RAS': {
            'model': 'kEpsilon',
            'turbulence': 'on',
            'printCoeffs': 'on',
            'viscosityModel': 'Newtonian'
        }
    }

    default_decomposeParDict = {
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

    example_U_property = {
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

    example_p_property = {
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

    example_nut_property = {
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

    example_k_property = {
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

    example_epsilon_property = {
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

    example_f_property = {
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

    @staticmethod
    def create_foam_file(
            proj_dir: str, location: Union[str, int, float],
            class_name: str, object_name: str, arg_dict: dict
    ):
        """
        object_name: such as controlDict
        """
        if isinstance(location, str):
            assert location in ['system', 'constant']

        location_dir = os.path.join(proj_dir, str(location))
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
        content.extend(OpenFoamUtils.convert_dict2content(arg_dict))
        with open(os.path.join(location_dir, object_name), 'w') as f:
            content = '\n'.join(content)
            f.write(content)

    @staticmethod
    def modify_boundary_type(proj_dir: str, modify_dict: dict):
        boundary_file = os.path.join(proj_dir, 'constant/polyMesh', 'boundary')
        content = []
        modify_flag, modify_type = False, None
        with open(boundary_file, 'r') as f:
            lines = f.readlines()
            for line_txt in lines:
                line_strip = line_txt.strip()
                if line_strip in modify_dict.keys():
                    modify_flag = True
                    modify_type = modify_dict[line_strip]
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
    def get_unit_scale_code(proj_dir: str, scale):
        code = f"cd {proj_dir}; "
        code += f"transformPoints \"scale={scale}\";"
        return code

    @staticmethod
    def create_snappy_blockMesh(
            self, location_dir, xmin, ymin, zmin, xmax, ymax, zmax, grid_x, grid_y, grid_z,
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

        with open(os.path.join(location_dir, 'blockMeshDict'), 'w') as f:
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
        raise NotImplementedError()

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

        raise NotImplementedError("Non Finish")
