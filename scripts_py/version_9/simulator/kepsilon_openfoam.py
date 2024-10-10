import numpy as np
import os
import shutil
import subprocess
from typing import Union, Dict, List
import pyvista

"""
copy *.stl to root/constant/triSurface/*.stl 
-> surfaceFeatureExtract(surfaceFeatureExtractDict)
-> blockMesh
-> snappyHexMesh -overwrite
-> topoSet
-> createPatch -overwrite
"""


class KEpsilonSimulator(object):
    def __init__(self, cfg: Dict, by_method):
        self.cfg = cfg
        self.proj_dir = self.cfg['proj_dir']
        self.mesh_file = None

        if os.path.exists(self.proj_dir):
            shutil.rmtree(self.proj_dir)
        os.mkdir(self.proj_dir)

        self.constant_dir = os.path.join(self.proj_dir, 'constant')
        os.mkdir(self.constant_dir)

        self.system_dir = os.path.join(self.proj_dir, 'system')
        os.mkdir(self.system_dir)

        self.start_dir = os.path.join(self.proj_dir, '0')
        os.mkdir(self.start_dir)

        self.prepare_project(by_method=by_method)

    @staticmethod
    def exec_cmd(code: str):
        res = subprocess.run(code, shell=True, check=True)
        return res

    @staticmethod
    def exec_bash(file: str):
        assert file.endswith('.sh')
        subprocess.run(f"bash {file}", shell=True, check=False)

    @staticmethod
    def write_file(content, save_dir, save_name):
        with open(os.path.join(save_dir, save_name), 'w') as f:
            content = '\n'.join(content)
            f.write(content)

    def prepare_project(self, by_method='gmsh'):
        control_cfg = self.cfg['controlDict']
        content = self.generate_controlDict_file(
            control_cfg['application'], control_cfg['endTime'], control_cfg['deltaT'], control_cfg['writeInterval']
        )
        self.write_file(content, self.system_dir, 'controlDict')

        content = self.generate_fvSchemes_file()
        self.write_file(content, self.system_dir, 'fvSchemes')

        content = self.generate_fvSolution_file()
        self.write_file(content, self.system_dir, 'fvSolution')

        transport_properties_cfg = self.cfg['transportProperties']
        content = self.generate_transportProperties_file(transport_properties_cfg['nu'])
        self.write_file(content, self.constant_dir, 'transportProperties')

        content = self.generate_turbulenceProperties_file()
        self.write_file(content, self.constant_dir, 'turbulenceProperties')

        self.mesh_file = os.path.join(self.proj_dir, 'mesh.stl')
        shutil.copy(self.cfg['mesh_file'], self.mesh_file)

        if by_method == 'gmsh':
            mesh_cfg = self.cfg['gmsh_mesh_cfg']
            geo_file, msh_file = self.generate_msh_by_stl(
                self.mesh_file, self.proj_dir, 'model', mesh_cfg['meshSizeMin'], mesh_cfg['meshSizeMax']
            )
            mesh_cfg['geo_file'] = geo_file
            mesh_cfg['msh_file'] = msh_file
            self.exec_cmd(f"cd {self.proj_dir}; gmshToFoam {mesh_cfg['msh_file']} >> gmsh2foam.log;")
        else:
            stl_name = os.path.basename(self.mesh_file).split('.')[0]

            tri_surface_dir = os.path.join(self.constant_dir, 'triSurface')
            if os.path.exists(tri_surface_dir):
                shutil.rmtree(tri_surface_dir)
            os.mkdir(tri_surface_dir)
            shutil.copy(self.mesh_file, os.path.join(tri_surface_dir, f"{stl_name}.stl"))

            content = self.generate_blockMeshDict_file(
                xmin=self.cfg['openfoam_mesh_cfg']['xmin'],
                xmax=self.cfg['openfoam_mesh_cfg']['xmax'],
                ymin=self.cfg['openfoam_mesh_cfg']['ymin'],
                ymax=self.cfg['openfoam_mesh_cfg']['ymax'],
                zmin=self.cfg['openfoam_mesh_cfg']['zmin'],
                zmax=self.cfg['openfoam_mesh_cfg']['zmax'],
                x_grid=self.cfg['openfoam_mesh_cfg']['x_grid'],
                y_grid=self.cfg['openfoam_mesh_cfg']['y_grid'],
                z_grid=self.cfg['openfoam_mesh_cfg']['z_grid'],
                tol=self.cfg['openfoam_mesh_cfg']['tol'],
                convertToMeters=self.cfg['openfoam_mesh_cfg']['convertToMeters'],
            )
            self.write_file(content, self.system_dir, 'blockMeshDict')

            content = self.generate_surfaceFeatureExtractDict_file(f"{stl_name}.stl")
            self.write_file(content, self.system_dir, 'surfaceFeatureExtractDict')

            content = self.generate_snappyHexMeshDict_file(
                stl_name,
                inside_x=self.cfg['openfoam_mesh_cfg']['inside_x'],
                inside_y=self.cfg['openfoam_mesh_cfg']['inside_y'],
                inside_z=self.cfg['openfoam_mesh_cfg']['inside_z'],
            )
            self.write_file(content, self.system_dir, 'snappyHexMeshDict')

            content = self.generate_meshQualityDict_file()
            self.write_file(content, self.system_dir, 'meshQualityDict')

            self.exec_cmd(f"cd {self.proj_dir}; surfaceFeatureExtract >> surfaceFeatureExtract.log;")
            self.exec_cmd(f"cd {self.proj_dir}; blockMesh >> blockMesh.log;")
            self.exec_cmd(f"cd {self.proj_dir}; snappyHexMesh -overwrite >> snappyHexMesh.log;")

        with open(os.path.join(self.proj_dir, 'para.foam'), 'w') as f:
            pass

        content = self.generate_topoSetDict_file(self.cfg['boundary_definition'])
        self.write_file(content, self.system_dir, 'topoSetDict')
        self.exec_cmd(f"cd {self.proj_dir}; topoSet >> topoSet.log;")

        content = self.generate_createPatchDict_file(self.cfg['boundary_definition'])
        self.write_file(content, self.system_dir, 'createPatchDict')
        self.exec_cmd(f"cd {self.proj_dir}; createPatch -overwrite >> createPatch.log;")

        self.modify_boundary_type(self.proj_dir, {'boundary': 'wall'})

        content = self.generate_U_file(self.cfg['background_cfg']['U']['internalField'], self.cfg['simulation_cfg'])
        self.write_file(content, self.start_dir, 'U')

        content = self.generate_p_file(self.cfg['background_cfg']['p']['internalField'], self.cfg['simulation_cfg'])
        self.write_file(content, self.start_dir, 'p')

        content = self.generate_nut_file(self.cfg['background_cfg']['nut']['internalField'], self.cfg['simulation_cfg'])
        self.write_file(content, self.start_dir, 'nut')

        content = self.generate_k_file(self.cfg['background_cfg']['k']['internalField'], self.cfg['simulation_cfg'])
        self.write_file(content, self.start_dir, 'k')

        content = self.generate_epsilon_file(
            self.cfg['background_cfg']['epsilon']['internalField'], self.cfg['simulation_cfg']
        )
        self.write_file(content, self.start_dir, 'epsilon')

    @staticmethod
    def generate_header(version: float, class_name: str, location: Union[str, float, int], object_name: str):
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
        return content

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
                sub_content = KEpsilonSimulator.dict2content(value, layer + 1)
                content.extend(sub_content)
                content.append(f"{tab_count}}}")
                content.append('')

            elif isinstance(value, (str, int, float)):
                content.append(f"{tab_count}{key} {value};")
            else:
                raise ValueError("[ERROR] Non-Valid dict value")
        return content

    @staticmethod
    def generate_U_file(internalField: List[float], simulation_cfg: Dict):
        file_content = KEpsilonSimulator.generate_header(2.0, 'volVectorField', location=0, object_name='U')
        file_content.extend([
            "dimensions [0 1 -1 0 0 0 0];\n",
            f"internalField uniform ({internalField[0]} {internalField[0]} {internalField[0]});\n",
            "boundaryField",
            "{"
        ])

        for name, cfg in simulation_cfg.items():
            u_cfg = cfg['U']
            sub_content = [f"\t{name}", "\t{"]

            if u_cfg['type'] == 'fixedValue':
                sub_content.extend([
                    "\t\ttype fixedValue;",
                    f"\t\tvalue uniform ({u_cfg['value'][0]} {u_cfg['value'][1]} {u_cfg['value'][2]});"
                ])
            elif u_cfg['type'] == 'zeroGradient':
                sub_content.append("\t\ttype zeroGradient;")
            elif u_cfg['type'] == 'noSlip':
                sub_content.append("\t\ttype noSlip;")
            else:
                raise ValueError

            sub_content.append("\t}\n")
            file_content.extend(sub_content)

        file_content.append("}")
        return file_content

    @staticmethod
    def generate_p_file(internalField: float, simulation_cfg: Dict):
        file_content = KEpsilonSimulator.generate_header(2.0, 'volScalarField', location=0, object_name='p')
        file_content.extend([
            "dimensions [0 2 -2 0 0 0 0];\n",
            f"internalField uniform {internalField};\n",
            "boundaryField",
            "{"
        ])

        for name, cfg in simulation_cfg.items():
            p_cfg = cfg['p']
            sub_content = [f"\t{name}", "\t{"]

            if p_cfg['type'] == 'zeroGradient':
                sub_content.append("\t\ttype zeroGradient;")
            elif p_cfg['type'] == 'fixedValue':
                sub_content.extend([
                    "\t\ttype fixedValue;",
                    f"\t\tvalue uniform {p_cfg['value']};"
                ])

            else:
                raise ValueError
            sub_content.append("\t}\n")
            file_content.extend(sub_content)

        file_content.append("}")
        return file_content

    @staticmethod
    def generate_k_file(internalField: float, simulation_cfg: Dict):
        file_content = KEpsilonSimulator.generate_header(2.0, 'volScalarField', location=0, object_name='k')
        file_content.extend([
            "dimensions [0 2 -2 0 0 0 0];\n",
            f"internalField uniform {internalField};\n",
            "boundaryField",
            "{"
        ])

        for name, cfg in simulation_cfg.items():
            k_cfg = cfg['k']
            sub_content = [f"\t{name}", "\t{"]

            if k_cfg['type'] == 'fixedValue':
                sub_content.extend([
                    "\t\ttype fixedValue;",
                    f"\t\tvalue uniform {k_cfg['value']};"
                ])
            elif k_cfg['type'] == 'zeroGradient':
                sub_content.append("\t\ttype zeroGradient;")
            elif k_cfg['type'] == 'kqRWallFunction':
                sub_content.extend([
                    "\t\ttype kqRWallFunction;",
                    f"\t\tvalue uniform {k_cfg['value']};"
                ])
            else:
                raise ValueError
            sub_content.append("\t}\n")
            file_content.extend(sub_content)

        file_content.append("}")
        return file_content

    @staticmethod
    def generate_epsilon_file(internalField: float, simulation_cfg: Dict):
        file_content = KEpsilonSimulator.generate_header(2.0, 'volScalarField', location=0, object_name='epsilon')
        file_content.extend([
            "dimensions [0 2 -3 0 0 0 0];\n",
            f"internalField uniform {internalField};\n",
            "boundaryField",
            "{"
        ])

        for name, cfg in simulation_cfg.items():
            epsilon_cfg = cfg['epsilon']
            sub_content = [f"\t{name}", "\t{"]

            if epsilon_cfg['type'] == 'fixedValue':
                sub_content.extend([
                    "\t\ttype fixedValue;",
                    f"\t\tvalue uniform {epsilon_cfg['value']};"
                ])
            elif epsilon_cfg['type'] == 'zeroGradient':
                sub_content.append("\t\ttype zeroGradient;")
            elif epsilon_cfg['type'] == 'epsilonWallFunction':
                sub_content.extend([
                    "\t\ttype epsilonWallFunction;",
                    f"\t\tvalue uniform {epsilon_cfg['value']};"
                ])
            else:
                raise ValueError

            sub_content.append("\t}\n")
            file_content.extend(sub_content)

        file_content.append("}")
        return file_content

    @staticmethod
    def generate_nut_file(internalField: float, simulation_cfg: Dict):
        file_content = KEpsilonSimulator.generate_header(2.0, 'volScalarField', location=0, object_name='nut')
        file_content.extend([
            "dimensions [0 2 -1 0 0 0 0];\n",
            f"internalField uniform {internalField};\n",
            "boundaryField",
            "{"
        ])

        for name, cfg in simulation_cfg.items():
            nut_cfg = cfg['nut']
            sub_content = [f"\t{name}", "\t{"]
            if nut_cfg['type'] == 'calculated':
                sub_content.extend([
                    "\t\ttype calculated;",
                    f"\t\tvalue uniform {nut_cfg['value']};"
                ])
            elif nut_cfg['type'] == 'nutkWallFunction':
                sub_content.extend([
                    "\t\ttype nutkWallFunction;",
                    f"\t\tvalue uniform {nut_cfg['value']};"
                ])

            else:
                raise ValueError
            sub_content.append("\t}\n")
            file_content.extend(sub_content)

        file_content.append("}")

        return file_content

    @staticmethod
    def generate_transportProperties_file(nu):
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'constant', 'transportProperties')
        content_json = {
            'transportModel': 'Newtonian',
            'nu': f'[0 2 -1 0 0 0 0] {nu}'
        }
        file_content.extend(KEpsilonSimulator.dict2content(content_json))
        return file_content

    @staticmethod
    def generate_turbulenceProperties_file():
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'constant', 'turbulenceProperties')
        content_json = {
            'simulationType': 'RAS',  # laminar
            'RAS': {
                'RASModel': 'kEpsilon',
                'turbulence': 'on',
                'printCoeffs': 'on'
            }
        }
        file_content.extend(KEpsilonSimulator.dict2content(content_json))
        return file_content

    @staticmethod
    def generate_controlDict_file(application='simpleFoam', endTime=5.0, deltaT=0.005, writeInterval=100):
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'controlDict')
        content_json = {
            'application': f'{application}',
            'startFrom': 'latestTime',
            'startTime': 0,
            'stopAt': 'endTime',
            'endTime': f'{endTime}',
            'deltaT': f'{deltaT}',
            'writeControl': 'timeStep',
            'writeInterval': f'{writeInterval}',
            'purgeWrite': 0,
            'writeFormat': 'ascii',
            'writePrecision': 6,
            'writeCompression': 'off',
            'timeFormat': 'general',
            'runTimeModifiable': 'true'
        }
        file_content.extend(KEpsilonSimulator.dict2content(content_json))
        return file_content

    @staticmethod
    def generate_fvSchemes_file():
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'fvSchemes')
        content_json = {
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
        file_content.extend(KEpsilonSimulator.dict2content(content_json))
        return file_content

    @staticmethod
    def generate_fvSolution_file():
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'fvSolution')
        content_json = {
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
        file_content.extend(KEpsilonSimulator.dict2content(content_json))
        return file_content

    # ------ generate elements based on openfoam
    @staticmethod
    def generate_blockMeshDict_file(
            xmin, xmax, ymin, ymax, zmin, zmax, x_grid=50, y_grid=50, z_grid=50, tol=1.0, convertToMeters=1.0
    ):
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'blockMeshDict')
        file_content.extend([
            f"convertToMeters {convertToMeters};",
            "vertices",
            "(",
            f"   ({xmin - tol} {ymin - tol} {zmin - tol})",
            f"   ({xmax + tol} {ymin - tol} {zmin - tol})",
            f"   ({xmax + tol} {ymax + tol} {zmin - tol})",
            f"   ({xmin - tol} {ymax + tol} {zmin - tol})",

            f"   ({xmin - tol} {ymin - tol} {zmax + tol})",
            f"   ({xmax + tol} {ymin - tol} {zmax + tol})",
            f"   ({xmax + tol} {ymax + tol} {zmax + tol})",
            f"   ({xmin - tol} {ymax + tol} {zmax + tol})",
            ");\n",
            "blocks",
            "(",
            f"    hex (0 1 2 3 4 5 6 7) ({x_grid} {y_grid} {z_grid}) simpleGrading (1 1 1)",
            ");\n",
            "boundary",
            "(",
            "\tallBoundary",
            "\t{",
            "\t\ttype patch;",
            "\t\tfaces",
            "\t\t(",
            "\t\t\t(3 7 6 2)",
            "\t\t\t(0 4 7 3)",
            "\t\t\t(2 6 5 1)",
            "\t\t\t(1 5 4 0)",
            "\t\t\t(0 3 2 1)",
            "\t\t\t(4 5 6 7)",
            "\t\t);",
            "\t}",
            ");\n"
        ])
        return file_content

    @staticmethod
    def generate_meshQualityDict_file():
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'meshQualityDict')
        file_content.extend(['#includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg"'])
        return file_content

    @staticmethod
    def generate_surfaceFeatureExtractDict_file(stl_name: str):
        assert stl_name.endswith('.stl')
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'surfaceFeatureExtractDict')
        file_content.extend([
            f"{stl_name}",
            "{",
            '\t#includeEtc "caseDicts/surface/surfaceFeatureExtractDict.cfg"',
            "}"
        ])
        return file_content

    @staticmethod
    def generate_snappyHexMeshDict_file(stl_name: str, inside_x, inside_y, inside_z):
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'snappyHexMeshDict')
        content_json = {
            '#includeEtc': '"caseDicts/mesh/generation/snappyHexMeshDict.cfg"',
            'castellatedMesh': 'on',
            'snap': 'on',
            'addLayers': 'off',
            'geometry': {
                f"{stl_name}.stl": {
                    'type': 'triSurfaceMesh',
                    'name': 'boundary'
                }
            },
            'castellatedMeshControls': {
                'maxLocalCells': 100000,
                'maxGlobalCells': 2000000,
                'minRefinementCells': 10,
                'nCellsBetweenLevels': 3,
                'features': "\n\t("
                            "\n\t {"
                            f"\n\t   file \"{stl_name}.eMesh\";"
                            "\n\t   level 2;"
                            "\n\t }"
                            "\n\t)",
                'refinementSurfaces': {
                    'boundary': {
                        'level': '(2 3)'
                    }
                },
                'refinementRegions': {
                    'boundary': {
                        'mode': 'inside',
                        'levels': '((1e-5 1))'
                    }
                },
                'resolveFeatureAngle': '30',
                'locationInMesh': f'({inside_x} {inside_y} {inside_z})'
            },
            'snapControls': {
                'explicitFeatureSnap': 'true',
                'nSmoothPatch': 3,
                'tolerance': 1.0,
                'nSolveIter': 300,
                'nRelaxIter': 5
            },
            'addLayersControls': {
                'layers': {
                    'boundary': {
                        'nSurfaceLayers': 3
                    }
                },
                'relativeSizes': 'true',  # false, usually with firstLayerThickness
                'minThickness': 1e-3,
                'expansionRatio': 1.1,
                'finalLayerThickness': 0.3,
                'nGrow': 0
            },
            'meshQualityControls': {},
            'mergeTolerance': 1e-6
        }
        file_content.extend(KEpsilonSimulator.dict2content(content_json))
        return file_content

    # ------ generate elements based on gmsh
    @staticmethod
    def generate_msh_by_stl(stl_file, proj_dir, model_name, meshSizeMax, meshSizeMin):
        geo_file = os.path.join(proj_dir, f"{model_name}.geo")
        msh_file = os.path.join(proj_dir, f"{model_name}.msh")

        content = [
            f"Merge \"{stl_file}\";",
            's[] = Surface {:};',
            'Surface Loop(1) = s[];',
            'Volume(1) = {1};',
            'Physical Surface("boundary", 2) = {1};',
            'Physical Volume("volume", 3) = {1};',
            'Mesh.CharacteristicLengthExtendFromBoundary = 0;',
            f'Mesh.MeshSizeMax = {meshSizeMax};',
            f'Mesh.MeshSizeMin = {meshSizeMin};'
            '//Mesh.CharacteristicLengthMax = 1.0;'
            '// Mesh.CharacteristicLengthMin = 1.0;'
        ]
        with open(geo_file, 'w') as f:
            content = '\n'.join(content)
            f.write(content)

        KEpsilonSimulator.exec_cmd(f"cd {proj_dir}; "
                                   f"gmsh {geo_file} -optimize_netgen -3 -format msh2 -o {msh_file} >> gmsh_export.log")
        return geo_file, msh_file

    # ------ define boundary
    @staticmethod
    def generate_topoSetDict_file(boundary_cfg: Dict):
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'topoSetDict')
        file_content.extend([
            "actions",
            "("
        ])

        for name, cfg in boundary_cfg.items():
            sub_content = []

            if cfg['method'] == 'cylinder':
                center = np.array(cfg['center'])
                vec = np.array(cfg['direction'])
                p1 = center + vec * cfg['tol1']
                p2 = center - vec * cfg['tol2']

                sub_content.extend([
                    "\t{",
                    f"\t\tname {name}_cylinder;",
                    "\t\ttype faceSet;",
                    "\t\taction  new;",
                    "\t\tsource  cylinderToFace;",
                    "\t\tsourceInfo",
                    "\t\t{",
                    f"\t\t\tp1 ({p1[0]} {p1[1]} {p1[2]});",
                    f"\t\t\tp2 ({p2[0]} {p2[1]} {p2[2]});",
                    f"\t\t\tradius {cfg['radius']};",
                    "\t\t}",
                    "\t}"
                ])
                sub_content.extend([
                    "\t{",
                    f"\t\tname {name};",
                    "\t\ttype faceSet;",
                    "\t\taction  new;",
                    "\t\tsource  patchToFace;",
                    "\t\tsourceInfo",
                    "\t\t{",
                    f"\t\t\tpatch {cfg['source']};",
                    "\t\t}",
                    "\t}"
                ])
                sub_content.extend([
                    "\t{",
                    f"\t\tname {name};",
                    "\t\ttype faceSet;",
                    "\t\taction  subset;",
                    "\t\tsource  faceToFace;",
                    "\t\tsourceInfo",
                    "\t\t{",
                    f"\t\t\tset {name}_cylinder;",
                    "\t\t}",
                    "\t}"
                ])

                sub_content.append(f"\t// ************** generate {name}.\n")
                file_content.extend(sub_content)
            else:
                raise ValueError

        file_content.append(");")
        return file_content

    @staticmethod
    def generate_createPatchDict_file(boundary_cfg: Dict):
        file_content = KEpsilonSimulator.generate_header(2.0, 'dictionary', 'system', 'createPatchDict')
        file_content.extend([
            "pointSync false;",
            "patches",
            "("
        ])

        for name in boundary_cfg.keys():
            sun_content = [
                "\t{",
                f"\t\tname {name};",
                f"patchInfo {{type patch;}}",
                "constructFrom set;",
                f"set {name};"
                "\t}"
            ]
            file_content.extend(sun_content)

        file_content.append(");")
        return file_content

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

    # ------ post process
    def run(self):
        self.exec_cmd(f"cd {self.proj_dir}; {self.cfg['controlDict']['application']} >> simulate.log;")
        self.exec_cmd(f"cd {self.proj_dir}; foamToVTK -ascii -latestTime -one-boundary -overwrite >> foam2vtk.log;")

        vtk_dir = os.path.join(self.proj_dir, 'VTK')
        for name in os.listdir(vtk_dir):
            if name.endswith('.vtm'):
                vtk_block = pyvista.read(os.path.join(vtk_dir, name))
                res_vtk = vtk_block['internal']
                return res_vtk


def main():
    cfg = {
        'proj_dir': '/home/admin123456/Desktop/work/path_examples/auto_openfoam_01/test',
        'mesh_file': '/home/admin123456/Desktop/work/path_examples/auto_openfoam_01/model.stl',
        'gmsh_mesh_cfg': {
            'meshSizeMin': 0.001,
            'meshSizeMax': 0.001
        },
        'openfoam_mesh_cfg': {
            'xmin': 0.012,
            'xmax': 0.0425,
            'ymin': 0.012,
            'ymax': 0.029,
            'zmin': -0.01,
            'zmax': 0.023,
            'x_grid': 50,
            'y_grid': 50,
            'z_grid': 50,
            'tol': 0.001,
            'convertToMeters': 1.0,
            'inside_x': 0.018,
            'inside_y': 0.024,
            'inside_z': 0.002,
        },
        'controlDict': {
            'application': 'simpleFoam',
            'endTime': 5.0,
            'deltaT': 0.005,
            'writeInterval': 100
        },
        'transportProperties': {
            'nu': 0.01
        },
        'boundary_definition': {
            'inlet': {
                'method': 'cylinder',
                'center': [0.0187, 0.0245, 0.0],
                'direction': [0.0, 0.0, 1.0],
                'tol1': 0.001,
                'tol2': 0.001,
                'radius': 0.003,
                'source': 'boundary'
            },
            'outlet0': {
                'method': 'cylinder',
                'center': [0.035, 0.0245, 0.0185],
                'direction': [-1.0, 0.0, 0.0],
                'tol1': 0.001,
                'tol2': 0.001,
                'radius': 0.003,
                'source': 'boundary'
            },
            'outlet1': {
                'method': 'cylinder',
                'center': [0.02, 0.0245, 0.0185],
                'direction': [1.0, 0.0, 0.0],
                'tol1': 0.001,
                'tol2': 0.001,
                'radius': 0.003,
                'source': 'boundary'
            },
        },
        'simulation_cfg': {
            'inlet': {
                'U': {
                    'type': 'fixedValue',
                    'value': [0.0, 0.0, 1.0]
                },
                'p': {'type': 'zeroGradient'},
                'nut': {
                    'type': 'calculated',
                    'value': 0
                },
                'k': {
                    'type': 'fixedValue',
                    'value': 0.04
                },
                'epsilon': {
                    'type': 'fixedValue',
                    'value': 2.4
                },
            },
            'outlet0': {
                'U': {'type': 'zeroGradient'},
                'p': {
                    'type': 'fixedValue',
                    'value': 0.0
                },
                'nut': {
                    'type': 'calculated',
                    'value': 0
                },
                'k': {'type': 'zeroGradient'},
                'epsilon': {'type': 'zeroGradient'}
            },
            'outlet1': {
                'U': {'type': 'zeroGradient'},
                'p': {
                    'type': 'fixedValue',
                    'value': 0.0
                },
                'nut': {
                    'type': 'calculated',
                    'value': 0
                },
                'k': {'type': 'zeroGradient'},
                'epsilon': {'type': 'zeroGradient'}
            },
            'boundary': {
                'U': {'type': 'noSlip'},
                'p': {'type': 'zeroGradient'},
                'nut': {
                    'type': 'nutkWallFunction',
                    'value': 0
                },
                'k': {
                    'type': 'kqRWallFunction',
                    'value': 0.04
                },
                'epsilon': {
                    'type': 'epsilonWallFunction',
                    'value': 2.4
                }
            }
        },
        'background_cfg': {
            'U': {'internalField': [0.0, 0.0, 0.0]},
            'p': {'internalField': 0.0},
            'nut': {'internalField': 0.0},
            'k': {'internalField': 0.04},
            'epsilon': {'internalField': 2.4},
        }
    }

    # simulator = KEpsilonSimulator(cfg, by_method='openfoam')
    simulator = KEpsilonSimulator(cfg, by_method='gmsh')


if __name__ == '__main__':
    main()
