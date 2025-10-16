import os
import shutil
from typing import Union, Literal

import numpy as np
import gmsh
import pyvista
import subprocess
import re


class PipeModule(object):
    def __init__(
        self,
        name: str,
        style: Literal["inlet", "outlet"],
        norm_vector: np.ndarray,
        center_point: np.ndarray,
        radius: float,
        length: float,
    ):
        '''
        Args:
            norm_vector: outward normal vector of the surface
        '''
        self.name: str = name
        self.style = style
        norm_vector = norm_vector / np.linalg.norm(norm_vector)
        if self.style == "inlet":
            self.center_point = center_point - norm_vector * length
        elif self.style == "outlet":
            self.center_point = center_point + norm_vector * length
            norm_vector = norm_vector * -1
        else:
            raise ValueError(f"Invalid style: {self.style}")
        self.norm_vector = norm_vector
        self.radius = radius
        self.length = length
        self.stl_file: str

    def set_U(
        self, method: Literal["fixedValue", "zeroGradient", "noSlip", "inletOutlet", "empty"], **kwargs,
    ):
        if method == "fixedValue":
            self.U_dict = {
                "type": "fixedValue",
                "value": f"uniform ({' '.join([str(i) for i in kwargs['value']])})",
                }
        elif method == "zeroGradient":
            self.U_dict = {"type": "zeroGradient"}
        elif method == "noSlip":
            self.U_dict = {"type": "noSlip"}
        elif method == "inletOutlet":
            self.U_dict = {
                "type": "inletOutlet",
                "inletValue": f"uniform ({' '.join([str(i) for i in kwargs['inletValue']])})",
                }
        elif method == "empty":
            self.U_dict = {"type": "empty"}
        else:
            raise ValueError(f"Invalid method: {method}")

    def set_p(self, method: Literal["fixedValue", "zeroGradient", "empty"], **kwargs):
        if method == "fixedValue":
            self.p_dict = {"type": "fixedValue", "value": f"uniform {kwargs['value']}"}
        elif method == "zeroGradient":
            self.p_dict = {"type": "zeroGradient"}
        elif method == "empty":
            self.p_dict = {"type": "empty"}
        else:
            raise ValueError(f"Invalid method: {method}")

    def set_k(
        self,
        method: Literal["fixedValue", "zeroGradient", "kqRWallFunction", "empty"],
        **kwargs,
    ):
        if method == "fixedValue":
            self.k_dict = {"type": "fixedValue", "value": f"uniform {kwargs['value']}"}
        elif method == "zeroGradient":
            self.k_dict = {"type": "zeroGradient"}
        elif method == "kqRWallFunction":
            self.k_dict = {"type": "kqRWallFunction", "value": f"uniform {kwargs['value']}"}
        elif method == "empty":
            self.k_dict = {"type": "empty"}
        else:
            raise ValueError(f"Invalid method: {method}")
    
    def set_epsilon(self, method: Literal["fixedValue", "zeroGradient", "epsilonWallFunction", "empty", "calculated"], **kwargs):
        if method == "fixedValue":
            self.epsilon_dict = {"type": "fixedValue", "value": f"uniform {kwargs['value']}"}
        elif method == "zeroGradient":
            self.epsilon_dict = {"type": "zeroGradient"}
        elif method == "epsilonWallFunction":
            self.epsilon_dict = {
                "type": "epsilonWallFunction",
                "Cmu": kwargs.get("Cmu", 0.09),
                "kappa": kwargs.get("kappa", 0.41),
                "E": kwargs.get("E", 9.8),
                "value": f"uniform {kwargs['value']}",
            }
        elif method == "empty":
            self.epsilon_dict = {"type": "empty"}
        else:
            raise ValueError(f"Invalid method: {method}")

    def set_nut(self, method: Literal["fixedValue", "zeroGradient", "empty", "calculated", "nutkWallFunction"], **kwargs):
        if method == "fixedValue":
            self.nut_dict = {"type": "fixedValue", "value": f"uniform {kwargs['value']}"}
        elif method == "zeroGradient":
            self.nut_dict = {"type": "zeroGradient"}
        elif method == "nutkWallFunction":
            self.nut_dict = {
                "type": "nutkWallFunction",
                "Cmu": kwargs.get("Cmu", 0.09),
                "kappa": kwargs.get("kappa", 0.41),
                "E": kwargs.get("E", 9.8),
                "value": f"uniform {kwargs['value']}",
            }
        elif method == "empty":
            self.nut_dict = {"type": "empty"}
        elif method == "calculated":
            self.nut_dict = {"type": "calculated", "value": f"uniform {kwargs['value']}"}
        else:
            raise ValueError(f"Invalid boundaryField method: {boundary_dict['method']}")
    
    def set_nuTilda(self, method: Literal["fixedValue", "zeroGradient", "empty", "calculated"], **kwargs):
        if method == "fixedValue":
            self.nuTilda_dict = {"type": "fixedValue", "value": f"uniform {kwargs['value']}"}
        elif method == "zeroGradient":
            self.nuTilda_dict = {"type": "zeroGradient"}
        elif method == "empty":
            self.nuTilda_dict = {"type": "empty"}
        elif method == "calculated":
            self.nuTilda_dict = {"type": "calculated", "value": f"uniform {kwargs['value']}"}
        else:
            raise ValueError(f"Invalid method: {method}")

    def set_topoWeight(self, method: Literal["fixedValue", "empty"], **kwargs):
        if method == "fixedValue":
            self.topoWeight_dict = {"type": "fixedValue", "value": f"uniform {kwargs['value']}"}
        elif method == "empty":
            self.topoWeight_dict = {"type": "empty"}
        else:
            raise ValueError(f"Invalid method: {method}")


class WallModule(PipeModule):
    def __init__(
        self, 
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
        name = 'walls' 

    ):
        self.name = name
        self.style = "wall"
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.stl_file: str
    

class FieldModule(object):

    def __init__(
        self,
        name: str,
        geometry_dir: str,
        mesh_max_size: float,
        mesh_min_size: float
    ):
        self.name = name
        self.geometry_dir = geometry_dir
        
        self.pipe_modules: dict[str, PipeModule] = {}
        self.wall_modules: WallModule
        
        self.mesh_max_size = mesh_max_size
        self.mesh_min_size = mesh_min_size

    def add_module(self, module: PipeModule|WallModule):
        if module.style != "wall":
            self.pipe_modules[module.name] = module
        else:
            self.wall_modules = module # type: ignore

    def to_stl(self):
        if not os.path.exists(self.geometry_dir):
            os.makedirs(self.geometry_dir)

        gmsh.initialize()
        gmsh.model.add("field_module")
        box_tag = gmsh.model.occ.addBox(
            self.wall_modules.x_min,
            self.wall_modules.y_min,
            self.wall_modules.z_min,
            self.wall_modules.x_max - self.wall_modules.x_min,
            self.wall_modules.y_max - self.wall_modules.y_min,
            self.wall_modules.z_max - self.wall_modules.z_min,
        )
        for i, pipe_module in enumerate(self.pipe_modules.values()):
            pipe_module_tag = gmsh.model.occ.addCylinder(
                pipe_module.center_point[0],
                pipe_module.center_point[1],
                pipe_module.center_point[2],
                pipe_module.norm_vector[0] * pipe_module.length,
                pipe_module.norm_vector[1] * pipe_module.length,
                pipe_module.norm_vector[2] * pipe_module.length,
                pipe_module.radius,
            )
            if i == 0:
                outDimTags, outDimTagsMap = gmsh.model.occ.fuse(
                    [(3, box_tag)],
                    [(3, pipe_module_tag)],
                    removeObject=True,
                    removeTool=True,
                )
            else:
                outDimTags, outDimTagsMap = gmsh.model.occ.fuse(
                    outDimTags,
                    [(3, pipe_module_tag)],
                    removeObject=True,
                    removeTool=True,
                )
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)

        physical_groups = {
            self.wall_modules.name: [],
            **{pipe_module.name: [] for pipe_module in self.pipe_modules.values()},
        }

        for dim, tag in gmsh.model.getEntities(2):
            """
            element_types: Types of elements (e.g., 2 for triangles in a 2D mesh).
            element_tags: Tags identifying each element (e.g., each triangle).
            node_tags: Lists of node tags associated with each element.
            """
            element_types, element_tags, node_tags = gmsh.model.mesh.getElements(
                dim, tag
            )
            unique_node_tags = np.unique(np.concatenate(node_tags))
            surface_pcd = []
            for node_tag in unique_node_tags:
                coord, parametric_coord, dim, tag = gmsh.model.mesh.getNode(node_tag)
                surface_pcd.append(coord)
            surface_pcd = np.array(surface_pcd)

            is_in_pipe = False
            for pipe_module in self.pipe_modules.values():
                pcd_distance = self.dist_pcd2surface(
                    surface_pcd, pipe_module.center_point, pipe_module.norm_vector
                )
                dist_min, dist_max = np.min(np.abs(pcd_distance)), np.max(
                    np.abs(pcd_distance)
                )
                if (dist_max - dist_min) < 1e-6 and dist_max < 1e-6:
                    physical_groups[pipe_module.name].append(tag)
                    is_in_pipe = True
                    break

            if not is_in_pipe:
                physical_groups[self.wall_modules.name].append(tag)

        for name, tags in physical_groups.items():
            gmsh.model.addPhysicalGroup(2, tags, name=name)

        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_max_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_min_size)
        # gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        # gmsh.option.setNumber("Mesh.Algorithm", 8)  # Delaunay
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(2)

        for pipe_module in self.pipe_modules.values():
            pipe_stl_file = os.path.join(self.geometry_dir, f"{pipe_module.name}.stl")
            self.save_physical_group_to_stl(pipe_module.name, pipe_stl_file)
            pipe_module.stl_file = pipe_stl_file

        wall_stl_file = os.path.join(self.geometry_dir, f"wall.stl")
        self.save_physical_group_to_stl(self.wall_modules.name, wall_stl_file)
        self.wall_modules.stl_file = wall_stl_file

        gmsh.finalize()
        # self.visualize_field()

    def save_physical_group_to_stl(self, physical_group_name, output_file):
        for group_dim, group_tag in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(group_dim, group_tag)
            if name != physical_group_name:
                continue

            with open(output_file, "w") as f:
                f.write(f"solid {physical_group_name} # PhysicalGroup: {name}\n")
                for surface_tag in gmsh.model.getEntitiesForPhysicalGroup(
                    group_dim, group_tag
                ):
                    element_types, element_tags, node_tags = (
                        gmsh.model.mesh.getElements(2, surface_tag)
                    )
                    for i, elem_type in enumerate(element_types):
                        if elem_type != 2:
                            continue

                        # Triangle elements
                        node_tags_i = np.array(node_tags[i]).reshape((-1, 3))
                        for nodes in node_tags_i:
                            coords = [
                                gmsh.model.mesh.getNode(node)[0] for node in nodes
                            ]
                            v1 = np.array(coords[1]) - np.array(coords[0])
                            v2 = np.array(coords[2]) - np.array(coords[0])
                            normal = np.cross(v1, v2)
                            normal /= np.linalg.norm(normal)
                            f.write(
                                f"facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
                            )
                            f.write("    outer loop\n")
                            for coord in coords:
                                f.write(
                                    f"        vertex {coord[0]:.6e} {coord[1]:.6e} {coord[2]:.6e}\n"
                                )
                            f.write("    endloop\n")
                            f.write("endfacet\n")

                f.write(f"endsolid {physical_group_name}\n")

            print(
                f"Saved custom STL with PhysicalGroup '{physical_group_name}' to {output_file}"
            )
            return

        print(f"PhysicalGroup '{physical_group_name}' not found.")

    @staticmethod
    def dist_pcd2surface(
        pcd: np.ndarray, center_point: np.ndarray, norm_vector: np.ndarray
    ):
        center2pcd = pcd - center_point
        return np.sum(center2pcd * norm_vector, axis=1)

    def gmsh_visualize(self):
        gmsh.fltk.run()

    def visualize_field(self):
        plotter = pyvista.Plotter()
        plotter.add_mesh(pyvista.read(self.wall_modules.stl_file), color="black", style="wireframe")  # type: ignore
        for pipe_module in self.pipe_modules.values():
            random_color = np.random.rand(3)
            plotter.add_mesh(pyvista.read(pipe_module.stl_file), color="red", style="wireframe")  # type: ignore
        plotter.show()

    def get_field_boundary(self):
        wall_mesh = pyvista.read(self.wall_modules.stl_file)
        x_min, x_max, y_min, y_max, z_min, z_max = wall_mesh.bounds  # type: ignore

        for pipe_module in self.pipe_modules.values():
            pipe_mesh = pyvista.read(pipe_module.stl_file)
            pipe_x_min, pipe_x_max, pipe_y_min, pipe_y_max, pipe_z_min, pipe_z_max = (
                pipe_mesh.bounds  # type: ignore
            )
            x_min = min(x_min, pipe_x_min)
            x_max = max(x_max, pipe_x_max)
            y_min = min(y_min, pipe_y_min)
            y_max = max(y_max, pipe_y_max)
            z_min = min(z_min, pipe_z_min)
            z_max = max(z_max, pipe_z_max)
        return x_min, x_max, y_min, y_max, z_min, z_max


class OpenFoamFluidSimulator(object):
    def __init__(self, openfoam_dir: str, field_module: FieldModule):
        self.openfoam_dir = openfoam_dir
        self.field_module = field_module
        self.mesh_vtk: pyvista.UnstructuredGrid

    def create_foam_file(
        self,
        location: Union[str, int, float],
        object_name: str,
        content: list[str],
        class_name="dictionary",
        file_name: str | None = None,
    ):
        location_dir = os.path.join(self.openfoam_dir, str(location))
        if not os.path.exists(location_dir):
            os.mkdir(location_dir)

        header = [
            "/*--------------------------------*- C++ -*----------------------------------*\\",
            "  =========                 |",
            "  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox",
            "   \\\\    /   O peration     | Website:  https://openfoam.org",
            "    \\\\  /    A nd           | Version:  dev",
            "     \\\\/     M anipulation  |",
            "\\*---------------------------------------------------------------------------*/",
            "FoamFile\n{",
            "    format      ascii;",
            f"    class       {class_name};",
            f'    location    "{location}";',
            f"    object      {object_name};",
            "}",
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n",
        ]
        content = header + content
        if file_name is None:
            file_name = object_name
        with open(os.path.join(location_dir, file_name), "w") as f:
            f.write("\n".join(content))

    def ndarray2content(self, arg_ndarray: np.ndarray, layer: int = 0):
        ndim = arg_ndarray.ndim
        assert ndim <= 2, "[ERROR] List dimension must be less than 2"
        tab_count = " " * layer * 4
        content = []
        if ndim == 1:
            content.append(f"{tab_count}({' '.join(arg_ndarray.tolist())});")
        elif ndim == 2:
            content.append(f"{tab_count}(")
            for i in range(arg_ndarray.shape[0]):
                content.append(f"{tab_count}({' '.join(arg_ndarray[i].tolist())})")
            content.append(f"{tab_count});")
        return content

    def list2content(self, arg_list: list, layer: int = 0):
        tab_count = " " * layer * 4
        content = []
        for item in arg_list:
            if isinstance(item, (str, int, float)):
                content.append(f"{tab_count}{item}")
            elif isinstance(item, list):
                content.append(f"{tab_count}(")
                content.extend(self.list2content(item, layer + 1))
                content.append(f"{tab_count})")
            elif isinstance(item, dict):
                content.append(f"{tab_count}{{")
                content.extend(self.dictionary2content(item, layer + 1))
                content.append(f"{tab_count}}}")
            else:
                raise ValueError("[ERROR] Non-Valid list value")
        return content

    def dictionary2content(self, arg_dict: dict, layer: int = 0):
        tab_count = " " * layer * 4
        content = []
        for key, value in arg_dict.items():
            if isinstance(value, (str, int, float)):
                content.append(f"{tab_count}{key}  {value};")
            elif isinstance(value, dict):
                content.append(f"{tab_count}{key}")
                content.append(f"{tab_count}{{")
                content.extend(self.dictionary2content(value, layer + 1))
                content.append(f"{tab_count}}}")
                content.append("")
            elif isinstance(value, np.ndarray):
                content.append(f"{tab_count}{key}")
                content.extend(self.ndarray2content(value, layer + 1))
                content.append("")
            elif isinstance(value, list):
                content.append(f"{tab_count}{key}")
                content.append(f"{tab_count}(")
                content.extend(self.list2content(value, layer + 1))
                content.append(f"{tab_count});")
                content.append("")
            else:
                raise ValueError("[ERROR] Non-Valid dict value")
        return content

    def create_controlDict(
        self,
        solver: Literal["incompressibleFluid", "potentialFoam", "topoIncompressibleFluid"] = "incompressibleFluid",
        startFrom: str = "startTime",
        startTime: float = 0,
        stopAt: str = "endTime",
        endTime: float = 10,
        deltaT: float = 0.005,
        writeControl: str = "timeStep",
        writeInterval: int = 100,
        purgeWrite: int = 0,
        writeFormat: str = "ascii",
        writePrecision: int = 6,
        timeFormat: str = "general",
        timePrecision: int = 6,
        runTimeModifiable: bool = True,
    ):
        content = self.dictionary2content(
            {
                "solver": solver,
                "startFrom": startFrom,
                "startTime": startTime,
                "stopAt": stopAt,
                "endTime": endTime,
                "deltaT": deltaT,
                "writeControl": writeControl,
                "writeInterval": writeInterval,
                "purgeWrite": purgeWrite,
                "writeFormat": writeFormat,
                "writePrecision": writePrecision,
                "timeFormat": timeFormat,
                "timePrecision": timePrecision,
                "runTimeModifiable": "true" if runTimeModifiable else "false",
            }
        )
        self.create_foam_file(
            location="system", object_name="controlDict", content=content
        )

    def create_fvSchemes(
        self,
        ddtSchemes: dict = {"default": "Euler"},
        gradSchemes: dict = {"default": "Gauss linear"},
        divSchemes: dict = {
            "default": "none",
            "div(phi,U)": "Gauss linearUpwind grad(U)",
            "div(phi,k)": "Gauss upwind",
            "div(phi,omega)": "bounded Gauss upwind",
            "div(phi,epsilon)": "Gauss upwind",
        },
        laplacianSchemes: dict = {"default": "Gauss linear corrected"},
        interpolationSchemes: dict = {"default": "linear"},
        snGradSchemes: dict = {"default": "corrected"},
    ):
        content = self.dictionary2content(
            {
                "ddtSchemes": ddtSchemes,
                "gradSchemes": gradSchemes,
                "divSchemes": divSchemes,
                "laplacianSchemes": laplacianSchemes,
                "interpolationSchemes": interpolationSchemes,
                "snGradSchemes": snGradSchemes,
            }
        )
        self.create_foam_file(
            location="system", object_name="fvSchemes", content=content
        )

    def create_state_solver(
        self,
        states: list[str],
        solver: Literal["smoothSolver", "GAMG", "PCG"] = "smoothSolver",
        tolerance: float = 1e-5,
        relTol: float = 0.1,
        smoother: str = "GaussSeidel",
        **kwargs,
    ):
        """
        How solver and smoother Work Together
        smoother is computationally inexpensive and run a few iterations to get a good initial guess for the solver before starting the solver
        """
        if len(states) > 1:
            name = f'"({"|".join(states)})"'
        else:
            name = states[0]

        solver_dict = {
            name: {
                "solver": solver,
                "tolerance": tolerance,
                "relTol": relTol,
                "smoother": smoother,
            }
        }
        if solver == "PCG":
            solver_dict[name]["preconditioner"] = kwargs.get("preconditioner", "DIC")
        return solver_dict

    def create_numerical_methods(
        self,
        method: Literal["PIMPLE", "PISO", "SIMPLE", "SIMPLEC"],
        nCorrectors: int = 1,
        nNonOrthogonalCorrectors: int = 0,
        **kwargs,
    ):
        if method == "PIMPLE":
            return {
                "name": method,
                "nCorrectors": nCorrectors,
                "nNonOrthogonalCorrectors": nNonOrthogonalCorrectors,
                "pRefCell": kwargs.get("pRefCell", 0),
                "pRefValue": kwargs.get("pRefValue", 0),
                "residualControl": {
                    "p": kwargs.get("pResidual", 1e-4),
                    "U": kwargs.get("UResidual", 1e-4),
                    "k": kwargs.get("kResidual", 1e-3),
                    "epsilon": kwargs.get("epsilonResidual", 1e-3),
                    "omega": kwargs.get("omegaResidual", 1e-3),
                    "nut": kwargs.get("nutResidual", 1e-3),
                    "nuTilda": kwargs.get("nuTildaResidual", 1e-3),
                },
            }
        elif method == "PISO":
            return {
                "name": method,
                "nCorrectors": nCorrectors,
            }
        elif method == "SIMPLE":
            return {
                "name": method,
                "consistent": "no",
                "nNonOrthogonalCorrectors": nNonOrthogonalCorrectors,
            }
        elif method == "SIMPLEC":
            return {
                "name": method,
                "consistent": "no",
                "nCorrectors": nCorrectors,
                "nNonOrthogonalCorrectors": nNonOrthogonalCorrectors,
            }
        else:
            raise ValueError(f"Invalid method: {method}")

    def create_fvSolution(
        self,
        state_solvers: list[dict],
        numerical_methods: dict,
        relaxationFactors: dict,
    ):
        state_solvers_dict = {}
        for state_solver in state_solvers:
            for state, solver_dict in state_solver.items():
                state_solvers_dict[state] = solver_dict

        name = numerical_methods.pop("name")
        content = self.dictionary2content(
            {
                "solvers": state_solvers_dict,
                name: numerical_methods,
                "relaxationFactors": relaxationFactors,
            }
        )
        self.create_foam_file(
            location="system", object_name="fvSolution", content=content
        )

    def create_blockMeshDict(
        self,
        convertToMeters: float = 1,
        grid_x: int = 10,
        grid_y: int = 10,
        grid_z: int = 10,
    ):
        x_min, x_max, y_min, y_max, z_min, z_max = self.field_module.get_field_boundary()
        x_dif = x_max - x_min
        y_dif = y_max - y_min
        z_dif = z_max - z_min

        x_min = x_min - x_dif / grid_x * 2
        x_max = x_max + x_dif / grid_x * 2
        y_min = y_min - y_dif / grid_y * 2
        y_max = y_max + y_dif / grid_y * 2
        z_min = z_min - z_dif / grid_z * 2
        z_max = z_max + z_dif / grid_z * 2

        content = [
            f"convertToMeters {convertToMeters};\n",
            "vertices\n(",
            f"  ({x_min} {y_min} {z_min})",
            f"  ({x_max} {y_min} {z_min})",
            f"  ({x_max} {y_max} {z_min})",
            f"  ({x_min} {y_max} {z_min})",
            f"  ({x_min} {y_min} {z_max})",
            f"  ({x_max} {y_min} {z_max})",
            f"  ({x_max} {y_max} {z_max})",
            f"  ({x_min} {y_max} {z_max})",
            ");\n",
            "blocks\n(",
            "   hex (0 1 2 3 4 5 6 7)"
            f"  ({grid_x} {grid_y} {grid_z})  simpleGrading  (1 1 1)",
            ");\n",
            "boundary\n(",
            "    left\n    {\n      type patch;\n      faces ( (0 3 7 4) );\n    }\n",
            "    right\n    {\n     type patch;\n      faces ( (1 2 6 5) );\n    }\n",
            "    front\n    {\n     type patch;\n      faces ( (0 1 5 4) );\n    }\n",
            "    back\n    {\n      type patch;\n      faces ( (3 7 6 2) );\n    }\n",
            "    bottom\n    {\n    type patch;\n      faces ( (0 1 2 3) );\n    }\n",
            "    top\n    {\n       type patch;\n      faces ( (4 5 6 7) );\n    }",
            ");",
        ]
        self.create_foam_file(
            location="system", object_name="blockMeshDict", content=content
        )

    def create_physicalProperties(self, nu: float = 1.5e-05):
        content = self.dictionary2content(
            {
                "viscosityModel": "constant",
                "nu": f"[0 2 -1 0 0 0 0] {nu}",
            }
        )
        self.create_foam_file(
            location="constant", object_name="physicalProperties", content=content
        )

    def create_momentumTransport(
        self,
        simulationType: str = "RAS",
        model: str = "kEpsilon",
        turbulence: bool = True,
    ):
        content = self.dictionary2content(
            {
                "simulationType": simulationType,
                simulationType: {
                    "model": model,
                    "turbulence": "on" if turbulence else "off",
                    "printCoeffs": "on",
                },
            }
        )
        self.create_foam_file(
            location="constant",
            object_name="RASProperties",
            content=content,
            file_name="momentumTransport",
        )

    def create_epsilon(self, internalField_dict: dict):
        if internalField_dict["method"] == "uniform":
            internalField = f"uniform {internalField_dict['value']}"
        else:
            raise ValueError(
                f"Invalid internalField method: {internalField_dict['method']}"
            )

        boundaryField = {}
        for pipe_module in self.field_module.pipe_modules.values():
            boundaryField[pipe_module.name] = pipe_module.epsilon_dict
        boundaryField[self.field_module.wall_modules.name] = self.field_module.wall_modules.epsilon_dict
        
        content = self.dictionary2content(
            {
                "dimensions": "[0 2 -3 0 0 0 0]",
                "internalField": internalField,
                "boundaryField": boundaryField,
            }
        )
        self.create_foam_file(location="0", object_name="epsilon", content=content)

    def create_k(self, internalField_dict: dict):
        if internalField_dict["method"] == "uniform":
            internalField = f"uniform {internalField_dict['value']}"
        else:
            raise ValueError(
                f"Invalid internalField method: {internalField_dict['method']}"
            )

        boundaryField = {}
        for pipe_module in self.field_module.pipe_modules.values():
            boundaryField[pipe_module.name] = pipe_module.k_dict
        boundaryField[self.field_module.wall_modules.name] = self.field_module.wall_modules.k_dict
        
        content = self.dictionary2content(
            {
                "dimensions": "[0 2 -2 0 0 0 0]",
                "internalField": internalField,
                "boundaryField": boundaryField,
            }
        )
        self.create_foam_file(location="0", object_name="k", content=content)

    def create_nut(self, internalField_dict: dict):
        if internalField_dict["method"] == "uniform":
            internalField = f"uniform {internalField_dict['value']}"
        elif internalField_dict["method"] == "nonuniform":
            assert isinstance(internalField_dict['value'], np.ndarray), "The value of nonuniform internalField must be a numpy array"
            assert internalField_dict["value"].ndim == 1, "The value of nonuniform internalField must be a 1D array"
            
            internalField = f"nonuniform List<scalar>\n{internalField_dict['value'].shape[0]}\n"
            internalField += "(\n"
            for sub_value in internalField_dict["value"]:
                internalField += f"{sub_value}\n"
            internalField += ")\n"
        else:
            raise ValueError(
                f"Invalid internalField method: {internalField_dict['method']}"
            )

        boundaryField = {}
        for pipe_module in self.field_module.pipe_modules.values():
            boundaryField[pipe_module.name] = pipe_module.nut_dict
        boundaryField[self.field_module.wall_modules.name] = self.field_module.wall_modules.nut_dict
        
        content = self.dictionary2content(
            {
                "dimensions": "[0 2 -1 0 0 0 0]",
                "internalField": internalField,
                "boundaryField": boundaryField,
            }
        )
        self.create_foam_file(location="0", object_name="nut", content=content)

    def create_nuTilda(self, internalField_dict: dict):
        if internalField_dict["method"] == "uniform":
            internalField = f"uniform {internalField_dict['value']}"
        else:
            raise ValueError(
                f"Invalid internalField method: {internalField_dict['method']}"
            )

        boundaryField = {}
        for pipe_module in self.field_module.pipe_modules.values():
            boundaryField[pipe_module.name] = pipe_module.nuTilda_dict
        boundaryField[self.field_module.wall_modules.name] = self.field_module.wall_modules.nuTilda_dict
        
        content = self.dictionary2content(
            {
                "dimensions": "[0 2 -1 0 0 0 0]",
                "internalField": internalField,
                "boundaryField": boundaryField,
            }
        )
        self.create_foam_file(location="0", object_name="nuTilda", content=content)

    def create_p(self, internalField_dict: dict):
        if internalField_dict["method"] == "uniform":
            internalField = f"uniform {internalField_dict['value']}"
        elif internalField_dict["method"] == "nonuniform":
            assert isinstance(internalField_dict['value'], np.ndarray), "The value of nonuniform internalField must be a numpy array"
            assert internalField_dict["value"].ndim == 1, "The value of nonuniform internalField must be a 1D array"
            
            internalField = f"nonuniform List<scalar>\n{internalField_dict['value'].shape[0]}\n"
            internalField += "(\n"
            for sub_value in internalField_dict["value"]:
                internalField += f"{sub_value}\n"
            internalField += ")\n"
        else:
            raise ValueError(
                f"Invalid internalField method: {internalField_dict['method']}"
            )
        boundaryField = {}
        for pipe_module in self.field_module.pipe_modules.values():
            boundaryField[pipe_module.name] = pipe_module.p_dict
        boundaryField[self.field_module.wall_modules.name] = self.field_module.wall_modules.p_dict
        
        content = self.dictionary2content(
            {
                "dimensions": "[0 2 -2 0 0 0 0]",
                "internalField": internalField,
                "boundaryField": boundaryField,
            }
        )
        self.create_foam_file(location="0", object_name="p", content=content)

    def create_U(self, internalField_dict: dict):
        if internalField_dict["method"] == "uniform":
            internalField = (
                f"uniform ({' '.join([str(i) for i in internalField_dict['value']])})"
            )
        elif internalField_dict["method"] == "nonuniform":
            assert isinstance(internalField_dict['value'], np.ndarray), "The value of nonuniform internalField must be a numpy array"
            assert internalField_dict["value"].ndim == 2, "The value of nonuniform internalField must be a 2D array"
            assert internalField_dict["value"].shape[1] == 3, "The value of nonuniform internalField must be a 2D array with 3 columns"
            
            internalField = f"nonuniform List<vector>\n{internalField_dict['value'].shape[0]}\n"
            internalField += "(\n"
            for sub_value in internalField_dict["value"]:
                internalField += f"({' '.join([str(i) for i in sub_value])})\n"
            internalField += ")\n"
        else:
            raise ValueError(
                f"Invalid internalField method: {internalField_dict['method']}"
            )

        boundaryField = {}
        for pipe_module in self.field_module.pipe_modules.values():
            boundaryField[pipe_module.name] = pipe_module.U_dict
        boundaryField[self.field_module.wall_modules.name] = self.field_module.wall_modules.U_dict
        
        content = self.dictionary2content(
            {
                "dimensions": "[0 1 -1 0 0 0 0]",
                "internalField": internalField,
                "boundaryField": boundaryField,
            }
        )
        self.create_foam_file(location="0", object_name="U", content=content)

    def create_topoWeight(self, internalField_dict: dict):
        if internalField_dict["method"] == "uniform":
            internalField = f"uniform {internalField_dict['value']}"
        elif internalField_dict["method"] == "nonuniform":
            assert isinstance(internalField_dict['value'], np.ndarray), "The value of nonuniform internalField must be a numpy array"
            assert internalField_dict["value"].ndim == 1, "The value of nonuniform internalField must be a 1D array"
            internalField = f"nonuniform List<scalar>\n{internalField_dict['value'].shape[0]}\n"
            internalField += "(\n"
            for sub_value in internalField_dict["value"]:
                internalField += f"{sub_value}\n"
            internalField += ")\n"
        else:
            raise ValueError(
                f"Invalid internalField method: {internalField_dict['method']}"
            )

        boundaryField = {}
        for pipe_module in self.field_module.pipe_modules.values():
            boundaryField[pipe_module.name] = pipe_module.topoWeight_dict
        boundaryField[self.field_module.wall_modules.name] = self.field_module.wall_modules.topoWeight_dict
        
        content = self.dictionary2content(
            {
                "dimensions": "[0 0 -1 0 0 0 0]",
                "internalField": internalField,
                "boundaryField": boundaryField,
            }
        )
        self.create_foam_file(location="0", object_name="topoWeight", content=content)

    def create_snappyHexMeshDict(
        self,
        inside_point: tuple[float, float, float],
        castellatedMesh: bool,
        snap: bool,
        addLayers: bool,
        max_local_cells: int = 100000,
        max_global_cells: int = 7000000,
        min_refinement_cells: int = 10,
        n_cells_between_levels: int = 2,
        castellatedMesh_levels=[2, 2],
        castellatedMesh_feature_levels=2,
        castellatedMesh_resolveFeatureAngle=60,
        castellatedMesh_allowFreeStandingZoneFaces=False,
        snap_nSmoothPatch: int = 5,
        snap_tolerance: float = 5.0,
        snap_nSolveIter: int = 20,
        snap_nRelaxIter: int = 10,
        snap_nFeatureSnapIter: int = 10,
        snap_explicitFeatureSnap: bool = True,
        snap_multiRegionFeatureSnap: bool = False,
        layers_relativeSizes: bool = True,
        layers_nSurfaceLayers: int = 3,
        layers_expansionRatio: float = 1.1,
        layers_finalLayerThickness: float = 0.3,
        layers_minThickness: float = 0.05,
        layers_nGrow: int = 1,
        layers_featureAngle: float = 130,
        layers_slipFeatureAngle: float = 30,
        layers_nRelaxIter: int = 5,
        layers_nSmoothSurfaceNormals: int = 2,
        layers_nSmoothNormals: int = 6,
        layers_nSmoothThickness: int = 2,
        layers_maxFaceThicknessRatio: float = 0.5,
        layers_maxThicknessToMedialRatio: float = 0.3,
        layers_minMedianAxisAngle: float = 90,
        layers_nBufferCellsNoExtrude: int = 0,
        layers_nLayerIter: int = 50,
        meshQualityControls_maxNonOrtho: float = 65,
        meshQualityControls_maxBoundarySkewness: float = 20,
        meshQualityControls_maxInternalSkewness: float = 4,
        meshQualityControls_maxConcave: float = 80,
        meshQualityControls_minFlatness: float = 0.5,
        meshQualityControls_minVol: float = 1e-16,
        meshQualityControls_minTetQuality: float = 1e-15,
        meshQualityControls_minTwist: float = 0.05,
        meshQualityControls_minDeterminant: float = 0.001,
        meshQualityControls_minFaceWeight: float = 0.05,
        meshQualityControls_minVolRatio: float = 0.01,
        meshQualityControls_nSmoothScale: int = 4,
        meshQualityControls_errorReduction: float = 0.75,
    ):
        constant_dir = os.path.join(self.openfoam_dir, "constant")
        if not os.path.exists(constant_dir):
            os.makedirs(constant_dir)
        geometry_dir = os.path.join(constant_dir, "geometry")
        if not os.path.exists(geometry_dir):
            os.makedirs(geometry_dir)

        geometry_dict = {}
        geometry_dict[self.field_module.wall_modules.name] = {
            "type": "triSurface",
            "file": f'"{os.path.basename(self.field_module.wall_modules.stl_file)}"',
        }
        for pipe_module in self.field_module.pipe_modules.values():
            geometry_dict[pipe_module.name] = {
                "type": "triSurface",
                "file": f'"{os.path.basename(pipe_module.stl_file)}"',
            }

        refinement_surfaces = {}
        refinement_surfaces[self.field_module.wall_modules.name] = {
            "level": f"({castellatedMesh_levels[0]} {castellatedMesh_levels[1]})",
            "patchInfo": {
                "type": "wall",
                # "inGroups": "(Wall)",
            },
        }
        for pipe_module in self.field_module.pipe_modules.values():
            refinement_surfaces[pipe_module.name] = {
                "level": f"({castellatedMesh_levels[0]} {castellatedMesh_levels[1]})",
                "patchInfo": {
                    "type": "patch",
                    # "inGroups": "(Inlet)" if pipe_module.style == "inlet" else "(Outlet)"
                },
            }

        castellatedMeshControls_feature = []
        for file in os.listdir(geometry_dir):
            if os.path.isfile(os.path.join(geometry_dir, file)):
                if file.endswith(".eMesh"):
                    castellatedMeshControls_feature.append(
                        {
                            "file": f'"{file}"',
                            "level": castellatedMesh_feature_levels,
                        }
                    )
        if len(castellatedMeshControls_feature) == 0:
            castellatedMeshControls_feature = "()"

        castellatedMeshControls = {
            "maxLocalCells": max_local_cells,
            "maxGlobalCells": max_global_cells,
            "minRefinementCells": min_refinement_cells,
            "nCellsBetweenLevels": n_cells_between_levels,
            "features": castellatedMeshControls_feature,
            "refinementSurfaces": refinement_surfaces,
            "resolveFeatureAngle": castellatedMesh_resolveFeatureAngle,
            "insidePoint": f"({inside_point[0]} {inside_point[1]} {inside_point[2]})",
            "allowFreeStandingZoneFaces": (
                "true" if castellatedMesh_allowFreeStandingZoneFaces else "false"
            ),
        }

        snapControls = {
            "nSmoothPatch": snap_nSmoothPatch,
            "tolerance": snap_tolerance,
            "nSolveIter": snap_nSolveIter,
            "nRelaxIter": snap_nRelaxIter,
            "nFeatureSnapIter": snap_nFeatureSnapIter,
            "explicitFeatureSnap": "true" if snap_explicitFeatureSnap else "false",
            "multiRegionFeatureSnap": (
                "true" if snap_multiRegionFeatureSnap else "false"
            ),
        }

        layers = {}
        layers[self.field_module.wall_modules.name] = {"nSurfaceLayers": layers_nSurfaceLayers}
        for pipe_module in self.field_module.pipe_modules.values():
            layers[pipe_module.name] = {"nSurfaceLayers": layers_nSurfaceLayers}

        addLayersControls = {
            "relativeSizes": layers_relativeSizes,
            "layers": layers,
            "expansionRatio": layers_expansionRatio,
            "finalLayerThickness": layers_finalLayerThickness,
            "minThickness": layers_minThickness,
            "nGrow": layers_nGrow,
            "featureAngle": layers_featureAngle,
            "slipFeatureAngle": layers_slipFeatureAngle,
            "nRelaxIter": layers_nRelaxIter,
            "nSmoothSurfaceNormals": layers_nSmoothSurfaceNormals,
            "nSmoothNormals": layers_nSmoothNormals,
            "nSmoothThickness": layers_nSmoothThickness,
            "maxFaceThicknessRatio": layers_maxFaceThicknessRatio,
            "maxThicknessToMedialRatio": layers_maxThicknessToMedialRatio,
            "minMedianAxisAngle": layers_minMedianAxisAngle,
            "nBufferCellsNoExtrude": layers_nBufferCellsNoExtrude,
            "nLayerIter": layers_nLayerIter,
        }

        meshQualityControls = {
            "maxNonOrtho": meshQualityControls_maxNonOrtho,
            "maxBoundarySkewness": meshQualityControls_maxBoundarySkewness,
            "maxInternalSkewness": meshQualityControls_maxInternalSkewness,
            "maxConcave": meshQualityControls_maxConcave,
            "minFlatness": meshQualityControls_minFlatness,
            "minVol": meshQualityControls_minVol,
            "minTetQuality": meshQualityControls_minTetQuality,
            "minTwist": meshQualityControls_minTwist,
            "minDeterminant": meshQualityControls_minDeterminant,
            "minFaceWeight": meshQualityControls_minFaceWeight,
            "minVolRatio": meshQualityControls_minVolRatio,
            "nSmoothScale": meshQualityControls_nSmoothScale,
            "errorReduction": meshQualityControls_errorReduction,
        }

        content = self.dictionary2content(
            {
                "castellatedMesh": "true" if castellatedMesh else "false",
                "snap": "true" if snap else "false",
                "addLayers": "true" if addLayers else "false",
                "geometry": geometry_dict,
                "castellatedMeshControls": castellatedMeshControls,
                "snapControls": snapControls,
                "addLayersControls": addLayersControls,
                "meshQualityControls": meshQualityControls,
                "mergeTolerance": 1e-6,
            }
        )
        self.create_foam_file(
            location="system", object_name="snappyHexMeshDict", content=content
        )

    def create_surfaceFeaturesDict(self, included_angle: float = 150):
        surface_features_dict = {}
        for pipe_module in self.field_module.pipe_modules.values():
            surface_features_dict[pipe_module.name] = {
                "surfaces": [f'"{os.path.basename(pipe_module.stl_file)}"'],
                "includedAngle": included_angle,
            }
        surface_features_dict[self.field_module.wall_modules.name] = {
            "surfaces": [f'"{os.path.basename(self.field_module.wall_modules.stl_file)}"'],
            "includedAngle": included_angle,
        }
        content = self.dictionary2content(surface_features_dict)
        self.create_foam_file(
            location="system", object_name="surfaceFeaturesDict", content=content
        )

    def compute_initial_value(self, turbulence_model: str = "kEpsilon", **kwargs):
        """
        Args:
            - velocity_ref: reference velocity (m/s)
            - length_ref: reference length scale (m)
            - turbulence_intensity: turbulence intensity (0.0-1.0)
            - viscosity_ratio: turbulent to laminar viscosity ratio
            - nu: kinematic viscosity (m²/s)
            - specific_values: directly provided k, epsilon, or nuTilda values
        """
        if turbulence_model == "kEpsilon":
            # compute k and epsilon
            k = 1.5 * (kwargs["velocity_ref"] * kwargs["turbulence_intensity"]) ** 2

            # Estimate epsilon using mixing length hypothesis
            # epsilon = C_mu^(3/4) * k^(3/2) / l
            C_mu = 0.09  # Standard k-epsilon constant
            mixing_length = 0.07 * kwargs["length_ref"]  # 7% of characteristic length
            epsilon = (C_mu**0.75) * (k**1.5) / mixing_length

            k = max(k, 1e-12)
            epsilon = max(epsilon, 1e-12)
            return {"k": k, "epsilon": epsilon}

        elif turbulence_model == "SpalartAllmaras":
            # compute nuTilda
            raise NotImplementedError(
                "SpalartAllmaras turbulence model is not implemented yet"
            )
        else:
            raise ValueError(f"Invalid turbulence model: {turbulence_model}")

    def foam2vtk(self, polyhedra: Literal["polyhedra", "none", "all"]):
        '''
        Args:
            - polyhedra: "polyhedra" means only polyhedra will be saved
            - none: means no polyhedra will be saved
                foamToVTK automatically decomposes polyhedral cells into a combination of tetrahedra and pyramids.
                the VTK files generated by foamToVTK in OpenFOAM can have a different number of points compared to 
                the constant/polyMesh/points file in your case directory.
            - all: means all meshes will be saved
        '''
        vtk_dir = os.path.join(self.openfoam_dir, "VTK")
        if os.path.exists(vtk_dir):
            shutil.rmtree(vtk_dir)
        
        subprocess.run(
            f"foamToVTK -ascii -latestTime -allPatches -polyhedra {polyhedra}", 
            shell=True, check=True, cwd=self.openfoam_dir
        )
        if not os.path.exists(vtk_dir):
            return

        vtk_files = []
        for file in os.listdir(vtk_dir):
            if file.endswith(".vtk") and (not os.path.isdir(os.path.join(vtk_dir, file))):
                vtk_files.append(os.path.join(vtk_dir, file))
        assert len(vtk_files) == 1, "There should be only one vtk file"
        
        self.mesh_vtk = pyvista.read(vtk_files[0])  # type: ignore
    
    def run_simulation(self):
        try:
            process = subprocess.Popen(
                "foamRun",
                shell=True,
                cwd=self.openfoam_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr with stdout
                bufsize=1,  # Line buffered
                universal_newlines=True  # Text mode
            )
            
            content = []
            for line in iter(process.stdout.readline, ''):  # type: ignore
                line = line.strip()
                if line:
                    # print('[LOG]:', line)  # Real-time output
                    content.append(line)
                    res_dict = self.divergence_check(content)
                    if res_dict['is_diverged'] or (res_dict['state'] == 'failed'):
                        process.terminate()
                        break
                    if res_dict['is_new_step']:
                        content = content[-1:]

            process.stdout.close()  # type: ignore
            return_code = process.wait()
            
            if return_code != 0:
                print(f"Simulation ended with error code: {return_code}")
            else:
                print("Simulation completed successfully")
                
        except Exception as e:
            print(f"Error running simulation: {e}")
    
    def divergence_check(self, content: list[str]) -> dict:
        res_dict = {'is_new_step': False, 'state': 'success'}
        if len(content) > 1 and content[-1].startswith('Time = '):
            res_dict['is_new_step'] = True

            content_str = '\n'.join(content[:-1]).strip()
            time_match = re.search(r'Time\s*=\s*([\d.]+)', content_str)
            if time_match:
                time_step = float(time_match.group(1))
                res_dict['time_step'] = time_step
            else:
                res_dict['state'] = 'failed'
            
            continuity_match = re.search(
                r'time step continuity errors\s*:\s*sum local\s*=\s*([\d.e+-]+),\s*global\s*=\s*([\d.e+-]+),\s*cumulative\s*=\s*([\d.e+-]+)', 
                content_str
            )
            if continuity_match:
                sum_local = float(continuity_match.group(1))
                global_error = float(continuity_match.group(2))
                cumulative = float(continuity_match.group(3))
                res_dict['sum_local'] = sum_local
                res_dict['global_error'] = global_error
                res_dict['cumulative'] = cumulative
                res_dict['is_diverged'] = sum_local > 1e2 or global_error > 1e2
            else:
                res_dict['state'] = 'failed'
            
            pattern = r'Solving for (.*?),\s*Initial residual\s*=\s*([\d.]+),\s*Final residual\s*=\s*([\d.]+),\s*No Iterations\s*=\s*(\d+)'
            field_match = re.search(pattern, content_str)
            
            if field_match:
                res_dict['field'] = field_match.group(1)
                res_dict['initial_residual'] = float(field_match.group(2))
                res_dict['final_residual'] = float(field_match.group(3))
                res_dict['iterations'] = int(field_match.group(4))
            else:
                res_dict['state'] = 'failed'
        
        return res_dict


def test_openfoam_kEpsilon():
    openfoam_dir = "/home/admin123456/Desktop/work/openfoam_database/test"

    # ------ 1. create field module ------
    field_module = FieldModule(
        name="field",
        geometry_dir=os.path.join(openfoam_dir, "constant", "geometry"),
        mesh_max_size=0.1,
        mesh_min_size=0.05,
    )
    openfoam_simulator = OpenFoamFluidSimulator(openfoam_dir, field_module)
    
    # ------ 2. compute initial value ------
    initial_value = openfoam_simulator.compute_initial_value(
        turbulence_model="kEpsilon",
        velocity_ref=5.0,
        length_ref=1.0,
        turbulence_intensity=0.05,
    )

    # ------ 3. create pipe modules ------
    wall_module = WallModule(
        x_min=0,
        x_max=8,
        y_min=0,
        y_max=2,
        z_min=0,
        z_max=2,
    )
    wall_module.set_U(method="noSlip")
    wall_module.set_p(method="zeroGradient")
    wall_module.set_k(method="kqRWallFunction", value=initial_value["k"])
    wall_module.set_epsilon(method="epsilonWallFunction", value=initial_value["epsilon"])
    wall_module.set_nut(method="nutkWallFunction", value=0.0)
    field_module.add_module(wall_module)
    
    inlet_pipe_module = PipeModule(
        name="inlet",
        style="inlet",
        norm_vector=np.array([0.0, 0.0, 1.0]),
        center_point=np.array([1.5, 1.0, 0.0]),
        radius=0.5,
        length=0.25,
    )
    inlet_pipe_module.set_U(method="fixedValue", value=[0.0, 0.0, 5.0])
    inlet_pipe_module.set_p(method="zeroGradient")
    inlet_pipe_module.set_k(method="fixedValue", value=initial_value["k"])
    inlet_pipe_module.set_epsilon(method="fixedValue", value=initial_value["epsilon"])
    inlet_pipe_module.set_nut(method="calculated", value=0.0)
    field_module.add_module(inlet_pipe_module)
    
    outlet_pipe_module = PipeModule(
        name="outlet",
        style="outlet",
        norm_vector=np.array([0.0, 0.0, 1.0]),
        center_point=np.array([6.5, 1.0, 2.0]),
        radius=0.5,
        length=0.25,
    )
    outlet_pipe_module.set_U(method="zeroGradient")
    outlet_pipe_module.set_p(method="fixedValue", value=0.0)
    outlet_pipe_module.set_k(method="zeroGradient")
    outlet_pipe_module.set_epsilon(method="zeroGradient")
    outlet_pipe_module.set_nut(method="calculated", value=0.0)
    field_module.add_module(outlet_pipe_module)

    field_module.to_stl()
    # field_module.visualize_field()

    # ------ 4. create openfoam files ------
    openfoam_simulator.create_U(internalField_dict={"method": "uniform", "value": [1.0, 0.0, 0.0]})
    openfoam_simulator.create_p(internalField_dict={"method": "uniform", "value": 0.0})
    openfoam_simulator.create_k(internalField_dict={"method": "uniform", "value": initial_value["k"]})
    openfoam_simulator.create_epsilon(internalField_dict={"method": "uniform", "value": initial_value["epsilon"]})
    openfoam_simulator.create_nut(internalField_dict={"method": "uniform", "value": 0.0})

    openfoam_simulator.create_controlDict()
    openfoam_simulator.create_fvSchemes(
        ddtSchemes={"default": "steadyState"},
        # ddtSchemes={"default": "Euler"},
    )

    numerical_methods = openfoam_simulator.create_numerical_methods(method="PIMPLE")
    state_solvers = [
        openfoam_simulator.create_state_solver(
            states=["U"],
            solver="smoothSolver",
            tolerance=1e-5,
            relTol=0.1,
            smoother="GaussSeidel",
        ),
        openfoam_simulator.create_state_solver(
            states=["p"],
            solver="GAMG",
            tolerance=1e-5,
            relTol=0.1,
            smoother="GaussSeidel",
        ),
        openfoam_simulator.create_state_solver(
            states=["k"],
            solver="smoothSolver",
            tolerance=1e-5,
            relTol=0.1,
            smoother="GaussSeidel",
        ),
        openfoam_simulator.create_state_solver(
            states=["epsilon"],
            solver="smoothSolver",
            tolerance=1e-5,
            relTol=0.1,
            smoother="GaussSeidel",
        ),
    ]
    relaxation_factors = {
        "U": 0.3,
        "p": 0.5,
        # "k": 0.9,
        # "epsilon": 0.9,
    }
    openfoam_simulator.create_fvSolution(
        state_solvers=state_solvers,
        numerical_methods=numerical_methods,
        relaxationFactors=relaxation_factors,
    )

    openfoam_simulator.create_physicalProperties()
    openfoam_simulator.create_momentumTransport()

    # ------ 5. create mesh ------
    openfoam_simulator.create_blockMeshDict(
        convertToMeters=1,
        grid_x=30,
        grid_y=30,
        grid_z=30,
    )
    subprocess.run("blockMesh", shell=True, check=True, cwd=openfoam_dir)

    openfoam_simulator.create_surfaceFeaturesDict(included_angle=30)
    subprocess.run("surfaceFeatures", shell=True, check=True, cwd=openfoam_dir)

    openfoam_simulator.create_snappyHexMeshDict(
        inside_point=(4.0, 1.0, 1.0),
        castellatedMesh=True,
        snap=True,
        addLayers=False,
    )
    subprocess.run("snappyHexMesh", shell=True, check=True, cwd=openfoam_dir)


def test_openfoam_topoIncompressibleFluid():
    openfoam_dir = "/home/admin123456/Desktop/work/openfoam_database/test2"
    
    # ------ 1. create field module ------
    field_module = FieldModule(
        name="field",
        geometry_dir=os.path.join(openfoam_dir, "constant", "geometry"),
        mesh_max_size=0.1,
        mesh_min_size=0.05,
    )
    openfoam_simulator = OpenFoamFluidSimulator(openfoam_dir, field_module)
    
    # ------ 2. compute initial value ------
    initial_value = openfoam_simulator.compute_initial_value(
        turbulence_model="kEpsilon",
        velocity_ref=2.0,
        length_ref=1.0,
        turbulence_intensity=0.05,
    )

    # ------ 3. create pipe modules ------
    wall_module = WallModule(
        x_min=0,
        x_max=4,
        y_min=0,
        y_max=4,
        z_min=0,
        z_max=2,
    )
    wall_module.set_U(method="noSlip")
    wall_module.set_p(method="zeroGradient")
    wall_module.set_k(method="kqRWallFunction", value=initial_value["k"])
    wall_module.set_epsilon(method="epsilonWallFunction", value=initial_value["epsilon"])
    wall_module.set_nut(method="nutkWallFunction", value=0.0)
    wall_module.set_topoWeight(method="fixedValue", value=0.0)
    field_module.add_module(wall_module)
    
    inlet_pipe_module = PipeModule(
        name="inlet",
        style="inlet",
        norm_vector=np.array([1.0, 0.0, 0.0]),
        center_point=np.array([0.0, 2.0, 1.0]),
        radius=0.5,
        length=0.25,
    )
    inlet_pipe_module.set_U(method="fixedValue", value=[2.0, 0.0, 0.0])
    inlet_pipe_module.set_p(method="zeroGradient")
    inlet_pipe_module.set_k(method="fixedValue", value=initial_value["k"])
    inlet_pipe_module.set_epsilon(method="fixedValue", value=initial_value["epsilon"])
    inlet_pipe_module.set_nut(method="calculated", value=0.0)
    inlet_pipe_module.set_topoWeight(method="fixedValue", value=0.0)
    field_module.add_module(inlet_pipe_module)
    
    outlet_pipe_module = PipeModule(
        name="outlet",
        style="outlet",
        norm_vector=np.array([1.0, 0.0, 0.0]),
        center_point=np.array([4.0, 2.0, 1.0]),
        radius=0.5,
        length=0.25,
    )
    outlet_pipe_module.set_U(method="zeroGradient")
    outlet_pipe_module.set_p(method="fixedValue", value=0.0)
    outlet_pipe_module.set_k(method="zeroGradient")
    outlet_pipe_module.set_epsilon(method="zeroGradient")
    outlet_pipe_module.set_nut(method="calculated", value=0.0)
    outlet_pipe_module.set_topoWeight(method="fixedValue", value=0.0)
    field_module.add_module(outlet_pipe_module)

    field_module.to_stl()
    # field_module.visualize_field()

    # ------ 4. create openfoam files ------
    openfoam_simulator.create_controlDict(
        solver="topoIncompressibleFluid"
        # solver="incompressibleFluid"
    )
    openfoam_simulator.create_fvSchemes(
        ddtSchemes={"default": "steadyState"},
        # ddtSchemes={"default": "Euler"},
    )

    numerical_methods = openfoam_simulator.create_numerical_methods(method="PIMPLE")
    state_solvers = [
        openfoam_simulator.create_state_solver(
            states=["U"],
            solver="smoothSolver",
            tolerance=1e-5,
            relTol=0.1,
            smoother="GaussSeidel",
        ),
        openfoam_simulator.create_state_solver(
            states=["p"],
            solver="GAMG",
            tolerance=1e-5,
            relTol=0.1,
            smoother="GaussSeidel",
        ),
        openfoam_simulator.create_state_solver(
            states=["k"],
            solver="smoothSolver",
            tolerance=1e-5,
            relTol=0.1,
            smoother="GaussSeidel",
        ),
        openfoam_simulator.create_state_solver(
            states=["epsilon"],
            solver="smoothSolver",
            tolerance=1e-5,
            relTol=0.1,
            smoother="GaussSeidel",
        ),
    ]
    relaxation_factors = {
        "U": 0.3,
        "p": 0.5,
        # "k": 0.9,
        # "epsilon": 0.9,
    }
    openfoam_simulator.create_fvSolution(
        state_solvers=state_solvers,
        numerical_methods=numerical_methods,
        relaxationFactors=relaxation_factors,
    )

    openfoam_simulator.create_physicalProperties()
    openfoam_simulator.create_momentumTransport()

    # ------ 5. create mesh ------
    openfoam_simulator.create_blockMeshDict(
        convertToMeters=1,
        grid_x=30,
        grid_y=30,
        grid_z=30,
    )
    subprocess.run("blockMesh", shell=True, check=True, cwd=openfoam_dir)

    openfoam_simulator.create_surfaceFeaturesDict(included_angle=30)
    subprocess.run("surfaceFeatures", shell=True, check=True, cwd=openfoam_dir)

    openfoam_simulator.create_snappyHexMeshDict(
        inside_point=(2.0, 2.0, 1.0),
        castellatedMesh=True,
        snap=True,
        addLayers=False,
    )
    subprocess.run("snappyHexMesh", shell=True, check=True, cwd=openfoam_dir)
    
    # ------ 6. create openfoam files ------
    openfoam_simulator.create_U(internalField_dict={"method": "uniform", "value": [0.0, 0.0, 0.0]})
    openfoam_simulator.create_p(internalField_dict={"method": "uniform", "value": 0.0})
    openfoam_simulator.create_k(internalField_dict={"method": "uniform", "value": initial_value["k"]})
    openfoam_simulator.create_epsilon(internalField_dict={"method": "uniform", "value": initial_value["epsilon"]})
    openfoam_simulator.create_nut(internalField_dict={"method": "uniform", "value": 0.0})
    
    openfoam_simulator.foam2vtk(polyhedra="polyhedra")
    mesh_cell_centers = np.asarray(openfoam_simulator.mesh_vtk.cell_centers().points)  # type: ignore    
    dists = np.linalg.norm(mesh_cell_centers[:, :2] - np.array([2.0, 2.0]), axis=1)
    weights = np.zeros(mesh_cell_centers.shape[0])
    weights[dists < 0.5] = 1e6
    openfoam_simulator.create_topoWeight(internalField_dict={"method": "nonuniform", "value": weights})
    # openfoam_simulator.create_topoWeight(internalField_dict={"method": "uniform", "value": 0.0})  # used for original test
    


if __name__ == "__main__":
    # test_openfoam_kEpsilon()
    # test_openfoam_topoIncompressibleFluid()
    pass
