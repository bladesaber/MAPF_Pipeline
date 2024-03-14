import pyvista
import gmsh
import os
import dolfinx
from typing import Union, List, Literal, Dict, Callable
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from dolfinx.io.gmshio import extract_geometry
from sklearn.neighbors import KDTree
from dolfinx import geometry
import collections

from .dolfinx_utils import MeshUtils
from .vis_mesh_utils import VisUtils


class ReMesher(object):
    @staticmethod
    def save_to_stl(domain: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid], stl_file: str):
        assert stl_file.endswith('.stl')
        if isinstance(domain, dolfinx.mesh.Mesh):
            domain = VisUtils.convert_to_grid(domain)
        pyvista.save_meshio(stl_file, domain)

    @staticmethod
    def save_remesh_stl_geo(stl_file: str, geo_file: str, minSize: float, maxSize: float, detection_angel: float):
        """
        Fail
        """

        assert stl_file.endswith('.stl')
        assert geo_file.endswith('.geo')

        with open(geo_file, 'w') as f:
            f.write(f"Merge '{stl_file}';\n")
            f.write(f"DefineConstant[\n"
                    f"  angle = {{{detection_angel}, Min 20, Max 120, Step 1, Name 'Surface Detection Angel'}},\n"
                    f"  forceParametrizablePatches = {{0, Choices{{0,1}}, Name 'Create Surfaces guaranteed "
                    f"to parametrizable'}},\n"
                    f"  includeBoundary = 1,\n"
                    f"  curveAngle = 180\n"
                    f"];\n")
            f.write(f"CreateGeometry;\n")
            f.write(f"Mesh.MeshSizeMax = {maxSize};\n")
            f.write(f"Mesh.MeshSizeMin = {minSize};\n")

    @staticmethod
    def save_remesh_msh_geo(stl_file: str, geo_file: str, minSize: float, maxSize: float):
        assert stl_file.endswith('.msh')
        assert geo_file.endswith('.geo')

        with open(geo_file, 'w') as f:
            f.write(f"Merge '{stl_file}';\n")
            f.write("CreateGeometry;\n")
            f.write(f"Mesh.MeshSizeMax = {maxSize};\n")
            f.write(f"Mesh.MeshSizeMin = {minSize};\n")

    @staticmethod
    def reconstruct_vertex_indices(orig_msh_file: str, domain: dolfinx.mesh.Mesh, check=False):
        """
        Create dof to vertex map.
        Needed in convert_domain_to_new_msh
        """
        gmsh.initialize()
        gmsh.model.add("Mesh from file")
        gmsh.merge(orig_msh_file)
        msh_xyzs = extract_geometry(gmsh.model)
        gmsh.finalize()

        tree = KDTree(msh_xyzs, metric='minkowski')
        geo_xyzs = domain.geometry.x
        neighbour_idxs = tree.query_radius(geo_xyzs, r=1e-8, return_distance=False)

        if check:
            for idx_np in neighbour_idxs:
                assert idx_np.shape[0] == 1
        vertex_indices = np.concatenate(neighbour_idxs, axis=-1)
        return vertex_indices

    @staticmethod
    def convert_domain_to_new_msh(
            orig_msh_file: str, new_msh_file: str,
            domain: dolfinx.mesh.Mesh, dim: int, vertex_indices: np.ndarray[np.int32]
    ):
        assert orig_msh_file.endswith('.msh')
        assert new_msh_file.endswith('.msh')

        vertex_xyzs = domain.geometry.x[:, :dim].copy()  # must copy here
        vertex_xyzs = vertex_xyzs[np.argsort(vertex_indices)]

        with open(orig_msh_file, "r", encoding="utf-8") as old_file:
            with open(new_msh_file, "w", encoding="utf-8") as new_file:
                node_section = False
                info_section = False
                subnode_counter = 0
                subwrite_counter = 0
                idcs = np.zeros(1, dtype=int)

                for line in old_file:
                    if line == "$EndNodes\n":
                        node_section = False

                    if not node_section:
                        new_file.write(line)
                    else:
                        split_line = line.split(" ")
                        if info_section:
                            new_file.write(line)
                            info_section = False
                        else:
                            if len(split_line) == 4:
                                num_subnodes = int(split_line[-1][:-1])
                                subnode_counter = 0
                                subwrite_counter = 0
                                idcs = np.zeros(num_subnodes, dtype=int)
                                new_file.write(line)

                            elif len(split_line) == 1:
                                idcs[subnode_counter] = int(split_line[0][:-1]) - 1
                                subnode_counter += 1
                                new_file.write(line)

                            elif len(split_line) == 3:
                                mod_line = ""
                                if dim == 2:
                                    mod_line = (
                                        f"{vertex_xyzs[idcs[subwrite_counter]][0]:.16f} "
                                        f"{vertex_xyzs[idcs[subwrite_counter]][1]:.16f} "
                                        f"0\n"
                                    )
                                elif dim == 3:
                                    mod_line = (
                                        f"{vertex_xyzs[idcs[subwrite_counter]][0]:.16f} "
                                        f"{vertex_xyzs[idcs[subwrite_counter]][1]:.16f} "
                                        f"{vertex_xyzs[idcs[subwrite_counter]][2]:.16f}\n"
                                    )
                                new_file.write(mod_line)
                                subwrite_counter += 1

                    if line == "$Nodes\n":
                        node_section = True
                        info_section = True

    @staticmethod
    def generate_msh_from_geo(geo_file: str, msh_file: str, dim: int):
        assert geo_file.endswith('.geo')
        assert msh_file.endswith('.msh')

        gmsh_cmd_list = ["gmsh", geo_file, f"-{int(dim):d}", "-o", msh_file]
        subprocess.run(gmsh_cmd_list, check=True, stdout=subprocess.DEVNULL)

    @staticmethod
    def remesh_run(
            domain: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid],
            vertex_indices: np.ndarray[np.int32],
            orig_msh_file: str,
            minSize: float, maxSize: float, dim: int,
            save_dir: str, model_name: str, tmp_dir: str = None
    ):
        with TemporaryDirectory() as tmp_dir:
            tmp_msh_file = os.path.join(tmp_dir, f"{model_name}_orig.msh")
            ReMesher.convert_domain_to_new_msh(orig_msh_file, tmp_msh_file, domain, dim, vertex_indices)

            tmp_geo_file = os.path.join(tmp_dir, f"{model_name}_tmp.geo")
            ReMesher.save_remesh_msh_geo(tmp_msh_file, tmp_geo_file, minSize=minSize, maxSize=maxSize)

            msh_file = os.path.join(save_dir, f"{model_name}.msh")
            ReMesher.generate_msh_from_geo(tmp_geo_file, msh_file, dim=dim)

            xdmf_file = os.path.join(save_dir, f"{model_name}.xdmf")
            MeshUtils.msh_to_XDMF(msh_file, output_file=xdmf_file, name=model_name, dim=dim)

        return msh_file, xdmf_file


class MeshQuality(object):
    @staticmethod
    def compute_cells_from_points(
            domain: dolfinx.mesh.Mesh,
            bb_tree: dolfinx.geometry.BoundingBoxTree = None,
            point_xyzs: np.ndarray = None,
            with_cell_counts=False
    ):
        if point_xyzs is None:
            point_xyzs = domain.geometry.x

        if bb_tree is None:
            bb_tree = geometry.bb_tree(domain, domain.topology.dim)

        cell_candidates: dolfinx.cpp.graph.AdjacencyList_int32 = geometry.compute_collisions_points(
            bb_tree, point_xyzs
        )
        colliding_cells: dolfinx.cpp.graph.AdjacencyList_int32 = geometry.compute_colliding_cells(
            domain, cell_candidates, point_xyzs
        )

        cells = []
        if with_cell_counts:
            cell_counts = []
            for i in range(point_xyzs.shape[0]):
                cells.append(colliding_cells.links(i))
                cell_counts.append(len(colliding_cells.links(i)))
            return cells, np.array(cell_counts)

        else:
            for i in range(point_xyzs.shape[0]):
                cells.append(colliding_cells.links(i))
            return cells

    @staticmethod
    def extract_connectivity(domain: dolfinx.mesh.Mesh):
        # grid = VisUtils.convert_to_grid(domain)
        # cells = []
        # for key in grid.cells_dict.keys():
        #     cells.append(grid.cells_dict[key])
        # connectivity = np.concatenate(cells, axis=0)

        connectivity = []
        adjacency_list: dolfinx.cpp.graph.AdjacencyList_int32 = domain.topology.connectivity(domain.topology.dim, 0)
        for i in range(adjacency_list.num_nodes):
            connectivity.append(adjacency_list.links(i))
        connectivity = np.array(connectivity)

        return connectivity

    @staticmethod
    def estimate_mesh_quality(
            obj: Union[dolfinx.mesh.Mesh, pyvista.UnstructuredGrid], quality_measure: str, tdim=None, entities=None
    ):
        """
        quality_measure:
            1. area
            2. radius_ratio
            3. skew            | Fail
            4. volume
            5. max_angle
            6. min_angle
            7. condition
        """

        # TODO so unstable why?

        if isinstance(obj, dolfinx.mesh.Mesh):
            grid = VisUtils.convert_to_grid(obj, tdim, entities)
        else:
            grid = obj
        qual = grid.compute_cell_quality(quality_measure=quality_measure, progress_bar=False)
        quality = qual['CellQuality']
        return quality


class MeshDeformationRunner(object):
    def __init__(
            self, domain: dolfinx.mesh.Mesh,
            volume_change: float = -1.0,
            quality_measures: Dict = {}
    ):
        """
        quality_measures:{
            measure_method["max_angle"]: {
                measure_type: max,
                tol_upper: float,
                tol_lower: float
            }
        }
        """
        self.domain = domain
        self.num_points = self.domain.geometry.x.shape[0]
        self.shape = self.domain.geometry.x.shape
        self.grid = VisUtils.convert_to_grid(domain)
        self.bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        self.tdim = self.domain.topology.dim
        self.fdim = self.tdim - 1

        self.connectivity = MeshQuality.extract_connectivity(self.domain)

        # 计算每个point都被哪些cell包含
        cell_counter: collections.Counter = collections.Counter(self.connectivity.flatten().tolist())
        self.occurrences = np.array([cell_counter[i] for i in range(self.num_points)])

        # ------ replace by pyvista
        # self.dg_function_space = dolfinx.fem.FunctionSpace(domain, element=("DG", 0))
        # self.cg_function_space = dolfinx.fem.VectorFunctionSpace(domain, element=("CG", 1))
        # self.transformation_container = dolfinx.fem.Function(self.cg_function_space)
        # self.ksp_option = {
        #     "ksp_type": "preonly",
        #     "pc_type": "jacobi",
        #     # "pc_jacobi_type": "diagonal",
        #     # "ksp_rtol": 1e-16,
        #     # "ksp_atol": 1e-20,
        #     # "ksp_max_it": 1000,
        # }
        # self.A_prior = ufl.TrialFunction(self.dg_function_space) * ufl.TestFunction(self.dg_function_space) * ufl.dx
        # self.l_prior = det(Identity(domain.geometry.dim) + grad(self.transformation_container)) * \
        #                ufl.TestFunction(self.dg_function_space) * ufl.dx
        # ------
        if self.tdim == 2:
            self.volume_measure_method = 'area'
        elif self.tdim == 3:
            self.volume_measure_method = 'volume'
        else:
            raise ValueError

        self.volume_change = volume_change
        self.validate_priori = False
        if self.volume_change > 0.0:
            self.validate_priori = True

        self.quality_measures = quality_measures
        self.validate_quality = False
        if len(self.quality_measures) > 0:
            self.validate_quality = True

    def detect_collision(self, domain: dolfinx.mesh.Mesh):
        cells, cell_counts = MeshQuality.compute_cells_from_points(domain, with_cell_counts=True)
        is_intersections = False
        if not np.all(cell_counts == self.occurrences):
            is_intersections = True
        return is_intersections

    def detect_valid_volume_change(self, displacement_np: np.ndarray, volume_change: float, **kwargs):
        """
        estimate the volume change of each cell
        """

        volume0 = MeshQuality.estimate_mesh_quality(self.grid, self.volume_measure_method)
        MeshUtils.move(self.domain, displacement_np)
        volume1 = MeshQuality.estimate_mesh_quality(self.grid, self.volume_measure_method)
        MeshUtils.move(self.domain, displacement_np * -1.0)
        uh = volume1 / volume0
        min_det, max_det = np.min(uh), np.max(uh)

        # ------ replace by pyvista
        # ------ compute the volume change percentage based on PETSC
        # displacement_petsc = PETScUtils.create_vec_from_x(displacement_np[:, :self.tdim].reshape(-1))
        # self.transformation_container.vector.aypx(0.0, displacement_petsc)
        #
        # uh = dolfinx.fem.Function(self.dg_function_space)  # volume1 / volume0
        # res_dict = LinearProblemSolver.solve_by_petsc_form(
        #     comm=self.domain.comm,
        #     uh=uh, a_form=self.A_prior, L_form=self.l_prior, bcs=[],
        #     ksp_option=self.ksp_option,
        #     **kwargs
        # )
        # if kwargs.get('with_debug', False):
        #     print(f"[DEBUG MeshDeformationRunner]: max_error:{res_dict['max_error']:.6f} "
        #           f"cost_time:{res_dict['cost_time']:.2f}")
        #
        # min_idx, min_det = uh.vector.min()
        # max_idx, max_det = uh.vector.max()
        # ------

        min_det = min_det - 1.0
        max_det = max_det - 1.0

        is_valid = (min_det >= -volume_change) and (max_det <= volume_change)
        return is_valid

    def detect_mesh_quality(self, domain):
        is_valid = True
        for measure_method in self.quality_measures.keys():
            meausre_info = self.quality_measures[measure_method]
            qualitys = MeshQuality.estimate_mesh_quality(domain, measure_method)
            if meausre_info['measure_type'] == 'min':
                quality = np.min(qualitys)
            elif meausre_info['measure_type'] == 'max':
                quality = np.max(qualitys)
            else:
                quality = np.mean(qualitys)

            is_valid = (quality >= meausre_info['tol_lower']) and (quality <= meausre_info['tol_upper'])
            if not is_valid:
                return is_valid

        return is_valid

    def compute_mesh_quality(self, domain):
        res = {}
        for measure_method in self.quality_measures.keys():
            qualitys = MeshQuality.estimate_mesh_quality(domain, measure_method)
            res[measure_method] = qualitys
        return res

    def move_mesh(self, displacement_np: np.ndarray, **kwargs):
        info = "Info:"

        if displacement_np.shape != self.shape:
            raise ValueError("[ERROR]: Shape UnCompatible")

        if self.validate_priori:
            is_valid = self.detect_valid_volume_change(displacement_np, self.volume_change, **kwargs)
            if not is_valid:
                info += "Validate Priori Fail"
                return False, info

        MeshUtils.move(self.domain, displacement_np)

        is_intersections = self.detect_collision(self.domain)
        if is_intersections:
            MeshUtils.move(self.domain, displacement_np * -1.0)  # revert mesh
            info += " |Mesh Intersect"
            return False, info

        if self.validate_quality:
            is_valid_quality = self.detect_mesh_quality(self.domain)
            if not is_valid_quality:
                MeshUtils.move(self.domain, displacement_np * -1.0)  # revert mesh
                info += " |Validate Quality Fail"
                return False, info

        info = "Success"
        return True, info

    def move_mesh_by_line_search(
            self,
            direction_np: np.ndarray,
            max_iter: int, init_stepSize=1.0, stepSize_lower=1e-4,
            detect_cost_valid_func: Callable = None,
            with_debug_info=False, **kwargs
    ):
        step_size = init_stepSize
        iteration = 0
        success_flag = False

        while True:
            if step_size < stepSize_lower:
                break

            displacement_np = direction_np * step_size
            valid_move_flag, info = self.move_mesh(displacement_np, **kwargs)
            # print(f"[DEBUG MeshDeformationRunner] Success_flag:{valid_move_flag} Info:{info}")

            if valid_move_flag:
                valid_cost_flag = True
                if detect_cost_valid_func is not None:
                    valid_cost_flag = detect_cost_valid_func()

                if valid_cost_flag:
                    success_flag = True
                    break

                else:
                    MeshUtils.move(self.domain, displacement_np * -1.0)  # revert mesh
                    info += '| Cost Detect Fail'

            step_size = step_size / 2.0
            iteration += 1

            if iteration > max_iter:
                break

        if with_debug_info:
            print(f"[DEBUG MeshDeformationRunner] Success_flag:{success_flag} Info:{info}")

        return success_flag, step_size

    def require_remesh(self):
        raise NotImplementedError
