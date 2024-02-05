import pyvista
import gmsh
import os
import dolfinx
from typing import Union, List
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from dolfinx.io.gmshio import extract_geometry
from sklearn.neighbors import KDTree
from dolfinx import geometry
import ufl

from .dolfinx_utils import MeshUtils
from .vis_mesh_utils import VisUtils
from .equation_solver import LinearProblemSolver


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
        gmsh.initialize()
        gmsh.model.add("Mesh from file")
        gmsh.merge(orig_msh_file)
        msh_xyzs = extract_geometry(gmsh.model)
        geo_xyzs = domain.geometry.x
        gmsh.finalize()

        tree = KDTree(msh_xyzs, metric='minkowski')
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
    def compute_collide_counts(domain: dolfinx.mesh.Mesh):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, domain.geometry.x)

        collide_counts = []
        for i in range(domain.geometry.x.shape[0]):
            link_cells: np.ndarray[int] = cell_candidates.links(i)
            collide_counts.append(link_cells.size)

        return np.array(collide_counts)

    @staticmethod
    def detect_collision(domain: dolfinx.mesh.Mesh, orig_collide_counts: np.ndarray, with_intersections=False):
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)

        # compute_collisions_points return a list of cells whose bounding box collide for each input points
        cell_candidates = geometry.compute_collisions_points(bb_tree, domain.geometry.x)

        # compute_colliding_cells measure the exact distance between point and cell
        # colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, domain.geometry.x)

        collide_counts = []
        for i in range(domain.geometry.x.shape[0]):
            link_cells: np.ndarray[int] = cell_candidates.links(i)
            collide_counts.append(link_cells.size)

        collide_counts = np.array(collide_counts)
        intersections_bool = collide_counts == orig_collide_counts
        is_intersection = not np.all(intersections_bool)

        if with_intersections:
            return intersections_bool, is_intersection
        else:
            return is_intersection


class MeshDeformation(object):
    @staticmethod
    def estimate_cell_topology_volume_change(
            domain: dolfinx.mesh.Mesh, transformation: dolfinx.fem.Function, **kwargs
    ):
        """
        TODO: Math Base Still need to explored
        estimate the volume change of each cell
        """
        dg_function_space = dolfinx.fem.FunctionSpace(domain, element=("DG", 0))

        a_form = ufl.TrialFunction(dg_function_space) * ufl.TestFunction(dg_function_space) * ufl.dx
        l_form = ufl.det(ufl.Identity(domain.geometry.dim) + ufl.grad(transformation)) * \
                 ufl.TestFunction(dg_function_space) * ufl.dx

        uh = dolfinx.fem.Function(transformation.function_space)
        res_dict = LinearProblemSolver.solve_by_petsc_form(
            comm=domain.comm,
            uh=uh,
            a_form=a_form,
            L_form=l_form,
            bcs=[],
            ksp_option={
                "ksp_type": "preonly",
                "pc_type": "jacobi",
                # "pc_jacobi_type": "diagonal",
            },
            **kwargs
        )
        return uh

    @staticmethod
    def estimate_mesh_quality(domain: dolfinx.mesh.Mesh, quality_measure: str, tdim=None, entities=None):
        """
        quality_measure:
            1. area
            2. radius_ratio
            3. skew
            4. volume
            5. max_angle
            6. min_angle
            7. condition
        """
        grid = VisUtils.convert_to_grid(domain, tdim, entities)
        qual = grid.compute_cell_quality(quality_measure=quality_measure, progress_bar=False)
        quality = qual['CellQuality']
        return quality
