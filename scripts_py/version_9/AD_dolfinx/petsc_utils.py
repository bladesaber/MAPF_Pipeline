from petsc4py import PETSc
import numpy as np
from typing import Union
from mpi4py import MPI
import time


class PETScUtils(object):
    Solver_methods = []

    @staticmethod
    def create_Mat(size=None):
        A_mat = PETSc.Mat().create()
        if size is not None:
            A_mat.setSizes(size)
        return A_mat

    @staticmethod
    def extract_CSR_idxs(mat: np.ndarray, v=0.0):
        assert mat.ndim == 2
        row_idxs, col_idxs = np.where(mat != v)
        return row_idxs, col_idxs, mat[row_idxs, col_idxs]

    @staticmethod
    def create_CSRMat(size: tuple, row_idxs: list[int], col_idxs: list[int], values):
        """
        size: eg. (3, 3)
        """
        A_mat = PETSc.Mat().create()
        A_mat.setSizes(size)
        for row_idx, col_idx, value in zip(row_idxs, col_idxs, values):
            A_mat.setValue(row_idx, col_idx, value)
        A_mat.assemble()
        return A_mat

    @staticmethod
    def getArray_from_Mat(row_idxs: list[int], col_idxs: list[int], mat: PETSc.Mat):
        return mat.getValues(row_idxs, col_idxs)

    @staticmethod
    def create_Vec_from_x(x: np.ndarray):
        assert x.ndim == 1
        vec = PETSc.Vec().createSeq(x.shape[0])
        vec.array[:] = x
        return vec

    @staticmethod
    def create_Vec(size: int = None):
        if size is not None:
            vec = PETSc.Vec().createSeq(size)
        else:
            vec = PETSc.Vec()
        return vec

    @staticmethod
    def getArray_from_Vec(vec: PETSc.Vec):
        return vec.array

    @staticmethod
    def get_Size(x: Union[PETSc.Mat, PETSc.Vec]):
        return x.getSize()

    @staticmethod
    def solve_linear_system_by_np(A_mat: np.ndarray, b_vec: np.ndarray):
        res = np.dot(np.linalg.inv(A_mat), b_vec)
        errors = np.dot(A_mat, res) - b_vec
        res_dict = {
            'res': res,
            'error_mean': np.mean(errors),
            'error_max': np.max(np.abs(errors))
        }
        return res_dict

    @staticmethod
    def solve_linear_system_by_PETSc(A_mat: PETSc.Mat, b_vec: PETSc.Vec, solver: PETSc.KSP):
        """
        solverType0: method to solve the problem
        solverType1: method to preprocessing the data
        """
        time0 = time.time()

        solver.setOperators(A_mat)
        res_vec = PETScUtils.create_Vec(PETScUtils.get_Size(b_vec))
        solver.solve(b_vec, res_vec)

        errors = (A_mat * res_vec).array - b_vec.array
        res_dict = {
            'res': res_vec,
            'error_mean': np.mean(errors),
            'error_max': np.max(np.abs(errors)),
            'cost_time': time.time() - time0
        }
        return res_dict

    Solver_setting = {
        "ksp_type": PETSc.KSP.Type.PREONLY,
        "pc_type": PETSc.PC.Type.LU
    }

    @staticmethod
    def create_PETSc_solver(comm=MPI.COMM_WORLD, solver_setting: dict = {}):
        """
        find more info from: https://petsc4py.readthedocs.io/en/stable/manual/ksp/
        """
        solver: PETSc.KSP = PETSc.KSP().create(comm)

        if solver_setting is None:
            solver_setting = PETScUtils.Solver_setting
        elif len(solver_setting) == 0:
            return solver

        # ------- easy use
        solver.setType(solver_setting["ksp_type"])
        solver.getPC().setType(solver_setting["pc_type"])

        # ------ detail use
        # solver.report = True
        # opts = PETSc.Options()
        # option_prefix = solver.getOptionsPrefix()
        #
        # for key in solver_setting.keys():
        #     opts[f"{option_prefix}{key}"] = solver_setting[key]
        #
        # opts[f"{option_prefix}ksp_type"] = solver_setting["ksp_type"]
        # opts[f"{option_prefix}pc_type"] = solver_setting["pc_type"]
        # opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

        solver.setFromOptions()

        return solver

    @staticmethod
    def set_solver_tolerances(solver, rtol=None, atol=None, divtol=None, mat_it=None):
        solver.setTolerances(rtol, atol, divtol, mat_it)

    @staticmethod
    def solver_func(solver, func: str):
        if func == 'destroy':
            solver.destroy()
        elif func == 'reset':
            solver.reset()
        elif func == 'getTolerances':
            return solver.getTolerances()
        else:
            raise NotImplementedError

    @staticmethod
    def get_solver_convergence_history(solver):
        residuals = solver.getConvergenceHistory()
        return residuals

    @staticmethod
    def save_data(x: Union[PETSc.Vec, PETSc.Mat], file: str):
        assert file.endswith('.dat')
        viewer = PETSc.Viewer().createBinary(file, 'w')
        viewer(x)

    @staticmethod
    def load_data(x: Union[PETSc.Vec, PETSc.Mat], file: str):
        """
        Please call create_Mat(size=None) or create_Vec(size=None) to create x first
        """
        assert file.endswith('.dat')
        viewer = PETSc.Viewer().createBinary(file, 'r')
        x.load(viewer)
