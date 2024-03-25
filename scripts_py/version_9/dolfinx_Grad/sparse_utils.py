from petsc4py import PETSc
import numpy as np
from typing import Union, Literal
from scipy import sparse
from scipy.sparse import linalg

"""
vec.axpy(x, y) = vec + x * y
vec.aypx(x, y) = vec * x + y
"""


class PETScUtils(object):
    @staticmethod
    def create_mat(size=None):
        A_mat = PETSc.Mat().create()
        if size is not None:
            A_mat.setSizes(size)
        return A_mat

    @staticmethod
    def create_vec(size: int = None):
        if size is not None:
            vec = PETSc.Vec().createSeq(size)
        else:
            vec = PETSc.Vec()
        return vec

    @staticmethod
    def create_mat_from_x(x: Union[np.ndarray, sparse.csr_matrix], v=0.0):
        if isinstance(x, np.ndarray):
            assert x.ndim == 2
            row_idxs, col_idxs = np.where(x != v)
            values = x[row_idxs, col_idxs]

            A_mat = PETScUtils.create_mat(size=x.shape)
            for row_idx, col_idx, value in zip(row_idxs, col_idxs, values):
                A_mat.setValue(row_idx, col_idx, value)

        elif isinstance(x, sparse.csr_matrix):
            A_mat = PETSc.Mat().createAIJ(size=x.shape, csr=(x.indptr, x.indices, x.data))

        else:
            raise NotImplementedError

        A_mat.assemble()
        return A_mat

    @staticmethod
    def create_vec_from_x(x: np.ndarray):
        assert x.ndim == 1
        vec = PETSc.Vec().createSeq(x.shape[0])
        vec.array[:] = x
        return vec

    @staticmethod
    def get_array_from_mat(x: PETSc.Mat, row_idxs: list[int] = None, col_idxs: list[int] = None):
        size = x.getSize()
        if row_idxs is None:
            row_idxs = range(0, size[0])
        if col_idxs is None:
            col_idxs = range(0, size[1])
        return x.getValues(row_idxs, col_idxs)

    @staticmethod
    def convert_mat_to_scipy(x: PETSc.Mat):
        indptr, indices, values = x.getValuesCSR()
        a_mat = sparse.csr_matrix((values, indices, indptr), shape=x.getSize())
        return a_mat

    @staticmethod
    def get_array_from_vec(vec: PETSc.Vec, with_copy=False):
        if with_copy:
            return vec.array.copy()
        return vec.array

    @staticmethod
    def get_size(x: Union[PETSc.Mat, PETSc.Vec]):
        return x.getSize()

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


class ScipySparseUtils(object):
    @staticmethod
    def create_coo_mat(values, row_idxs, col_idxs, shape) -> sparse.coo_matrix:
        return sparse.coo_matrix((values, (row_idxs, col_idxs)), shape=shape)

    @staticmethod
    def create_csr_mat(values, indices, indptr, shape) -> sparse.csr_matrix:
        return sparse.csr_matrix((values, indices, indptr), shape=shape)

    @staticmethod
    def create_csc_mat(values, indices, indptr, shape) -> sparse.csc_matrix:
        return sparse.csc_matrix((values, indices, indptr), shape=shape)

    @staticmethod
    def convert_format(
            x: Union[sparse.coo_matrix, sparse.csr_matrix, sparse.csc_matrix],
            method: Literal["coo", "csr", "csc", "dense", "array"]
    ):
        if method == 'coo':
            return x.tocoo()
        elif method == 'csc':
            return x.tocsc()
        elif method == 'csr':
            return x.tocsr()
        elif method == 'dense':
            return x.todense()
        elif method == 'array':
            return x.toarray()
        else:
            raise NotImplementedError

    @staticmethod
    def linear_solve(
            A_mat: Union[sparse.coo_matrix, sparse.csr_matrix, sparse.csc_matrix],
            b_vec: np.ndarray
    ):
        return sparse.linalg.spsolve(A_mat, b_vec)
