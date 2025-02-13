from petsc4py import PETSc
import numpy as np
from typing import Union, Literal
from scipy import sparse
from scipy.sparse import linalg

"""
vec.axpy(x, y) = vec + x * y
vec.aypx(x, y) = vec * x + y
"""


class PetscUtils(object):
    @staticmethod
    def create_mat(size=None):
        """please assemble when use"""
        mat = PETSc.Mat().create()
        if size is not None:
            mat.setSizes(size)
        return mat

    @staticmethod
    def create_vec(size: int = None):
        if size is not None:
            return PETSc.Vec().createSeq(size)
        else:
            return PETSc.Vec()

    @staticmethod
    def convert_x_to_mat(x: Union[np.ndarray, sparse.csr_matrix], v=0.0):
        if isinstance(x, np.ndarray):
            assert x.ndim == 2
            row_idxs, col_idxs = np.where(x != v)
            values = x[row_idxs, col_idxs]
            mat = PetscUtils.create_mat(size=x.shape)
            for row_idx, col_idx, value in zip(row_idxs, col_idxs, values):
                mat.setValue(row_idx, col_idx, value)
        elif isinstance(x, sparse.csr_matrix):
            mat = PETSc.Mat().createAIJ(size=x.shape, csr=(x.indptr, x.indices, x.data))
        else:
            raise ValueError('[Error]: Non-valid Type.')
        mat.assemble()
        return mat

    @staticmethod
    def create_np_to_vec(x: np.ndarray):
        assert x.ndim == 1
        vec = PETSc.Vec().createSeq(x.shape[0])
        vec.array[:] = x
        return vec

    @staticmethod
    def convert_mat_to_np(mat: PETSc.Mat):
        indptr, indices, values = mat.getValuesCSR()
        mat_np = sparse.csr_matrix((values, indices, indptr), shape=mat.getSize())
        return mat_np.toarray()

    @staticmethod
    def convert_mat_to_scipy(x: PETSc.Mat) -> sparse.csr_matrix:
        indptr, indices, values = x.getValuesCSR()
        mat = sparse.csr_matrix((values, indices, indptr), shape=x.getSize())
        return mat

    @staticmethod
    def convert_vec_to_np(vec: PETSc.Vec, with_copy=False):
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
        """Please call create_Mat(size=None) or create_Vec(size=None) to create x first"""
        assert file.endswith('.dat')
        viewer = PETSc.Viewer().createBinary(file, 'r')
        x.load(viewer)


class ScipySparseUtils(object):
    @staticmethod
    def create_coo_mat(values, row_idxs, col_idxs, shape) -> sparse.coo_matrix:
        return sparse.coo_matrix((values, (row_idxs, col_idxs)), shape=shape)

    @staticmethod
    def create_csr_mat(values, row_idxs, col_idxs, shape) -> sparse.csr_matrix:
        mat = sparse.coo_matrix((values, (row_idxs, col_idxs)), shape=shape)
        return mat.tocsr()

    @staticmethod
    def create_csc_mat(values, row_idxs, col_idxs, shape) -> sparse.csc_matrix:
        mat = sparse.coo_matrix((values, (row_idxs, col_idxs)), shape=shape)
        return mat.tocsc()

    @staticmethod
    def linear_solve(
            A_mat: Union[sparse.coo_matrix, sparse.csr_matrix, sparse.csc_matrix], b_vec: np.ndarray,
            method: Literal['spsolve', 'bicg', 'gmres', 'lsqr'] = 'bicg'
    ):
        if method == 'spsolve':
            return sparse.linalg.spsolve(A_mat, b_vec)
        elif method == 'bicg':
            (x, state_code) = sparse.linalg.bicg(A_mat, b_vec, maxiter=None)
            return x, state_code == 0
        elif method == 'gmres':
            (x, state_code) = sparse.linalg.gmres(A_mat, b_vec, maxiter=None)
            return x, state_code == 0
        elif method == 'lsqr':
            res = sparse.linalg.lsqr(A_mat, b_vec)
            x, state_code = res[0], res[1]
            return x, state_code == 1
        else:
            raise ValueError('[Error]: Non-valid Method.')


def debug():
    a: sparse.csr_matrix = sparse.random(4, 4, format='csr', density=0.1)
    a[2, 2] = 10.0
    print(a)


if __name__ == '__main__':
    debug()
