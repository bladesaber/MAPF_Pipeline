from petsc4py import PETSc
import numpy as np
from typing import Union, Literal
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


class PetscUtils(object):
    @staticmethod
    def create_mat(size: list[int] | None = None, assemble: bool = False) -> PETSc.Mat:
        """please assemble when use"""
        mat = PETSc.Mat().create()
        if size is not None:
            mat.setSizes(size)
        if assemble:
            mat.assemble()
        return mat

    @staticmethod
    def create_vec(size: int | None = None) -> PETSc.Vec:
        if size is not None:
            return PETSc.Vec().createSeq(size)
        else:
            return PETSc.Vec()

    @staticmethod
    def convert_x_to_mat(x: Union[np.ndarray, sparse.csr_matrix], v=0.0) -> PETSc.Mat:
        if isinstance(x, np.ndarray):
            assert x.ndim == 2
            row_idxs, col_idxs = np.where(x != v)
            values = x[row_idxs, col_idxs]
            mat = PetscUtils.create_mat(size=list(x.shape))
            for row_idx, col_idx, value in zip(row_idxs, col_idxs, values):
                mat.setValue(row_idx, col_idx, value)
        elif isinstance(x, sparse.csr_matrix):
            mat = PETSc.Mat().createAIJ(size=x.shape, csr=(x.indptr, x.indices, x.data))
        else:
            raise ValueError('[Error]: non-valid type.')
        mat.assemble()
        return mat

    @staticmethod
    def np_to_vec(x: np.ndarray) -> PETSc.Vec:
        assert x.ndim == 1
        vec = PETSc.Vec().createSeq(x.shape[0])
        vec.array[:] = x
        return vec

    @staticmethod
    def mat_to_np(mat: PETSc.Mat) -> np.ndarray:
        return PetscUtils.mat_to_scipy(mat).toarray()

    @staticmethod
    def mat_to_scipy(x: PETSc.Mat) -> sparse.csr_matrix:
        # TODO: i don't know why, values contain 0.0
        indptr, indices, values = x.getValuesCSR()
        mat = sparse.csr_matrix((values, indices, indptr), shape=x.getSize())
        return mat

    @staticmethod
    def vec_to_np(vec: PETSc.Vec, with_copy: bool = False) -> np.ndarray:
        if with_copy:
            return vec.array.copy()
        return vec.array

    @staticmethod
    def get_size(x: Union[PETSc.Mat, PETSc.Vec]) -> list[int]|int:
        return x.getSize()

    @staticmethod
    def save_data(x: Union[PETSc.Vec, PETSc.Mat], file: str):
        assert file.endswith('.dat')
        viewer = PETSc.Viewer().createBinary(file, 'w')
        viewer(x)

    @staticmethod
    def load_data(x: Union[PETSc.Vec, PETSc.Mat], file: str):
        '''please call create_mat(size=None) or create_vec(size=None) at first'''
        assert file.endswith('.dat')
        viewer = PETSc.Viewer().createBinary(file, 'r')
        x.load(viewer)

    @staticmethod
    def clean_zeros(x: PETSc.Mat):
        mat_scipy = PetscUtils.mat_to_scipy(x)
        mat_scipy.eliminate_zeros()
        return PetscUtils.convert_x_to_mat(mat_scipy)


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
            a_mat: Union[sparse.coo_matrix, sparse.csr_matrix, sparse.csc_matrix], b_vec: np.ndarray,
            method: Literal['spsolve', 'bicg', 'gmres', 'lsqr'] = 'bicg'
    ):
        if method == 'spsolve':
            return sparse_linalg.spsolve(a_mat, b_vec)
        elif method == 'bicg':
            (x, state_code) = sparse_linalg.bicg(a_mat, b_vec, maxiter=None)
            return x, state_code == 0
        elif method == 'gmres':
            (x, state_code) = sparse_linalg.gmres(a_mat, b_vec, maxiter=None)
            return x, state_code == 0
        elif method == 'lsqr':
            res = sparse_linalg.lsqr(a_mat, b_vec)
            x, state_code = res[0], res[1]
            return x, state_code == 1
        else:
            raise ValueError('[Error]: non-valid method.')


def debug():
    vec = PetscUtils.create_vec(size=5)
    vec.array[:] = 1.0
    vec.scale(2.0)
    print(vec.array)

if __name__ == '__main__':
    debug()
