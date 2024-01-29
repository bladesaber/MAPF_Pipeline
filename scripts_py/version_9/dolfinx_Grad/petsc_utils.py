from petsc4py import PETSc
import numpy as np
from typing import Union
from mpi4py import MPI
import time
import pandas as pd
import os

"""
vec.axpy(x, y) = vec + x * y
vec.aypx(x, y) = vec * x + y
"""


class PETScUtils(object):
    # ----------------------- PETSc Matrix Operation
    @staticmethod
    def create_mat(size=None):
        A_mat = PETSc.Mat().create()
        if size is not None:
            A_mat.setSizes(size)
        return A_mat

    @staticmethod
    def create_mat_from_x(x: np.ndarray, v=0.0):
        assert x.ndim == 2
        row_idxs, col_idxs = np.where(x != v)
        values = x[row_idxs, col_idxs]

        A_mat = PETSc.Mat().create()
        A_mat.setSizes(tuple(x.shape))
        for row_idx, col_idx, value in zip(row_idxs, col_idxs, values):
            A_mat.setValue(row_idx, col_idx, value)
        A_mat.assemble()

    @staticmethod
    def get_array_from_mat(x: PETSc.Mat, row_idxs: list[int] = None, col_idxs: list[int] = None):
        size = x.getSize()
        if row_idxs is None:
            row_idxs = range(size[0])
        if col_idxs is None:
            col_idxs = range(size[1])
        return x.getValues(row_idxs, col_idxs)

    @staticmethod
    def create_vec_from_x(x: np.ndarray):
        assert x.ndim == 1
        vec = PETSc.Vec().createSeq(x.shape[0])
        vec.array[:] = x
        return vec

    @staticmethod
    def create_vec(size: int = None):
        if size is not None:
            vec = PETSc.Vec().createSeq(size)
        else:
            vec = PETSc.Vec()
        return vec

    @staticmethod
    def get_array_from_vec(vec: PETSc.Vec):
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
