import dolfinx
import ufl

import numpy as np
import pyvista
from ufl.sobolevspace import L2

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.dolfinx_utils import DolfinUtils
from scripts_py.dolfinx_adjoint.shape_fluid_utils.shape_fluid import FluidPipeBoundary
from scripts_py.eqn_solver import LinearProblemSolver


class WeightTopologyMerger(object):
    """
    Only used for Topology Optimization
    """
    
    ksp_option = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1
    }
    
    def __init__(self, mesh: MeshObj, ksp_option:dict = ksp_option):
        self.mesh: MeshObj = mesh
        self.gradient_dict: dict[str, np.ndarray] = {}

    def update_grad(self, key: str, grad: np.ndarray):
        assert grad.ndim == 1, f'[Error]: topology gradient should be 1D array, but got {grad.ndim}D array.'
        assert grad.shape[0] == self.mesh.num_points, f'[Error]: gradient shape:{grad.shape}, mesh num_points:{self.mesh.num_points}.'
        self.gradient_dict[key] = grad

    def merge_norm_gradient(self, weight: dict={}):
        grad_agg = 0.0
        for name, grad in self.gradient_dict.items():
            grad_agg += grad[:, :self.mesh.tdim] * weight.get(name, 1.0)
        grad_agg = grad_agg / float(len(self.gradient_dict))
        grad_norm = np.linalg.norm(grad_agg, axis=1, ord=2)
        grad_agg = grad_agg / np.max(np.abs(grad_norm))
        return grad_agg

