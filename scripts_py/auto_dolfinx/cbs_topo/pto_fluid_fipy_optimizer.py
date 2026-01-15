import os

import numpy as np

from scripts_py.mesh_utils import MeshObj, MeshUtils
from scripts_py.simulator.fipy_simulator import FipyStokeSimulator, FluidBoundary


"""
如果不使用基于alpha的仿真方法,那每次仿真前都必须re-mesh,非常低效率
"""


class PtoFluidFipyOptimizer(object):
    def __init__(self,
        name: str,
        mesh: MeshObj,
        msh_file: str,
        viscosity: float,
    ):
        self.name = name