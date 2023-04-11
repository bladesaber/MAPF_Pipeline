import numpy as np
import math

from scripts.continueAstar.instance import Instance

from build import mapf_pipeline

class HybridAstarWrap(object):
    def __init__(self, instance:Instance, constrainTable:np.array):
        self.instance = instance
        self.constrainTable = constrainTable

        self.allNodes = {}

    def findPath(self):
        pass

    def release():
        pass

    
