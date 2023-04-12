import numpy as np
import math

from scripts.continueAstar.dubinsCruve import compute_dubinsPath3D

from build import mapf_pipeline

class Instance(object):
    def __init__(
        self, 
        num_of_rows, num_of_cols, num_of_z, radius = 1.0,
        cell_size = 1.0, horizon_range = 2.0 * math.pi, vertical_range = math.pi, 
        horizon_discrete_num = 36, vertical_discrete_num = 18
    ):
        self.num_of_rows = num_of_rows
        self.num_of_cols = num_of_cols
        self.num_of_z = num_of_z
        self.radius = radius

        self.cell_size = cell_size
        self.horizon_range = horizon_range
        self.vertical_range = vertical_range
        self.horizon_discrete_num = horizon_discrete_num
        self.vertical_discrete_num = vertical_discrete_num
        self.horizon_delRadian = self.horizon_range / self.horizon_discrete_num
        self.vertical_delRadian = self.vertical_range / self.vertical_discrete_num

        self.step_length = math.sqrt(math.pow(self.cell_size, 2) * 3.0)

        self.alpha_deltas = [
            np.deg2rad(-30.0), np.deg2rad(0.0), np.deg2rad(30.0)
        ]
        self.beta_deltas = [
            np.deg2rad(-30.0), np.deg2rad(0.0), np.deg2rad(30.0)
        ]

    def getRoundCoordinate(self,  pos):
        '''
        pos: (x, y, z, alpha, beta)
        '''
        return (
            int(round(pos[0] / self.cell_size, 0)),
            int(round(pos[1] / self.cell_size, 0)),
            int(round(pos[2] / self.cell_size, 0)),
            int(round(pos[3] / self.horizon_delRadian, 0)),
            int(round(pos[4] / self.vertical_delRadian, 0)),
        )

    def getEulerDistance(self, pos0, pos1):
        return np.linalg.norm(pos0[:3]-pos1[:3], ord=2)

    def getManhattanDistance(self, pos0, pos1):
        return np.linalg.norm(pos0[:3]-pos1[:3], ord=1)
    
    def getDubinsDistance(self, pos0, pos1):
        solutions, cost, invert_yz = compute_dubinsPath3D(pos0, pos1, self.radius)
        return cost, (solutions, invert_yz)

    def isValidGrid(self, pos):
        if pos[0] > self.num_of_cols:
            return False
        elif pos[0] < 0:
            return False
        elif pos[1] > self.num_of_rows:
            return False
        elif pos[1] < 0:
            return False
        elif pos[2] > self.num_of_z:
            return False
        elif pos[2] < 0:
            return False
        
        return True

    def getNeighbors(self, pos):
        x, y, z, alpha, beta = pos

        neighborTable = []
        for alpha_delta in self.alpha_deltas:
            for beta_delta in self.beta_deltas:
                new_alpha = alpha + alpha_delta
                new_beta = beta + beta_delta
                
                vec = mapf_pipeline.polar3D_to_vec3D(new_alpha, new_alpha, self.step_length)
                new_x = x + vec[0]
                new_y = y + vec[1]
                new_z = z + vec[2]

                neighborTable.append([new_x, new_y, new_z, new_alpha, new_beta])

        return neighborTable
    
    def compute_pathDist(self, path):
        dists = np.linalg.norm(path[:, 1:] - path[:, :-1], ord=2, axis=1)
        total_dist = np.sum(dists)
        return total_dist
