import numpy as np
from scipy.spatial import KDTree

class ConstrainTable(object):
    def __init__(self):
        self.CT = []

    def insert2CT(self, x, y, z, radius):
        self.CT.append([
            x, y, z, radius
        ])

    def update_numpy(self):
        CT_np = np.array(self.CT)
        self.CT_xyz = CT_np[:, :3]
        self.CT_radius = CT_np[:, 3:4]

    def coodrIsConstrained(self, x, y, z, radius):
        node_np = np.array([x, y, z])
        dist = np.linalg.norm(self.CT_xyz - node_np, ord=2, axis=1)
        isConflict = np.any(self.CT_radius - (dist + radius) < 0.0)
        return isConflict

    def lineIsConstrained(self, path, radius):
        kd_tree = KDTree(path)

        for xyz, conflict_radius in zip(self.CT_xyz, self.CT_radius):
            conflict_idxs = kd_tree.query_ball_point(x=xyz, r=radius+conflict_radius)
            if len(conflict_idxs) > 0:
                return True
        
        return False

if __name__ == '__main__':
    path = np.array([
        [1., 1., 1.],
        [1., 1., 2.],
        [1., 2., 3.]
    ])
    conflict = np.array([
        [0., 0., 0.],
        [5., 5., 6.],
        [1., 2., 8.]
    ])

    r = conflict[:, None, :] - path[:, :]
    r = r.reshape((-1, 3))
    print(r)
