import numpy as np
import matplotlib.pyplot as plt

from build import mapf_pipeline

from scripts_py import vis_utils

def extractPath(pathXYZ):
    detailPath = []
    for i in pathXYZ:
        x = i.getX()
        y = i.getY()
        z = i.getZ()
        detailPath.append([x, y, z])
    detailPath = np.array(detailPath)
    return detailPath

### -------------------------------------------------------

cond_param = {
    'x': 10,
    'y': 10,
    'z': 10
}

instance = mapf_pipeline.Instance(
    cond_param['x'], cond_param['y'], cond_param['z']
)

detailPath0 = [
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 1.0),
    (2.0, 0.0, 0.0, 2.0),
    (3.0, 0.0, 0.0, 3.0),
    (4.0, 0.0, 0.0, 4.0),
    (4.0, 1.0, 0.0, 5.0),
    (4.0, 2.0, 0.0, 6.0),
    (4.0, 3.0, 0.0, 7.0),
    (4.0, 4.0, 0.0, 8.0),
]
detailPath1 = [
    (0.0, 1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0, 1.0),
    (2.0, 1.0, 0.0, 2.0),
    (3.0, 1.0, 0.0, 3.0),
    (3.0, 2.0, 0.0, 4.0),
    (3.0, 3.0, 0.0, 5.0),
    (3.0, 4.0, 0.0, 6.0),
]

# detailPath0_np = np.array(detailPath0)
# detailPath1_np = np.array(detailPath1)
# plt.scatter(detailPath0_np[:, 0], detailPath0_np[:, 1], c='r')
# plt.scatter(detailPath1_np[:, 0], detailPath1_np[:, 1], c='b')
# plt.show()

smoother = mapf_pipeline.RandomStep_Smoother(0.05)
smoother.addAgentObj(agentIdx=0, radius=0.1, detailPath=detailPath0)
smoother.addAgentObj(agentIdx=1, radius=0.1, detailPath=detailPath1)

smoother.wSmoothness = 0.25
smoother.wCurvature = 1.0
smoother.wObstacle = 3.0

oldPath0, oldPath1 = np.array(detailPath0), np.array(detailPath1)
for _ in range(100):
    smoother.smoothPath(instance=instance, updateTimes=1)

    newPath0 = extractPath(smoother.agentMap[0].pathXYZ)
    newPath1 = extractPath(smoother.agentMap[1].pathXYZ)

    plt.scatter(oldPath0[:, 0], oldPath0[:, 1], c='r', s=10.0)
    plt.scatter(newPath0[:, 0], newPath0[:, 1], c='g')

    plt.scatter(oldPath1[:, 0], oldPath1[:, 1], c='b', s=10.0)
    plt.scatter(newPath1[:, 0], newPath1[:, 1], c='g')

    oldPath0 = newPath0
    oldPath1 = newPath1

    print('---------------\n')

    plt.show()

# print("finish")