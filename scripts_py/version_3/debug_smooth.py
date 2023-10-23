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

def show(old_path, new_path):
    # ax = vis_utils.create_Graph3D(xmax=cond_param['x'], ymax=cond_param['y'], zmax=cond_param['z'])
    # ax.scatter(detailPath_np[:, 0], detailPath_np[:, 1], detailPath_np[:, 2], c='r', s=6.0)
    # ax.scatter(new_detailPath_np[:, 0], new_detailPath_np[:, 1], new_detailPath_np[:, 2], c='g')

    plt.scatter(old_path[:, 0], old_path[:, 1], c='r', s=10.0)
    plt.scatter(new_path[:, 0], new_path[:, 1], c='g')

    plt.show()

### -------------------------------------------------------

cond_param = {
    'x': 13,
    'y': 13,
    'z': 13
}

instance = mapf_pipeline.Instance(
    cond_param['x'], cond_param['y'], cond_param['z']
)

# detailPath = [
#     (0.0, 0.0, 0.0, 0.0),
#     (1.0, 0.0, 0.0, 1.0),
#     (2.0, 0.0, 0.0, 2.0),
#     (2.0, 1.0, 0.0, 3.0),
#     (2.0, 2.0, 0.0, 4.0),
# ]

# detailPath = [
#     (0.0, 3.0, 0.0, 0.0),
#     (1.0, 3.0, 0.0, 1.0),
#     (2.0, 3.0, 0.0, 2.0),
#     (3.0, 3.0, 0.0, 3.0),
#     (4.0, 3.0, 0.0, 4.0),
#     (5.0, 3.0, 0.0, 5.0),
#     (5.0, 4.0, 0.0, 6.0),
#     (5.0, 5.0, 0.0, 7.0),
#     (5.0, 6.0, 0.0, 8.0),
#     (5.0, 7.0, 0.0, 9.0),
#     (5.0, 8.0, 0.0, 10.0),
#     (6.0, 8.0, 0.0, 11.0),
#     (7.0, 8.0, 0.0, 12.0),
#     (8.0, 8.0, 0.0, 13.0),
#     (9.0, 8.0, 0.0, 14.0),
#     (10.0, 8.0, 0.0, 15.0),
#     (11.0, 8.0, 0.0, 16.0),
# ]

detailPath = [
    (0.0, 3.0, 0.0, 0.0),
    (0.25, 3.0, 0.0, 0.0),
    # (1.0, 3.0, 0.0, 1.0),
    # (10.0, 8.0, 0.0, 15.0),
    (10.75, 8.0, 0.0, 0.0),
    (11.0, 8.0, 0.0, 16.0),
]

smoother = mapf_pipeline.RandomStep_Smoother(0.02)

detailPath = smoother.detailSamplePath(detailPath, 0.25)
# plt.scatter(np.array(detailPath)[:, 0], np.array(detailPath)[:, 1])
# plt.axis('equal')
# plt.show()

smoother.addAgentObj(agentIdx=0, radius=0.5, detailPath=detailPath)

smoother.wSmoothness = 1.0
smoother.wCurvature = 0.0
smoother.wObstacle = 0.0

oldPath = np.array(detailPath)
for _ in range(100):
    smoother.smoothPath(instance=instance, updateTimes=10)

    newPath = extractPath(smoother.agentMap[0].pathXYZ)
    show(oldPath, newPath)
    oldPath = newPath

    print('---------------\n')

# print("finish")