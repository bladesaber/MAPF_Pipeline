import numpy as np
import matplotlib.pyplot as plt

from build import mapf_pipeline

path0 = [
    (0.0, 0.0, 0.0, 0.5),
    (1.0, 0.0, 0.0, 0.5),
    (2.0, 0.0, 0.0, 0.5),
    (3.0, 0.0, 0.0, 0.5),
    (3.0, 1.0, 0.0, 0.5),
    (3.0, 2.0, 0.0, 0.5),
    (3.0, 3.0, 0.0, 0.5),
    (3.0, 4.0, 0.0, 0.5),
]
startDire0 = (0.0, 0.0)
endDire0 = (np.deg2rad(90.0), 0.0)

groupPath = mapf_pipeline.GroupPath(0)
groupPath.insertPath(pathIdx=0, path_xyzr=path0, startDire=startDire0, endDire=endDire0)

# print(groupPath.pathIdxs_set)
# for key in groupPath.nodeMap.keys():
#     node = groupPath.nodeMap[key]
#     print('GroupIdx:%d nodeIdx:%d x:%f y:%f z:%f radius:%f alpha:%f theta:%f' % (
#         node.nodeIdx, node.groupIdx, node.x, node.y, node.z, node.radius, node.alpha, node.theta
#     ))

path1 = [
    (8.0, 0.0, 0.0, 0.5),
    (7.0, 0.0, 0.0, 0.5),
    (6.0, 0.0, 0.0, 0.5),
    (5.0, 0.0, 0.0, 0.5),
    (4.0, 0.0, 0.0, 0.5),
    (3.1, 0.0, 0.0, 0.5),
    (3.1, 1.0, 0.0, 0.5),
    (3.1, 2.0, 0.0, 0.5),
    (3.5, 3.0, 0.0, 0.5),
    (4.0, 4.0, 0.0, 0.5),
    (5.0, 4.0, 0.0, 0.5),
]
startDire1 = (np.deg2rad(180.0), 0.0)
endDire1 = (np.deg2rad(0.0), 0.0)
groupPath.insertPath(pathIdx=1, path_xyzr=path1, startDire=startDire1, endDire=endDire1)

groupPath.updatePathDirection()

for pathIdx in groupPath.pathIdxs_set:
    nodeIdxs_path = groupPath.extractPath(pathIdx)

    path_xy = []
    alphas = []
    for nodeIdx in nodeIdxs_path:
        node = groupPath.nodeMap[nodeIdx]
        path_xy.append([node.x, node.y])
        alphas.append(node.alpha)

    path_xy = np.array(path_xy)

    # plt.plot(path_xy[:, 0], path_xy[:, 1], '-*')
    plt.scatter(path_xy[:, 0], path_xy[:, 1], s=10.0)
    for (x, y), alpha in zip(path_xy, alphas):
        plt.arrow(x, y, 0.3 * np.cos(alpha), 0.3 * np.sin(alpha))

plt.show()
