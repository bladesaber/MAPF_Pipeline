import numpy as np
import matplotlib.pyplot as plt

from build import mapf_pipeline

cond_params = {
    'y': 9,
    'x': 9,
    'z': 1,
}

# detailPath0 = [
#     (0.0, 0.0, 0.0, 0.0),
#     (1.0, 1.0, 0.0, 0.0),
#     (2.0, 2.0, 0.0, 0.0),
#     (3.0, 3.0, 0.0, 0.0),
#     (3.0, 4.0, 0.0, 0.0),
#     (3.0, 5.0, 0.0, 0.0),
#     (3.0, 6.0, 0.0, 0.0),
#     (2.0, 7.0, 0.0, 0.0),
#     (1.0, 8.0, 0.0, 0.0),
#     (0.0, 9.0, 0.0, 0.0),
# ]
# detailPath1 = [
#     (6.0, 0.0, 0.0, 0.0),
#     (5.0, 1.0, 0.0, 0.0),
#     (4.0, 2.0, 0.0, 0.0),
#     (3.1, 3.0, 0.0, 0.0),
#     (3.1, 4.0, 0.0, 0.0),
#     (3.1, 5.0, 0.0, 0.0),
#     (3.1, 6.0, 0.0, 0.0),
#     (4.0, 7.0, 0.0, 0.0),
#     (5.0, 8.0, 0.0, 0.0),
#     (6.0, 9.0, 0.0, 0.0),
# ]

detailPath0 = [
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0, 0.0),
    (3.0, 1.0, 0.0, 0.0),
    (3.0, 2.0, 0.0, 0.0),
    (3.0, 3.0, 0.0, 0.0),
]

detailPath1 = [
    (6.0, 0.0, 0.0, 0.0),
    (5.0, 0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0, 0.0),
    (3.1, 0.0, 0.0, 0.0),
    (3.1, 1.0, 0.0, 0.0),
    (3.1, 2.0, 0.0, 0.0),
    (3.1, 3.0, 0.0, 0.0),
]

smoother = mapf_pipeline.RandomStep_Smoother(
    0.0, cond_params['x'] + 2.0, 
    0.0, cond_params['y'] + 2.0,
    0.0, cond_params['z'],
    stepReso = 0.05
)

### ------ 1st padding path
detailPath0 = smoother.paddingPath(
    detailPath0,
    startPadding=(-1.0, 0.0, 0.0), endPadding=(0.0, 1.0, 0.0),
    x_shift=1.0, y_shift=1.0, z_shift=0
)
detailPath1 = smoother.paddingPath(
    detailPath1,
    startPadding=(1.0, 0.0, 0.0), endPadding=(0.0, 1.0, 0.0),
    x_shift=1.0, y_shift=1.0, z_shift=0
)

# detailPath0_np = np.array(detailPath0)
# detailPath1_np = np.array(detailPath1)
# plt.scatter(detailPath0_np[:, 0], detailPath0_np[:, 1])
# plt.scatter(detailPath1_np[:, 0], detailPath1_np[:, 1])
# plt.show()
### --------------------------------

### ------ 2st debug
detailPath0 = smoother.detailSamplePath(
    detailPath0, stepLength=0.4
)
detailPath1 = smoother.detailSamplePath(
    detailPath1, stepLength=0.4
)

# detailPath0_np = np.array(detailPath0)
# detailPath1_np = np.array(detailPath1)
# plt.scatter(detailPath0_np[:, 0], detailPath0_np[:, 1])
# plt.scatter(detailPath1_np[:, 0], detailPath1_np[:, 1])
# plt.show()

### ------ 3st add path
smoother.addDetailPath(groupIdx=0, pathIdx=0, detailPath=detailPath0, radius=0.5)
smoother.addDetailPath(groupIdx=0, pathIdx=1, detailPath=detailPath1, radius=0.5)

# for groupIdx in smoother.groupMap.keys():
#     groupInfo = smoother.groupMap[groupIdx]
#     groupPath = groupInfo.path

#     print('GroupIdx:', groupIdx)
#     print('  StartIdxMap:',groupPath.startPathIdxMap)
#     print('  EndIdxMap:', groupPath.endPathIdxMap)
#     print('  pathIdxs_set:', groupPath.pathIdxs_set)
#     print('  NodeMap Size:', len(groupPath.nodeMap))

#     for pathIdx in groupPath.pathIdxs_set:
#         path = groupPath.extractPath(pathIdx)
#         path = np.array(groupPath.extractPath(pathIdx))
#         plt.scatter(path[:, 0], path[:, 1])
# plt.show()

## ------ 4st smooth path
smoother.wSmoothness = 1.0
smoother.wCurvature = 0.25
smoother.wObstacle = 0.0

oldPaths = {}
for groupIdx in smoother.groupMap.keys():
    groupInfo = smoother.groupMap[groupIdx]
    groupPath = groupInfo.path

    if groupIdx not in oldPaths:
        oldPaths[groupIdx] = {}

    for pathIdx in groupPath.pathIdxs_set:
        path = groupPath.extractPath(pathIdx)
        path = np.array(groupPath.extractPath(pathIdx))
        oldPaths[groupIdx][pathIdx] = path[:, :3]

newPaths = {}
for _ in range(50):
    smoother.smoothPath(updateTimes=5)

    ### --- vis
    for groupIdx in smoother.groupMap.keys():
        groupInfo = smoother.groupMap[groupIdx]
        groupPath = groupInfo.path

        if groupIdx not in newPaths:
            newPaths[groupIdx] = {}

        for pathIdx in groupPath.pathIdxs_set:
            path = groupPath.extractPath(pathIdx)
            path = np.array(groupPath.extractPath(pathIdx))
            newPaths[groupIdx][pathIdx] = path[:, :3]

    for groupKey in oldPaths.keys():
        for pathKey in oldPaths[groupKey].keys():
            plt.scatter(oldPaths[groupKey][pathKey][:, 0], oldPaths[groupKey][pathKey][:, 1], c='r')
            plt.scatter(newPaths[groupKey][pathKey][:, 0], newPaths[groupKey][pathKey][:, 1], c='b')
    plt.show()

    oldPaths = newPaths

