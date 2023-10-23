import numpy as np
import matplotlib.pyplot as plt

from build import mapf_pipeline

detailPath0 = [
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 1.0, 0.0, 0.0),
    (2.0, 2.0, 0.0, 0.0),
    (3.0, 3.0, 0.0, 0.0),
    (3.0, 4.0, 0.0, 0.0),
    (3.0, 5.0, 0.0, 0.0),
    (3.0, 6.0, 0.0, 0.0),
    (2.0, 7.0, 0.0, 0.0),
    (1.0, 8.0, 0.0, 0.0),
    (0.0, 9.0, 0.0, 0.0),
]

detailPath1 = [
    (6.0, 0.0, 0.0, 0.0),
    (5.0, 1.0, 0.0, 0.0),
    (4.0, 2.0, 0.0, 0.0),
    (3.1, 3.0, 0.0, 0.0),
    (3.1, 4.0, 0.0, 0.0),
    (3.1, 5.0, 0.0, 0.0),
    (3.1, 6.0, 0.0, 0.0),
    (4.0, 7.0, 0.0, 0.0),
    (5.0, 8.0, 0.0, 0.0),
    (6.0, 9.0, 0.0, 0.0),
]

# detailPath0_np = np.array(detailPath0)
# detailPath1_np = np.array(detailPath1)
# plt.plot(detailPath0_np[:, 0], detailPath0_np[:, 1])
# plt.plot(detailPath1_np[:, 0], detailPath1_np[:, 1])
# plt.show()

groupPath = mapf_pipeline.GroupPath(0)
groupPath.insertPath(0, detailPath0, 0.5)
groupPath.insertPath(1, detailPath1, 0.5)

# print(groupPath.startPathIdxMap)
# print(groupPath.endPathIdxMap)
# for key in groupPath.nodeMap.keys():
#     print('key: ', key)
#     groupNode = groupPath.nodeMap[key]
#     print('info: idx:%d x:%f y:%f z:%f radius:%f' % (groupNode.nodeIdx, groupNode.x, groupNode.y, groupNode.z, groupNode.radius))

path0 = groupPath.extractPath(0)
path1 = groupPath.extractPath(1)

path0_np = np.array(path0)
path1_np = np.array(path1)
plt.plot(path0_np[:, 0], path0_np[:, 1])
plt.plot(path1_np[:, 0], path1_np[:, 1])
plt.show()

print('finish')
