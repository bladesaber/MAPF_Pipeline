import numpy as np
import math

import matplotlib.pyplot as plt

from build import mapf_pipeline

constrains = [
    (0., 7., 7., 0.5),
    (2., 0., 6., 0.5),
    (0., 4., 7., 0.5),
    (7., 3., 6., 0.5),
    (7., 5., 1., 0.5),
    (0., 0., 1., 0.5),
    (1., 4., 7., 0.5),
    (7., 0., 0., 0.5),
    (4., 7., 4., 0.5),
    (1., 0., 4., 0.5),
    (0.5, 6.25, 6.75, 0.5)
]

instance = mapf_pipeline.Instance(8, 8, 8)

model = mapf_pipeline.AngleAStar(0.5)
cbs = mapf_pipeline.CBS()

path = model.findPath(
    constraints = constrains,
    instance = instance,
    start_state = (1, 7, 6),
    goal_state = (0, 0, 5)
)
path_xyz = []
for loc in path:
    (x, y, z) = instance.getCoordinate(loc)
    path_xyz.append([x, y, z])
path_xyz = np.array(path_xyz)

detailPath = cbs.sampleDetailPath(path, instance, 0.5)
detailPath_np = np.array(detailPath)

# print(path_xyz)
# print(detailPath_np[:, :3])
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.grid(True)
# # ax.axis("equal")
# ax.scatter(detailPath_np[:, 0], detailPath_np[:, 1], detailPath_np[:, 2], c='r', s=30)
# ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], c='g')
# plt.show()

print('--------------------------\n')

constrain_np = np.array(constrains)
constrain_np = constrain_np[:, :3]
for xyz in detailPath_np[:, :3]:
    dist = np.linalg.norm(xyz - constrain_np, ord=2, axis=1)
    conflict_num = (dist < 1.0).sum()
    if conflict_num > 0:
        print('conflict: ', xyz, ' dist: ', dist.min())

constraint_table = mapf_pipeline.ConstraintTable()
for constrain in constrains:
    constraint_table.insert2CT(constrain)

for i in range(len(path) - 1):
    cur = path[i]
    next = path[i+1]

    (cur_x, cur_y, cur_z) = instance.getCoordinate(cur)
    (next_x, next_y, next_z) = instance.getCoordinate(next)

    print('cur pos: x:%f y:%f z:%f' % (cur_x, cur_y, cur_z))
    print('next pos: x:%f y:%f z:%f' % (next_x, next_y, next_z))

    isOnlight = constraint_table.islineOnSight(instance, cur, next, 0.5)
    print('conflict: ', isOnlight)

    # test1 = constraint_table.isConstrained(instance, cur, next, 0.5)
    # test2 = constraint_table.islineOnSight(instance, cur, next, 0.5)
    # print(test1, test2)

    break