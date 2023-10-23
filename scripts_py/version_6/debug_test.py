import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib import patches

from build import mapf_pipeline

pathRes = np.load('/home/quan/Desktop/MAPF_Pipeline/scripts_py/version_6/app_dir/res.npy', allow_pickle=True).item()
pathRes = pathRes[4]

smoother = mapf_pipeline.SmootherXYZG2O()
smoother.initOptimizer()

xmin=0.0 - 2.5
ymin=0.0 - 2.5
zmin=0.0 - 2.5
xmax=63.0 + 2.5
ymax=63.0 + 2.5
zmax=2.0 + 2.5
smoother.setBoundary(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)

for path_xyzrl in pathRes:
    # print(path_xyzrl)

    path_xyzrl[:, 1] = 1.0
    path_xyzrl = np.concatenate([
        path_xyzrl[:, 0:1],
        path_xyzrl[:, 2:3],
        path_xyzrl[:, 1:2],
        path_xyzrl[:, 3:4]
    ], axis=1)

    path_xyzrl = pd.DataFrame(path_xyzrl, columns=['x', 'y', 'z', 'radius']).drop_duplicates(subset=['x', 'y', 'z']).values

    # print(path_xyzrl)
    # plt.plot(path_xyzrl[:, 0], path_xyzrl[:, 1])

    path_xyzr = []
    for xyzrl in path_xyzrl:
        path_xyzr.append((xyzrl[0], xyzrl[1], xyzrl[2], xyzrl[3]))
    smoother.add_Path(0, path_xyzr)
# plt.show()

# ### -----------------------------------------------------
# pathIdxs = smoother.groupMap[0].extractPath(
#     start_x=36, start_y=49, start_z=1,
#     end_x=17, end_y=0, end_z=1,
# )
# path_xy = []
# for idx in pathIdxs:
#     node = smoother.groupMap[0].pathNodeMap[idx]
#     path_xy.append([node.x, node.y])
# path_xy = np.array(path_xy)
# plt.plot(path_xy[:, 0], path_xy[:, 1], '-*')

# pathIdxs = smoother.groupMap[0].extractPath(
#     start_x=0, start_y=33, start_z=1,
#     end_x=0, end_y=52, end_z=1,
# )
# path_xy = []
# for idx in pathIdxs:
#     node = smoother.groupMap[0].pathNodeMap[idx]
#     path_xy.append([node.x, node.y])
# path_xy = np.array(path_xy)
# plt.plot(path_xy[:, 0], path_xy[:, 1], '-*')

# pathIdxs = smoother.groupMap[0].extractPath(
#     start_x=0, start_y=33, start_z=1,
#     end_x=30, end_y=62, end_z=1,
# )
# path_xy = []
# for idx in pathIdxs:
#     node = smoother.groupMap[0].pathNodeMap[idx]
#     path_xy.append([node.x, node.y])
# path_xy = np.array(path_xy)
# plt.plot(path_xy[:, 0], path_xy[:, 1], '-*')
# plt.show()
# ### -----------------------------------------------------

success = smoother.add_OptimizePath(
    groupIdx=0, pathIdx=0,
    start_x=36, start_y=49, start_z=1,
    end_x=17, end_y=0, end_z=1,
    startDire=(-1.57, 0), endDire=(-1.57, 0),
    startFlexRatio=0.0, endFlexRatio=0.2
)
print(success)
# success = smoother.add_OptimizePath(
#     groupIdx=0, pathIdx=1,
#     start_x=0, start_y=33, start_z=1,
#     end_x=0, end_y=52, end_z=1,
#     startDire=(0, 0), endDire=(3.14, 0),
#     startFlexRatio=0.0, endFlexRatio=0.75
# )
# print(success)
# success = smoother.add_OptimizePath(
#     groupIdx=0, pathIdx=2,
#     start_x=0, start_y=33, start_z=1,
#     end_x=30, end_y=62, end_z=1,
#     startDire=(0, 0), endDire=(1.57, 0),
#     startFlexRatio=0.0, endFlexRatio=0.2
# )
# print(success)

smoother.setMaxRadius(0, 0.5)

# groupPath = smoother.groupMap[0]
# for pathIdx in groupPath.graphPathMap.keys():
#     print(groupPath.graphPathMap[pathIdx])

### -------------------------------------
for outer_i in range(100):

    ### Step 2.1 Build Graph 
    success = smoother.build_graph(
        elasticBand_weight=0.1,
        kinematic_weight=10.0,
        obstacle_weight=0.0,
        pipeConflict_weight=0.0,
        boundary_weight=0.0
    )
    # print("build Graph:", success)

    if outer_i == 0:
        smoother.info()
    #     smoother.loss_info(
    #         elasticBand_weight=0.0,
    #         kinematic_weight=0.0,
    #         obstacle_weight=0.0,
    #         pipeConflict_weight=0.0,
    #         boundary_weight=1.0
    #     )
    #     print()

    ### Step 2.2 Optimize
    smoother.optimizeGraph(10, False)

    smoother.loss_info(
        elasticBand_weight=1.0,
        kinematic_weight=1.0,
        obstacle_weight=0.0,
        pipeConflict_weight=0.0,
        boundary_weight=0.0
    )
    print()

    ### Step 3.3 Update Vertex to Node
    smoother.update2groupVertex()

    # ### Step 2.4 Clear Graph
    smoother.clear_graph()

    ### Debug Vis
    figure, axes = plt.subplots()
    groupPath = smoother.groupMap[0]
    for pathIdx in groupPath.graphPathMap.keys():
        nodeIdxs_path = groupPath.graphPathMap[pathIdx]

        path_xyz = []
        for idx, nodeIdx in enumerate(nodeIdxs_path):
            node = groupPath.graphNodeMap[nodeIdx]
            path_xyz.append([node.x, node.y, node.z])

            if (idx == 0) or (idx == len(nodeIdxs_path)-1):
                axes.arrow(node.x, node.y, 2.0 * np.cos(node.alpha), 2.0 * np.sin(node.alpha), head_width=0.5)
            axes.add_patch( patches.Circle((node.x, node.y), radius=node.radius, fill=False) )

        path_xyz = np.array(path_xyz)
        # print(path_xyz)
        axes.plot(path_xyz[:, 0], path_xyz[:, 1], '-*', c='r')

    axes.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='g')
    # axes.plot(path_xyzr_2_np[:, 0], path_xyzr_2_np[:, 1], '-*', c='b')
    axes.axis("equal")
    plt.show()
### -------------------------------------


print('Finsih')
