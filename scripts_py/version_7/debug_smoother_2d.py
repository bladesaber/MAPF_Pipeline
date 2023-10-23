import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib import patches

from build import mapf_pipeline

path_xyzr_0 = [
    (0.0, 0.0, 1.0, 0.5),
    (0.0, 1.0, 1.0, 0.5),
    (0.0, 2.0, 1.0, 0.5),
    (0.0, 3.0, 1.0, 0.5),
    (0.0, 4.0, 1.0, 0.5),
    (0.0, 5.0, 1.0, 0.5),
    (0.0, 6.0, 1.0, 0.5),
    (0.0, 7.0, 1.0, 0.5),
    (0.0, 8.0, 1.0, 0.5),
    (0.0, 9.0, 1.0, 0.5),
    (0.0, 10.0, 1.0, 0.5),
]
startDire0 = (0.0, 0.0)
endDire0 = (3.14, 0.0)

path_xyzr_1 = [
    (0.0, 0.0, 1.0, 0.5),
    (0.0, 1.0, 1.0, 0.5),
    (0.0, 2.0, 1.0, 0.5),
    (0.0, 3.0, 1.0, 0.5),
    (0.0, 4.0, 1.0, 0.5),
    (0.0, 5.0, 1.0, 0.5),
    (0.0, 6.0, 1.0, 0.5),
    (0.0, 7.0, 1.0, 0.5),
    (0.0, 8.0, 1.0, 0.5),
    (0.0, 9.0, 1.0, 0.5),
    (0.0, 10.0, 1.0, 0.5),
    (1.0, 10.0, 1.0, 0.5),
    (2.0, 10.0, 1.0, 0.5),
    (3.0, 10.0, 1.0, 0.5),
    (4.0, 10.0, 1.0, 0.5),
    (5.0, 10.0, 1.0, 0.5),
]
startDire1 = (0.0, 0.0)
endDire1 = (1.57, 0.0)

path_xyzr_2 = [
    (0.0, 0.0, 1.0, 0.5),
    (1.0, 0.0, 1.0, 0.5),
    (2.0, 0.0, 1.0, 0.5),
    (3.0, 0.0, 1.0, 0.5),
    (4.0, 0.0, 1.0, 0.5),
    (5.0, 0.0, 1.0, 0.5),
    (6.0, 0.0, 1.0, 0.5),
    (7.0, 0.0, 1.0, 0.5),
    (7.0, 1.0, 1.0, 0.5),
    (7.0, 2.0, 1.0, 0.5),
    (7.0, 3.0, 1.0, 0.5),
    (7.0, 4.0, 1.0, 0.5),
    (7.0, 5.0, 1.0, 0.5),
    (7.0, 6.0, 1.0, 0.5),
    (7.0, 7.0, 1.0, 0.5),
    (6.0, 7.0, 1.0, 0.5),
    (5.0, 7.0, 1.0, 0.5),
    (4.0, 7.0, 1.0, 0.5),
    (3.0, 7.0, 1.0, 0.5),
    (2.0, 7.0, 1.0, 0.5),
    (1.0, 7.0, 1.0, 0.5),
    (0.0, 7.0, 1.0, 0.5),
]
path_xyzr_2_np = np.array(path_xyzr_2)
startDire2 = (0.0, 0.0)
endDire2 = (3.14, 0.0)

path_xyzr_3 = [
    (0.0, 5.0, 1.0, 1.0),
    (1.0, 5.0, 1.0, 1.0),
    (2.0, 5.0, 1.0, 1.0),
    (3.0, 5.0, 1.0, 1.0),
    (4.0, 5.0, 1.0, 1.0),
    (5.0, 5.0, 1.0, 1.0),
    (6.0, 5.0, 1.0, 1.0),
    (7.0, 5.0, 1.0, 1.0),
    (8.0, 5.0, 1.0, 1.0),
    (9.0, 5.0, 1.0, 1.0),
    (10.0, 5.0, 1.0, 1.0),
]
startDire3 = (-1.57, 0.0)
endDire3 = (-1.57, 0.0)

# ### --------- debug extractPath
# groupPath = mapf_pipeline.GroupPath(0)
# groupPath.insertPath(path_xyzr_0)
# groupPath.insertPath(path_xyzr_1)

# pathIdxs = groupPath.extractPath(
#     start_x=0.0, start_y=0.0, start_z=0.0,
#     end_x=4.0, end_y=10.0, end_z=0.0
# )

# path_xyz = []
# for idx in pathIdxs:
#     node = groupPath.pathNodeMap[idx]
#     path_xyz.append([node.x, node.y, node.z])
# path_xyz = np.array(path_xyz)
# print(path_xyz)
# ### ----------------------------------------------------------

smoother = mapf_pipeline.SmootherXYZG2O()
smoother.initOptimizer()

xmin=-0.5
ymin=-0.5
zmin=0
xmax=10.5
ymax=10.5
zmax=2.0
smoother.setBoundary(xmin=xmin, ymin=ymin, zmin=zmin, xmax=xmax, ymax=ymax, zmax=zmax)

# smoother.add_Path(0, path_xyzr_2)
# success = smoother.add_OptimizePath(
#     groupIdx=0, 
#     pathIdx=0,
#     start_x=path_xyzr_2[0][0], 
#     start_y=path_xyzr_2[0][1], 
#     start_z=path_xyzr_2[0][2],
#     end_x=path_xyzr_2[-1][0], 
#     end_y=path_xyzr_2[-1][1], 
#     end_z=path_xyzr_2[-1][2],
#     startDire=startDire2,
#     endDire=endDire2
# )

# path_xyzr_0_tem = path_xyzr_0.copy()
# path_xyzr_0 = []
# for (x, y, z, r, l) in mapf_pipeline.sampleDetailPath(path_xyzr_0_tem, 0.5):
#     path_xyzr_0.append((x, y, z, r))

# path_xyzr_1_tem = path_xyzr_1.copy()
# path_xyzr_1 = []
# for (x, y, z, r, l) in mapf_pipeline.sampleDetailPath(path_xyzr_1_tem, 0.5):
#     path_xyzr_1.append((x, y, z, r))

# smoother.add_Path(0, path_xyzr_0)
# smoother.add_Path(0, path_xyzr_1)

path_xyzr_3_tem = path_xyzr_3.copy()
path_xyzr_3 = []
for (x, y, z, r, l) in mapf_pipeline.sampleDetailPath(path_xyzr_3_tem, 0.5):
    path_xyzr_3.append((x, y, z, r))
smoother.add_Path(0, path_xyzr_3)
print(len(path_xyzr_3))

success = smoother.add_OptimizePath(
    groupIdx=0, 
    pathIdx=0,
    start_x=path_xyzr_3[0][0], 
    start_y=path_xyzr_3[0][1], 
    start_z=path_xyzr_3[0][2],
    end_x=path_xyzr_3[-1][0], 
    end_y=path_xyzr_3[-1][1], 
    end_z=path_xyzr_3[-1][2],
    startDire=startDire3,
    endDire=endDire3,
    startFlexRatio=0.0, endFlexRatio=0.0
)
# success = smoother.add_OptimizePath(
#     groupIdx=0, 
#     pathIdx=1,
#     start_x=path_xyzr_1[0][0], 
#     start_y=path_xyzr_1[0][1], 
#     start_z=path_xyzr_1[0][2],
#     end_x=path_xyzr_1[-1][0], 
#     end_y=path_xyzr_1[-1][1], 
#     end_z=path_xyzr_1[-1][2],
#     startDire=startDire1,
#     endDire=endDire1
# )

smoother.setMaxRadius(0, 0.5)

# groupPath = smoother.groupMap[0]
# for pathIdx in groupPath.graphPathMap.keys():
#     print(groupPath.graphPathMap[pathIdx])

for outer_i in range(100):

    ### Step 2.1 Build Graph 
    success = smoother.build_graph(
        elasticBand_weight=1.0,
        kinematic_weight=1.0,
        obstacle_weight=0.0,
        pipeConflict_weight=0.0,
        boundary_weight=0.0
    )
    # print("build Graph:", success)

    if outer_i == 0:
        # smoother.info()
        smoother.loss_report(
            groupIdx=0, 
            pathIdx=0,
            elasticBand_weight=1.0,
            kinematic_weight=1.0,
            obstacle_weight=0.0,
            pipeConflict_weight=0.0,
            boundary_weight=0.0
        )
        print()

    ### Step 2.2 Optimize
    smoother.optimizeGraph(1, False)

    ### Step 3.3 Update Vertex to Node
    smoother.update2groupVertex()

    smoother.loss_report(
            groupIdx=0, 
            pathIdx=0,
            elasticBand_weight=1.0,
            kinematic_weight=1.0,
            obstacle_weight=0.0,
            pipeConflict_weight=0.0,
            boundary_weight=0.0
        )
    print()

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
                axes.arrow(node.x, node.y, 0.3 * np.cos(node.alpha), 0.3 * np.sin(node.alpha), head_width=0.1)
            axes.add_patch( patches.Circle((node.x, node.y), radius=node.radius, fill=False) )

        path_xyz = np.array(path_xyz)
        # print(path_xyz)
        axes.plot(path_xyz[:, 0], path_xyz[:, 1], '-*', c='r')

    axes.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='g')
    # axes.plot(path_xyzr_2_np[:, 0], path_xyzr_2_np[:, 1], '-*', c='b')
    axes.axis("equal")
    plt.show()

print('Finsih')