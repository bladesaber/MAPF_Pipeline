import numpy as np
import os

from scripts.version_3.cbs import CBSSolver
from build import mapf_pipeline
from scripts.visulizer import VisulizerVista
from scripts import vis_utils
import matplotlib.pyplot as plt

cond_params = {
    'y': 8,
    'x': 8,
    'z': 8,
    'num_of_agents': 8,
    'radius_choices': [
        0.5, 
        # 0.85, 
        # 1.25
    ],

    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_3/map',
    "load": True,

    'save_dir': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_3/',
}

def extractPath(pathXYZ):
    detailPath = []
    for i in pathXYZ:
        x = i.getX()
        y = i.getY()
        z = i.getZ()
        detailPath.append([x, y, z])
    detailPath = np.array(detailPath)
    return detailPath

class MapGen(object):
    def __init__(self):
        self.agentInfos = {}

    def create_agentPos(self, num):
        dires = np.random.choice(['x', 'y', 'z'], size=num)

        records = []
        for agentIdx, dire in enumerate(dires):
            while True:
                (startPos, startDire), (endPos, endDire) = self.create_Pos(dire)
                radius = np.random.choice(cond_params['radius_choices'], size=1)

                valid = self.checkValid(startPos, endPos, radius)
                if valid:
                    break

            records.append(startPos)
            records.append(endPos)
            self.agentInfos[agentIdx] = {
                'agentIdx': agentIdx,
                'startPos': startPos,
                'endPos': endPos,
                'radius': radius,
                'startDire': startDire,
                'endDire': endDire
            }

    def create_Pos(self, dire):
        if dire == 'x':
            pos1 = (
                0, 
                np.random.randint(0, cond_params['y']),
                np.random.randint(0, cond_params['z'])
            )
            pos2 = (
                cond_params['x'] - 1, 
                np.random.randint(0, cond_params['y']),
                np.random.randint(0, cond_params['z'])
            )
            startDire = (-1.0, 0.0, 0.0)
            endDire = (1.0, 0.0, 0.0)
        
        elif dire == 'y':
            pos1 = (
                np.random.randint(0, cond_params['x']),
                0,
                np.random.randint(0, cond_params['z'])
            )
            pos2 = (
                np.random.randint(0, cond_params['x']),
                cond_params['y'] - 1, 
                np.random.randint(0, cond_params['z'])
            )
            startDire = (0.0, -1.0, 0.0)
            endDire = (0.0, 1.0, 0.0)

        elif dire == 'z':
            pos1 = (
                np.random.randint(0, cond_params['x']),
                np.random.randint(0, cond_params['y']),
                0
            )
            pos2 = (
                np.random.randint(0, cond_params['x']),
                np.random.randint(0, cond_params['y']),
                cond_params['z'] - 1 
            )
            startDire = (0.0, 0.0, -1.0)
            endDire = (0.0, 0.0, 1.0)
        
        if np.random.uniform(0.0, 1.0) > 0.5:
            return (pos1, startDire), (pos2, endDire)
        else:
            return (pos2, endDire), (pos1, startDire)

    def checkValid(self, startPos, endPos, radius):
        for agentIdx in self.agentInfos.keys():
            agentInfo = self.agentInfos[agentIdx]

            ref_Pos = np.array([
                agentInfo['startPos'],
                agentInfo['endPos'],
            ])

            if startPos == agentInfo['startPos']:
                return False
            
            if endPos == agentInfo['endPos']:
                return False

            dist = (np.linalg.norm(np.array(startPos) - ref_Pos, ord=2, axis=1)).min()
            if dist < agentInfo['radius'] + radius:
                return False
            
            dist = (np.linalg.norm(np.array(endPos) - ref_Pos, ord=2, axis=1)).min()
            if dist < agentInfo['radius'] + radius:
                return False

        return True

    def save(self):
        np.save(cond_params['save_path'], self.agentInfos)

    def load(self):
        self.agentInfos = np.load(os.path.join(cond_params['save_path']+'.npy'), allow_pickle=True).item()

def planner_find_Path():
    print("Start Creating Map......")
    map = MapGen()
    if cond_params['load']:
        map.load()
    else:
        map.create_agentPos(cond_params['num_of_agents'])
        map.save()

    agentInfos = map.agentInfos

    planner = CBSSolver(cond_params, agentInfos)
    success_node = planner.solve()

    if success_node is not None:        
        for agentIdx in success_node.agentMap.keys():
            agent = success_node.agentMap[agentIdx]

            detailPath = agent.getDetailPath()
            agentInfos[agentIdx]['detailPath'] = detailPath
        
        np.save(os.path.join(cond_params['save_dir'], 'agentInfo'), agentInfos)

def smoothPath():
    smoother = mapf_pipeline.RandomStep_Smoother(
        xmin = 0.0, xmax = cond_params['x'] + 2.0,
        ymin = 0.0, ymax = cond_params['y'] + 2.0,
        zmin = 0.0, zmax = cond_params['z'] + 2.0,
        stepReso = 0.02
    )
    smoother.wSmoothness = 1.0
    smoother.wCurvature = 0.0
    smoother.wObstacle = 3.0

    agentInfos = np.load(
        os.path.join(cond_params['save_dir'], 'agentInfo')+'.npy', allow_pickle=True
    ).item()

    # agentInfos = {0: agentInfos[0]}

    # ### -------- Just For Debug
    # random_colors = np.random.uniform(0.0, 1.0, size=(len(agentInfos), 3))

    # ax = vis_utils.create_Graph3D(
    #     xmax=cond_params['x'] + 2.0, 
    #     ymax=cond_params['y'] + 2.0, 
    #     zmax=cond_params['z'] + 2.0
    # )
    # for agentIdx in agentInfos.keys():
    #     paddingPath = smoother.paddingPath(
    #         agentInfos[agentIdx]['detailPath'], 
    #         agentInfos[agentIdx]['startDire'],
    #         agentInfos[agentIdx]['endDire'],
    #         x_shift=1.0, y_shift=1.0, z_shift=1.0
    #     )
    #     paddingPath = smoother.detailSamplePath(paddingPath, 0.3)

    #     paddingPath_XYZ = np.array(paddingPath)[:, :3]
    #     # print('startPos:', agentInfos[agentIdx]['startPos'])
    #     # print('endPos:', agentInfos[agentIdx]['endPos'])
    #     # print('startDire:', agentInfos[agentIdx]['startDire'])
    #     # print('endDire:', agentInfos[agentIdx]['endDire'])
    #     # print('detailPath first: ', agentInfos[agentIdx]['detailPath'][0])
    #     # print('detailPath last: ', agentInfos[agentIdx]['detailPath'][-1])
    #     # print('paddingPath first: ', paddingPath[0])
    #     # print('paddingPath last: ', paddingPath[-1])
    #     vis_utils.plot_Path3D(ax, paddingPath_XYZ, color=tuple(random_colors[agentIdx]))
    # plt.show()

    # vis = VisulizerVista()
    # for agentIdx in agentInfos.keys():
    #     paddingPath = smoother.paddingPath(
    #         agentInfos[agentIdx]['detailPath'], 
    #         agentInfos[agentIdx]['startDire'],
    #         agentInfos[agentIdx]['endDire'],
    #         x_shift=1.0, y_shift=1.0, z_shift=1.0
    #     )
    #     paddingPath_XYZ = np.array(paddingPath)[:, :3]
    
    #     tube_mesh = vis.create_tube(paddingPath_XYZ, radius=agentInfos[agentIdx]['radius'])
    #     vis.plot(tube_mesh, color=tuple(random_colors[agentIdx]))
    # vis.show()
    # ### ----------------------------

    oldPaths = {}
    for agentIdx in agentInfos.keys():
        paddingPath = smoother.paddingPath(
            agentInfos[agentIdx]['detailPath'], 
            agentInfos[agentIdx]['startDire'],
            agentInfos[agentIdx]['endDire'],
            x_shift=1.0, y_shift=1.0, z_shift=1.0
        )
        paddingPath = smoother.detailSamplePath(paddingPath, 0.25)

        agentInfos[agentIdx]['paddingPath'] = paddingPath
            
        smoother.addAgentObj(
            agentIdx=agentIdx, 
            radius=agentInfos[agentIdx]['radius'],
            detailPath=agentInfos[agentIdx]['paddingPath']
        )

        oldPaths[agentIdx] = np.array(agentInfos[agentIdx]['paddingPath'])[:, :3]

    for _ in range(5):
        smoother.smoothPath(updateTimes=30)

        ax = vis_utils.create_Graph3D(
            xmax=cond_params['x'] + 2.0, 
            ymax=cond_params['y'] + 2.0, 
            zmax=cond_params['z'] + 2.0
        )
        for agentIdx in agentInfos.keys():
            smoothXYZ = extractPath(smoother.agentMap[agentIdx].pathXYZ)
            old_path = np.array(oldPaths[agentIdx])
            vis_utils.plot_Path3D(ax, old_path, color='r')
            vis_utils.plot_Path3D(ax, smoothXYZ, color='b')
            oldPaths[agentIdx] = smoothXYZ
        plt.show()

        for agentIdx in agentInfos.keys():
            smoothXYZ = extractPath(smoother.agentMap[agentIdx].pathXYZ)
            agentInfos[agentIdx]['smoothXYZ'] = smoothXYZ

    random_colors = np.random.uniform(0.0, 1.0, size=(len(agentInfos), 3))
    vis = VisulizerVista()
    for agentIdx in agentInfos.keys():
        tube_mesh = vis.create_tube(agentInfos[agentIdx]['smoothXYZ'], radius=agentInfos[agentIdx]['radius'])
        vis.plot(tube_mesh, color=tuple(random_colors[agentIdx]))
    vis.show()

if __name__ == '__main__':
    # planner_find_Path()
    smoothPath()
