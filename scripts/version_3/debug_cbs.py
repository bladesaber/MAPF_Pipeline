import numpy as np
import os

from scripts.version_3.cbs import CBSSolver

cond_params = {
    'y': 15,
    'x': 15,
    'z': 15,
    'num_of_agents': 10,
    'radius_choices': [0.45, 0.85, 1.25],

    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_3/map',
    "load": False,
}

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

def main():
    print("Start Creating Map......")
    map = MapGen()
    if cond_params['load']:
        map.load()
    else:
        map.create_agentPos(cond_params['num_of_agents'])
        map.save()

    planner = CBSSolver(cond_params, map.agentInfos)
    planner.solve()

if __name__ == '__main__':
    main()
