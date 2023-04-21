import numpy as np
import os

cond_params = {
    'y': 8,
    'x': 8,
    'z': 8,
    'num_of_agents': 5,

    'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_3/map',
    "load": False,
}

class MapGen(object):
    def __init__(self):
        self.startPoses = []
        self.endPoses = []

    def create_agentPos(self, num):
        dires = np.random.choice(['x', 'y', 'z'], size=num)
        for dire in dires:
            startPos, endPos = self.create_Pos(dire)

            self.startPoses.append(startPos)
            self.endPoses.append(endPos)

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
        
        if np.random.uniform(0.0, 1.0) > 0.5:
            return pos1, pos2
        else:
            return pos2, pos1

    def save(self):
        np.save(
            cond_params['save_path'],
            {
                'startPoses': self.startPoses,
                'endPoses': self.endPoses
            }
        )

    def load(self):
        d = np.load(os.path.join(cond_params['save_path']+'.npy'), allow_pickle=True).item()
        self.startPoses = d['startPoses']
        self.endPoses = d['endPoses']

def main():
    map = MapGen()
    if cond_params['load']:
        map.load()
    else:
        map.create_agentPos(cond_params['num_of_agents'])
        map.save()

if __name__ == '__main__':
    main()
