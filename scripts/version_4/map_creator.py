import numpy as np
import os

from scripts.visulizer import VisulizerVista

class MapGen(object):
    def __init__(self, config):
        self.config = config
        self.agentInfos = {}

        self.min_length = np.linalg.norm([
            self.config['x'] / 2.0, 
            self.config['y'] / 2.0, 
            self.config['z'] / 2.0
        ], ord=2)
        self.agentCount = 0

    def generateMap(self, num):
        record = []

        for groupIdx in range(num):
            pipe_type = np.random.choice(self.config['pipe_type'])

            if pipe_type == 'one2one':
                self.generate_one2one_Group(groupIdx, record)
            elif pipe_type == 'more2one':
                self.generate_one2more_Group(groupIdx, record)

        return self.agentInfos

    def generate_one2one_Group(self, groupIdx, record):
        record_copy = record.copy()

        while True:
            radius = np.random.choice(self.config['radius_choices'])

            startPos, start_paddingDire = self.generate_randomPos()
            if self.checkValid(record_copy, startPos, radius):
                record_copy.append([
                    startPos[0], startPos[1], startPos[2], radius
                ])
                break
        
        while True:
            endPos, end_paddingDire = self.generate_randomPos()
            if self.checkValid(record_copy, endPos, radius):

                if np.linalg.norm(startPos - endPos, ord=2) > self.min_length:
                    break
        
        record.append([
            startPos[0], startPos[1], startPos[2], radius
        ])
        record.append([
            endPos[0], endPos[1], endPos[2], radius
        ])

        self.agentInfos[self.agentCount] = {
            'groupIdx': groupIdx,
            'agentIdx': self.agentCount,
            
            'startPos': tuple(startPos),
            'start_paddingDire': start_paddingDire,

            'endPos': tuple(endPos),
            'end_paddingDire': end_paddingDire,

            'radius': radius,
        }
        self.agentCount += 1

    def generate_one2more_Group(self, groupIdx, record):
        record_copy = record.copy()

        while True:
            radius = np.random.choice(self.config['radius_choices'])

            startPos, start_paddingDire = self.generate_randomPos()
            if self.checkValid(record_copy, startPos, radius):
                record_copy.append([
                    startPos[0], startPos[1], startPos[2], radius
                ])
                break
        
        output_num = np.random.choice(self.config['pipe_choice'])
        endPos_list = []

        for _ in range(output_num):
            while True:
                endPos, end_paddingDire = self.generate_randomPos()
                if self.checkValid(record_copy, endPos, radius):
                    if np.linalg.norm(startPos - endPos) > self.min_length:
                        
                        endPos_list.append((endPos, end_paddingDire))
                        record_copy.append([
                            endPos[0], endPos[1], endPos[2], radius
                        ])

                        break
        
        record.append([
            startPos[0], startPos[1], startPos[2], radius
        ])

        for endPos, end_paddingDire in endPos_list:
            record.append([
                endPos[0], endPos[1], endPos[2], radius
            ])
            
            self.agentInfos[self.agentCount] = {
                'groupIdx': groupIdx,
                'agentIdx': self.agentCount,
                
                'startPos': tuple(startPos),
                'start_paddingDire': start_paddingDire,

                'endPos': tuple(endPos),
                'end_paddingDire': end_paddingDire,

                'radius': radius,
            }
            self.agentCount += 1

    def generate_randomPos(self):
        pos = np.array([
            np.random.randint(0, self.config['x']), 
            np.random.randint(0, self.config['y']),
            np.random.randint(0, self.config['z'])
        ])
        face = np.random.choice(['x', 'y', 'z'], size=1)
        dire = np.random.uniform(0.0, 1.0)
        if face == 'x':
            if dire > 0.5:
                pos[0] = 0.0
                paddingDire = (-1.0, 0.0, 0.0)
            else:
                pos[0] = self.config['x'] - 1
                paddingDire = (1.0, 0.0, 0.0)

        elif face == 'y':
            if dire > 0.5:
                pos[1] = 0.0
                paddingDire = (0.0, -1.0, 0.0)
            else:
                pos[1] = self.config['y'] - 1
                paddingDire = (0.0, 1.0, 0.0)

        else:
            if dire > 0.5:
                pos[2] = 0.0
                paddingDire = (0.0, 0.0, -1.0)
            else:
                pos[2] = self.config['z'] - 1
                paddingDire = (0.0, 0.0, 1.0)

        return pos, paddingDire

    def checkValid(self, record, pos, radius):
        record_np = np.array(record)
        pos_np = np.array(pos)

        if record_np.shape[0] > 0:
            dist = np.linalg.norm(pos_np - record_np[:, :3], ord=2, axis=1)
            conflict_num = (dist < (record_np[:, 3] + radius)).sum()

            if conflict_num > 0:
                return False
        
        return True

    def save(self):
        np.save(self.config['save_path'], self.agentInfos)

    def load(self):
        self.agentInfos = np.load(os.path.join(self.config['save_path']+'.npy'), allow_pickle=True).item()

if __name__ == '__main__':
    config = {
        'y': 10,
        'x': 10,
        'z': 10,
        'num_of_agents': 4,
        'radius_choices': [
            0.5, 
            # 0.85, 
            # 1.25
        ],
        'pipe_type': [
            'one2one', 
            'more2one'
        ],
        'pipe_choice': [2, 3],

        'save_path': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_4/map',
        "load": False,

        'save_dir': '/home/quan/Desktop/MAPF_Pipeline/scripts/version_4/',
    }

    maper = MapGen(config)
    maper.generateMap(config['num_of_agents'])

    random_colors = np.random.uniform(low=0.0, high=1.0, size=(config['num_of_agents'], 3))
    vis = VisulizerVista()
    for key in maper.agentInfos.keys():
        agentInfo = maper.agentInfos[key]
        print(agentInfo)

        mesh = vis.create_sphere(
            np.array(agentInfo['startPos']), radius=agentInfo['radius']
        )
        vis.plot(mesh, tuple(random_colors[agentInfo['groupIdx']]))
        mesh = vis.create_sphere(
            np.array(agentInfo['endPos']), radius=agentInfo['radius']
        )
        vis.plot(mesh, tuple(random_colors[agentInfo['groupIdx']]))
    
    vis.show()