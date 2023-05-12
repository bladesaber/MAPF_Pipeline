import numpy as np
import pandas as pd
import os
import open3d as o3d

from scripts.visulizer import VisulizerVista

center_radius = 2.5
radShift_high = 30.0
radShift_low = -30.0
x_shift = 10.0
y_shift = 10.0
z_shift = 0.0
radius = 1.0

agentsInfo = {
    0:{
        'groupIdx': 0,
        'agentIdx': 0,
        'startPos': [
            6.5 + x_shift, 
            0.0 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(0.0 + radShift_high)) + x_shift, 
            center_radius * np.sin(np.deg2rad(0.0 + radShift_high)) + y_shift, 
            15.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (-1.0, 0.0, 0.0),
        'radius': radius,
    },
    1:{
        'groupIdx': 0,
        'agentIdx': 1,
        'startPos': [
            6.5 + x_shift, 
            0.0 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(180.0 + radShift_high)) + x_shift, 
            center_radius * np.sin(np.deg2rad(180.0 + radShift_high)) + y_shift, 
            15.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },
    2:{
        'groupIdx': 0,
        'agentIdx': 2,
        'startPos': [
            6.5 + x_shift, 
            0.0 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(0.0 + radShift_low)) + x_shift, 
            center_radius * np.sin(np.deg2rad(0.0 + radShift_low)) + y_shift, 
            8.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },
    3:{
        'groupIdx': 0,
        'agentIdx': 3,
        'startPos': [
            6.5 + x_shift, 
            0.0 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(180.0 + radShift_low)) + x_shift, 
            center_radius * np.sin(np.deg2rad(180.0 + radShift_low)) + y_shift, 
            8.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },

    4:{
        'groupIdx': 1,
        'agentIdx': 4,
        'startPos': [
            0.0 + x_shift, 
            6.5 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(120.0 + radShift_low)) + x_shift, 
            center_radius * np.sin(np.deg2rad(120.0 + radShift_low)) + y_shift, 
            8.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },
    5:{
        'groupIdx': 1,
        'agentIdx': 5,
        'startPos': [
            0.0 + x_shift, 
            6.5 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(300.0 + radShift_low)) + x_shift, 
            center_radius * np.sin(np.deg2rad(300.0 + radShift_low)) + y_shift, 
            8.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },


    6:{
        'groupIdx': 2,
        'agentIdx': 6,
        'startPos': [
            -6.5 + x_shift, 
            0.0 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(120.0 + radShift_high)) + x_shift, 
            center_radius * np.sin(np.deg2rad(120.0 + radShift_high)) + y_shift, 
            15.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },
    7:{
        'groupIdx': 2,
        'agentIdx': 7,
        'startPos': [
            -6.5 + x_shift, 
            0.0 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(300.0 + radShift_high)) + x_shift, 
            center_radius * np.sin(np.deg2rad(300.0 + radShift_high)) + y_shift, 
            15.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },
    8:{
        'groupIdx': 2,
        'agentIdx': 8,
        'startPos': [
            -6.5 + x_shift, 
            0.0 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(240.0 + radShift_low)) + x_shift, 
            center_radius * np.sin(np.deg2rad(240.0 + radShift_low)) + y_shift, 
            8.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },
    9:{
        'groupIdx': 2,
        'agentIdx': 9,
        'startPos': [
            -6.5 + x_shift, 
            0.0 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(60.0 + radShift_low)) + x_shift, 
            center_radius * np.sin(np.deg2rad(60.0 + radShift_low)) + y_shift, 
            8.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },

    10:{
        'groupIdx': 3,
        'agentIdx': 10,
        'startPos': [
            0.0 + x_shift, 
            -6.5 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(60.0 + radShift_high)) + x_shift, 
            center_radius * np.sin(np.deg2rad(60.0 + radShift_high)) + y_shift, 
            15.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },
    11:{
        'groupIdx': 3,
        'agentIdx': 11,
        'startPos': [
            0.0 + x_shift, 
            -6.5 + y_shift, 
            0.0 + z_shift
        ],
        'endPos': [
            center_radius * np.cos(np.deg2rad(240.0 + radShift_high)) + x_shift, 
            center_radius * np.sin(np.deg2rad(240.0 + radShift_high)) + y_shift, 
            15.0 + z_shift
        ],
        # 'start_paddingDire': (0., 0., -1.),
        # 'end_paddingDire': (1.0, 0.0, 0.0),
        'radius': radius,
    },    
}

def printAgentsInfo(agentsInfo):
    for agentIdx in agentsInfo.keys():
        agentInfo = agentsInfo[agentIdx]
        
        print('AgentIdx:%d GroupIdx:%d' % (agentInfo['agentIdx'], agentInfo['groupIdx']))
        print('  startPos:', agentInfo['startPos'])
        print('  endPos:', agentInfo['endPos'])
        print('  radius:', agentInfo['radius'])

random_colors = np.random.uniform(0.0, 1.0, size=(4, 3))
vis = VisulizerVista()
for agentIdx in agentsInfo.keys():
    agentInfo = agentsInfo[agentIdx]

    mesh = vis.create_sphere(
        xyz=np.array(agentInfo['startPos']), radius=agentInfo['radius']
    )
    vis.plot(mesh, tuple(random_colors[agentInfo['groupIdx']]))
    mesh = vis.create_sphere(
        xyz=np.array(agentInfo['endPos']), radius=agentInfo['radius']
    )
    vis.plot(mesh, tuple(random_colors[agentInfo['groupIdx']]))

stl_file = '/home/quan/Desktop/MAPF_Pipeline/scripts/application/company.STL'
mesh = vis.read_file(stl_file)
vis.plot(mesh, (0.0, 1.0, 0.0))

vis.show()