import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from build import mapf_pipeline

# unit: mm
pipe_config = {
    0: [
        {
            'name': 'p',
            'position': (0.0, 67.5, 67.0),
            'radius': 5.0,
            'alpha': 0.0, 
            'theta': 0.0,
        },
        {
            'name': 'p1',
            'position': (125, 67.5, 67.0),
            'radius': 5.0,
            'alpha': 0.0,
            'theta': 0.0,
        },
        {
            'name': 'M1',
            'position': (0.0, 67.5, 105.0),
            'radius': 2.5,
            'alpha': np.deg2rad(180.0),
            'theta': 0.0,
        },
        {
            'name': 'p_valve',
            'position': (61.7, 60.3, 125.0),
            'radius': 2.5,
            'alpha': 0.0,
            'theta': np.deg2rad(90.0),
        }
    ],
    1: [
        {
            'name': 'M3',
            'position': (53.0, 0.0, 65.0),
            'radius': 5.0,
            'alpha': np.deg2rad(-90.0), 
            'theta': 0.0,
        },
        {
            'name': 'B_valve',
            'position': (53.0, 50.0, 125.0),
            'radius': 3.0,
            'alpha': 0.0, 
            'theta': np.deg2rad(90.0),
        },
        {
            'name': 'B',
            'position': (55.0, 17.0, 0.0),
            'radius': 5.0,
            'alpha': 0.0, 
            'theta': np.deg2rad(90.0),
        }
    ],
    2: [
        {
            'name': 'T_valve',
            'position': (61.7, 39.7, 125.0),
            'radius': 3.0,
            'alpha': 0.0, 
            'theta': np.deg2rad(-90.0),
        },
        {
            'name': 'T',
            'position': (98.0, 42.5, 0.0),
            'radius': 7.5,
            'alpha': 0.0, 
            'theta': np.deg2rad(-90.0),
        },
        {
            'name': 'A2T',
            'position': (35.0, 98.0 - 3.0, 110.0),
            'radius': 3.0,
            'alpha': np.deg2rad(-90.0), 
            'theta': 0.0,
        },
    ],
    3: [
        {
            'name': 'A_valve',
            'position': (70.5, 50.0, 125.0),
            'radius': 2.5,
            'alpha': 0.0, 
            'theta': np.deg2rad(-90.0),
        },
        {
            'name': 'A2valve_01',
            'position': (85.0, 50.0, 102.0),
            'radius': 5.0,
            'alpha': 0.0, 
            'theta': 0.0,
        },
        {
            'name': 'A2valve_02',
            'position': (72.0, 87.0 - 5.0, 110.0),
            'radius': 5.0,
            'alpha': np.deg2rad(90.0), 
            'theta': 0.0,
        }
    ],
    4: [
        {
            'name': 'valve_01',
            'position': (100.0, 50.0, 95.6 - 6.5),
            'radius': 5.0,
            'alpha': 0.0, 
            'theta': np.deg2rad(-90.0),
        },
        {
            'name': 'valve_02',
            'position': (72.0, 98.0, 95.65 + 3.0),
            'radius': 5.0,
            'alpha': 0.0, 
            'theta': np.deg2rad(-90.0),
        },
        {
            'name': 'valve_03',
            'position': (35.0, 109.0, 95.65 + 3.0),
            'radius': 5.0,
            'alpha': 0.0, 
            'theta': np.deg2rad(-90.0),
        },
        {
            'name': 'A',
            'position': (35.0, 109.0, 0.0),
            'radius': 5.0,
            'alpha': 0.0, 
            'theta': np.deg2rad(-90.0),
        },
        {
            'name': 'M2',
            'position': (0.0, 33.0, 45.0),
            'radius': 2.5,
            'alpha': 0.0, 
            'theta': np.deg2rad(-90.0),
        }
    ]
}

obstacle_config = [
    {
        'name': 'support',
        'position': (12.0, 114.0, 125.0),
        'radius': 10.0,
        'height': 125.0,
        'type': 'Z_support',
        'sample_num': 500,
    },
    {
        'name': 'support',
        'position': (12.0, 12.0, 125.0),
        'radius': 10.0,
        'height': 125.0,
        'type': 'Z_support',
        'sample_num': 500,
    },
    {
        'name': 'support',
        'position': (103.0, 12.0, 125.0),
        'radius': 10.0,
        'height': 125.0,
        'type': 'Z_support',
        'sample_num': 500,
    },
    {
        'name': 'support',
        'position': (103.0, 114.0, 125.0),
        'radius': 10.0,
        'height': 125.0,
        'type': 'Z_support',
        'sample_num': 500,
    },
    {
        'name': 'screw',
        'position': (42.7, 66.25, 125.0),
        'radius': 2.0,
        'height': 14.0,
        'type': 'Z_screw',
        'sample_num': 50,
    },
    {
        'name': 'screw',
        'position': (42.7, 33.75, 125.0),
        'radius': 2.0,
        'height': 14.0,
        'type': 'Z_screw',
        'sample_num': 50,
    },
    {
        'name': 'screw',
        'position': (83.2, 66.25, 125.0),
        'radius': 2.0,
        'height': 14.0,
        'type': 'Z_screw',
        'sample_num': 50,
    },
    {
        'name': 'screw',
        'position': (83.2, 33.75, 125.0),
        'radius': 2.0,
        'height': 14.0,
        'type': 'Z_screw',
        'sample_num': 50,
    },
    {
        'name': 'valve',
        'position': (125.0, 50.0, 102.0),
        'radius': 6.35,
        'height': 23.0,
        'type': 'X_valve',
        'sample_num': 200,
    },
    {
        'name': 'valve',
        'position': (72.0, 98.0, 125.0),
        'radius': 10.5,
        'height': 19.5,
        'type': 'Z_valve',
        'sample_num': 200,
    },
    {
        'name': 'valve',
        'position': (35.0, 109.0, 125.0),
        'radius': 10.5,
        'height': 19.5,
        'type': 'Z_valve',
        'sample_num': 200,
    },
]

env_config = {
    'x': 84,
    'y': 84,
    'z': 84,
    'resolution': 1.5,
    'stepLength': 1.0,
}
