env_config = {
    "warning": "这里的所有管道半径都必须加上管道壁厚",

    "projectDir": "/home/quan/Desktop/tempary/application_pipe",

    "x": 125,
    "y": 125,
    "z": 125,

    "scale": 0.35,
    "stepLength": 1.0,
    "obstacle_resolution": 1.0,

    "pipeConfig":[
        {
            "groupIdx": 0,
            "pipe": [
                {
                    "name": "p",
                    "position": [0.0, 67.5, 67.0],
                    "radius": 5.0,
                    "alpha": 0.0, 
                    "theta": 0.0
                },
                {
                    "name": "p1",
                    "position": [125, 67.5, 67.0],
                    "radius": 5.0,
                    "alpha": 0.0,
                    "theta": 0.0
                },
                {
                    "name": "M1",
                    "position": [0.0, 67.5, 105.0],
                    "radius": 2.5,
                    "alpha": 3.14,
                    "theta": 0.0
                },
                {
                    "name": "p_valve",
                    "position": [61.7, 60.3, 125.0],
                    "radius": 2.5,
                    "alpha": 0.0,
                    "theta": 1.57
                }
            ]
        },
        {
            "groupIdx": 1,
            "pipe": [
                {
                    "name": "M3",
                    "position": [53.0, 0.0, 65.0],
                    "radius": 5.0,
                    "alpha": -1.57, 
                    "theta": 0.0
                },
                {
                    "name": "B_valve",
                    "position": [53.0, 50.0, 125.0],
                    "radius": 3.0,
                    "alpha": 0.0, 
                    "theta": 1.57
                },
                {
                    "name": "B",
                    "position": [55.0, 17.0, 0.0],
                    "radius": 5.0,
                    "alpha": 0.0, 
                    "theta": 1.57
                }
            ]
        },
        {
            "groupIdx": 2,
            "pipe": [
                {
                    "name": "T_valve",
                    "position": [61.7, 39.7, 125.0],
                    "radius": 3.0,
                    "alpha": 0.0, 
                    "theta": -1.57
                },
                {
                    "name": "T",
                    "position": [98.0, 42.5, 0.0],
                    "radius": 7.5,
                    "alpha": 0.0, 
                    "theta": -1.57
                },
                {
                    "name": "A2T",
                    # "position": [35.0, 95.0, 115.25],
                    "position": [35.0, 98.5, 110.0],
                    "radius": 3.0,
                    "alpha": -1.57, 
                    "theta": 0.0
                }
            ]
        },
        {
            "groupIdx": 3,
            "pipe": [
                {
                    "name": "A_valve",
                    "position": [70.5, 50.0, 125.0],
                    "radius": 2.5,
                    "alpha": 0.0, 
                    "theta": -1.57
                },
                {
                    "name": "A2valve_01",
                    "position": [102, 50.0, 102.0],
                    "radius": 5.0,
                    "alpha": 0.0, 
                    "theta": 0.0
                },
                {
                    "name": "A2valve_02",
                    "position": [72.0, 87.5, 110.0],
                    "radius": 5.0,
                    "alpha": 1.57, 
                    "theta": 0.0
                }
            ]
        },
        {
            "groupIdx": 4,
            "pipe": [
                {
                    "name": "valve_01",
                    "position": [113.5, 50.0, 95.65],
                    "radius": 5.0,
                    "alpha": 0.0, 
                    "theta": -1.57
                },
                {
                    "name": "valve_02",
                    "position": [72.0, 98.0, 105.5],
                    "radius": 5.0,
                    "alpha": 0.0, 
                    "theta": -1.57
                },
                {
                    "name": "valve_03",
                    "position": [35.0, 109.0, 105.5],
                    "radius": 5.0,
                    "alpha": 0.0, 
                    "theta": -1.57
                },
                {
                    "name": "A",
                    "position": [35.0, 109.0, 0.0],
                    "radius": 5.0,
                    "alpha": 0.0, 
                    "theta": -1.57
                },
                {
                    "name": "M2",
                    "position": [0.0, 33.0, 45.0],
                    "radius": 2.5,
                    "alpha": 0.0, 
                    "theta": -1.57
                }
            ]
        }
    ],   
}