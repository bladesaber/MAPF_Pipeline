import subprocess


config_path = '/home/admin123456/Desktop/work/gmsh_test/t40/config.json'
cmd = f'mpirun -n 4 python scripts_py/auto_dolfinx/cbs_topo/fipy_obstacle_simulator.py --cfg_file {config_path}'
subprocess.run(
   cmd, shell=True,
   cwd='/home/admin123456/Desktop/work/MAPF_Pipeline'
)