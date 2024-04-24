import os
import shutil
import numpy as np
import dolfinx
import ufl
from ufl import grad, dot, inner, div
from functools import partial
import argparse
import json
import pyvista
from typing import List, Union

from scripts_py.version_9.dolfinx_Grad.collision_objs import MeshCollisionObj, ObstacleCollisionObj
from scripts_py.version_9.dolfinx_Grad.dolfinx_utils import MeshUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Find Good Naiver Stoke Initiation")
    parser.add_argument('--json_files', type=str, nargs='+', default=[])
    parser.add_argument('--obstacle_json_files', type=str, nargs='+', default=[])
    parser.add_argument('--obstacle_dir', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    mesh_objs = []
    for run_cfg_file in args.json_files:
        with open(run_cfg_file, 'r') as f:
            run_cfg: dict = json.load(f)

        if run_cfg.get('recombine_cfgs', False):
            with open(os.path.join(run_cfg['proj_dir'], run_cfg['recombine_cfgs'][0]), 'r') as f:
                run_cfg = json.load(f)

        domain, cell_tags, facet_tags = MeshUtils.read_XDMF(
            file=os.path.join(run_cfg['proj_dir'], 'model.xdmf'),
            mesh_name='model', cellTag_name='model_cells', facetTag_name='model_facets'
        )
        input_markers = [int(marker) for marker in run_cfg['input_markers'].keys()]
        output_markers = run_cfg['output_markers']
        bry_markers = run_cfg['bry_free_markers'] + run_cfg['bry_fix_markers']
        whole_markers = input_markers + output_markers + bry_markers
        mesh_obj = MeshCollisionObj(
            run_cfg['name'], domain, facet_tags, cell_tags,
            bry_markers=whole_markers,
            point_radius=run_cfg['point_radius']
        )
        mesh_objs.append(mesh_obj)

    for obs_cfg_file in args.obstacle_json_files:
        with open(obs_cfg_file, 'r') as f:
            obs_cfg: dict = json.load(f)

        obs_dir = obs_cfg['obstacle_dir']
        obs_file = os.path.join(obs_dir, f"{obs_cfg['name']}.{obs_cfg['file_format']}")
        mesh = pyvista.read(obs_file)

        coords = np.array(mesh.points[:, :obs_cfg['dim']]).astype(float)
        coords = ObstacleCollisionObj.remove_intersection_points(coords, mesh_objs, obs_cfg['point_radius'])

        filter_obs_name = f"{obs_cfg['name']}_filter.vtu"
        filter_obs = pyvista.PointSet(coords)
        pyvista.save_meshio(os.path.join(obs_dir, filter_obs_name), filter_obs)

        obs_cfg['filter_obs'] = filter_obs_name

        with open(obs_cfg_file, 'w') as f:
            json.dump(obs_cfg, f, indent=4)


if __name__ == '__main__':
    main()
