import numpy as np
import argparse
import json
import os

from scripts_py.version_9.mapf_pkg import networker_path


class SimpleBlocker(object):
    @staticmethod
    def create_block_network(pipes_cfg: dict, blocks_seq: list, with_debug=False):
        group_networker = {}
        for pipe_name in pipes_cfg:
            group_idx = pipes_cfg[pipe_name]['group_idx']
            if group_idx not in group_networker.keys():
                group_networker[group_idx] = {'network': networker_path.HyperRadiusGrowthTree(), 'main_leaf': None}
            group_networker[group_idx]['network'].add_item(
                pipe_name, pipes_cfg[pipe_name]['radius'], pipes_cfg[pipe_name]['position']
            )

        for block_info in blocks_seq:
            for group_info in block_info['groups']:
                group_idx = group_info['group_idx']
                res_list, main_leaf = group_networker[group_idx]['network'].compute_block(
                    block_names=group_info['names'], main_leaf=group_networker[group_idx]['main_leaf']
                )
                group_networker[group_idx]['main_leaf'] = main_leaf
                group_info['sequence'] = res_list

                if with_debug:
                    for res in res_list:
                        print(res)
                    # group_networker[group_idx]['network'].draw_connect_network(with_plot=False)
                    # group_networker[group_idx]['network'].draw_sequence_network(res_list, with_plot=True)

        return blocks_seq


def parse_args():
    parser = argparse.ArgumentParser(description="Fluid Tool")
    parser.add_argument(
        '--setup_json', type=str,
        default='/home/admin123456/Desktop/work/path_examples/example1/algorithm_setup.json',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists(args.setup_json):
        return -1

    with open(args.setup_json, 'r') as f:
        setup_json = json.load(f)

    pipes_cfg = setup_json['pipes']
    blocks_seq = setup_json['search_tree']['block_sequence']
    if setup_json['search_tree']['method'] == 'simple':
        SimpleBlocker.create_block_network(pipes_cfg, blocks_seq, with_debug=False)


if __name__ == '__main__':
    main()
