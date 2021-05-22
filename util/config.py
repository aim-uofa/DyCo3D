'''
config.py
Written by Li Jiang
'''

import argparse
import yaml
import os

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/pointgroup_default_scannet.yaml', help='path to config file')

    ### pretrain
    parser.add_argument('--pretrain', type=str, default=None, help='path to pretrain model')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--threshold_ins', type=float, default=0.5)
    parser.add_argument('--min_pts_num', type=int, default=50)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--use_backbone_transformer', action='store_true', default=False)


    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


cfg = get_parser()
# setattr(cfg, 'exp_path', os.path.join('exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
setattr(cfg, 'exp_path', cfg.output_path)