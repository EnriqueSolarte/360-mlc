import os
from pathlib import Path
from mlc.config.cfg import read_config, read_omega_cfg
from mlc.utils.io_utils import create_mlc_label_dirs
from mlc.mlc import iterator_room_scenes, compute_and_store_mlc_labels
import argparse


def get_passed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt',
        default="mp3d",
        help='ckpt pre-trained model. Options zind, mp3d, st3d, panos2d3d. Default mp3d')
    
    parser.add_argument(
        '--cfg',
        default="config/create_mlc_labels.yaml",
        help='Config File. Default config/create_mlc_labels.yaml')
    
    args = parser.parse_args()
    return args
         
if __name__ == "__main__":
    args = get_passed_args()
    # ! Reading configuration
    cfg_file = Path(args.cfg).resolve()
    assert os.path.exists(cfg_file), f"File does not exits: {cfg_file}"
    
    cfg = read_omega_cfg(cfg_file)
    cfg.ckpt = args.ckpt
    
    create_mlc_label_dirs(cfg)
    
    for list_ly in iterator_room_scenes(cfg):
        compute_and_store_mlc_labels(list_ly, save_vis=True)
    