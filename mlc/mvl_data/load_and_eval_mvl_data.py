import os
from pathlib import Path
from mlc.config.cfg import read_config, read_omega_cfg
from mlc.utils.io_utils import create_mlc_label_dirs
from mlc.utils.image_utils import draw_boundaries_phi_coords, plot_image
from mlc.mlc import iterator_room_scenes, compute_and_store_mlc_labels
from mvl_challenge.utils.vispy_utils import plot_list_ly
from imageio import imwrite
import argparse
import numpy as np


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
    
    parser.add_argument(
        '--target',
        default="mp3d_fpe",
        help='Target mp3d_fpe')
    
    args = parser.parse_args()
    return args
         
if __name__ == "__main__":
    args = get_passed_args()
    # ! Reading configuration
    cfg_file = Path(args.cfg).resolve()
    assert os.path.exists(cfg_file), f"File does not exits: {cfg_file}"
    
    cfg = read_omega_cfg(cfg_file)
    cfg.ckpt = args.ckpt
    cfg.target_dataset = args.target

    create_mlc_label_dirs(cfg)
    
    for list_ly in iterator_room_scenes(cfg):
        for ly in list_ly:
            img = ly.img
            draw_boundaries_phi_coords(
                img, phi_coords=np.vstack([ly.phi_coord[0], ly.phi_coord[1]])
            )
            # imwrite("test.jpg", img)
            plot_image(image=img, caption=ly.idx)
        plot_list_ly(list_ly)

    