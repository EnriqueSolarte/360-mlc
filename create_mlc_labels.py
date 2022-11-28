import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from mlc.config.cfg import read_config


if __name__ == "__main__":
    # ! Reading configuration
    cfg_file = Path("config/create_mlc_labels.yaml").resolve()

    assert os.path.exists(cfg_file), f"File does not exits: {cfg_file}"
    
    cfg = read_config(cfg_file)

    print(cfg)
    # # ! Create directories
    # create_directory(cfg.output_dir, delete_prev=True)
    # create_directory(os.path.join(cfg.output_dir, "label", cfg.runners.mvl.label))
    # create_directory(os.path.join(cfg.output_dir, "label", "std"))
    # create_directory(os.path.join(cfg.output_dir, "vis"))
    # save_cfg(os.path.join(cfg.output_dir, "cfg.yaml"), cfg)

    # for list_ly in iterator_room_scenes(cfg):
    #     create_mlc_labels(list_ly)
    