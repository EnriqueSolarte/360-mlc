import os
from pathlib import Path
from mlc.config.cfg import read_config, read_omega_cfg
from mlc.utils.io_utils import create_mlc_label_dirs
from mlc.mlc import iterator_room_scenes, compute_and_store_mlc_labels

if __name__ == "__main__":
    # ! Reading configuration
    cfg_file = Path("config/create_mlc_labels.yaml").resolve()

    assert os.path.exists(cfg_file), f"File does not exits: {cfg_file}"
    
    cfg = read_omega_cfg(cfg_file)

    create_mlc_label_dirs(cfg)
    
    for list_ly in iterator_room_scenes(cfg):
        compute_and_store_mlc_labels(list_ly, save_vis=True)
    