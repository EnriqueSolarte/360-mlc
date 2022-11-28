import os
import logging
import shutil
from mlc.config.cfg import save_cfg


def create_directory(output_dir, delete_prev=True):
    if os.path.exists(output_dir) and delete_prev:
        logging.warning(f"This directory will be deleted: {output_dir}")
        input("This directory will be deleted. PRESS ANY KEY TO CONTINUE...")
        shutil.rmtree(output_dir, ignore_errors=True)
    logging.info(f"Dir created: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)


def create_mlc_label_dirs(cfg):
    """
    Creates the necessary directories to store mlc labels given a cfg file
    """

    create_directory(cfg.output_dir, delete_prev=True)
    create_directory(os.path.join(cfg.output_dir, "label", cfg.runners.mvl.label))
    create_directory(os.path.join(cfg.output_dir, "label", "std"))
    create_directory(os.path.join(cfg.output_dir, "vis"))
    save_cfg(os.path.join(cfg.output_dir, "cfg.yaml"), cfg)

    return True

    
