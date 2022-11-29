from mlc.models.utils import load_layout_model
from mlc.config.cfg import read_config
from mlc.data_loaders.mlc_simple_dataloader import MLC_SimpleDataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
import os

if __name__ == "__main__":
    # ! Reading configuration
    cfg_file = "config/train_mlc.yaml"
     
    # ! Reading configuration
    cfg_file = Path("config/train_mlc.yaml").resolve()

    assert os.path.exists(cfg_file), f"File does not exits: {cfg_file}"   
    
    cfg = read_config(cfg_file)
    logging.info(f"Training model: {cfg.id_exp}")
    
    model = load_layout_model(cfg)
    model.prepare_for_training()

    model.set_valid_dataloader()
    model.valid_h_loop()
    model.valid_iou_loop()
    model.save_current_scores()
    while model.is_training:
        model.train_loop()
        model.valid_h_loop()
        model.valid_iou_loop()
        model.save_current_scores()
