from mlc.models.utils import load_layout_model
from mlc.config.cfg import read_config, read_omega_cfg
from mlc.data_loaders.mlc_simple_dataloader import MLC_SimpleDataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
import os
import argparse


def get_passed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt',
        default="mp3d",
        help='ckpt pre-trained model. Options zind, mp3d, st3d, panos2d3d. Default mp3d')
    
    parser.add_argument(
        '--cfg',
        default="config/train_mlc.yaml",
        help='Config File. Default config/train_mlc.yaml')
    
    parser.add_argument(
        '--desc',
        default="training_test",
        help='Give a description for this training. Default training_test')
    
    parser.add_argument(
        '--mlc',
        default=None,
        help='Define a custom mlc-label. By default hn_<ckpt>__mp3d_fpe__mlc is used')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_passed_args()
     
    # ! Reading configuration
    cfg_file = Path().resolve()
    cfg_file = Path(args.cfg).resolve()

    assert os.path.exists(cfg_file), f"File does not exits: {cfg_file}"   
    
    cfg = read_omega_cfg(cfg_file)
    cfg.ckpt = args.ckpt
    cfg.id_exp = f"{cfg.id_exp}__{args.desc}"
    
    if args.mlc is not  None:
        cfg.mlc_label = args.mlc
        
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
