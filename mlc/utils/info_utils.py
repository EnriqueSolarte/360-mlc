import json
import logging 
import numpy as np
import argparse

def print_run_information(cfg):
    print("\n")
    logging.info(f"Experiment ID: {cfg.id_exp}")
    logging.info(f"Output_dir: {cfg.output_dir}")
    print("\n")

def get_mean_mse_h(dict_data):
    """
    data = {
        scene: {
            0.1: xxx
            0.2: xxx
            ...
        }
        ...
    }
    """
    dt_out = {grid: [] for grid in list(list(dict_data.values())[0].keys())}
    for grid, dt_list in dt_out.items():
        # ! appending H for each grid in all scenes
        [dt_list.append(sc[grid]) for sc in list(dict_data.values())]
        dt_out[grid] = np.mean(dt_list)
    return dt_out
