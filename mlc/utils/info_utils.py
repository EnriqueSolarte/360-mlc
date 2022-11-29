import json
import logging 
import numpy as np
import argparse

def print_run_information(cfg):
    # print("\n")
    logging.info(f"Experiment ID: {cfg.id_exp}")
    logging.info(f"Output_dir: {cfg.output_dir}")
    # print("\n")