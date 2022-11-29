from email.mime import image
import os
import numpy as np
import pathlib
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from PIL import Image
import json
import torch
from mlc.utils.io_utils import read_csv_file
import logging

  
class MLC_MixedDataDataLoader(data.Dataset):
    '''
    Dataloader that handles MLC dataset format which sample from GT and MLC pseudo labels
    '''

    def __init__(self, cfg):
        self.cfg = cfg
        assert cfg.get('mix_data_dir') is not None, "key mix_data_dir is not defined in th cfg file"
        # ! List of data defined for mixing
        self.list_images = []
        self.list_labels = []
        self.list_std = []
        for dt_mix_key in list(cfg.get('mix_data_dir').keys()):
            # ! Except for "active" the rest of key define a dataset
            """
            dt: 
              scene_list:
              size:
              data_dir:
                  img_dir:
                  labels_dir:
            """
            if dt_mix_key == 'active':
                continue
            
            dt = cfg['mix_data_dir'][dt_mix_key]
            if dt.get('scene_list', '') == '':
                # ! Reading from available labels data       
                list_frames = os.listdir(
                    os.path.join(dt.data_dir.labels_dir, dt_mix_key)
                    )
            else:
                assert os.path.exists(dt.scene_list)
                raw_data = json.load(open(dt.scene_list))
                list_rooms = list(raw_data.keys())
                list_frames = [raw_data[room] for room in list_rooms]
                list_frames = [item for sublist in list_frames for item in sublist]
                
            # ! define __data --> files
            data = []    
            if dt.get('size', -1) < 0:
                [data.append(pathlib.Path(fn).stem) for fn in list_frames]
            elif dt.size < 1:
                np.random.shuffle(list_frames)
                [data.append(pathlib.Path(fn).stem) for fn in list_frames[:int(dt.size * list_frames.__len__())]] 
            else:
                np.random.shuffle(list_frames)
                [data.append(pathlib.Path(fn).stem) for fn in list_frames[:dt.size]]
                
            [self.list_images.append(os.path.join(dt.data_dir.img_dir, fn)) 
                for fn in data]
            
            [self.list_labels.append(
                os.path.join(dt.data_dir.labels_dir, dt_mix_key, f"{fn}.npy")) 
                for fn in data]
            
            [self.list_std.append(
                os.path.join(dt.data_dir.labels_dir, 'std', f"{fn}.npy")) 
                for fn in data]
        
            logging.info(f"MIX MLC dataloader initialized with: {dt.data_dir.img_dir}.")
            logging.info(f"Total frames in dataloader: {self.list_images.__len__()}.")
            
                
    def __len__(self):
        return self.list_images.__len__()

    def __getitem__(self, idx):
        # ! iteration per each self.data given a idx
        image_fn = self.list_images[idx]
        if os.path.exists(image_fn + '.jpg'):
            image_fn += '.jpg'
        elif os.path.exists(image_fn + '.png'):
            image_fn += '.png'
            
        label_fn = self.list_labels[idx]
        std_fn = self.list_std[idx]
        
        img = np.array(Image.open(image_fn), np.float32)[..., :3] / 255.
        
        label = np.load(label_fn)
        
        # Random flip
        if self.cfg.get('flip', False) and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=len(label.shape) - 1)

        # Random horizontal rotate
        if self.cfg.get('rotate', False):
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            label = np.roll(label, dx, axis=len(label.shape) - 1)

        # Random gamma augmentation
        if self.cfg.get('gamma', False):
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img**p
        
        if os.path.exists(std_fn):
            std = np.load(std_fn)
        else:
            std = np.ones_like(label)
        
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        label = torch.FloatTensor(label.copy())
        std = torch.FloatTensor(std.copy())
        return [x, label, std]

