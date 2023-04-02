from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from imageio import imread
from mlc.utils.io_utils import get_files_given_a_pattern
from mlc.utils.io_utils import save_obj, read_csv_file
import yaml, json
import argparse
from glob import glob
from shutil import copyfile


def process_data_scene(scene, data_scene):
    list_rooms = [r for r in list(data_scene.keys()) if "room." in r]
    scene_name = str.join("_", (data_scene["data.scene"], data_scene["data.scene_version"]))
    list_poses = get_list_camera_poses(scene)
    data = dict()
    for room in list_rooms:  
        scene_name = f"{scene_name}_room{room.split('.')[-1]}" 
        list_kf = data_scene[room]['list_kf']

        for kf in list_kf:
            data[f"{scene_name}_{kf}"] = dict(
                img= os.path.join( scene, f"rgb/{kf}.png"),
                cam=list_poses[kf-1])
    return data    
        
def get_list_camera_poses(scene):
    cam_poses_fn = os.path.join(scene, 'frm_ref.txt')
    assert os.path.isfile(cam_poses_fn), f'Cam pose file {cam_poses_fn} does not exist'
    cam_poses = read_csv_file(cam_poses_fn)
    return cam_poses
          
def process_mp3d_fpe(args):
    data_dir = args.path
    output_dir = Path(args.output).resolve().__str__()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "geometry_info"), exist_ok=True)
    
    list_scenes = get_files_given_a_pattern(data_dir=data_dir, flag_file="frm_ref.txt", exclude=["depth", 'rgb', 'hn_mp3d'])
    
    for scene in tqdm(list_scenes, desc="reding scenes..."):
        
        room_gt_file = os.path.join(scene, "metadata", "room_gt_v0.0.yaml")
        assert os.path.isfile(room_gt_file)
        
        with open(room_gt_file, 'r') as f:
            data_scene = yaml.safe_load(f)
        
        data = process_data_scene(scene, data_scene)
        
        for scene, val in data.items():
            src = val['img']
            if not os.path.exists(src):
                continue
            
            ext = val['img'].split(".")[-1]
            if args.copy:
                copyfile(
                    src, 
                    os.path.join(output_dir, "img", f"{scene}.{ext}"))
            else:
                os.symlink(
                    src, 
                    os.path.join(output_dir, "img", f"{scene}.{ext}"))      
            
            geom = dict(
                translation=[float(v) for v in val['cam'].split(" ")[1:4]],
                quaternion=[float(v) for v in val['cam'].split(" ")[3:-1]])
            
            with open(os.path.join(output_dir, "geometry_info", f'{scene}.json'), 'w') as outfile:
                json.dump(geom, outfile)
         
def get_passed_args():
    this_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        # required=True,
        default="/media/public_dataset/MP3D_360_FPE/SINGLE_ROOM_SCENES",
        help='MP3D-FPE path')
    
    parser.add_argument(
        '-n', type=int,
        default=10,
        help='Min number of frames per room')
    
    parser.add_argument(
        '-output', type=str,
        default=f"{this_dir}/../assets/mp3d_fpe_dataset",
        help='Output directory. By default /assets/mp3d_fpe_dataset')
    
    parser.add_argument(
        '-copy', action='store_true',
        help='Copy a img file instead of creating symbolic links. By default symbolic links will be created')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_passed_args()
    
    process_mp3d_fpe(args)
    