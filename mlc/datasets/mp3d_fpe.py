import os
import json
from traceback import print_tb
from tqdm import tqdm
from mlc.utils.layout_utils import label_cor2ly_phi_coord
from mlc.utils.geometry_utils import eulerAnglesToRotationMatrix, extend_array_to_homogeneous
from mlc.utils.geometry_utils import tum_pose2matrix44
from mlc.utils.projection_utils import phi_coords2bearings
from mlc.data_structure import Layout, CamPose
import numpy as np
import logging
from imageio import imread
from pyquaternion import Quaternion
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MP3D_FPE:
    def __init__(self, cfg):
        logging.info("Initializing MP3D-FPE dataloader...")
        
        self.cfg = cfg
        self.set_paths()
        self.load_imgs = True
        self.load_npy = False
        
        logging.info("MP3D-FPE dataloader successfully initialized")

    def set_paths(self):
        # * Paths for MP3D_FPE dataset
        self.DIR_GEOM_FILES = self.cfg.runners.mvl.data_dir.geometry_info
        self.DIR_IMGS = self.cfg.runners.mvl.data_dir.img_dir
        
        assert os.path.exists(self.DIR_GEOM_FILES)
        assert os.path.exists(self.DIR_IMGS)
        
        # * Set main lists of files
        #! List of scene names
        assert os.path.exists(self.cfg.runners.mvl.scene_list)
        logging.info(f"Scene list: {self.cfg.runners.mvl.scene_list}")
        self.data_scenes = json.load(
            open(self.cfg.runners.mvl.scene_list))
        
        if self.cfg.runners.mvl.get("size", -1) > 0:
            self.list_scenes = list(self.data_scenes.keys())
            np.random.shuffle(self.list_scenes)
            self.list_scenes = self.list_scenes[:self.cfg.runners.mvl.size]
            return
        self.list_scenes = list(self.data_scenes.keys())

    def get_list_ly(self, idx=0, scene_name=""):
        """
        Returns a list of layout described by scene_name or scene idx
        """
        if scene_name == "":
            # When no scene_name has been passed
            scene_data = self.data_scenes[self.list_scenes[idx]]
        else:
            scene_data = self.data_scenes[scene_name]

        self.list_ly = []
        for frame in tqdm(scene_data, desc=f"Loading data scene..."):

            ly = Layout(self.cfg)
            ly.idx = os.path.splitext(frame)[0]

            if self.load_imgs:
                # try:
                ly.img = np.array(Image.open(os.path.join(self.DIR_IMGS, f"{ly.idx}.jpg")))
                # ly.img = imread(os.path.join(self.DIR_IMGS, f"{ly.idx}.jpg"))
                # except:
                #     print(os.path.join(self.DIR_IMGS, f"{ly.idx}.jpg"))
                #     input("Something went wrong ")
                     
            # ! Loading geometry
            geom = json.load(
                open(os.path.join(self.DIR_GEOM_FILES, f"{ly.idx}.json")))

            self.set_geom_info(layout=ly, geom=geom)
            
            # ! Setting in WC
            ly.cam_ref = 'WC'

            # ! Loading gt labels
            if self.load_npy:
                raise NotImplemented()
            self.list_ly.append(ly)            
        
        return self.list_ly

    @staticmethod
    def set_geom_info(layout, geom):
        
        layout.pose = CamPose(layout.cfg)
        layout.pose.t = np.array(geom['translation'])  # * geom['scale']
        qx, qy, qz, qw = geom['quaternion']
        q = Quaternion(qx=qx, qy=qy, qz=qz, qw=qw)
        layout.pose.rot = q.rotation_matrix
        layout.pose.idx = layout.idx
        layout.camera_height = geom.get(["cam_h"], 1)