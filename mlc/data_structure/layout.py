import numpy as np

from .cam_pose import CAM_REF
from mlc.utils.geometry_utils import extend_array_to_homogeneous
from mlc.utils.projection_utils import phi_coords2bearings


class Layout:
    def __init__(self, cfg):
        self.cfg = cfg

        self.boundary_floor = None
        self.boundary_ceiling = None

        self.cam2boundary = None
        self.cam2boundary_mask = None

        self.bearings_floor = None
        self.bearings_ceiling = None

        self.img = None
        self.pose = None
        self.idx = None

        self.phi_coord = None
        self.cam_ref = CAM_REF.CC
        self.ceiling_height = None  # ! Must be None by default
        self.camera_height = 1
        self.primary = False
        self.scale = 1

    def apply_vo_scale(self, scale):

        if self.cam_ref == CAM_REF.WC_SO3:
            self.boundary_floor = self.boundary_floor + \
                (scale/self.pose.vo_scale) * \
                np.ones_like(self.boundary_floor) * self.pose.t.reshape(3, 1)

            self.boundary_ceiling = self.boundary_ceiling + \
                (scale/self.pose.vo_scale) * \
                np.ones_like(self.boundary_ceiling) * self.pose.t.reshape(3, 1)

            self.cam_ref = CAM_REF.WC

        elif self.cam_ref == CAM_REF.WC:
            delta_scale = scale - self.pose.vo_scale
            self.boundary_floor = self.boundary_floor + \
                (delta_scale/self.pose.vo_scale) * \
                np.ones_like(self.boundary_floor) * self.pose.t.reshape(3, 1)

            self.boundary_ceiling = self.boundary_ceiling + \
                (delta_scale/self.pose.vo_scale) * \
                np.ones_like(self.boundary_ceiling) * self.pose.t.reshape(3, 1)

        self.pose.vo_scale = scale

        return True

    def estimate_height_ratio(self):
        """
        Estimates the height ratio that describes the distance ratio of camera-floor over the
        camera-ceiling distance. This information is important to recover the 3D
        structure of the predicted Layout
        """
        floor = np.abs(self.ly_data[1, :])
        ceiling = np.abs(self.ly_data[0, :])

        ceiling[ceiling > np.radians(80)] = np.radians(80)
        ceiling[ceiling < np.radians(5)] = np.radians(5)
        floor[floor > np.radians(80)] = np.radians(80)
        floor[floor < np.radians(5)] = np.radians(5)

        self.height_ratio = np.mean(np.tan(ceiling)/np.tan(floor))

    def compute_cam2boundary(self):
        """
        Computes the horizontal distance for every boundary point w.r.t camera pose. 
        The boundary can be in any reference coordinates
        """
        if self.cam_ref == CAM_REF.WC_SO3 or self.cam_ref == CAM_REF.CC:
            # ! Boundary reference still in camera reference
            self.cam2boundary = np.linalg.norm(self.boundary_floor[(0, 2), :], axis=0)
    
        else:
            assert self.cam_ref == CAM_REF.WC
            pcl = np.linalg.inv(self.pose.SE3_scaled())[
                :3, :] @ extend_array_to_homogeneous(self.boundary_floor)
            self.cam2boundary = np.linalg.norm(pcl[(0, 2), :], axis=0)
    
        # self.cam2boundary_mask = np.zeros_like(self.cam2boundary)
        # self.cam2boundary_mask = self.cam2boundary < np.quantile(self.cam2boundary, 0.25)
        
    def recompute_data(self, phi_coord=None):
        if phi_coord is not None:
            self.phi_coord = phi_coord
            
        # ! Compute bearings
        self.bearings_ceiling = phi_coords2bearings(
            phi_coords=self.phi_coord[0, :])
        self.bearings_floor = phi_coords2bearings(
            phi_coords=self.phi_coord[1, :])

        # ! Compute floor boundary
        ly_scale = self.camera_height / self.bearings_floor[1, :]
        pcl = ly_scale * self.bearings_floor * self.scale
        self.cam_ref = CAM_REF.WC
        self.boundary_floor = self.pose.SE3_scaled()[:3,
                                            :] @ extend_array_to_homogeneous(pcl)

        # from mlc.utils.vispy_utils.vispy_utils import plot_pcl
        # ! Compute ceiling boundary
        if self.ceiling_height is None:
            # ! forcing consistency between floor and ceiling
            scale_ceil = np.linalg.norm(
                pcl[(0, 2), :], axis=0) / np.linalg.norm(self.bearings_ceiling[(0, 2), :], axis=0)
            pcl = scale_ceil * self.bearings_ceiling
            # plot_pcl(pcl)
        else:
            ly_scale = (self.ceiling_height-self.camera_height) / \
                self.bearings_ceiling[1, :]
            pcl = ly_scale * self.bearings_ceiling * self.scale

        self.boundary_ceiling = self.pose.SE3_scaled()[:3,:] @ extend_array_to_homogeneous(pcl)
        self.compute_cam2boundary()
        
    def transform_to_WC_SO3(self):

        self.boundary_ceiling = self.boundary_ceiling - \
            self.pose.t.reshape(3, 1)
        self.boundary_floor = self.boundary_floor - self.pose.t.reshape(3, 1)

        self.cam_ref = CAM_REF.WC_SO3
        