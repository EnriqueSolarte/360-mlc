import torch.nn.functional as F
from mlc.utils.projection_utils import phi_coords2bearings
from shapely.geometry import Polygon
import numpy as np


def compute_L1_loss(y_est, y_ref):
        return F.l1_loss(y_est, y_ref)     
    
def compute_weighted_L1(y_est, y_ref, std, min_std=1E-2):
    return F.l1_loss(y_est/(std + min_std)**2, y_ref/(std + min_std)**2) 
    # return F.l1_loss(y_est, y_ref) 
    

def eval_2d3d_iuo(est_bon, gt_bon, losses, ch=-1.6):
    est_bearing_ceiling = phi_coords2bearings(est_bon[:, 0, :].squeeze())
    est_bearing_floor = phi_coords2bearings(est_bon[:, 1, :].squeeze())
    gt_bearing_ceiling = phi_coords2bearings(gt_bon[:, 0, :].squeeze())
    gt_bearing_floor = phi_coords2bearings(gt_bon[:, 1, :].squeeze())

    # Project bearings into a xz plane, ch: camera height
    est_scale_floor = ch / est_bearing_floor[1, :]
    est_pcl_floor = est_scale_floor * est_bearing_floor

    gt_scale_floor = ch / gt_bearing_floor[1, :]
    gt_pcl_floor = gt_scale_floor * gt_bearing_floor

    # Calculate height
    est_scale_ceiling = est_pcl_floor[2] / est_bearing_ceiling[2]
    est_pcl_ceiling = est_scale_ceiling * est_bearing_ceiling
    est_h = abs(est_pcl_ceiling[1, :].mean() - ch)

    gt_scale_ceiling = gt_pcl_floor[2] / gt_bearing_ceiling[2]
    gt_pcl_ceiling = gt_scale_ceiling * gt_bearing_ceiling
    gt_h = abs(gt_pcl_ceiling[1, :].mean() - ch)
    try:
        est_poly = Polygon(zip(est_pcl_floor[0], est_pcl_floor[2]))
        gt_poly = Polygon(zip(gt_pcl_floor[0], gt_pcl_floor[2]))
            
        if not gt_poly.is_valid:
            print("[ERROR] Skip ground truth invalid")
            return

        # 2D IoU
        try:
            area_dt = est_poly.area
            area_gt = gt_poly.area
            area_inter = est_poly.intersection(gt_poly).area
            iou2d = area_inter / (area_gt + area_dt - area_inter)
        except:
            iou2d = 0

        # 3D IoU
        try:
            area3d_inter = area_inter * min(est_h, gt_h)
            area3d_pred = area_dt * est_h
            area3d_gt = area_gt * gt_h
            iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
        except:
            iou3d = 0
    except:
        iou2d = 0
        iou3d = 0
        
    losses["2DIoU"].append(iou2d)
    losses["3DIoU"].append(iou3d)


def eval_mse(ref_mse, mse):
    
    diff_h = [v - mse[k] for k,v in ref_mse.items()]
    results = dict(
        max_diff_h = np.max(diff_h),
        max_diff_h_res = list(ref_mse.keys())[np.argmax(diff_h)],
        diff_h = diff_h,
        max_diff_h_idx = np.argmax(diff_h),
        sum_diff_h = np.sum(diff_h),
        mean_diff_h = np.mean(diff_h),
        std_diff_h = np.std(diff_h),
    )
    
    return results
    