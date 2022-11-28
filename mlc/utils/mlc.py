import numpy as np
import logging
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
from mlc.scale_recover.scale_recover import ScaleRecover
from mlc.utils.image_utils import draw_boundaries_uv, draw_uncertainty_map
from mlc.utils.color_utils import *
from mlc.utils.layout_utils import filter_out_noisy_layouts
from mlc.utils.geometry_utils import extend_array_to_homogeneous, stack_camera_poses
from mlc.utils.projection_utils import uv2xyz, uv2sph, xyz2uv
from mlc.utils.bayesian_utils import apply_kernel
from mlc.models.utils import load_layout_model
from mlc.datasets.utils import load_mvl_dataset


def reproject_ly_boundaries(list_ly, ref_ly, shape=(512, 1024)):
    """
    Reproject a list of layouts into the passed reference using comprehension lists
    """
    pose_W2ref = np.linalg.inv(ref_ly.pose.SE3_scaled())

    boundaries_xyz = []
    [(boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(ly.boundary_floor)),
      boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(ly.boundary_ceiling)))
     for ly in list_ly]

    # [(boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(
    #     ly.boundary_floor[:, ly.cam2boundary < np.quantile(ly.cam2boundary, 0.5)])),
    #   boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(
    #       ly.boundary_ceiling[:, ly.cam2boundary < np.quantile(ly.cam2boundary, 0.5)])))
    #  for ly in list_ly]

    # [boundaries_xyz.append(pose_W2ref[:3, :] @ extend_array_to_homogeneous(ly.boundary_ceiling))
    #  for ly in list_ly]

    uv = xyz2uv(np.hstack(boundaries_xyz), shape)

    return uv


def compute_pseudo_labels(list_frames, ref_frame, shape=(512, 1024)):
    """
    Computes the pseudo labels based on the passed list of frames wrt the ref_frame
    """
    cfg = ref_frame.cfg

    uv = reproject_ly_boundaries(list_frames, ref_frame, shape)

    uv[0] = np.clip(uv[0], 0, shape[1] - 1)
    uv[1] = np.clip(uv[1], 0, shape[0] - 1)
    # ! Aggregating projected points into a histogram
    uv, counts = np.unique(list(uv.T), return_counts=True, axis=0)

    prj_map = np.zeros(shape)
    prj_map[uv[:, 1], uv[:, 0]] = counts

    _prj_map = prj_map.copy()

    # ! Floor
    floor_map = _prj_map[512//2:, :]
    v_floor = np.argmax(floor_map, axis=0)
    std_floor = get_std(floor_map, v_floor, cfg.kernels_mlc.std_kernel)
    v_floor += 512//2

    # ! Ceiling
    ceiling_map = _prj_map[:512//2, :]
    v_ceiling = np.argmax(ceiling_map, axis=0)
    std_ceiling = get_std(ceiling_map, v_ceiling, cfg.kernels_mlc.std_kernel)

    u = np.linspace(0, ceiling_map.shape[1]-1, ceiling_map.shape[1]).astype(int)
    return np.vstack((u, v_ceiling)), np.vstack((u, v_floor)), std_ceiling, std_floor, _prj_map


def get_std(hw_map, peak, kernel):
    """
    https://www.statology.org/histogram-standard-deviation/
    https://math.stackexchange.com/questions/857566/how-to-get-the-standard-deviation-of-a-given-histogram-image
    """
    # ! To avoid zero STD
    hw_map = apply_kernel(hw_map.copy(), size=(kernel[0], kernel[1]), sigma=kernel[2])
    m = np.linspace(0, hw_map.shape[0]-1, hw_map.shape[0]) + 0.5
    miu = np.repeat(peak.reshape(1, -1), hw_map.shape[0], axis=0)
    mm = np.repeat(m.reshape(-1, 1), hw_map.shape[1], axis=1)
    N = np.sum(hw_map, axis=0)
    std = np.sqrt(np.sum(hw_map*(mm-miu)**2, axis=0)/N)
    return (std / hw_map.shape[0]) * np.pi * 0.5


def median(c):
    px_val = c[c > 0]
    med_val = np.median(px_val)
    std_val = np.std(px_val)
    v_val = np.argmin(abs(c - med_val))
    return v_val, std_val


def iterator_room_scenes(cfg):
    """
    Creates a generator which yields a list of layout from a defined 
    dataset.
    """
    model = load_layout_model(cfg)
    dataset = load_mvl_dataset(cfg)
    scale_recover = ScaleRecover(cfg)
    dataset.load_imgs = True
    dataset.load_gt_labels = False
    dataset.load_npy = False

    # ! Data Selection from cfg.data_selection cfg
    if cfg.get("data_selection.active", False):
        data = json.load(open(cfg.data_selection.selected_scenes_file))
        selected_scene = data[cfg.data_selection.quantile]

    for scene in tqdm(dataset.list_scenes, desc="Reading MVL scenes..."):
        if cfg.get("data_selection.active", False):
            if scene not in selected_scene:
                continue
            logging.warning(f"Forcing selected scenes only")
            logging.warning(f"Selection Scenes from{cfg.data_selection.selected_scenes_file}")

        cfg._room_scene = scene
        logging.info(f"Scene Name: {scene}")
        logging.info(f"Experiment ID: {cfg.id_exp}")
        logging.info(f"Output_dir: {cfg.output_dir}")
        list_ly = dataset.get_list_ly(scene_name=scene)

        # ! Overwrite phi_coord by the estimates
        model.estimate_within_list_ly(list_ly)
        filter_out_noisy_layouts(
            list_ly=list_ly,
            max_room_factor_size=cfg.runners.mvl.max_room_factor_size
        )
        if cfg.runners.mvl.apply_scale_recover:
            scale_recover.fully_vo_scale_estimation(list_ly=list_ly)

        yield list_ly


def compute_and_store_mlc_labels(list_ly, save_vis=False):
    _output_dir = ref.cfg.output_dir
    _cfg = ref.cfg
    for ref in list_ly:
        uv_ceiling_ps, uv_floor_ps, std_ceiling, std_floor, prj_map = compute_pseudo_labels(
            list_frames=sampling_frames(list_ly),
            ref_frame=ref,
        )

        # ! Saving pseudo labels
        uv = np.vstack((uv_ceiling_ps[1], uv_floor_ps[1]))
        std = np.vstack((std_ceiling, std_floor))
        phi_bon = (uv / 512 - 0.5) * np.pi
        np.save(os.path.join(_output_dir, "label", _cfg.runners.mvl.label, ref.idx), phi_bon)
        np.save(os.path.join(_output_dir, "label", "std", ref.idx), std)

        if not save_vis:
            return

        uv_ceiling_hat = xyz2uv(ref.bearings_ceiling)
        uv_floor_hat = xyz2uv(ref.bearings_floor)
        img = ref.img.copy()

        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_ceiling_hat,
            color=COLOR_CYAN
        )
        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_floor_hat,
            color=COLOR_CYAN
        )

        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_ceiling_ps,
            color=COLOR_MAGENTA
        )
        draw_boundaries_uv(
            image=img,
            boundary_uv=uv_floor_ps,
            color=COLOR_MAGENTA
        )

        sigma_map = draw_uncertainty_map(
            peak_boundary=np.hstack((uv_ceiling_ps, uv_floor_ps)),
            sigma_boundary=np.hstack((std_ceiling, std_floor))
        )

        plt.figure(0, dpi=200)
        plt.clf()
        plt.subplot(211)
        plt.suptitle(ref.idx)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(sigma_map)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.savefig(os.path.join(_output_dir, "vis",
                    f"{ref.idx}.jpg"), bbox_inches='tight')
