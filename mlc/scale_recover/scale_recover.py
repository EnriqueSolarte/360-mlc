import numpy as np
from tqdm import tqdm
import yaml
from .vo_scale_recover import VO_ScaleRecover, get_ocg_map, filter_inf_points
import matplotlib.pyplot as plt
import os
import yaml
import skimage.filters
import logging


class ScaleRecover:
    def force_vo_scale(self, list_ly, vo_scale):
        self.list_ly = list_ly
        self.vo_scale = vo_scale
        self.apply_vo_scale(list_ly, vo_scale)

    def __init__(self, cfg):
        self.cfg = cfg
        self.hist_scales = []
        self.hist_vo_scales = []

        self.internal_idx = 0
        self.vo_scale_recover = VO_ScaleRecover(self.cfg)
        self.vo_scale = 1
        self.list_ly = None

    def get_next_batch(self):
        if self.cfg.scale_recover.random_batches:
            idx = np.random.randint(0, self.list_ly.__len__())
            batch = [self.list_ly[idx % self.list_ly.__len__()] for idx in range(
                idx, idx + self.cfg.scale_recover.sliding_windows)]
        else:
            batch = self.list_ly[
                self.internal_idx: self.internal_idx
                + self.cfg.scale_recover.sliding_windows
            ]
            self.internal_idx = (
                self.internal_idx + self.cfg.scale_recover.sliding_windows
            )
        # print(f"Reading Batch LY.idx's: {batch[0].idx} - {batch[-1].idx}")
        return batch

    @staticmethod
    def apply_vo_scale(list_ly, scale):
        [ly.apply_vo_scale(scale) for ly in list_ly]
        logging.info("VO-Scale {0:.6f} was successfully applied.".format(scale))

    def vo_scale_recovery_by_batches(self):
        """
        Estimates the vo-scale by linear search in a coarse-to-fine manner
        """

        max_scale = 50 * self.cfg.scale_recover.scale_step
        init_scale = -50 * self.cfg.scale_recover.scale_step

        for iteration in range(
            self.cfg.scale_recover.max_loops_iterations *
                self.list_ly.__len__()
        ):
            batch = self.get_next_batch()
            if batch.__len__() == 0:
                continue
            self.apply_vo_scale(batch, self.vo_scale)

            scale = self.vo_scale_recover.estimate_scale(
                # !Estimation using coarse-to-fine approach and only the last planes
                list_ly=batch,
                max_scale=max_scale,
                initial_scale=init_scale,
                scale_step=self.cfg.scale_recover.scale_step,
                plot=False,
            )
            self.update_vo_scale(self.vo_scale + scale)

            if self.cfg.scale_recover.early_stop:
                if self.hist_vo_scales.__len__() > self.cfg.scale_recover.min_estimations:
                    mean_vo_scale = np.mean(self.hist_vo_scales)

                    if abs(mean_vo_scale - self.hist_vo_scales[-1]) < self.cfg.scale_recover.min_vo_scale_diff:
                        print(" >>> Scale Recover Early Stop")
                        break

            if (
                self.internal_idx + self.cfg.scale_recover.sliding_windows
                >= self.list_ly.__len__()
            ):
                self.internal_idx = 0

            if iteration > self.list_ly.__len__() * 0.2:
                if (
                    np.std(self.hist_vo_scales[-100:])
                    < self.cfg.scale_recover.min_scale_variance
                ):
                    break

        # self.apply_vo_scale(self.list_ly, self.vo_scale)

        return True

    def recover_initial_vo_scale(self, batch=None):
        """
        Recovers an initial vo-scale (initial guess), which later will be used as
        a pivot to refine the global vo-scale
        """

        if batch is None:
            # ! Number of frames for initial scale recovering
            self.internal_idx = self.cfg.scale_recover.initial_batch
            batch = self.list_ly[: self.internal_idx]

        # > We need good LYs for initialize
        scale = self.vo_scale_recover.estimate_by_searching_in_range(
            list_ly=batch,
            max_scale=self.cfg.scale_recover.max_vo_scale,
            initial_scale=self.cfg.scale_recover.min_vo_scale,
            scale_step=self.cfg.scale_recover.scale_step,
            plot=False,
        )
        if scale < self.cfg.scale_recover.min_vo_scale:
            return False

        self.update_vo_scale(scale)

        self.apply_vo_scale(batch, self.vo_scale)
        return True

    def update_vo_scale(self, scale):
        """
        Sets an estimated scale to the system
        """
        self.hist_scales.append(scale)
        self.vo_scale = np.mean(self.hist_scales)
        self.hist_vo_scales.append(self.vo_scale)

    def fully_vo_scale_estimation(self, list_ly):
        """
        Recovers VO-scale by Entropy Minimization [360-DFPE]
        https://arxiv.org/abs/2112.06180
        """

        self.list_ly = list_ly

        logging.info("Estimating VO-SCALE")
        # ! For scale recovering every layout must be reference as WC_SO3 (only rotation)
        [ly.transform_to_WC_SO3() for ly in self.list_ly]

        # # ! Filtering out noisy estimation
        # thr_size = np.mean([ly.cam2boundary.max() for ly in self.list_ly])
        # self.list_ly = [ly for ly in self.list_ly if ly.cam2boundary.max() < 2* thr_size]

        try:
            self.recover_initial_vo_scale()
        except Exception as e:
            print(e)
            raise ValueError("Initial vo-scale failed")

        try:
            self.vo_scale_recovery_by_batches()
        except Exception as e:
            print(e)
            raise ValueError("Vo-scale recovering failed")

        self.apply_vo_scale(list_ly, self.vo_scale)

    def recompute_vo_scale(self, list_ly):
        self.list_ly = list_ly
        try:
            self.vo_scale_recovery_by_batches()
        except Exception as e:
            print(e)
            raise ValueError("Vo-scale recovering failed")

        self.apply_vo_scale(list_ly, self.vo_scale)
        
    def save_estimation(self, output_filename):
        """
        Saves the scale estimation into a directory
        """

        # ! Save image LY aligned
        plt.figure("Scale Recover", dpi=500)
        plt.clf()
        plt.title(
            f"{os.path.split(output_filename)[-1]}\nScale recover - VO-Scale:{self.vo_scale:0.3f}")

        pcl_ly = np.hstack([
            ly.boundary_floor for ly in self.list_ly])

        pcl_ly = filter_inf_points(pcl_ly)
        grid, _, _ = get_ocg_map(
            pcl_ly, grid_size=self.cfg.scale_recover.grid_size, padding=10, clip=None)
        grid = grid/grid.max()
        grid[grid == 0] = -1
        grid[grid > 0.1] = 1
        # grid[grid > self.cfg.scale_recover.vis_threhold] = 1
        grid = skimage.filters.gaussian(
            grid, sigma=(1, 1), channel_axis=True)

        plt.imshow(grid)
        plt.axis('off')
        plt.draw()

        plt.savefig(output_filename, bbox_inches='tight')
        # plt.savefig(f"{output_filename}.pdf", bbox_inches='tight')
