import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class VO_ScaleRecover:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hist_entropy = []
        self.hist_scale = []

    def apply_vo_scale(self, scale):
        # ! note that every LY is referenced as WC_SO3_REF (only Rot was applied)
        return np.hstack(
            [
                obj.boundary_floor
                + (scale / obj.pose.vo_scale)
                * np.ones_like(obj.boundary_floor)
                * obj.pose.t.reshape(3, 1)
                for obj in self.list_ly
            ]
        )

    def reset_all(self):
        self.hist_entropy = []
        self.hist_scale = []
        self.hist_best_scale = []

    def estimate_scale(self, list_ly, max_scale, initial_scale, scale_step, plot=False):

        self.list_ly = list_ly
        scale = initial_scale
        self.reset_all()

        best_scale_hist = []
        # for c2f in tqdm(range(self.config.scale_recover.coarse_levels),
        #                 desc="...Estimating Scale"):
        for c2f in range(self.cfg.scale_recover.coarse_levels):
            scale = initial_scale
            self.reset_all()
            scale_step = (max_scale - initial_scale) / 10
            while True:
                # ! Applying scale
                pcl = self.apply_vo_scale(scale=scale)
                # ! Computing Entropy
                h = compute_entropy_from_pcl(
                    pcl=pcl, grid_size=self.cfg.scale_recover.grid_size,
                    min_likelihood=self.cfg.scale_recover.min_likelihood_percent,
                    max_ocg_size=self.cfg.scale_recover.max_ocg_size,
                )
                if h is None:
                    break
                
                if plot and self.hist_entropy.__len__() > 0:
                    grid, xedges, zedges = get_ocg_map(
                        pcl=pcl, grid_size=self.cfg.scale_recover.grid_size
                    )
                    grid = grid / np.max(grid)
                    fig = plt.figure("Optimization", figsize=(10, 4))
                    ax1 = fig.add_subplot(121)
                    ax1.clear()
                    ax1.set_title("OCG map @ scale:{0:0.4f}".format(scale))
                    ax1.imshow(grid)

                    ax2 = fig.add_subplot(122)
                    ax2.clear()
                    ax2.set_title("Entropy Optimization")
                    ax2.plot(self.hist_scale, self.hist_entropy)
                    idx_min = np.argmin(self.hist_entropy)
                    best_scale = self.hist_scale[idx_min]
                    ax2.scatter(
                        best_scale,
                        np.min(self.hist_entropy),
                        label="Best Scale:{0:0.2f}\nLowest H:{1:0.2f}".format(
                            best_scale, np.min(self.hist_entropy)
                        ),
                        c="red",
                    )
                    ax2.set_xlabel("Scale")
                    ax2.set_ylabel("Entropy")
                    ax2.grid()
                    plt.draw()
                    plt.waitforbuttonpress(0.01)
                    # if wait is None:
                    #     wait = input("\nPress enter: >>>>")

                self.hist_entropy.append(h)
                self.hist_scale.append(scale)
                scale += scale_step
                if scale > max_scale:
                    if (
                        np.max(self.hist_entropy) - np.min(self.hist_entropy)
                        < self.cfg.scale_recover.min_scale_variance
                    ):
                        best_scale_hist.append(0)
                    else:
                        idx_min = np.argmin(self.hist_entropy)
                        best_scale_hist.append(self.hist_scale[idx_min])

                    initial_scale = np.mean(best_scale_hist) - scale_step * 2
                    max_scale = np.mean(best_scale_hist) + scale_step * 2
                    break
        
        if best_scale_hist.__len__() == 0:
            return 0
        return np.mean(best_scale_hist)

    def estimate_by_searching_in_range(
        self, list_ly, max_scale, initial_scale, scale_step, plot=False
    ):
        assert (
            np.random.choice(list_ly, size=1)[0].cam_ref == "WC_SO3"
        ), "WC_SO3 references is need for Initial Guess in Scale Recovering"
        self.list_ly = list_ly

        scale = initial_scale
        self.reset_all()

        while True:
            # ! Applying scale
            pcl = self.apply_vo_scale(scale=scale)
            # ! Computing Entropy
            h = compute_entropy_from_pcl(
                pcl=pcl, grid_size=self.cfg.scale_recover.grid_size, 
                min_likelihood=self.cfg.scale_recover.min_likelihood_percent, 
                max_ocg_size=self.cfg.scale_recover.max_ocg_size
            )
            
            if h is None:
                best_scale = -1
                break
            
            if plot and self.hist_entropy.__len__() > 0:
                grid, xedges, zedges = get_ocg_map(
                    pcl=pcl, grid_size=self.cfg.scale_recover.grid_size
                )
                grid = grid / np.max(grid)
                fig = plt.figure("Optimization", figsize=(10, 4))
                plt.clf()
                ax1 = fig.add_subplot(121)
                ax1.clear()
                ax1.set_title("OCG map @ scale:{0:0.4f}".format(scale))
                ax1.imshow(grid)

                ax2 = fig.add_subplot(122)
                ax2.clear()
                ax2.set_title("Entropy Optimization")
                ax2.plot(self.hist_scale, self.hist_entropy)
                idx_min = np.argmin(self.hist_entropy)
                best_scale = self.hist_scale[idx_min]
                ax2.scatter(
                    best_scale,
                    np.min(self.hist_entropy),
                    label="Best Scale:{0:0.2f}\nLowest H:{1:0.2f}".format(
                        best_scale, np.min(self.hist_entropy)
                    ),
                    c="red",
                )
                ax2.set_xlabel("Scale")
                ax2.set_ylabel("Entropy")
                ax2.grid()
                plt.draw()
                plt.waitforbuttonpress(0.01)
                # if wait is None:
                #     wait = input("\nPress enter: >>>>")

            self.hist_entropy.append(h)
            self.hist_scale.append(scale)
            scale += scale_step
            if scale > max_scale:
                idx_min = np.argmin(self.hist_entropy)
                best_scale = self.hist_scale[idx_min]
                break
        return best_scale


def get_ocg_map(
    pcl, grid_size=None, weights=None, xedges=None, zedges=None, padding=100, clip=None
):
    """
    Compute a 2d histogram (ocg_map) for the passed PCL
    """
    x = pcl[0, :]
    z = pcl[2, :]

    if (xedges is None) or (zedges is None):
        xedges = np.mgrid[
            np.min(x)
            - padding * grid_size: np.max(x)
            + padding * grid_size: grid_size
        ]
        zedges = np.mgrid[
            np.min(z)
            - padding * grid_size: np.max(z)
            + padding * grid_size: grid_size
        ]

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights /= np.max(weights)

    grid, xedges, zedges = np.histogram2d(
        x, z, weights=1 / weights, bins=(xedges, zedges)
    )
    # grid = grid/np.sum(grid)
    # if clip is not None:
    #     mask = grid > clip
    #     grid[mask] = clip
    #     grid = grid / clip

    return grid, xedges, zedges

def filter_inf_points(pcl):
    pcl = pcl[:,~np.isnan(pcl[0, :])]
    center = np.median(pcl, axis=1)
    dist_meas = np.linalg.norm(pcl - center.reshape(3, 1), axis=0)
    mask = dist_meas < np.quantile(dist_meas, 0.99)
    return pcl[:, mask]
 
def compute_entropy_from_pcl(pcl, grid_size=0.1, weights=None, xedges=None, zedges=None, min_likelihood=0.1, max_ocg_size=10000):
    pcl = filter_inf_points(pcl)
    
    grid, _, _ = get_ocg_map(
        pcl=pcl, grid_size=grid_size, weights=weights, xedges=xedges, zedges=zedges
    )
    if np.max(grid.shape) > max_ocg_size: 
        return None
    return compute_entropy_from_ocg_map(grid, min_likelihood)


def compute_entropy_from_ocg_map(ocg_map, min_likelihood_percent=0.1):
    # mask = ocg_map > 0
    # # * Entropy
    # H = np.sum(-ocg_map[mask] * np.log2(ocg_map[mask]))
    # return H
    min_likelihood_percent = 0.000000001
    px = ocg_map.copy()
    px[px < min_likelihood_percent * px.max()] = min_likelihood_percent * px.max()
    px = px/np.sum(px)  # ! as density function
    return -np.sum(px * np.log2(px))
    # return np.sum((ocg_map/ocg_map.max()))
