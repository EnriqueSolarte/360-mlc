import numpy as np
from mlc.scale_recover.vo_scale_recover import get_ocg_map


def masking_entropy_map(ocg_map):
    """
    Masking zero values in the ocg map 
    """
    # q = np.quantile(ocg_map[ocg_map > 0], 0.5)
    new_map = ocg_map.copy()
    new_map[new_map == 0] = -new_map.max()
    new_map = new_map / new_map.max()
    new_map[new_map > 0.1] = 1
    return new_map


def filter_inf_points(pcl):
    center = np.median(pcl, axis=1)
    dist_meas = np.linalg.norm(pcl - center.reshape(3, 1), axis=0)
    mask = dist_meas < np.quantile(dist_meas, 0.99)
    return pcl[:, mask]

def eval_entropy_from_boundaries(list_boundaries, xedges=None, zedges=None, grid_size=0.1, min_likelihood=1E-4, padding=10):
    """
    Evaluates the entropy for a list of boundaries. A boundary is defined as 
    the projection in 2D of a layout
    """

    pcl_xyz = np.hstack(list_boundaries)
    pcl_xyz = filter_inf_points(pcl_xyz)
    ocg_map, xedges, zedges = get_ocg_map(
        pcl_xyz, zedges=zedges, xedges=xedges, grid_size=grid_size, padding=padding, clip=None)

    entropy = eval_entropy(ocg_map, min_likelihood)

    return dict(xzedges=(xedges, zedges), ocg_map=ocg_map, entropy=entropy)


def pdf_normalization(ocg_map):
    if np.sum(ocg_map) == 1:
        return ocg_map
    else:
        ocg_map = ocg_map/np.sum(ocg_map)
    return ocg_map


def max_normalization(ocg_map):
    ocg_map = ocg_map/np.max(ocg_map)
    return ocg_map


def eval_entropy(ocg_map, min_likelihood=1E-4):
    # px = ocg_map.copy()
    # px[px < min_likelihood * px.max()] = min_likelihood * px.max()
    # px = px/np.sum(px)  # ! as density function
    px = pdf_normalization(ocg_map)
    mask = px > 0
    return -np.sum(px[mask] * np.log2(px[mask]))


def eval_cross_entropy(ocg_ref, ocg, min_likelihood=1E-4):
    # px = ocg_ref.copy()
    # px[px < min_likelihood * px.max()] = min_likelihood * px.max()
    # px = px/np.sum(px)  # ! as density function
    px = pdf_normalization(ocg_ref)

    # qx = ocg.copy()
    # qx[qx < min_likelihood * qx.max()] = min_likelihood * qx.max()
    # qx = qx / np.sum(qx)
    qx = pdf_normalization(ocg)

    mask1 = px > 0
    mask2 = qx > 0

    return - np.sum(px[mask1*mask2] * np.log2(qx[mask1*mask2]))


def eval_kl_div(ocg_ref, ocg, min_likelihood=1E-4):

    # px = ocg_ref.copy()
    # px[px < min_likelihood * px.max()] = min_likelihood * px.max()
    # px = px/np.sum(px)  # ! as density function
    px = pdf_normalization(ocg)

    # qx = ocg.copy()
    # qx[qx < min_likelihood * qx.max()] = min_likelihood * qx.max()
    # qx = qx / np.sum(qx)
    qx = pdf_normalization(ocg_ref)

    mask_px = px > 0
    mask_qx = qx > 0

    entropy_ref = np.sum(px[mask_px] * np.log2(px[mask_px]))
    entropy_div = np.sum(px[mask_qx] * np.log2(qx[mask_qx]))
    kl_div = entropy_ref - entropy_div
    return kl_div


def eval_jsd_div(ocg_ref, ocg, min_likelihood=1E-4):
    """
    https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    """
    m = (ocg_ref + ocg)*0.5
    m = pdf_normalization(m)
    jsd = 0.5 * eval_kl_div(ocg_ref, m, min_likelihood) + 0.5 * eval_kl_div(ocg, m, min_likelihood)
    return jsd


def eval_divergence_from_boundaries(list_boundaries, entropy_ref):
    """
    Computes the LK-div for two list of boundaries. A boundary is defined as 
    the projection in 2D of a layout
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """

    xedges, zedges, ocg_map_ref = entropy_ref['xzedges'][0], entropy_ref['xzedges'][1], entropy_ref['ocg_map']

    pcl_xyz = np.hstack(list_boundaries)
    pcl_xyz = filter_inf_points(pcl_xyz)

    ocg_map, _, _, = get_ocg_map(
        pcl_xyz, zedges=zedges, xedges=xedges, clip=None)

    h_ref = eval_entropy(ocg_map_ref)
    h_curr = eval_entropy(ocg_map)

    kl_div = eval_kl_div(
        ocg_ref=ocg_map_ref,
        ocg=ocg_map,
    )

    cross_h = eval_cross_entropy(
        ocg_ref=ocg_map_ref,
        ocg=ocg_map,
    )

    jsd_div = eval_jsd_div(
        ocg_ref=ocg_map_ref,
        ocg=ocg_map,
    )

    return dict(xaedges=(xedges, zedges),
                ocg_map_ref=ocg_map_ref,
                ocg_map=ocg_map,
                kl_div=kl_div,
                cross_h=cross_h,
                jsd_div=jsd_div,
                h_ref=h_ref,
                entropy=h_curr
                )
