import numpy as np
import skimage.filters
from tqdm import tqdm

from mlc.utils.projection_utils import xyz2uv


def draw_uncertainty_boundary(image, sigma_boundary, peak_boundary, color=(1, 0, 1)):
    H, W, C = image.shape
    for u,v, sigma in  zip(peak_boundary[0, :], peak_boundary[1, :], sigma_boundary[1, :]):
        sampled_points = np.random.normal(v, sigma, 100).astype(np.int)
        sampled_points[sampled_points >= H] = H-1
        sampled_points[sampled_points <= 0] = 0
        
        u_point = (np.ones_like(sampled_points) * u).astype(np.int)
        image[sampled_points, u_point, :] = np.array(color)
        image[sampled_points, u_point, :] = np.array(color)

    return image

def draw_uncertainty_map(sigma_boundary, peak_boundary, shape=(512, 1024)):
    H, W = shape
    img_map = np.zeros((H, W))
    for u,v, sigma in  zip(peak_boundary[0, :], peak_boundary[1, :], sigma_boundary):
        sigma_bon = (sigma / np.pi) * shape[0]

        sampled_points = np.random.normal(v, sigma_bon, 512).astype(np.int)
        sampled_points[sampled_points >= H] = H-1
        sampled_points[sampled_points <= 0] = 0
        
        u_point = (np.ones_like(sampled_points) * u).astype(np.int)
        img_map[sampled_points, u_point] = 1
       
    img_map = skimage.filters.gaussian(
                img_map, sigma=(15, 5), 
                truncate=5,
                channel_axis=True)
        
    img_map = img_map/img_map.max()
    return img_map

def draw_pcl_on_image(image, pcl, color=(0,255.0,0)):
    """Draws in the passed image the passed point cloud. 
    This function assumes that the passed point cloud is 
    already in the needed reference coordinates
    """
    if image.shape.__len__() == 3:
        H, W, C = image.shape
        uv = xyz2uv(pcl, (H, W))
        image[uv[1], uv[0], :] = np.array(color)
    else:
        H, W = image.shape
        uv = xyz2uv(pcl, (H, W))
        image[uv[1], uv[0]] = 255
    
    return image
    
def draw_boundaries_uv(image, boundary_uv, color=(0,1,0), size=2):
    if image.shape.__len__() == 3:
        for i in range(size):
            image[(boundary_uv[1]+i)% image.shape[0], boundary_uv[0], :] = np.array(color)
            # image[(boundary_uv[1]-i)% 0, boundary_uv[0], :] = np.array(color)
    else:
        for i in range(size):
            image[(boundary_uv[1]+i)% image.shape[0], boundary_uv[0]] = 255
            # image[(boundary_uv[1]-i)% 0, boundary_uv[0]] = 255
            
    return image

