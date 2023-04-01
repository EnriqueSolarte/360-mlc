import numpy as np
import skimage.filters
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from mlc.utils.projection_utils import xyz2uv, phi_coords2bearings
import matplotlib.pyplot as plt


def draw_uncertainty_boundary(image, sigma_boundary, peak_boundary, color=(1, 0, 1)):
    H, W, C = image.shape
    for u, v, sigma in zip(peak_boundary[0, :], peak_boundary[1, :], sigma_boundary[1, :]):
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
    for u, v, sigma in zip(peak_boundary[0, :], peak_boundary[1, :], sigma_boundary):
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


def draw_pcl_on_image(image, pcl, color=(0, 255.0, 0)):
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


def draw_boundaries_uv(image, boundary_uv, color=(0, 1, 0), size=2):
    if image.shape.__len__() == 3:
        for i in range(size):
            image[(boundary_uv[1]+i) % image.shape[0], boundary_uv[0], :] = np.array(color)
            # image[(boundary_uv[1]-i)% 0, boundary_uv[0], :] = np.array(color)
    else:
        for i in range(size):
            image[(boundary_uv[1]+i) % image.shape[0], boundary_uv[0]] = 255
            # image[(boundary_uv[1]-i)% 0, boundary_uv[0]] = 255

    return image


def draw_boundaries_phi_coords(image, phi_coords, color=(0, 255, 0), size=2):

    # ! Compute bearings
    bearings_ceiling = phi_coords2bearings(phi_coords=phi_coords[0, :])
    bearings_floor = phi_coords2bearings(phi_coords=phi_coords[1, :])

    uv_ceiling = xyz2uv(bearings_ceiling)
    uv_floor = xyz2uv(bearings_floor)

    draw_boundaries_uv(image=image, boundary_uv=uv_ceiling, color=color, size=size)
    draw_boundaries_uv(image=image, boundary_uv=uv_floor, color=color, size=size)

    return image


def add_caption_to_image(image, caption):
    img_obj = Image.fromarray(image)
    img_draw = ImageDraw.Draw(img_obj)
    font_obj = ImageFont.truetype("FreeMono.ttf", 20)
    img_draw.text((20, 20), f"{caption}", font=font_obj, fill=(255, 0, 0))
    return np.array(img_obj)


def plot_image(image, caption, figure=0):
    plt.figure(figure)
    plt.clf()
    image = add_caption_to_image(image, caption)
    plt.imshow(image)
    plt.draw()
    plt.waitforbuttonpress(0.01)
