from __future__ import division
import numpy as np
import math


def bearings2layout_3d(bound, scale=1):
    """
    Projects a set of bearing vectors defined at the unit sphere into a 3D layout
    """

    ly_scale = scale/bound[1, :]
    pcl = ly_scale * bound

    return pcl


def uv2xyz(uv, shape=(512, 1024)):
    """
    Projects uv vectors to xyz vectors (bearing vector)
    """
    sph = uv2sph(uv, shape)
    theta = sph[0]
    phi = sph[1]

    x = np.cos(phi) * np.sin(theta)
    y = -np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    return np.vstack((x, y, z))


def uv2sph(uv, shape):
    """
    Projects a set of uv points into spherical coordinates (theta, phi)
    """
    H, W = shape
    theta = 2 * np.pi * ((uv[0] + 0.5) / W - 0.5)
    phi = np.pi * ((uv[1] + 0.5) / H - 0.5)
    return np.vstack((theta, phi))

def sph2xyz(sph):
    """
    Projects spherical coordinates (theta, phi) to euclidean space xyz
    """
    theta = sph[:, 0]
    phi = sph[:, 1]

    x = math.cos(phi) * math.sin(theta)
    y = -math.sin(phi)
    z = math.cos(phi) * math.cos(theta)

    return np.vstack((x, y, z))



def sph2uv(sph, shape):
    H, W = shape
    u = W * (sph[0]/(2*np.pi) + 0.5)
    v = H * (sph[1]/np.pi + 0.5)
    return np.vstack((
        np.clip(u, 0, W-1),
        np.clip(v, 0, H-1)
    )).astype(int)


def phi_coords2bearings(phi_coords):
    """
    Returns 3D bearing vectors (on the unite sphere) from phi_coords
    """
    W = phi_coords.__len__()
    u = np.linspace(0, W - 1, W)
    theta_coords = (2 * np.pi * u / W) - np.pi
    bearings_y = -np.sin(phi_coords)
    bearings_x = np.cos(phi_coords) * np.sin(theta_coords)
    bearings_z = np.cos(phi_coords) * np.cos(theta_coords)
    return np.vstack((bearings_x, bearings_y, bearings_z))


def phi_coords2uv(phi_coord, shape=(512, 1024)):
    """
    Converts a set of phi_coordinates (2, W), defined by ceiling and floor boundaries encoded as 
    phi coordinates, into uv pixels 
    """
    H, W = shape
    u = np.linspace(0, W - 1, W)
    theta_coords = (2 * np.pi * u / W) - np.pi
    uv_c = sph2uv(np.vstack((theta_coords, phi_coord[0])), shape)
    uv_f = sph2uv(np.vstack((theta_coords, phi_coord[1])), shape)

    return uv_c, uv_f
     

def xyz2uv(xyz, shape=(512, 1024)):
    """
    Projects XYZ array into uv coord
    """
    xyz_n = xyz / np.linalg.norm(xyz, axis=0, keepdims=True)

    normXZ = np.linalg.norm(xyz[(0, 2), :], axis=0, keepdims=True)

    phi_coord = -np.arcsin(xyz_n[1, :])
    theta_coord = np.sign(xyz[0, :]) * np.arccos(xyz[2, :] / normXZ)

    u = np.clip(np.floor((0.5 * theta_coord / np.pi + 0.5) * shape[1] + 0.5), 0, shape[1] - 1)
    v = np.clip(np.floor((phi_coord / np.pi + 0.5) * shape[0]+0.5), 0, shape[0] - 1)
    return np.vstack((u, v)).astype(int)


def xyz2sph(xyz):
    """
    Project xyz coordinates into spherical coordinates (theta, phi)
    """
    normXZ = math.sqrt(math.pow(xyz[0], 2) + math.pow(xyz[2], 2))
    if normXZ < 0.000001:
        normXZ = 0.000001

    normXYZ = math.sqrt(math.pow(xyz[0], 2) + math.pow(xyz[1], 2) + math.pow(xyz[2], 2))

    # ! Spherical coordinates
    phi = -math.asin(xyz[1] / normXYZ)
    theta = math.acos(xyz[2] / normXZ)

    # # ! Don't needed anymore, since we are using acos (-1, 1)--> (-pi, pi)
    # if xyz[2] > 0 and theta > 0:
    #     theta = math.pi - theta
    # elif xyz[2] > 0 and theta < 0:
    #     theta = -math.pi - theta

    # if xyz[0] == 0 and xyz[2] > 0:
    #     theta = math.pi

    uv = (theta, phi)
    return uv


