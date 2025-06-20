# Functions for reading data from .txt files (ShapeNet-SRN) 
# and .npy files (CO3D)

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from utils.graphics_utils import focal2fov, fov2focal


class CameraInfoCo3d(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class CameraInfoShapeNet(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth_map: np.array
    mask_map: torch.Tensor
    img_xyz: torch.Tensor
    image_path: str
    image_name: str
    depth_path: str
    mask_path: str
    width: int
    height: int


def pixel_to_world(depth, K, w2c):
    """

    Args:
        depth: [128, 128]
        K: [3, 3]
        w2c: [4, 4], the last row is [0, 0, 0, 1]

    Returns: [128, 128, 3]

    """
    height, width = depth.shape

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    pixels = np.stack((x.flatten(), y.flatten(), np.ones_like(x.flatten())), axis=-1).T

    camera_coords = np.linalg.inv(K) @ pixels

    camera_coords *= depth.flatten()

    camera_coords = np.vstack((camera_coords, np.ones((1, camera_coords.shape[1]))))

    world_coords = np.linalg.inv(w2c) @ camera_coords

    world_coords = world_coords / world_coords[3]

    world_coords = world_coords[:3].T
    world_coords = world_coords.reshape(height, width, -1)

    return world_coords


def improved_normalize_with_mask(seen_xyz, mask):

    mask_3d = mask.unsqueeze(-1).expand_as(seen_xyz)


    foreground_points = seen_xyz[~mask_3d].view(-1, 3)

    mean = foreground_points.mean(dim=0)
    std = foreground_points.std(dim=0)

    normalized_xyz = (seen_xyz - mean) / (std + 1e-8)

    background_scale = 0.3
    normalized_xyz[mask_3d] *= background_scale

    normalized_xyz = torch.tanh(normalized_xyz)

    return normalized_xyz

def normalize(seen_xyz):
    seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    return seen_xyz


def readCamerasFromTxt(rgb_paths, pose_paths, depth_paths, mask_paths, K, idxs):
    cam_infos = []
    # Transform fov from degrees to radians
    fovx = 51.98948897809546 * 2 * np.pi / 360

    for idx in idxs:
        cam_name = pose_paths[idx]
        # SRN cameras are camera-to-world transforms
        # no need to change from SRN camera axes (x right, y down, z away) 
        # it's the same as COLMAP (x right, y down, z forward)
        c2w = np.loadtxt(cam_name, dtype=np.float32).reshape(4, 4)

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code，(Transpose R in the loading phase, but to calculate w2c matrix, it transpose it back...)
        T = w2c[:3, 3]

        image_path = rgb_paths[idx]
        image_name = Path(cam_name).stem
        # SRN images already are RGB with white background
        image = Image.open(image_path)

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        # new
        depth_path = depth_paths[idx]
        depth_map = np.array(Image.open(depth_path))

        mask_path = mask_paths[idx]
        mask_map = Image.open(mask_path)
        mask_map = np.array(mask_map).sum(-1) < 127 #(low) True indicates bg
        mask_map = torch.from_numpy(mask_map)

        depth_map[mask_map] = 255

        world_coords = pixel_to_world(depth_map, K, w2c) #[128, 128, 3]
        world_coords[mask_map] = float('inf')
        # world_coords = improved_normalize_with_mask(torch.from_numpy(world_coords), mask_map)

        world_coords = normalize(torch.from_numpy(world_coords))

        # world_coords = world_coords.reshape(-1, 3)
        # colors = np.array(image)[..., :3].reshape(-1, 3)
        # import plotly.graph_objects as go
        # import plotly.io as pio
        # pio.renderers.default = "browser"
        #
        # fig = go.Figure(data=[go.Scatter3d(
        #     x=world_coords[:, 0],
        #     y=world_coords[:, 1],
        #     z=world_coords[:, 2],
        #     mode='markers',
        #     marker=dict(
        #         size=2,
        #         color=['rgb({},{},{})'.format(r, g, b) for r, g, b in colors],
        #         opacity=0.8
        #     )
        # )])
        #
        # fig.update_layout(
        #     scene=dict(
        #         xaxis_title='X',
        #         yaxis_title='Y',
        #         zaxis_title='Z',
        #         aspectmode='data'
        #     ),
        #     width=800,
        #     height=800,
        #     title='3D Point Cloud Visualization'
        # )
        #
        # fig.show()


        #
        cam_infos.append(CameraInfoShapeNet(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth_map=depth_map, mask_map=mask_map, img_xyz=world_coords.permute(-1, 0, 1),
                        image_path=image_path, image_name=image_name, depth_path=depth_path, mask_path=mask_path,
                                    width=image.size[0], height=image.size[1]))
        
    return cam_infos

def readCamerasFromNpy(folder_path, 
                       w2c_Rs_rmo=None, 
                       w2c_Ts_rmo=None, 
                       focals_folder_path=None):
    # Set every_5th_in for the testing set
    cam_infos = []
    # Transform fov from degrees to radians
    fname_order_path = os.path.join(folder_path, "frame_order.txt")
    c2w_T_rmo_path = os.path.join(folder_path, "c2w_T_rmo.npy")
    c2w_R_rmo_path = os.path.join(folder_path, "c2w_R_rmo.npy")
    if focals_folder_path is None:
        focals_folder_path = folder_path
    focal_lengths_path = os.path.join(focals_folder_path, "focal_lengths.npy")

    with open(fname_order_path, "r") as f:
        fnames = f.readlines()
    fnames = [fname.split("\n")[0] for fname in fnames]

    if w2c_Ts_rmo is None:
        c2w_T_rmo = np.load(c2w_T_rmo_path)
    if w2c_Rs_rmo is None:
        c2w_R_rmo = np.load(c2w_R_rmo_path)
    focal_lengths = np.load(focal_lengths_path)[:, 0, :]

    camera_transform_matrix = np.eye(4)
    camera_transform_matrix[0, 0] *= -1
    camera_transform_matrix[1, 1] *= -1
    
    # assume shape 128 x 128
    image_side = 128

    for f_idx, fname in enumerate(fnames):

        w2c_template = np.eye(4)
        if w2c_Rs_rmo is None:
            w2c_R = np.transpose(c2w_R_rmo[f_idx])
        else:
            w2c_R = w2c_Rs_rmo[f_idx]
        if w2c_Ts_rmo is None:
            w2c_T = - np.matmul(c2w_T_rmo[f_idx], w2c_R)
        else:
            w2c_T = w2c_Ts_rmo[f_idx]
        w2c_template[:3, :3] = w2c_R
        # at this point the scene scale is approx. that of shapenet cars
        w2c_template[3:, :3] = w2c_T

        # Pytorch3D cameras have (x left, y right, z away axes)
        # need to transform to COLMAP / OpenCV (x right, y down, z forward)
        # transform axes and transpose to column major order
        w2c = np.transpose(np.matmul(w2c_template, camera_transform_matrix))

        # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_name = fname.split(".png")

        focal_lengths_ndc = focal_lengths[f_idx]
        focal_lengths_px = focal_lengths_ndc * image_side / 2

        FovY = focal2fov(focal_lengths_px[1], image_side) 
        FovX = focal2fov(focal_lengths_px[0], image_side)

        cam_infos.append(CameraInfoCo3d(uid=fname, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                        image_path=None, image_name=image_name, width=image_side, height=image_side))
        
    return cam_infos
