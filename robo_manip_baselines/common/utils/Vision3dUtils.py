import numpy as np
import pytorch3d.ops as torch3d_ops
import torch


def rotate_pointcloud(
    pointcloud: np.ndarray, rotation: np.ndarray, center: np.ndarray = None
):
    """
    Apply rotation to point cloud around a center (default: origin).

    Parameters:
        pointcloud: (N, 3) or (N, >=3) array
        rotation: (3, 3) rotation matrix
        center: (3,) array to rotate around (default: no translation)
    Returns:
        np.ndarray: rotated pointcloud (same shape as input)
    """
    xyz_array = pointcloud[:, :3]
    if center is not None:
        xyz_array = xyz_array - center
    xyz_array_rot = xyz_array @ rotation.T
    if center is not None:
        xyz_array_rot = xyz_array_rot + center

    pointcloud_rot = pointcloud.copy()
    pointcloud_rot[:, :3] = xyz_array_rot
    return pointcloud_rot


def crop_pointcloud_bb(pointcloud: np.ndarray, min_bound=None, max_bound=None):
    """
    Crop a point cloud using axis-aligned bounding box limits.

    Args:
        pointcloud (np.ndarray): input (N, 3) or (N, D) array.
        min_bound (array-like): lower (x, y, z) bound. Ignored if None.
        max_bound (array-like): upper (x, y, z) bound. Ignored if None.

    Returns:
        np.ndarray: cropped point cloud.
    """
    if min_bound is not None:
        mask = np.all(pointcloud[:, :3] > min_bound, axis=1)
        pointcloud = pointcloud[mask]
    if max_bound is not None:
        mask = np.all(pointcloud[:, :3] < max_bound, axis=1)
        pointcloud = pointcloud[mask]
    return pointcloud


def downsample_pointcloud_fps(pointcloud: np.ndarray, num_points: int = 512):
    """
    Downsample a point cloud using farthest point sampling (FPS).

    Args:
        pointcloud (np.ndarray): input (N, 3) or (N, D) array.
        num_points (int): number of points to sample.

    Returns:
        np.ndarray: downsampled point cloud of shape (num_points, D).
    """
    pointcloud_tensor = torch.from_numpy(pointcloud).unsqueeze(0)
    num_points_tensor = torch.tensor([num_points])
    _, sampled_indices = torch3d_ops.sample_farthest_points(
        pointcloud_tensor, K=num_points_tensor
    )
    pointcloud = pointcloud[sampled_indices.squeeze(0).numpy()]
    return pointcloud
