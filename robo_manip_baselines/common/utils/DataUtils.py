from functools import reduce

import numpy as np

from ..data.DataKey import DataKey
from .MathUtils import get_rel_pose_from_se3, get_se3_from_rel_pose


def normalize_data(data, stats):
    """Normalize data."""
    if "norm_config" in stats:
        norm_type = stats["norm_config"]["type"]
    else:
        norm_type = "gaussian"

    if norm_type == "gaussian":
        return (data - stats["mean"]) / stats["std"]
    elif norm_type == "limits":
        norm_config = stats["norm_config"]
        scale = (norm_config["out_max"] - norm_config["out_min"]) / stats["range"]
        return scale * (data - stats["min"]) + norm_config["out_min"]
    else:
        raise ValueError(f"[normalize_data] Invalid normalization type: {norm_type}")


def denormalize_data(data, stats):
    """Denormalize data."""
    if "norm_config" in stats:
        norm_type = stats["norm_config"]["type"]
    else:
        norm_type = "gaussian"

    if norm_type == "gaussian":
        return stats["std"] * data + stats["mean"]
    elif norm_type == "limits":
        norm_config = stats["norm_config"]
        scale = stats["range"] / (norm_config["out_max"] - norm_config["out_min"])
        return scale * (data - norm_config["out_min"]) + stats["min"]
    else:
        raise ValueError(f"[denormalize_data] Invalid normalization type: {norm_type}")


def _aggregate_data_seq_with_skip(data_seq, skip, agg_func):
    """
    Aggregate elements along the first axis (axis=0) in chunks of skip.

    Args:
        data_seq (np.ndarray): Input data sequence.
        skip (int): Group size.
        agg_func (callable): A function that takes a group (shape: (skip, ...)) as input and returns the aggregated result.

    Returns:
        np.ndarray: New data sequence where the specified function has been applied to grouped elements along the first axis.
    """
    # Add zero padding to the first axis
    size = data_seq.shape[0]
    padding = (skip - size % skip) % skip
    if padding > 0:
        pad_shape = [(0, padding)] + [(0, 0)] * (data_seq.ndim - 1)
        data_seq = np.pad(
            data_seq, pad_width=pad_shape, mode="constant", constant_values=0
        )

    # Reshape into skip groups along the first axis
    grouped_data_seq = data_seq.reshape(-1, skip, *data_seq.shape[1:])

    # Aggregate array by applying the user-defined function to each group
    aggregated_data_seq = np.array([agg_func(arr) for arr in grouped_data_seq])

    return aggregated_data_seq


def get_skipped_data_seq(data_seq, key, skip):
    """Get skipped data sequence."""
    if skip == 1:
        return data_seq

    if key in (
        DataKey.MEASURED_JOINT_POS_REL,
        DataKey.COMMAND_JOINT_POS_REL,
        DataKey.MEASURED_GRIPPER_JOINT_POS_REL,
        DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
    ):

        def sum_regular(arr):
            return np.sum(arr, axis=0)

        skipped_data_seq = _aggregate_data_seq_with_skip(data_seq, skip, sum_regular)
    elif key in (
        DataKey.MEASURED_EEF_POSE_REL,
        DataKey.COMMAND_EEF_POSE_REL,
    ):

        def sum_rel_pose(arr):
            def add_rel_pose(rel_pose_arr1, rel_pose_arr2):
                return np.concatenate(
                    [
                        get_rel_pose_from_se3(
                            get_se3_from_rel_pose(rel_pose1)
                            * get_se3_from_rel_pose(rel_pose2)
                        )
                        for rel_pose1, rel_pose2 in zip(
                            rel_pose_arr1.reshape(-1, 6), rel_pose_arr2.reshape(-1, 6)
                        )
                    ]
                )

            return reduce(add_rel_pose, arr)

        skipped_data_seq = _aggregate_data_seq_with_skip(data_seq, skip, sum_rel_pose)
    else:
        skipped_data_seq = data_seq[::skip]

    return skipped_data_seq


def get_skipped_single_data(data_seq, time_idx, key, skip):
    """Get skipped single data."""

    return get_skipped_data_seq(data_seq[time_idx : time_idx + skip], key, skip)[0]
