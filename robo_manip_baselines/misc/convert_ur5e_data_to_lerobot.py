"""
Script to convert UR5e hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import json

import einops
import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "base",
        "shoulder",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "gripper",
    ]
    cameras = [
        "front_rgb",
        "side_rgb",
        "hand_rgb",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/measured_joint_vel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_images_per_camera_from_mp4(ep_path: Path, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        video_path = ep_path.parent / f"{camera}_image.rmb.mp4"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        array = np.reshape(frame,(1,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3))
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = np.reshape(frame,(1,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3))       
            array = np.append(array,frame,axis=0)
        cap.release()

        imgs_per_cam[camera] = array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/measured_joint_pos"][:])
        action = torch.from_numpy(ep["/command_joint_pos"][:])

        velocity = None
        if "/measured_joint_vel" in ep:
            velocity = torch.from_numpy(ep["/measured_joint_vel"][:])

        effort = None
        if "/effort" in ep:
            effort = torch.from_numpy(ep["/effort"][:])

    imgs_per_cam = load_raw_images_per_camera_from_mp4(
        ep_path,
        [
            "front_rgb",
            "side_rgb",
            "hand_rgb",
        ],
    )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        raw_file_string = ep_path.parents[1].name
        if "MujocoUR5eCable" in raw_file_string:
            task = "pass the cable between two poles"
        elif "MujocoUR5eRing" in raw_file_string:
            task = "pick a ring and put it around the pole"
        elif "MujocoUR5eParticle" in raw_file_string:
            task = "scoop up particles"
        elif "MujocoUR5eCloth" in raw_file_string:
            task = "roll up the cloth"
        elif "MujocoUR5eDoor" in raw_file_string:
            task = "open the door"

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        if ep_idx == 0:
            with open(dataset.root / "meta" / "modality.json", "w") as f:
                modality = {
                    "state": {
                        "qpos": {
                            "start": 0,
                            "end": 7
                        }
                    },
                    "action": {
                        "action": {
                            "start": 0,
                            "end": 7
                        }
                    },
                    "video": {
                        "front_rgb": {
                            "original_key": "observation.images.front_rgb"
                        },
                        "side_rgb": {
                            "original_key": "observation.images.side_rgb"
                        },
                        "hand_rgb": {
                            "original_key": "observation.images.hand_rgb"
                        }
                    },
                    "annotation": {
                        "human.action.task_description": {
                            "original_key": "task_index"
                        }
                    }
                }
                json.dump(modality, f, indent=4)

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert c < h and c < w, f"expect channel first images, but instead {batch[key].shape}"

            # sanity check that images are float32 in range [0,1]
            assert batch[key].dtype == torch.float32, f"expect torch.float32, but instead {batch[key].dtype=}"
            assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def create_seeded_dataloader(dataset, batch_size, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    return dataloader


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    serialized_dict = {key: value.tolist() for key, value in flatten_dict(stats).items()}
    return unflatten_dict(serialized_dict)


def port_ur5e(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    hdf5_files = sorted(raw_dir.glob("./*/main.rmb.hdf5"))

    dataset = create_empty_dataset(
        repo_id,
        robot_type="ur5e",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
    )
    dataset.consolidate()

    meta_stats = dataset.meta.stats

    stats_patterns = get_stats_einops_patterns(dataset, 8)

    data_num = len(dataset)
    q01, q99 = {}, {}
    data_dir = {}
    
    for key, pattern in stats_patterns.items():
        if key in dataset.meta.camera_keys:
            continue
        data_dir[key] = []
        for i in range(data_num):
            data_dir[key].append(dataset[i][key].float())
        data_dir[key] = torch.stack(data_dir[key], dim=0)
        
        q01[key] = torch.quantile(data_dir[key], 0.01, 0)
        q99[key] = torch.quantile(data_dir[key], 0.99, 0)
    
    for key in stats_patterns:
        if key in dataset.meta.camera_keys:
            continue
        meta_stats[key]["q01"] = q01[key]
        meta_stats[key]["q99"] = q99[key]

    serialized_stats = serialize_dict(meta_stats)
    
    with open(dataset.root / "meta" / "stats.json", "w") as f:
        json.dump(serialized_stats, f, indent=4)

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_ur5e)
