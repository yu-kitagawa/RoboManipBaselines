import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
    normalize_data,
)


class DiffusionPolicy3dDataset(DatasetBase):
    """Dataset to train 3D diffusion policy."""

    def setup_variables(self):
        # Set chunk_info_list
        self.chunk_info_list = []
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        # Set pad_before and pad_after to values one less than n_obs_steps and n_action_steps, respectively
        # Ref: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/task/pusht_image.yaml#L36-L37
        pad_before = self.model_meta_info["data"]["n_obs_steps"] - 1
        pad_after = self.model_meta_info["data"]["n_action_steps"] - 1

        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                for start_time_idx in range(
                    -1 * pad_before, episode_len - (horizon - 1) + pad_after
                ):
                    self.chunk_info_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load pointcloud
            camera_name = self.model_meta_info["image"]["camera_names"][0]
            pointcloud = rmb_data[DataKey.get_pointcloud_key(camera_name)][::skip][
                time_idxes
            ]

        # Pre-convert data
        state, action, pointcloud = self.pre_convert_data(state, action, pointcloud)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        pointcloud_tensor = torch.tensor(pointcloud, dtype=torch.float32)

        # Augment data
        state_tensor, action_tensor, _ = self.augment_data(
            state_tensor, action_tensor, None
        )

        # Convert to data structure of policy input and output
        data = {"obs": {}, "action": action_tensor}
        if len(self.model_meta_info["state"]["keys"]) > 0:
            data["obs"]["state"] = state_tensor
        data["obs"]["point_cloud"] = pointcloud_tensor
        return data

    def pre_convert_data(self, state, action, pointcloud):
        """Pre-convert data. Arguments must be numpy arrays (not torch tensors)."""
        state = normalize_data(state, self.model_meta_info["state"])
        action = normalize_data(action, self.model_meta_info["action"])
        pointcloud = normalize_data(pointcloud, self.model_meta_info["pointcloud"])

        return state, action, pointcloud
